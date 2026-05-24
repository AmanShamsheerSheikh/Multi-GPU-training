import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
import pynvml
from tqdm import tqdm
from helper import plot_graphs_log_data, parse_args, concat_images
from constants import TRAIN, TUNE, ACCUMULATION_STEPS, BATCH_SIZE, DATA_LOADER_SEED, DATA_LOADER_BUFFER_SIZE
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


"""DDP Code"""

def setup_ddp():
  """
  tells that nccl will be used for communication between GPU's
  """
  dist.init_process_group(
    backend="nccl"
  )

def get_device():
  """
  When the code is ran each process run the main hence this function gets called n times for n Processes
  so each GPU in a node gets assign depending on its index given by cuda and the local_rank
  """
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  return torch.device("cuda", local_rank)

def get_dataloader(batch_size, dataset, world_size, rank, epoch):
  sharded = dataset.shard(num_shards=world_size, index=rank)
  shuffled = sharded.shuffle(seed=DATA_LOADER_SEED + epoch, buffer_size=DATA_LOADER_BUFFER_SIZE)
  dataloader = DataLoader(
    shuffled,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True
  )
  return dataloader

def get_model(device, model):
  model = model.to(device)
  model = DDP(model, device_ids=[device.index])
  return model

def check_gradients(model):
  with torch.no_grad():
    p = next(model.parameters()).grad

    grad_max = p.clone()
    grad_min = p.clone()

    dist.all_reduce(grad_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(grad_min, op=dist.ReduceOp.MIN)

    max_diff = (grad_max - grad_min).abs().max()

    if dist.get_rank() == 0:
      print("Gradient max diff:", max_diff.item())

def get_gpu_util(gpu_number):
  handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_number)  # GPU 0
  util = pynvml.nvmlDeviceGetUtilizationRates(handle)
  return util.gpu


def train(model, dataset, accumulation_steps, device, world_size, epochs, batch_size):
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
  )
  loss_per_step = []
  time_per_step = []
  time_per_epoch = []
  gpu_utilizations = []
  starter = torch.cuda.Event(enable_timing=True)
  ender = torch.cuda.Event(enable_timing=True)
  rank = dist.get_rank()
  for epoch in range(epochs):
    dataloader = get_dataloader(batch_size, dataset, world_size, rank, epoch)
    # sampler.set_epoch(epoch)  # it sets the epoch for the shuffling of the images per epoch 
    i = 0
    if rank == 0:
      epoch_start = time.time()
      pbar = tqdm(desc=f"Epoch {epoch+1}/{epochs}")
    optimizer.zero_grad(set_to_none=True)

    for batch in dataloader:
      input_ids = batch["input_ids"].to(device, non_blocking=True)
      attention_mask = batch["attention_mask"].to(device, non_blocking=True)
      labels = batch["labels"].to(device, non_blocking=True)
      if rank == 0:
        torch.cuda.synchronize()
        starter.record()
      if (i + 1) % accumulation_steps == 0:
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          labels=labels
        )
        raw_loss = outputs.loss
        loss = raw_loss / accumulation_steps # later read a bit more why this is done.
        loss.backward()   # all-reduce happens here
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
      else:
        with model.no_sync():
          outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
          )
          raw_loss = outputs.loss
          loss = raw_loss / accumulation_steps
          loss.backward()
      if epoch == 0 and i == 0:
        check_gradients(model)

      if rank == 0:
        ender.record()
        torch.cuda.synchronize()
        time_per_step.append(starter.elapsed_time(ender))

      with torch.no_grad():
        reduced_loss = raw_loss.detach()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss /= world_size
        if i%10 == 0 and dist.get_rank() == 0:
          utils = [get_gpu_util(j) for j in range(torch.cuda.device_count())]
          gpu_utilizations.append(utils)
      if rank == 0:
        loss_per_step.append(reduced_loss.item())
        pbar.update(1)
        pbar.set_postfix(loss=reduced_loss.item())
      i += 1
    if dist.get_rank() == 0:
      torch.cuda.synchronize()
      time_per_epoch.append(time.time() - epoch_start)
      pbar.close()
  if rank == 0:
    return loss_per_step, time_per_epoch, time_per_step, gpu_utilizations
  else:
    return None, None, None, None
  
def preprocess_dataset(dataset, tokenizer, text_column: str):
  def tokenize(batch):
      tokens = tokenizer(
          batch[text_column],
          truncation=True,
          max_length=512,
          padding="max_length",
      )
      labels = tokens["input_ids"].copy()
      tokens["labels"] = [
        [-100 if t == tokenizer.pad_token_id else t for t in label]
        for label in labels
      ]
      return tokens
  sample = next(iter(dataset))
  original_columns = list(sample.keys())
  return dataset.map(tokenize, batched=True, remove_columns=original_columns)
  
def load_model_struct_dataset(model_name, dataset_name, text_column):
  config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
  model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
  dataset = load_dataset(
    dataset_name,
    streaming=True,
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  dataset = preprocess_dataset(dataset, tokenizer, text_column)
  return model, dataset

def main(model, dataset, job_type, epochs, gpu_count):
  if job_type == TRAIN:
    setup_ddp()
    pynvml.nvmlInit()
    world_size = dist.get_world_size()
    batch_size= BATCH_SIZE // world_size
    accumulation_steps = ACCUMULATION_STEPS
    device = get_device()
    model = get_model(device, model)
    model.train()
    training_start_time = time.time()
    loss_per_step, time_per_epoch, time_per_step, gpu_utilizations = train(model, dataset, accumulation_steps, device, world_size, epochs, batch_size)
    total_time_taken = time.time() - training_start_time
    if dist.get_rank() == 0:
      plot_graphs_log_data(loss_per_step, time_per_epoch, time_per_step, gpu_utilizations, batch_size, accumulation_steps, total_time_taken, world_size)
      concat_images(gpu_count)
      torch.save(model.module.state_dict(), f'./model_{dist.get_rank()}_{world_size}.pt')
    dist.destroy_process_group()
  elif job_type == TUNE:
    print('coming soon')

if __name__ == '__main__':
  args = parse_args()
  model, dataset = load_model_struct_dataset(args.model_name, args.dataset_name, args.text_column_name)
  main(model, dataset, args.job_type, args.epochs, args.gpu_count)