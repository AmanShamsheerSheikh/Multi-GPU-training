import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import time
import pynvml
from tqdm import tqdm
from helper import plot_graphs_log_data, parse_args, concat_images, log_progress
from constants import TUNE, ACCUMULATION_STEPS, BATCH_SIZE, DATA_LOADER_SEED, DATA_LOADER_BUFFER_SIZE
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

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
  sharded = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
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

def get_gpu_util(gpu_number) -> int:
  handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_number)  # GPU 0
  util = pynvml.nvmlDeviceGetUtilizationRates(handle)
  return util.gpu


def train(model, dataset, accumulation_steps, device, world_size, epochs, batch_size, optimizer, output_dir):
  loss_per_step = []
  time_per_step = []
  time_per_epoch = []
  gpu_utilizations = []
  starter = torch.cuda.Event(enable_timing=True)
  ender = torch.cuda.Event(enable_timing=True)
  rank = dist.get_rank()
  if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/plots_GPU_{world_size}', exist_ok=True)
  for epoch in range(epochs):
    dataloader = get_dataloader(batch_size, dataset, world_size, rank, epoch)
    # sampler.set_epoch(epoch)  # it sets the epoch for the shuffling of the images per epoch 
    i = 0
    if rank == 0:
      epoch_start = time.time()
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
          gpu_utilizations = []
          utils = [get_gpu_util(j) for j in range(torch.cuda.device_count())]
          gpu_utilizations.append(utils)
          log_progress(output_dir, epoch, epochs, i, reduced_loss.item(), gpu_utilizations)
      if rank == 0:
        loss_per_step.append(reduced_loss.item())
      i += 1
    if dist.get_rank() == 0:
      torch.cuda.synchronize()
      time_per_epoch.append(time.time() - epoch_start)
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
  original_columns = dataset.column_names
  dataset = dataset.with_format("torch")
  return dataset.map(tokenize, batched=True, remove_columns=original_columns)
  
def load_model_struct_dataset(model_name, dataset_name, text_column, job_type, task_type):
  if job_type == TUNE:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    from peft import get_peft_model, LoraConfig, TaskType
    from peft.utils import get_linear_names
    TASK_TYPE_MAP = {
      "causal_lm": TaskType.CAUSAL_LM,
      "seq2seq_lm": TaskType.SEQ_2_SEQ_LM,
      "image_classification": TaskType.IMAGE_CLASSIFICATION,
      "token_cls": TaskType.TOKEN_CLS,
      "seq_cls": TaskType.SEQ_CLS,
    }
    task_type_enum = TASK_TYPE_MAP.get(task_type, TaskType.CAUSAL_LM)
    lora_config = LoraConfig(
      task_type=task_type_enum,
      r=16,
      lora_alpha=32,
      lora_dropout=0.1,
      target_modules=get_linear_names(model),
    )
    model = get_peft_model(model, lora_config)
  else:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
  dataset = load_dataset(
    dataset_name,
    streaming=True,
  )
  if hasattr(dataset, "keys") and callable(dataset.keys):
    split_name = 'train' if 'train' in dataset.keys() else list(dataset.keys())[0]
    dataset = dataset[split_name]

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
      
  dataset = preprocess_dataset(dataset, tokenizer, text_column)
  return model, dataset, optimizer

def main(model, dataset, epochs, gpu_count, optimizer, output_dir):
  setup_ddp()
  pynvml.nvmlInit()
  world_size = dist.get_world_size()
  batch_size= BATCH_SIZE // world_size
  accumulation_steps = ACCUMULATION_STEPS
  device = get_device()
  model = get_model(device, model)
  model.train()
  training_start_time = time.time()
  loss_per_step, time_per_epoch, time_per_step, gpu_utilizations = train(model, dataset, accumulation_steps, device, world_size, epochs, batch_size, optimizer, output_dir)
  total_time_taken = time.time() - training_start_time
  if dist.get_rank() == 0:
    plot_graphs_log_data(loss_per_step, time_per_epoch, time_per_step, gpu_utilizations, batch_size, accumulation_steps, total_time_taken, world_size, output_dir)
    concat_images(gpu_count, output_dir)
    torch.save(model.module.state_dict(), f'{output_dir}/model_{dist.get_rank()}_{world_size}.pt')
  dist.destroy_process_group()

if __name__ == '__main__':
  args = parse_args()
  output_dir = f"/runpod-volume/{args.job_id}"
  model, dataset, optimizer = load_model_struct_dataset(args.model_name, args.dataset_name, args.text_column_name, args.job_type, args.task_type)
  main(model, dataset, args.epochs, args.gpu_count, optimizer, output_dir)