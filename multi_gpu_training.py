import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import time
import pynvml
from helper import log_final_metrics, parse_args
from constants import TUNE, DATA_LOADER_SEED, DATA_LOADER_BUFFER_SIZE, CASUAL_LM, SFT, CHAT
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import wandb
from huggingface_hub import HfApi
from torch.profiler import profile, ProfilerActivity, schedule
from dotenv import load_dotenv
load_dotenv()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

def get_dataloader(batch_size, dataset, world_size, rank, epoch, num_workers):
  sharded = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
  shuffled = sharded.shuffle(seed=DATA_LOADER_SEED + epoch, buffer_size=DATA_LOADER_BUFFER_SIZE)
  dataloader = DataLoader(
    shuffled,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True
  )
  return dataloader

def get_model(device, model):
  model = model.to(device)
  model = DDP(model, device_ids=[device.index])
  return model

def get_gpu_util(gpu_number) -> int:
  handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_number)  # GPU 0
  util = pynvml.nvmlDeviceGetUtilizationRates(handle)
  return util.gpu

def get_profiler(output_dir):
  return profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
      wait=1,      # skip first step
      warmup=1,    # warmup for one step
      active=3,    # profile 3 steps
      repeat=2     # do this once
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        f'{output_dir}/profiler'
    ),
    record_shapes=True,
    profile_memory=True,
    with_stack=False
  )

def train(model, dataset, accumulation_steps, device, world_size, epochs, batch_size, optimizer, output_dir, job_id, model_name, dataset_name, job_type, num_workers):
  time_per_step = []
  gpu_utilizations = []
  starter = torch.cuda.Event(enable_timing=True)
  ender = torch.cuda.Event(enable_timing=True)
  rank = dist.get_rank()
  if rank == 0:
    wandb_api_key = os.getenv("wabdb_api_key")
    wandb.login(key=wandb_api_key)
    wandb.init(
      project="distrain",
      name=job_id,
      config={
        "model": model_name,
        "dataset": dataset_name,
        "epochs": epochs,
        "step": 0,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "gpu_count": world_size,
        "job_type": job_type,
      }
    )
  global_step = 0
  amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  prof = get_profiler(output_dir) if rank == 0 else None
  if prof:
    prof.start()
  for epoch in range(epochs):
    dataloader = get_dataloader(batch_size, dataset, world_size, rank, epoch, num_workers)
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
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
          outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
          )
        raw_loss = outputs.loss
        loss = raw_loss / accumulation_steps
        loss.backward()   # all-reduce happens here
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
      else:
        with model.no_sync():
          with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              labels=labels
            )
          raw_loss = outputs.loss
          loss = raw_loss / accumulation_steps
          loss.backward()

      if rank == 0:
        ender.record()
        torch.cuda.synchronize()
        time_per_step.append(starter.elapsed_time(ender))
        wandb.log({"time_per_step": starter.elapsed_time(ender)})

      with torch.no_grad():
        reduced_loss = raw_loss.detach()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss /= world_size
        if i%10 == 0 and rank == 0:
          utils = [get_gpu_util(j) for j in range(torch.cuda.device_count())]
          gpu_utilizations.append(utils)
          wandb.log({
            "epoch": epoch,
            "step": global_step,
            "gpu_utilization": utils,
          })
          for j, util in enumerate(utils):
            wandb.log({f"gpu_util_rank_{j}": util})
      if rank == 0:
        wandb.log({"loss_per_step": reduced_loss.item()})
      i += 1
      global_step += 1
      if prof:
        prof.step()
    remaining_steps = i % accumulation_steps
    if remaining_steps != 0:
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
    if rank == 0:
      torch.cuda.synchronize()
      time_step = time.time() - epoch_start
      wandb.log({"time_per_epoch": time_step})
  if prof:
    prof.stop()
  if rank == 0:
    return time_per_step, gpu_utilizations
  else:
    return None, None
  
def get_tokens(tokenizer, input, isTruncate, max_length, padding):
  return tokenizer(
    input,
    truncation=isTruncate,
    max_length=max_length,
    padding=padding,
  )
  
def preprocess_dataset(dataset, tokenizer, columns, task_type, max_length):
  def tokenize(batch):
    if task_type == CASUAL_LM:
      tokens = get_tokens(tokenizer, batch[columns[0]], True, max_length if max_length else tokenizer.model_max_length, "max_length")
      labels = tokens["input_ids"].copy()
      tokens["labels"] = [
        [-100 if t == tokenizer.pad_token_id else t for t in label]
        for label in labels
      ]
      return tokens
    elif task_type == SFT:
      full_text = f"""
        ### Instruction:
        {batch[columns[0]]}

        ### Response:
        {batch[columns[1]]}
      """
      tokens = get_tokens(tokenizer, full_text, True, max_length if max_length else tokenizer.model_max_length, "max_length")
      input_ids = tokens["input_ids"]
      labels = input_ids.copy()
      prompt_text = f"""
        ### Instruction:
        {batch[columns[0]]}

        ### Response:
      """
      prompt_tokens =  tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length if max_length else tokenizer.model_max_length,
        add_special_tokens=False
      )["input_ids"]
      prompt_len = len(prompt_tokens)
      labels[:prompt_len] = [-100] * prompt_len
      labels = [
        -100 if token == tokenizer.pad_token_id else token
        for token in labels
      ]
      tokens["labels"] = labels
      return tokens
    elif task_type == CHAT:
      text = tokenizer.apply_chat_template(
        batch[columns[0]],
        tokenize=False
      )
      tokens = get_tokens(tokenizer, text, True, max_length if max_length else tokenizer.model_max_length, "max_length")
      labels = tokens["input_ids"].copy()
      labels = [
        -100 if token == tokenizer.pad_token_id else token
        for token in labels
      ]
      tokens["labels"] = labels
      return tokens
  original_columns = dataset.column_names
  if task_type == CASUAL_LM:
    dataset = dataset.map(tokenize, batched=True, remove_columns=original_columns)
  elif task_type == SFT or task_type == CHAT:
    dataset = dataset.map(tokenize, batched=False, remove_columns=original_columns)
  return dataset.with_format("torch")
  
def load_model_struct_dataset(model_name, dataset_name, columns, job_type, task_type, max_length):
  if job_type == TUNE:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
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
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
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
      
  dataset = preprocess_dataset(dataset, tokenizer, columns, task_type, max_length)
  return model, dataset

def main(model, dataset, epochs, output_dir, job_id, model_name, dataset_name, job_type, batch_size, accumulation_steps, hf_repo_id, num_workers):
  setup_ddp()
  pynvml.nvmlInit()
  world_size = dist.get_world_size()
  batch_size= batch_size
  accumulation_steps = accumulation_steps
  device = get_device()
  model = get_model(device, model)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
  model.train()
  training_start_time = time.time()
  time_per_step, gpu_utilizations = train(model, dataset, accumulation_steps, device, world_size, epochs, batch_size, optimizer, output_dir, job_id, model_name, dataset_name, job_type, num_workers)
  total_time_taken = time.time() - training_start_time
  if dist.get_rank() == 0:
    log_final_metrics(time_per_step, gpu_utilizations, batch_size, accumulation_steps, total_time_taken, world_size, output_dir)
    api = HfApi()
    model_dir = f'{output_dir}/model_{dist.get_rank()}_{world_size}.pt'
    torch.save(model.module.state_dict(), model_dir)
    api.upload_file(
      path_or_fileobj=model_dir,
      path_in_repo="model_final.pt",
      repo_id=hf_repo_id,
      token=os.environ.get("HF_TOKEN")
    )
    api.upload_folder(
      folder_path=f'{output_dir}/profiler',
      repo_id=hf_repo_id,
      path_in_repo='profiler',
      token=os.environ.get("HF_TOKEN")
    )
  dist.destroy_process_group()

if __name__ == '__main__':
  args = parse_args()
  args = args.training_config
  output_dir = f"/runpod-volume/{args.job_id}"
  model, dataset = load_model_struct_dataset(args.model_name, args.dataset_config.dataset_name, args.dataset_config.columns, args.job_type, args.dataset_config.task_type, args.dataset_config.max_length)
  main(model, dataset, args.epochs, output_dir, args.job_id, args.model_name, args.dataset_config.dataset_name, args.job_type, args.batch_size, args.accumulation_steps, args.hf_repo_id, args.dataloader_workers)