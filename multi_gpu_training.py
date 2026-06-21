from contextlib import nullcontext
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
from constants import TUNE, DATA_LOADER_SEED, DATA_LOADER_BUFFER_SIZE, CASUAL_LM, SFT, CHAT, ddp, fsdp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import wandb
from huggingface_hub import HfApi
from torch.profiler import profile, ProfilerActivity, schedule
import importlib
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
  FullStateDictConfig,
  FullyShardedDataParallel,
  ShardedStateDictConfig,
  StateDictType
)
from torch.optim.lr_scheduler import LambdaLR
import json
import threading
from huggingface_hub import hf_hub_download, list_repo_files
from torch.cuda import memory_allocated, max_memory_allocated, reset_peak_memory_stats
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(
  level=logging.INFO,
  format="[%(asctime)s] [rank %(process)d] %(message)s"
)
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
upload_thread = None
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.fsdp")
from torch.utils.data import DistributedSampler

"""DDP Code"""

def setup_communication():
  """
  tells that nccl will be used for communication between GPU's
  """
  # RunPod-specific workaround: NCCL P2P/CUMEM transport deadlock on SYS-topology
  # dual-L4 environments. Disable P2P to force fallback to SHM/NET transport.
  # Remove this if running on NVLink-connected GPUs (A100, H100) — P2P is beneficial there.
  # os.environ["NCCL_P2P_DISABLE"] = "1"
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
  if hasattr(dataset, '__len__'):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    sampler.set_epoch(epoch)
    dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      sampler=sampler,
      num_workers=num_workers,
      pin_memory=True
    )
  else:
    sharded = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    shuffled = sharded.shuffle(seed=DATA_LOADER_SEED + epoch, buffer_size=DATA_LOADER_BUFFER_SIZE)
    dataloader = DataLoader(
      shuffled,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True
    )
  return dataloader

def get_gpu_util(gpu_number) -> int:
  handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_number)  # GPU 0
  util = pynvml.nvmlDeviceGetUtilizationRates(handle)
  return util.gpu

def get_profiler(output_dir):
  profiler_dir = f'{output_dir}/profiler'
  os.makedirs(profiler_dir, exist_ok=True)
  return profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
      wait=1,      # skip first step
      warmup=1,    # warmup for one step
      active=3,    # profile 3 steps
      repeat=2     # do this once
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
      profiler_dir
    ),
    record_shapes=True,
    profile_memory=True,
    with_stack=False
  )

def upload_to_hf(save_dir, repo_id, index):
  try:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_folder(
      folder_path=save_dir,
      repo_id=repo_id,
      repo_type="model",
      path_in_repo=f'checkpoint/slot_{index}',
      commit_message=f"checkpoint slot {index}"
    )
  except Exception as e:
    print("Error :", e)

def save_checkpoint(model, optimizer, scheduler, rank, index, loss, epoch, global_step, upload_every_n_steps, hf_repo_id, training_type):
  save_dir = f'./checkpoint/slot_{index}/'
  os.makedirs(save_dir, exist_ok=True)
  global upload_thread
  if training_type == fsdp:
    sharded_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
      sharded_state = model.state_dict()
      optim_state = FullyShardedDataParallel.optim_state_dict(model, optimizer)
    torch.save(sharded_state, f'{save_dir}/shard_{rank}.pt')
    torch.save(optim_state, f'{save_dir}/optimizer_{rank}.pt')

    if rank == 0:
      torch.save(scheduler.state_dict(), f'{save_dir}/scheduler.pt')
      data = {
        'loss': loss,
        'global_step': global_step,
        'epoch': epoch
      }
      with open(f"{save_dir}/meta.json", "w") as f:
        json.dump(data, f, indent=4)
    dist.barrier()
    if global_step % upload_every_n_steps == 0:
      if rank == 0:
        upload_thread = threading.Thread(target=upload_to_hf, args=(save_dir, hf_repo_id, index))
        upload_thread.daemon = True
        upload_thread.start()
  elif training_type == ddp:
    if rank == 0:
      torch.save(model.module.state_dict(), f'{save_dir}/model.pt')
      torch.save(optimizer.state_dict(), f'{save_dir}/optimizer.pt')
      torch.save(scheduler.state_dict(), f'{save_dir}/scheduler.pt')
      data = {
        'loss': loss,
        'global_step': global_step,
        'epoch': epoch
      }
      with open(f"{save_dir}/meta.json", "w") as f:
        json.dump(data, f, indent=4)
      if global_step % upload_every_n_steps == 0:
        upload_thread = threading.Thread(target=upload_to_hf, args=(save_dir, hf_repo_id, index))
        upload_thread.daemon = True
        upload_thread.start()

def load_checkpoint(model, optimizer, scheduler, rank, device, repo_id, training_type):
  logger.info("Checking for checkpoint")
  hf_token = os.environ.get("HF_TOKEN")
  try:
    files = list(list_repo_files(repo_id, token=hf_token))
  except Exception:
    logger.info("No checkpoint found")
    return None, model, optimizer, scheduler 

  latest_slot = None
  latest_step = -1
  for slot in range(2):
    try:
      meta_path = hf_hub_download(repo_id, f'checkpoint/slot_{slot}/meta.json', token=hf_token)
      with open(meta_path) as f:
        meta = json.load(f)
      if meta['global_step'] > latest_step:
        latest_step = meta['global_step']
        latest_slot = slot
        latest_meta = meta
    except Exception:
      continue

  if latest_slot is None:
    logger.info("No checkpoint found")
    return None, model, optimizer, scheduler
  if training_type == fsdp:
    logger.info("Loading checkpoint.")
    model_path = hf_hub_download(repo_id, f'checkpoint/slot_{latest_slot}/shard_{rank}.pt', token=hf_token)
    optim_path = hf_hub_download(repo_id, f'checkpoint/slot_{latest_slot}/optimizer_{rank}.pt', token=hf_token)

    sharded_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
      sharded_state = torch.load(model_path, map_location='cpu', mmap=True)
      model.load_state_dict(sharded_state)

      optim_state = torch.load(optim_path, map_location='cpu', mmap=True)
      optim_state = FullyShardedDataParallel.optim_state_dict_to_load(model, optimizer, optim_state)
      optimizer.load_state_dict(optim_state)

    if rank == 0:
      sched_path = hf_hub_download(repo_id, f'checkpoint/slot_{latest_slot}/scheduler.pt', token=hf_token)
      scheduler_state = torch.load(sched_path, map_location='cpu', mmap=True)
    else:
      scheduler_state = None
    scheduler_state_list = [scheduler_state]
    dist.broadcast_object_list(scheduler_state_list, src=0)
    scheduler.load_state_dict(scheduler_state_list[0])
    dist.barrier()
    logger.info("Checkpoint Loaded")
    return latest_meta, model, optimizer, scheduler
  else:
    # All ranks download independently and simultaneously
    model_path = hf_hub_download(repo_id, f'checkpoint/slot_{latest_slot}/model.pt', token=hf_token)
    optim_path = hf_hub_download(repo_id, f'checkpoint/slot_{latest_slot}/optimizer.pt', token=hf_token)
    sched_path = hf_hub_download(repo_id, f'checkpoint/slot_{latest_slot}/scheduler.pt', token=hf_token)

    model_state = torch.load(model_path, map_location='cpu')
    model.module.load_state_dict(model_state)

    optim_state = torch.load(optim_path, map_location='cpu')
    optimizer.load_state_dict(optim_state)

    scheduler_state = torch.load(sched_path, map_location='cpu')
    scheduler.load_state_dict(scheduler_state)

    dist.barrier()
    logger.info("Checkpoint Loaded")
    return latest_meta, model, optimizer, scheduler


def train(training_type, model, dataset, accumulation_steps, device, world_size, epochs, batch_size, optimizer, output_dir, num_workers, scheduler, save_every_n_steps, upload_every_n_steps, hf_repo_id, max_steps):
  logger.info("Training started")
  time_per_step = []
  gpu_utilizations = []
  starter = torch.cuda.Event(enable_timing=True)
  ender = torch.cuda.Event(enable_timing=True)
  rank = dist.get_rank()
  amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  prof = get_profiler(output_dir) if rank == 0 else None
  start_epoch = 0
  global_step = 0
  if prof:
    prof.start()
  index_to_save = 0
  is_memory_stats_stored = False
  pbar = tqdm(desc="Training", disable=(rank != 0))
  for epoch in range(start_epoch, epochs):
    logger.info(f"epoch: {epoch}")
    dataloader = get_dataloader(batch_size, dataset, world_size, rank, epoch, num_workers)
    i = 0
    if rank == 0:
      epoch_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for batch in dataloader:
      input_ids = batch["input_ids"].to(device, non_blocking=True)
      attention_mask = batch["attention_mask"].to(device, non_blocking=True)
      labels = batch["labels"].to(device, non_blocking=True)

      if rank == 0 and i % accumulation_steps == 0:
        torch.cuda.synchronize()
        starter.record()

      if (i + 1) % accumulation_steps == 0:
        if not is_memory_stats_stored:
          reset_peak_memory_stats()
          baseline = memory_allocated() / 1024 / 1024
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
          outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
          )
        raw_loss = outputs.loss
        loss = raw_loss / accumulation_steps
        if not is_memory_stats_stored:
          post_forward = memory_allocated() / 1024 / 1024
          loss.backward()
          post_backward = memory_allocated() / 1024 / 1024
          peak = max_memory_allocated() / 1024 / 1024
          is_memory_stats_stored = True
          if rank == 0: 
            write_model_details(output_dir, baseline, post_forward, post_backward, peak)
        else:
          loss.backward()
        if training_type == fsdp:
          model.clip_grad_norm_(1.0)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
          ender.record()
          torch.cuda.synchronize()
          time_per_step.append(starter.elapsed_time(ender))
          wandb.log({"time_per_step": starter.elapsed_time(ender)})
      else:
        ctx = model.no_sync() if training_type == ddp  else nullcontext()
        with ctx:
          with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              labels=labels
            )
          raw_loss = outputs.loss
          loss = raw_loss / accumulation_steps
        

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
      if rank == 0:
        pbar.update(1)
        pbar.set_postfix({"loss": f"{reduced_loss.item():.4f}", "step": global_step, "epoch": epoch})
      if training_type == ddp and rank == 0:
        if global_step % save_every_n_steps == 0:
          save_checkpoint(model, optimizer, scheduler, rank, index_to_save, reduced_loss.item(), epoch, global_step, upload_every_n_steps, hf_repo_id, training_type)
          index_to_save = 0 if index_to_save == 1 else 1
      else:
        if global_step % save_every_n_steps == 0:
          save_checkpoint(model, optimizer, scheduler, rank, index_to_save, reduced_loss.item(), epoch, global_step, upload_every_n_steps, hf_repo_id, training_type)
          index_to_save = 0 if index_to_save == 1 else 1

      if prof:
        prof.step()
      if global_step >= max_steps:
        break
    remaining_steps = i % accumulation_steps
    if remaining_steps != 0:
      if training_type == fsdp:
        model.clip_grad_norm_(1.0)
      else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step() 
      optimizer.zero_grad(set_to_none=True)
    if rank == 0:
      torch.cuda.synchronize()
      time_step = time.time() - epoch_start
      wandb.log({"time_per_epoch": time_step})
    if global_step >= max_steps:
      break
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
  
def load_model_struct_dataset(training_type, model_name, dataset_name, columns, job_type, task_type, max_length):
  config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
  if training_type == ddp:
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
  else:
    with torch.device("meta"):
      model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

  if os.path.exists(dataset_name):
    logger.info("Loading from disk")
    from datasets import load_from_disk
    dataset = load_from_disk(dataset_name)
    if hasattr(dataset, "keys") and callable(dataset.keys):
      split_name = 'train' if 'train' in dataset.keys() else list(dataset.keys())[0]
      dataset = dataset[split_name]
    dataset = dataset.select(range(10000))
  else:
    dataset = load_dataset(dataset_name, streaming=True)
    if hasattr(dataset, "keys") and callable(dataset.keys):
      split_name = 'train' if 'train' in dataset.keys() else list(dataset.keys())[0]
      dataset = dataset[split_name]

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  dataset = preprocess_dataset(dataset, tokenizer, columns, task_type, max_length)
  return model, dataset

def get_wrap_policy(model):
  if not hasattr(model, "_no_split_modules"):
    raise ValueError(
      f"{model.__class__.__name__} does not define _no_split_modules"
    )

  layer_classes = set()

  module = importlib.import_module(model.__class__.__module__)

  for layer_name in model._no_split_modules:
    layer_classes.add(
      getattr(module, layer_name)
    )

  return functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls=layer_classes,
  )

def get_scheduler(optimizer, warmup_steps, max_steps):
  def lr_lambda(step):
      if step < warmup_steps:
          return step / warmup_steps  # linear warmup
      return max(0.0, (max_steps - step) / (max_steps - warmup_steps))  # linear decay
  return LambdaLR(optimizer, lr_lambda)

def write_model_details(output_dir, baseline, post_forward, post_backward, peak):
  os.makedirs(f"{output_dir}/logs", exist_ok=True)
  with open(f"{output_dir}/logs/memory_stats.txt", "w") as f:
    f.write(f"Weights + Optimizer: {baseline:.2f} MB\n")
    f.write(f"Activations: {post_forward - baseline:.2f} MB\n")
    f.write(f"Gradients: {post_backward - post_forward:.2f} MB\n")
    f.write(f"Peak Total: {peak:.2f} MB\n")

def main(training_type, model, dataset, epochs, output_dir, job_id, model_name, dataset_name, job_type, batch_size, accumulation_steps, hf_repo_id, num_workers, max_steps, warmup_steps, save_every_n_steps, upload_every_n_steps, peak_theoretical_flops, seq_length):
  logger.info("Main function")
  setup_communication()
  pynvml.nvmlInit()
  world_size = dist.get_world_size()
  batch_size= batch_size
  accumulation_steps = accumulation_steps
  device = get_device()
  if training_type == ddp:
    model = model.to(device)
    model = DDP(model, device_ids=[device.index])
  elif training_type == fsdp:
    model = FullyShardedDataParallel(
      model,
      auto_wrap_policy=get_wrap_policy(model),
      device_id=device,
      sync_module_states=False,
      param_init_fn=lambda m: m.to_empty(device=device, recurse=False)
    )
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
  scheduler = get_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps)
  meta, model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, dist.get_rank(), device, hf_repo_id, training_type)
  model.train()
  training_start_time = time.time()
  if dist.get_rank() == 0:
    wandb_api_key = os.getenv("wandb_api_key")
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
  time_per_step, gpu_utilizations = train(training_type, model, dataset, accumulation_steps, device, world_size, epochs, batch_size, optimizer, output_dir, num_workers, scheduler, save_every_n_steps, upload_every_n_steps, hf_repo_id, max_steps)
  total_time_taken = time.time() - training_start_time
  if training_type == fsdp:
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
      state_dict = model.state_dict()
      optim_state = FullyShardedDataParallel.optim_state_dict(model, optimizer)
  if dist.get_rank() == 0:
    log_final_metrics(time_per_step, gpu_utilizations, batch_size, accumulation_steps, world_size, output_dir, peak_theoretical_flops, model, seq_length, warmup_steps)
    api = HfApi()
    model_dir = f'{output_dir}/model_final.pt'
    optim_dir = f'{output_dir}/optim_final.pt'
    scheduler_dir = f'{output_dir}/scheduler_final.pt'
    if training_type == ddp:
      torch.save(model.module.state_dict(), model_dir)
      torch.save(optimizer.state_dict(), optim_dir)
    else:
      torch.save(state_dict, model_dir)
      torch.save(optim_state, optim_dir)
    torch.save(scheduler.state_dict(), scheduler_dir) 
    api.upload_file(
      path_or_fileobj=model_dir,
      path_in_repo="model_final.pt",
      repo_id=hf_repo_id,
      token=os.environ.get("HF_TOKEN")
    )
    api.upload_file(
      path_or_fileobj=optim_dir,
      path_in_repo="optim_final.pt",
      repo_id=hf_repo_id,
      token=os.environ.get("HF_TOKEN")
    )
    api.upload_file(
      path_or_fileobj=scheduler_dir,
      path_in_repo="scheduler_final.pt",
      repo_id=hf_repo_id,
      token=os.environ.get("HF_TOKEN")
    )
    if os.path.exists(f'{output_dir}/profiler'):
      api.upload_folder(
        folder_path=f'{output_dir}/profiler',
        repo_id=hf_repo_id,
        path_in_repo='profiler',
        token=os.environ.get("HF_TOKEN")
      )
    if os.path.exists(f'{output_dir}/logs'):
      api.upload_folder(
        folder_path=f'{output_dir}/logs',
        repo_id=hf_repo_id,
        path_in_repo='logs',
        token=os.environ.get("HF_TOKEN")
      )
  if dist.get_rank() == 0 and upload_thread:
    upload_thread.join()
  dist.barrier()
  dist.destroy_process_group()
  pynvml.nvmlShutdown()

if __name__ == '__main__':
  args = parse_args()
  args = args.training_config
  output_dir = f"./data/{args.job_id}"
  os.makedirs(output_dir, exist_ok=True)
  model, dataset = load_model_struct_dataset(args.training_type, args.model_name, args.dataset_config.dataset_name, args.dataset_config.columns, args.job_type, args.dataset_config.task_type, args.dataset_config.max_length)
  main(args.training_type, model, dataset, args.epochs, output_dir, args.job_id, args.model_name, args.dataset_config.dataset_name, args.job_type, args.batch_size, args.accumulation_steps, args.hf_repo_id, args.dataloader_workers, args.max_steps, args.warmup_steps, args.save_every_n_steps, args.upload_every_n_steps, args.peak_theoretical_flops, args.dataset_config.max_length)