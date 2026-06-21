import argparse
import json
import wandb
from dataclasses import dataclass
import os

def create_n_array(gpu_utils):
  per_gpu_util = []
  temp_arr= []
  for i in range(len(gpu_utils[0])):
    temp_arr = []
    for j in range(len(gpu_utils)):
      temp_arr.append(gpu_utils[j][i])
    per_gpu_util.append(temp_arr)
  return per_gpu_util

def log_final_metrics(time_per_step, gpu_utilizations, batch_size, accumulation_steps, world_size, output_dir, peak_theoretical_flops, model, seq_length, warump_steps):
  os.makedirs(f"{output_dir}/logs", exist_ok=True)
  per_gpu_utils = create_n_array(gpu_utilizations)
  time_per_step = time_per_step[warump_steps:]
  avg_step_time_s = sum(time_per_step) / len(time_per_step) / 1000 # divide by 1000 to convert ms to s
  global_batch_size = batch_size * world_size
  effective_batch_size = global_batch_size * accumulation_steps
  throughput = effective_batch_size / avg_step_time_s

  num_params = sum(p.numel() for p in model.parameters())
  peak_flops = peak_theoretical_flops * world_size  # total across all GPUs
  flops_per_sample = 6 * num_params * seq_length  # 6 * N * T is standard estimate
  flops_per_step = flops_per_sample * effective_batch_size
  mfu = flops_per_step / (avg_step_time_s * peak_flops) * 100
    
  with open(f"{output_dir}/logs/training_metadata.txt", "w") as f:
    f.write(f"throughput: {throughput:.2f} samples/sec\n")
    f.write(f"avg_step_time: {avg_step_time_s:.4f} sec\n")
    f.write(f"total_time_taken: {avg_step_time_s * len(time_per_step):.4f} sec\n")
    f.write(f"mfu: {mfu:.4f}%\n")
    for i in range(len(per_gpu_utils)):
      avg_util = sum(per_gpu_utils[i]) / len(per_gpu_utils[i])
      f.write(f"avg_gpu_util_GPU_{i}: {avg_util:.2f}%\n")

@dataclass
class DatasetConfig:
  dataset_name: str
  task_type: str
  columns: list[str]
  max_length: int | None = None

@dataclass
class TrainingConfig:
  training_type: str
  max_steps: int
  warmup_steps: int
  save_every_n_steps: int
  upload_every_n_steps: int
  job_id: str
  model_name: str
  job_type: str
  dataset_config: DatasetConfig
  peak_theoretical_flops: float
  epochs: int = 10
  gpu_count: int = 1
  batch_size: int = 8
  accumulation_steps: int = 1
  hf_repo_id: str = ""
  dataloader_workers: int = 4

def parse_training_config(value):
    if not os.path.exists(value):
      raise argparse.ArgumentTypeError(f"Config file not found at: {value}")
        
    try:
      with open(value, 'r') as f:
        data = json.load(f)

      data['dataset_config'] = DatasetConfig(**data['dataset_config'])

      return TrainingConfig(**data)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format in file {value}: {e}")
    except TypeError as e:
        raise argparse.ArgumentTypeError(f"JSON schema mismatch with dataclass fields: {e}")

def parse_args():
  parser = argparse.ArgumentParser(description="PyTorch DDP Training Cluster")
  parser.add_argument(
    "--training_config",
    type=parse_training_config,
    required=True
  )
  return parser.parse_args()