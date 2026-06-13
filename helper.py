import argparse
import json
import wandb
from dataclasses import dataclass

def create_n_array(gpu_utils):
  per_gpu_util = []
  temp_arr= []
  for i in range(len(gpu_utils[0])):
    temp_arr = []
    for j in range(len(gpu_utils)):
      temp_arr.append(gpu_utils[j][i])
    per_gpu_util.append(temp_arr)
  return per_gpu_util

def log_final_metrics(time_per_step, gpu_utilizations, batch_size, accumulation_steps, total_time_taken, world_size, output_dir):
  per_gpu_utils = create_n_array(gpu_utilizations)
  avg_step_time_s = sum(time_per_step) / len(time_per_step) / 1000 # divide by 1000 to convert ms to s
  global_batch_size = batch_size * world_size
  effective_batch_size = global_batch_size * accumulation_steps
  throughput = effective_batch_size / avg_step_time_s
  wandb.summary["throughput"] = throughput
  wandb.summary["total_time_taken"] = total_time_taken
  wandb.summary["avg_step_time"] = avg_step_time_s
  for i in range(len(per_gpu_utils)):
      avg_util = sum(per_gpu_utils[i]) / len(per_gpu_utils[i])
      wandb.summary[f"avg_gpu_util_GPU_{i}"] = avg_util
  wandb.finish()

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
  epochs: int = 10
  gpu_count: int = 1
  batch_size: int = 8
  accumulation_steps: int = 1
  hf_repo_id: str = ""
  dataloader_workers: int = 4

def parse_training_config(value):
  data = json.loads(value)
  data['dataset_config'] = DatasetConfig(**data['dataset_config'])
  return TrainingConfig(**data)

def parse_args():
  parser = argparse.ArgumentParser(description="PyTorch DDP Training Cluster")
  parser.add_argument(
    "--training_config",
    type=parse_training_config,
    required=True
  )
  return parser.parse_args()