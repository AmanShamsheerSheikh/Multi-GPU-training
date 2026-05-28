import matplotlib.pyplot as plt
import os
import argparse
import cv2
import wandb

def create_n_array(gpu_utils):
  per_gpu_util = []
  temp_arr= []
  for i in range(len(gpu_utils[0])):
    temp_arr = []
    for j in range(len(gpu_utils)):
      temp_arr.append(gpu_utils[j][i])
    per_gpu_util.append(temp_arr)
  return per_gpu_util

def plot_graph(value, title, xlabel, ylabel, filename, world_size, output_dir):
  plt.figure()
  plt.plot(value)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  folder = f"{output_dir}/plots_GPU_{world_size}"
  os.makedirs(folder, exist_ok=True)
  plot_path = f"{folder}/{filename}"
  plt.savefig(plot_path)
  plt.close()

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

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DDP Training Cluster")
    parser.add_argument(
      "--job_id", 
      type=str,
      required=True, 
      help="Id of the job."
    )
    parser.add_argument(
      "--model_name", 
      type=str,
      required=True, 
      help="The specific model architecture to train."
    )
    parser.add_argument(
      "--dataset_name", 
      type=str, 
      required=True, 
      help="The dataset name for training or tuning"
    )
    parser.add_argument(
      "--job_type", 
      type=str, 
      required=True, 
      choices=['train', 'tune'],
      help="Training or tuning"
    )
    parser.add_argument(
      "--epochs", 
      type=int, 
      default=10, 
      help="Number of epochs to train."
    )
    parser.add_argument(
      "--text_column_name", 
      type=str, 
      required=True, 
      help="text column name of dataset"
    )
    parser.add_argument(
      "--gpu_count", 
      type=int, 
      default=1,
      help="Number of gpu used"
    )
    parser.add_argument(
      "--task_type", 
      type=str, 
      default='',
      help="finetuning model type"
    )
    parser.add_argument(
      "--batch_size", 
      type=int,
      help="batchsize for training"
    )
    parser.add_argument(
      "--accumulation_steps", 
      type=int,
      default=1,
      help="accumulation steps for training"
    )
    parser.add_argument(
      "--hf_repo_id", 
      type=str,
      help="hugging face model repo"
    )
    parser.add_argument(
      "--dataloader_workers", 
      type=int,
      default=1,
      help="workers for dataloader"
    )
    return parser.parse_args()