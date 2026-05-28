import matplotlib.pyplot as plt
import os
import argparse
import cv2

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


def log_progress(output_dir, epoch, total_epochs, step, loss, gpu_memory):
  import json
  with open(f"{output_dir}/progress.txt", "w") as f:
    f.write(f"total_epochs: {total_epochs}\n")
    f.write(f"epoch: {epoch}\n")
    f.write(f"step: {step}\n")
    f.write(f"loss: {loss}\n")
    f.write(f"gpu_memory: {json.dumps(gpu_memory)}\n")

def plot_graphs_log_data(loss_per_step, time_per_epoch, time_per_step, gpu_utilizations, batch_size, accumulation_steps, total_time_taken, world_size, output_dir):
  plot_graph(loss_per_step, "Training Loss", "Step", "Loss", "loss.png", world_size, output_dir)
  plot_graph(time_per_epoch, "Time per Epoch", "Epoch", "Time (s)", "epoch_time.png", world_size, output_dir)
  plot_graph(time_per_step, "Time per Step", "Step", "Time (ms)", "step_time.png", world_size, output_dir)
  per_gpu_utils = create_n_array(gpu_utilizations)
  for i, gpu_util in enumerate(per_gpu_utils):
    plot_graph(gpu_util, f"GPU Util {i}", "Step", "%", f"gpu_{i}.png", world_size, output_dir)
  avg_step_time_s = sum(time_per_step) / len(time_per_step) / 1000 # divide by 1000 to convert ms to s
  global_batch_size = batch_size * world_size
  effective_batch_size = global_batch_size * accumulation_steps
  throughput = effective_batch_size / avg_step_time_s
  with open(f"{output_dir}/plots_GPU_{world_size}/training_metrics.txt", "w") as f:
    f.write(f"Throughput: {throughput}\n")
    f.write(f"Total time taken: {total_time_taken}\n")
    f.write(f"Average step time: {avg_step_time_s}\n")
    for i in range(len(per_gpu_utils)):
      avg_util = sum(per_gpu_utils[i]) / len(per_gpu_utils[i])
      f.write(f"Average GPU utilization for GPU {i}: {avg_util}\n")

def concat_images(gpu_count, output_dir):
  folder_path = f'{output_dir}/plots_GPU_{gpu_count}'
  reads = []
  for images in os.listdir(folder_path):
    if images.startswith('gpu'):
      full_image_path = os.path.join(folder_path, images)
      reads.append(cv2.imread(full_image_path))
  concat = cv2.hconcat(reads)
  cv2.imwrite(f'{output_dir}/plots_GPU_{gpu_count}/plot_GPU_{gpu_count}.png',concat)

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
    return parser.parse_args()