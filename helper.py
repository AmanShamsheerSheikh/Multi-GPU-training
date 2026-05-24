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

def plot_graph(value, title, xlabel, ylabel, filename, world_size):
  plt.figure()
  plt.plot(value)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  folder = f"../plots_GPU_{world_size}"
  os.makedirs(folder, exist_ok=True)
  plot_path = f"{folder}/{filename}"
  plt.savefig(plot_path)
  plt.close()

def plot_graphs_log_data(loss_per_step, time_per_epoch, time_per_step, gpu_utilizations, batch_size, accumulation_steps, total_time_taken, world_size):
  plot_graph(loss_per_step, "Training Loss", "Step", "Loss", "loss.png", world_size)
  plot_graph(time_per_epoch, "Time per Epoch", "Epoch", "Time (s)", "epoch_time.png", world_size)
  plot_graph(time_per_step, "Time per Step", "Step", "Time (ms)", "step_time.png", world_size)
  per_gpu_utils = create_n_array(gpu_utilizations)
  for i, gpu_util in enumerate(per_gpu_utils):
    plot_graph(gpu_util, f"GPU Util {i}", "Step", "%", f"gpu_{i}.png", world_size)
  avg_step_time_s = sum(time_per_step) / len(time_per_step) / 1000 # divide by 1000 to convert ms to s
  global_batch_size = batch_size * world_size
  effective_batch_size = global_batch_size * accumulation_steps
  throughput = effective_batch_size / avg_step_time_s
  print("throughput: ", throughput)
  print("Total time taken: ", total_time_taken)
  print("average step time: ", avg_step_time_s)
  for i in range(len(per_gpu_utils)):
    print(f"average gpu utilization for gpu {i}: ", sum(per_gpu_utils[i])/len(per_gpu_utils[i]))

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DDP Training Cluster")
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
        help="The dataset for training or tuning"
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
    return parser.parse_args()

def concat_images():
  GPUS = [1,2,4]
  for gpu in GPUS:
      folder_path = f'./data/plots_GPU_{gpu}'
      reads = []
      for images in os.listdir(folder_path):
          if images.startswith('gpu'):
              print(images)
              full_image_path = os.path.join(folder_path, images)
              reads.append(cv2.imread(full_image_path))
      concat = cv2.hconcat(reads)
      cv2.imwrite(f'./data/plots_GPU_{gpu}/plot_GPU_{gpu}.png',concat)