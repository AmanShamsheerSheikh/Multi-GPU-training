import torch.nn as nn
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import time
import pynvml

class ResNetBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.batchNorm1 = nn.BatchNorm2d(num_features=out_channels)
    self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
    self.batchNorm2 = nn.BatchNorm2d(num_features=out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.shortcut = nn.Identity()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_channels=in_channels, out_channels=out_channels,
              kernel_size=1, stride=stride, bias=False
          ),
          nn.BatchNorm2d(num_features=out_channels)
      )

  def forward(self, x):
    out = self.conv1(x)
    out = self.batchNorm1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.batchNorm2(out)
    out = out + self.shortcut(x)
    return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
      super().__init__()

      self.in_channels = 64

      self.conv1 = nn.Conv2d(
        3, 64, kernel_size=7, stride=2, padding=3, bias=False
      )
      self.bn1 = nn.BatchNorm2d(64)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      self.layer1 = self._make_layer(64,  2, stride=1)
      self.layer2 = self._make_layer(128, 2, stride=2)
      self.layer3 = self._make_layer(256, 2, stride=2)
      self.layer4 = self._make_layer(512, 2, stride=2)

      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
      layers = []

      layers.append(ResNetBlock(self.in_channels, out_channels, 3, stride))
      self.in_channels = out_channels

      for _ in range(1, num_blocks):
          layers.append(ResNetBlock(out_channels, out_channels, 3))

      return nn.Sequential(*layers)

    def forward(self, x):
      x = self.relu(self.bn1(self.conv1(x)))
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)

      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)

      return x

model = ResNet18(num_classes=1000)
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(y.shape)

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

def get_dataloader(batch_size):
  from torchvision.datasets import CIFAR10
  from torchvision.transforms import ToTensor

  dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
  )

  sampler = DistributedSampler(dataset)

  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
  )

  return dataloader, sampler

def get_model(device):
  model = ResNet18(num_classes=10)
  model = model.to(device)
  model = DDP(model, device_ids=[device.index])
  return model

def plot_graph(value, title, xlabel, ylabel):
  plt.plot(value)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

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
  mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
  return util.gpu


def train(model, dataloader, sampler, device):
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
  criterion = torch.nn.CrossEntropyLoss()
  loss_per_step = []
  time_per_step = []
  time_per_epoch = []
  gpu_utilizations = []
  starter = torch.cuda.Event(enable_timing=True)
  ender = torch.cuda.Event(enable_timing=True)
  rank = dist.get_rank()
  for epoch in range(5):
    sampler.set_epoch(epoch)  # it sets the epoch for the shuffling of the images per epoch 
    i = 0
    if rank == 0:
      epoch_start = time.time()

    for images, labels in dataloader:
      images = images.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)

      if rank == 0:
        torch.cuda.synchronize()
        starter.record()

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()   # all-reduce happens here
      if epoch == 0 and i == 0:
        check_gradients(model)
      optimizer.step()

      if rank == 0:
        ender.record()
        torch.cuda.synchronize()
        time_per_step.append(starter.elapsed_time(ender))

      if rank == 0:
        with torch.no_grad():
          reduced_loss = loss.detach()
          dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
          reduced_loss /= dist.get_world_size()
          loss_per_step.append(reduced_loss.item())
      i += 1
    if dist.get_rank() == 0:
      torch.cuda.synchronize()
      time_per_epoch.append(time.time() - epoch_start)
      gpu_utilizations.append(get_gpu_util(rank))
  if rank == 0:
    return loss_per_step, time_per_epoch, time_per_step, gpu_utilizations
  else:
    return None, None, None, None


def main():
  setup_ddp()
  pynvml.nvmlInit()
  device = get_device()
  batch_size=64
  dataloader, sampler = get_dataloader(batch_size)
  model = get_model(device)

  training_start_time = time.time()
  loss_per_step, time_per_epoch, time_per_step, gpu_utilizations = train(model, dataloader, sampler, device)
  total_time_taken = time.time() - training_start_time
  if dist.get_rank() == 0:
    plot_graph(value=loss_per_step, title='Training Loss Over Steps', xlabel='Step (Index)', ylabel='Loss')
    plot_graph(value=time_per_epoch, title="Time per Epoch", xlabel='Epoch', ylabel='Time (s)')
    plot_graph(value=time_per_step, title="Time per Step", xlabel='Step', ylabel='Time (ms)')
    plot_graph(value=gpu_utilizations, title="GPU Utilizations", xlabel="Epoch", ylabel='GPU Utilization (%)')
    avg_step_time_s = sum(time_per_step) / len(time_per_step) / 1000
    global_batch_size = batch_size * dist.get_world_size()
    throughput = global_batch_size / avg_step_time_s
    print("throughput: ", throughput)
    print("Total time taken: ", total_time_taken)
  dist.destroy_process_group()