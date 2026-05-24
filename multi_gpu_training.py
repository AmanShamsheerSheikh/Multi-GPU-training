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
from constants import TRAIN, TUNE, ACCUMULATION_STEPS, BATCH_SIZE

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

def get_dataloader(batch_size, dataset):
  sampler = DistributedSampler(dataset)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
  )
  return dataloader, sampler

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


def train(model, dataloader, sampler, accumulation_steps, device, world_size):
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
  criterion = torch.nn.CrossEntropyLoss()
  loss_per_step = []
  time_per_step = []
  time_per_epoch = []
  gpu_utilizations = []
  starter = torch.cuda.Event(enable_timing=True)
  ender = torch.cuda.Event(enable_timing=True)
  rank = dist.get_rank()
  num_epochs = 5
  total_steps = len(dataloader) * num_epochs
  global_pbar = tqdm(total=total_steps, disable=(rank != 0))
  for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # it sets the epoch for the shuffling of the images per epoch 
    i = 0
    if rank == 0:
      epoch_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for images, labels in dataloader:
      images = images.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)

      if rank == 0:
        torch.cuda.synchronize()
        starter.record()

      if (i + 1) % accumulation_steps == 0 or (i+1) == len(dataloader):
        outputs = model(images)
        raw_loss = criterion(outputs, labels)
        loss = raw_loss / accumulation_steps # later read a bit more why this is done.
        loss.backward()   # all-reduce happens here
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
      else:
        with model.no_sync():
          outputs = model(images)
          raw_loss = criterion(outputs, labels)
          loss = raw_loss / accumulation_steps
          loss.backward()
      if epoch == 0 and i == 0:
        check_gradients(model)

      if rank == 0:
        ender.record()
        torch.cuda.synchronize()
        time_per_step.append(starter.elapsed_time(ender))
        global_pbar.update(1)

      with torch.no_grad():
        reduced_loss = raw_loss.detach()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss /= world_size
        if i%10 == 0 and dist.get_rank() == 0:
          utils = [get_gpu_util(i) for i in range(torch.cuda.device_count())]
          gpu_utilizations.append(utils)
      if rank == 0:
          loss_per_step.append(reduced_loss.item())
      i += 1
    if dist.get_rank() == 0:
      torch.cuda.synchronize()
      time_per_epoch.append(time.time() - epoch_start)
  global_pbar.close()
  if rank == 0:
    return loss_per_step, time_per_epoch, time_per_step, gpu_utilizations
  else:
    return None, None, None, None

def main(model, dataset, job_type, epochs):
  if job_type == TRAIN:
    setup_ddp()
    if dist.get_rank() == 0:
      pynvml.nvmlInit()
    world_size = dist.get_world_size()
    batch_size= BATCH_SIZE // world_size
    accumulation_steps = ACCUMULATION_STEPS
    device = get_device()
    dataloader, sampler = get_dataloader(batch_size, dataset)
    model = get_model(device, model)
    model.train()
    training_start_time = time.time()
    loss_per_step, time_per_epoch, time_per_step, gpu_utilizations = train(model, dataloader, sampler, accumulation_steps, device, world_size)
    total_time_taken = time.time() - training_start_time
    if dist.get_rank() == 0:
      plot_graphs_log_data(loss_per_step, time_per_epoch, time_per_step, gpu_utilizations, batch_size, accumulation_steps, total_time_taken, world_size)
      concat_images()
    torch.save(model.module.state_dict(), f'./model_{dist.get_rank()}_{world_size}.pt')
    dist.destroy_process_group()
  elif job_type == TUNE:
    print('coming soon')

if __name__ == '__main__':
  args = parse_args()
  main(args.model_name, args.dataset_name, args.job_type, args.epochs)