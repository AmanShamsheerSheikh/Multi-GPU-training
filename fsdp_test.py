import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
from torch.distributed.fsdp import (
   FullyShardedDataParallel
)
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch
from transformers import GPT2LMHeadModel
import torch.distributed as dist
import pynvml

def setup_fsdp():
  """
  tells that nccl will be used for communication between GPU's
  """
  dist.init_process_group(
    backend="nccl"
  )
  
def get_fake_batch(batch_size, seq_len, vocab_size, device):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    return input_ids, labels, attention_mask

def get_device():
  """
  When the code is ran each process run the main hence this function gets called n times for n Processes
  so each GPU in a node gets assign depending on its index given by cuda and the local_rank
  """
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  return torch.device("cuda", local_rank)

def get_gpu_util(gpu_number) -> int:
  handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_number)  # GPU 0
  util = pynvml.nvmlDeviceGetUtilizationRates(handle)
  return util.gpu


def train(model, optimizer, epochs, device):
    for i in range(epochs):
        input_ids, labels, attention_mask = get_fake_batch(25, 10, 50257, device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if dist.get_rank() == 0:
            utils = [get_gpu_util(j) for j in range(torch.cuda.device_count())]
            for j, util in enumerate(utils):
                print({f"gpu_util_rank_{j}": util})
            print(f"epoch {i} loss {loss.item():.4f}")

def main(model_name):
    setup_fsdp()
    pynvml.nvmlInit()
    device = get_device()
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPT2Block}
    )
    fsdp_model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=wrap_policy,
    )
    total_params = sum(p.numel() for p in fsdp_model.parameters())
    print(f"rank {dist.get_rank()} params: {total_params}")
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=2e-4)
    fsdp_model.train()
    train(fsdp_model, optimizer, 10, device)
    dist.destroy_process_group()

if __name__ == '__main__':
    main("gpt2")