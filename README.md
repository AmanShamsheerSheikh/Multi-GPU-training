# Distrain: Distributed LLM Training with DDP & FSDP

## Project Overview

Distrain is a distributed LLM training framework built with PyTorch. It supports both Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP) training, checkpoint save/resume, mixed precision training, gradient accumulation, and multi-GPU benchmarking.

The project was developed to study the tradeoffs between data parallelism and parameter sharding for large language model training.

---

## System Architecture

### DDP Pipeline

* torchrun launches one process per GPU
* DistributedSampler shards the dataset across workers
* Each GPU holds a full model replica
* Gradients are synchronized using NCCL AllReduce
* Communication overlaps with backward computation through gradient bucketing

### FSDP Pipeline

* Parameters, gradients, and optimizer states are sharded across GPUs
* Each transformer block is wrapped using transformer_auto_wrap_policy
* FSDP performs ReduceScatter and AllGather operations during training
* Memory footprint scales approximately with 1/N as GPU count increases

---

## Features

* Distributed Data Parallel (DDP)
* Fully Sharded Data Parallel (FSDP / ZeRO-3)
* Mixed precision training (bf16)
* Gradient accumulation
* Distributed checkpointing
* Resume-from-checkpoint support
* Throughput benchmarking
* Memory profiling
* MFU (Model FLOP Utilization) measurement
* Multi-GPU scaling analysis

---

## DDP Benchmark — GPT-2 XL

Model: GPT-2 XL (1,557,611,200 parameters)

### Results

| GPUs | Throughput (samples/sec) | Speedup | Efficiency | Step Time (sec) | MFU    |
| ---- | ------------------------ | ------- | ---------- | --------------- | ------ |
| 1    | 30.77                    | 1.00x   | 100%       | 1.0400          | 47.54% |
| 2    | 60.57                    | 1.97x   | 98.5%      | 1.0567          | 46.79% |
| 4    | 121.19                   | 3.94x   | 98.5%      | 1.0562          | 46.81% |

### Memory Usage

| GPUs | Peak Memory/GPU | Weights + Optimizer |
| ---- | --------------- | ------------------- |
| 1    | 63,795 MB       | 34,501 MB           |
| 2    | 63,795 MB       | 34,500 MB           |
| 4    | 63,793 MB       | 34,498 MB           |

### Observations

* Near-linear scaling up to 4 GPUs
* Scaling efficiency remains above 98%
* Memory usage remains constant because each GPU stores a full model replica
* MFU remains stable, indicating communication overhead is small relative to compute

---

## FSDP Benchmark — Mistral 7B

Model: Mistral 7B (7,241,732,096 parameters)

### Results

| GPUs | Throughput (samples/sec) | Speedup | Step Time (sec) | MFU    |
| ---- | ------------------------ | ------- | --------------- | ------ |
| 2    | 12.64                    | 1.00x   | 0.6327          | 22.71% |
| 4    | 26.70                    | 2.11x   | 0.5992          | 11.99% |

### Memory Usage

| GPUs | Peak Memory/GPU | Weights + Optimizer |
| ---- | --------------- | ------------------- |
| 2    | 21,732 MB       | 14,455 MB           |
| 4    | 18,280 MB       | 11,003 MB           |

### Observations

* Memory footprint decreases as GPU count increases
* Parameter sharding significantly reduces per-GPU memory requirements
* FSDP enables training larger models while maintaining manageable memory usage
* Throughput increases from 12.64 to 26.70 samples/sec when scaling from 2 to 4 GPUs

---

## DDP vs FSDP

| Metric            | DDP                            | FSDP                                   |
| ----------------- | ------------------------------ | -------------------------------------- |
| Parameter Storage | Full Replica                   | Sharded                                |
| Optimizer State   | Full Replica                   | Sharded                                |
| Gradient Storage  | Full Replica                   | Sharded                                |
| Memory Scaling    | O(1)                           | O(1/N)                                 |
| Communication     | AllReduce                      | ReduceScatter + AllGather              |
| Best Use Case     | Models fitting on a single GPU | Large models requiring memory sharding |

---

## Checkpointing

Distrain supports distributed checkpoint save and resume.

Stored state includes:

* Model weights
* Optimizer state
* Scheduler state
* Epoch
* Global step

Resume functionality was verified through restart testing.

---

## Key Learnings

* DDP provides excellent scaling efficiency when models fit comfortably in GPU memory.
* FSDP trades additional communication for significant memory savings.
* Memory sharding becomes increasingly important as model size grows.
* Communication patterns and GPU topology strongly influence distributed training performance.

## Engineering Notes

**NCCL P2P Deadlock Debug**
On a RunPod multi-GPU environment, NCCL initialization succeeded but the first collective operation hung. The issue was traced to NCCL P2P transport behavior on the underlying GPU topology. Training was successfully restored by setting NCCL_P2P_DISABLE=1, forcing NCCL to use alternative communication paths.

**Checkpointing Strategy**
Mid-training checkpoints use `SHARDED_STATE_DICT` — each rank saves its own shard independently without synchronization overhead. End-of-training consolidation uses `FULL_STATE_DICT` (rank 0 only) for HuggingFace upload. Rotating 2-slot local checkpoint scheme prevents disk exhaustion. Background upload threads prevent blocking training.