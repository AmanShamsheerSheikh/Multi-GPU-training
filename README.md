## Project Overview
This project implements a distributed training pipeline that enables data-parallel training of models across multiple GPUs. It uses torchrun to initialize processes, DistributedSampler to partition data across GPUs, and DistributedDataParallel (DDP) for synchronized training. Gradient computation is handled by PyTorch autograd, while NCCL is used for efficient inter-GPU communication and gradient synchronization.
	
## System Architecture
The distributed training pipeline consists of three main components:

### • Torchrun:
- Launches one process per GPU.
- Initializes the distributed environment, including world_size, rank, and local_rank.
- Each process is mapped to a specific GPU using local_rank.

### • DistributedSampler:
- Splits the dataset across processes based on rank and world_size.
- Ensures each GPU processes a unique subset of data.
- Shuffles data across epochs to maintain randomness and improve generalization.

### • Distributed Data Parallel (DDP):
- Replicates the model across all processes.
- Uses PyTorch autograd to compute gradients during backward propagation.
- Groups gradients into buckets and overlaps communication with computation.
- Triggers asynchronous all-reduce (via NCCL) for each bucket to synchronize gradients across GPUs.
	
## Implementation Details

### • Model:
- ResNet18 used as the base architecture.
- Input images resized to 224×224.

### • Data Pipeline:
- Dataset: CIFAR-10
- Transformations: Resize + ToTensor.
- DistributedSampler used to partition data across GPUs and reshuffle each epoch.

### • Distributed Setup:
- Backend: NCCL
- Initialized using torchrun and init_process_group
- Each process mapped to a GPU using local_rank.

### • Training Loop:
- Forward pass executed independently on each GPU (data-parallel execution).
- Backward pass computes gradients via PyTorch autograd.
- Gradient synchronization occurs during backward via DDP hooks using all-reduce.
		
## Experimentation Setup

- Experiments were conducted using 1, 2, and 4 GPUs to evaluate scaling behavior.
- Batch size per GPU was kept constant at 256, resulting in increased global batch size with more GPUs.
- Each configuration was trained under identical conditions for fair comparison.

## Results + Observations

- The throughput i.e. the number of images processed per sec increases with the number of GPU's.
- The total training time decreases with the number of GPU's.
- But the efficiency of training decreases from 83 to 67, The drop in efficiency is due to increased communication overhead during gradient synchronization.  
Hence the GPU's are not utilized fully, this is evident from the average step time increase.

## Results Table

| Number of GPU's | Throughput (images/sec) | Speedup | Efficiency (in %) | Total training time (in sec) | Average Step time (in sec) |
|----------------|------------------------|---------|-------------------|------------------------------|----------------------------|
| 4              | 6854.503027363261      | 2.7093820357034555 | 67 | 47.76641893386841 | 0.14939084510024714 |
| 2              | 3707.3587985062445     | 1.6690667177063556 | 83 | 77.53882813453674 | 0.13810370881995376 |
| 1              | 2289.9372186446903     | NA | NA | 129.41747736930847 | 0.11179345787982552 |
