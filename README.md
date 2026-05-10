# Distributed Training & Optimization with PyTorch DDP

## Project Overview
This project implements a distributed training pipeline that enables data-parallel training of models across multiple GPUs. It uses `torchrun` to initialize processes, `DistributedSampler` to partition data across GPUs, and `DistributedDataParallel` (DDP) for synchronized training. Gradient computation is handled by PyTorch autograd, while NCCL is used for efficient inter-GPU communication and gradient synchronization.

## 🚀 Quick Start & Installation

**Prerequisites:**
* Python 3.8+
* PyTorch (with CUDA support)
* torchvision
* 1 to 4 GPUs (NVIDIA recommended for NCCL backend)

**Installation:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd Multi-GPU-training/
pip install -r requirements.txt
```bash

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

## Reasons for sublinear speedup and efficiency as the number of GPUs increases

	1. Communication:
	As the number of GPUs increases the communication overhead for performing all reduce also increases.
	Simply this means the time takes to sync gradients between 2 GPUs will be less than the time taken.
This is evident from the increase in average step time from 0.11 for 1 GPU to 0.13 for 2 GPUs to 0.14 for 4 GPUs.

	1. Bucket size:
	Communication can get too slow if the bucket size is not set properly as there is no global bucket size that is correct for all training it should be set according to the model. (Although the default bucket size is 25MB). The bucket size can be set using bucket_size_mb.
	Consider an example of a small model if the bucket size if too high than the computation may finish early and the communication of the bucket could still be going on, on the other hand if the bucket size is too small it could lead to too many all-reduce calls which will increase the communication overhead.
	Hence a suitable bucket size should be chosen depending on the model size.
	
	2. Number of GPUs:
	The number of GPUs play a large part in training models, as the number of GPUs increase the overhead for communication for all reduce between these GPUs also increase which would lead to less efficiency and sublinear speedup.
	This is evident from the data as the number of GPUs increase the efficiency drops from 83% for 2 GPUs to 67% for 4 GPUs.
hence the number of GPUs should be tested and chosen wisely.


## Optimization Strategies

	1. Gradient accumulation - Instead of doing all-reduce every step it can be done after every n steps this would theoretically give the same results and would reduce the number of all reduce synchronizations. This is done using the no sync API.
	2. Changing bucket size - The bucket size should be changed to solve two problems first reduce the number of all reduce sync calls and better overlap of communication and computation.
	3. Round robin process groups -  In round-robin process groups, multiple process groups are created to enable parallel communication. Gradient buckets are deterministically assigned to these process groups (e.g., bucket 1 and 3 use process group 1, while bucket 2 and 4 use process group 2).
	During training, AllReduce operations for different buckets are launched using their assigned process groups, allowing multiple communication operations to run concurrently.
	This improves communication efficiency by better utilizing available bandwidth and overlapping multiple AllReduce operations, which can reduce per-step training time and improve overall GPU efficiency.
	4. Gradient order prediction - Initially when the buckets are created they are created using greedy bucketing algorithm that means all the params while the bucket limit is not set and then move on to the next bucket.
After a couple of iterations the buckets are recreated by observing the backward computation order of the parameters, the buckets are created throughout all the GPUs and are same.
	5. Layer Dropping - Layer dropping (e.g., stochastic depth) can be used to reduce overfitting by skipping layers during the forward pass. While DDP correctly handles skipped parameters, it does not reduce communication overhead because gradient synchronization operates at the bucket level. To address this, buckets can be aligned with layers so that entire buckets can be skipped when layers are dropped, enabling potential communication savings. However, this requires coordination across all processes to ensure consistent behavior.
	6. Prioritizing the initial layers buckets - In this optimization strategy the all reduce of the buckets from the initial layer is prioritized over the later layers as the forward for the next iteration can start even if the communication of the previous layers are remaining.



## Post Optimization Result

As evident from the table below using accumulation step increases the throughput and efficiency of training, as the communication overhead gets reduced due to less all reduce calls. The all-reduce communication occurs every 4 steps instead of every step. This approximates training with a larger effective batch size while reducing synchronization frequency.

Number of GPU's	Throughput  (images/sec)	Speedup	Efficiency (in %)	Total training time (in sec)	Average Step time (in sec)
4	18765.065797789888	3.0612928867019487	76	59.15190935134888	0.05456948624824991
2	11106.43343900814	1.8303386327000886	91	98.93323349952698	0.0921988148241627
1	5988.976761275212	NA	NA	181.0813193321228	0.17098079368435606 

