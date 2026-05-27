import runpod
import subprocess

def handler(job):
    job_input = job["input"]
    
    job_id = job_input.get('job_id')
    model_name = job_input.get('model_name')
    dataset_name = job_input.get('dataset_name')
    job_type = job_input.get('job_type')
    epochs = job_input.get('epochs', 10)
    text_column_name = job_input.get('text_column_name')
    gpu_count = job_input.get('gpu_count', 1)
    task_type = job_input.get('task_type')

    print(f"[RunPod Worker] Launching DDP Cluster across {gpu_count} GPUs...")

    cmd = [
        "torchrun",
        f"--nproc_per_node={gpu_count}",
        "multi_gpu_training.py",
        "--model_name", str(model_name),
        "--dataset_name", str(dataset_name),
        "--job_type", str(job_type),
        "--epochs", str(epochs),
        "--text_column_name", str(text_column_name),
        "--job_id", str(job_id),
        "--gpu_count", str(gpu_count),
        "--task_type", str(task_type)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        return {
            "status": "COMPLETED",
            "message": f"Successfully completed training job for {model_name} with job id: {job_id}",
            "logs": result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        print(f"[CRITICAL] DDP Training execution hard crashed: {e.stderr}")
        return {
            "status": "FAILED",
            "error": e.stderr,
            "partial_logs": e.stdout
        }

runpod.serverless.start({"handler": handler})