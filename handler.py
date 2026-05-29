import runpod
import subprocess
import json

def handler(job):
    job_input = job["input"]
    
    training_config = job_input.get('training_config')

    print(f"[RunPod Worker] Launching DDP Cluster across {training_config['gpu_count']} GPUs...")

    cmd = [
        "torchrun",
        f"--nproc_per_node={training_config['gpu_count']}",
        "multi_gpu_training.py",
        '--config', json.dumps(training_config)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        return {
            "status": "COMPLETED",
            "message": f"Successfully completed training job for {training_config['model_name']} with job id: {training_config['job_id']}",
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