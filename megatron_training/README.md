# Fine-Tuning LLM with Megatron-Bridge and Ray Train

This example demonstrates how to run **Megatron-Bridge** training using **Ray Train** for multi-GPU distributed training on Anyscale. It performs Supervised Fine-Tuning (SFT) on a Qwen/Qwen2.5-1.5B model.


## Run as an Anyscale Job

This is the simplest way to execute the training. The job will automatically build the environment, provision resources, and run the script.

### 1. Install Anyscale CLI
If you haven't already:
```bash
pip install -U anyscale
anyscale login
```

### 2. Submit the Job
Clone the repository and submit the job using the provided YAML configuration:

```bash
# Clone the repository
git clone https://github.com/anyscale/examples.git
cd examples/megatron_training

# Submit the job
anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN
```

**Note:** The `--env HF_TOKEN=$HF_TOKEN` flag passes your HuggingFace token to the job. Make sure you have `HF_TOKEN` set in your local environment.

**What this job does:**
1. **Builds** a Docker image with Megatron-Bridge and dependencies (using `Dockerfile`).
2. **Provisions** 8 GPUs (Tested working with 1 node with 8xH100 and 2 nodes with 4xL4 GPUs).
3. **Runs** the distributed training script `llm_sft_ray_train_megatron.py`.

