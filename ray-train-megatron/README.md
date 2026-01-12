# Fine-Tuning LLM with Megatron-Bridge and Ray Train

This example demonstrates how to run **Megatron-Bridge** training using **Ray Train** for multi-GPU distributed training on Anyscale. It performs Supervised Fine-Tuning (SFT) on a Qwen/Qwen2.5-0.5B model.


## Option 1: Run as an Anyscale Job

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
cd examples/ray-train-megatron

# Submit the job
anyscale job submit -f job.yaml
```

**What this job does:**
1. **Builds** a Docker image with Megatron-Bridge and dependencies (using `Dockerfile`).
2. **Provisions** 8 GPUs (default: 2 nodes with 4xL4 GPUs each).
3. **Runs** the distributed training script `llm_sft_ray_train_megatron.py`.

---

## Option 2: Run in an Anyscale Workspace (Interactive)

Use a Workspace for interactive development, debugging, or modifying the code.

### 1. Build the Container Image

To ensure all dependencies are installed, you need to build a custom image.

Follow the [Build Farm guide](https://docs.anyscale.com/container-image/build-image#build-farm) and create a new container image named `megatron-bridge-ray-train` on Anyscale using the following configuration:

### 2. Create a Workspace

1. Start a new Workspace.
2. Select the `megatron-bridge-ray-train` image you just built.
3. Configure the **Compute**:
   - **Head Node:** 1x CPU node (e.g., `m5.xlarge`).
   - **Worker Nodes:** Select the `Auto-select nodes` option. It will automatically use 4xL4 GPUs in your cloud. Make sure you have the available GPUs.

### 3. Run the Training

Once your Workspace is running, open a terminal (VS Code or Jupyter) and execute the following:

```bash
# 1. Clone the repository
git clone https://github.com/anyscale/examples.git
cd examples/ray-train-megatron

# 2. Set environment variables
export RAY_TRAIN_V2_ENABLED=1
export MEGATRON_BRIDGE_ROOT=/app/Megatron-Bridge
export PYTHONPATH=$PYTHONPATH:/app/Megatron-Bridge/src:/app/Megatron-Bridge/3rdparty/Megatron-LM
export HF_HOME=/mnt/cluster_storage/huggingface
export PYTHONUNBUFFERED=1

# 3. Run the training script
python llm_sft_ray_train_megatron.py \
    --hf_model_path Qwen/Qwen2.5-0.5B \
    --num_workers 8 \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 2 \
    --train_iters 100 \
    --global_batch_size 8 \
    --micro_batch_size 1 \
    --seq_length 512 \
    --storage_path /mnt/cluster_storage/megatron_experiment
```

> **Note:** The configuration must satisfy `TP * PP * DP = Total GPUs`. For example, when using 8 GPUs (`--num_workers 8`), setting `TP=2` (`--tensor_parallel_size 2`) and `PP=2` (`--pipeline_parallel_size 2`) implies `DP = 8 / (2 * 2) = 2`. If you are using fewer than 8 GPUs, you must adjust these parameters accordingly.

### 4. Locate the checkpoints
After the training, you can locate the checkpoints in `/mnt/cluster_storage/megatron_experiment/megatron_outputs/checkpoints`.

