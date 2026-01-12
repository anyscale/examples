# Ray Train Integration for Megatron-Bridge: LLM Supervised Fine-Tuning

This guide explains how to run Megatron-Bridge training using Ray Train for distributed orchestration.

## Prerequisites

### 1. Create a Container Image

Follow the [Build Farm guide](https://docs.anyscale.com/container-image/build-image#build-farm) and create a new container image named `megatron-bridge-te` on Anyscale using the following configuration:

```dockerfile
# Containerfile for Megatron-Bridge with transformer_engine
FROM anyscale/ray:2.53.0-py312-cu128

# Install core dependencies
RUN pip install --no-cache-dir \
    transformers>=4.57.1 \
    datasets \
    accelerate \
    omegaconf>=2.3.0 \
    tensorboard>=2.19.0 \
    typing-extensions \
    rich \
    wandb>=0.19.10 \
    pyyaml>=6.0.2 \
    tqdm>=4.67.1 \
    "hydra-core>1.3,<=1.3.2" \
    timm \
    megatron-energon

# Install NVIDIA packages - transformer_engine is the key dependency
RUN pip install --no-cache-dir nvidia-modelopt
RUN pip install --no-cache-dir nvidia-resiliency-ext
RUN pip install --no-cache-dir --no-build-isolation transformer_engine[pytorch]

WORKDIR /app
```

### 2. Create a Workspace

1. Create a new workspace and select the `megatron-bridge-te` container image you just created.
2. **Important:** Add a worker node group. Select **4xL4 GPU** instances with autoscaling set to **0-2**.

### 3. Set Environment Variables

Configure the following environment variables in the workspace dependencies/settings:

```bash
RAY_TRAIN_V2_ENABLED=1
HF_HOME=/mnt/cluster_storage/huggingface
PYTHONPATH=./src:./3rdparty/Megatron-LM
NCCL_DEBUG=INFO
PYTHONUNBUFFERED=1
```

### 4. Setup the Repository

Clone the Megatron-Bridge repository and initialize the submodules:

```bash
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git submodule update --init 3rdparty/Megatron-LM
```

### 5. Download Training Script

We created a training script to finetune (SFT) a small Qwen/Qwen2.5-0.5B model using Megatron-Bridge with Ray Train.
Download the Ray Train integration script directly to `scripts/training/finetune_decoder_ray.py`:

```bash
curl -L -o scripts/training/finetune_decoder_ray.py https://raw.githubusercontent.com/tohtana/Megatron-Bridge/tohtana/run_with_ray_train/scripts/training/finetune_decoder_ray.py
```

### 6. Run the Training

Execute the training script with the following command:

```bash
python scripts/training/finetune_decoder_ray.py \
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
Note: With 8 GPUs, setting TP=2 and PP=2 implies DP=\(8 / (2 \times 2) = 2\). Ray will launch 2 worker nodes (each with 4x L4 GPUs) to provide the 8 GPUs.

### 7. Next steps:
Feel free to use your own dataset, or choose an larger LLM; be careful to select a larger GPU node based on the model size. 