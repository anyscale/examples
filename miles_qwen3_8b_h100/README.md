# GRPO Training for Qwen3-8B with MILES

This example demonstrates reinforcement learning fine-tuning of Qwen3-8B using **Group Relative Policy Optimization (GRPO)** on the DAPO-Math-17k dataset. It uses the [MILES](https://github.com/radixark/miles) framework for distributed RL training with disaggregated rollouts on Anyscale.

The training runs on a single node with **8x H100-80GB GPUs**, using:
- **4 GPUs for training** (TP=2, DP=2 with Megatron-LM)
- **4 GPUs for rollout inference** (disaggregated SGLang engines)

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Submit the job

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/miles_qwen3_8b_h100
```

Submit the job.

```bash
anyscale job submit -f job.yaml
```

The entrypoint will automatically download the model and dataset, convert weights to Megatron format, and start training. Training progress can be monitored via TensorBoard logs in `/mnt/cluster_storage/tensorboard_logs`.

## Understanding the example

- **Algorithm**: This example uses GRPO with DAPO-style asymmetric clipping (ε_low=0.2, ε_high=0.28), which is particularly effective for math reasoning tasks.
- **Dataset**: [DAPO-Math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) contains 17k integer math problems with deterministic reward signals based on answer correctness.
- **Disaggregated architecture**: Training and rollout happen on separate GPUs for maximum throughput. The 4 SGLang rollout engines run inference in parallel while the training GPUs perform gradient updates.
- **Weight conversion**: On the first run, HuggingFace weights are converted to Megatron-LM's `torch_dist` format. Converted weights are cached in `/mnt/cluster_storage/Qwen3-8B_torch_dist` for subsequent runs.
- **Async training**: The pipeline uses `train_async.py` which overlaps rollout generation and policy updates for better GPU utilization.
