# Qwen3-8B GRPO Training on Anyscale (H100)

Single-node RL training of Qwen3-8B with GRPO on **8x H100-80GB** using Anyscale, following the pattern from [anyscale/examples#43](https://github.com/anyscale/examples/pull/43).

## Cluster Layout

```
Head node (m5.2xlarge):  driver only, no GPUs
Worker 0 (8x H100-80GB):
  GPU 0-3: Training (TP=2, DP=2)
  GPU 4-7: Rollout (3 SGLang engines + 1 driver)
```

- **Training**: 4 GPUs — TP=2 x DP=2 (Megatron backend)
- **Rollout**: 3 GPUs — disaggregated SGLang inference, 1 GPU per engine (1 GPU reserved for driver)
- **Algorithm**: GRPO with DAPO-style asymmetric clipping
- **Dataset**: DAPO-Math-17k (integer math, deterministic reward)

## Files

| File | Description |
|------|-------------|
| `job.yaml` | Anyscale job config (`m5.2xlarge` head + 1x `p5.48xlarge` worker) |
| `Dockerfile.anyscale` | Docker image with Miles, Megatron-LM, SGLang, flash-attn, TE |
| `entrypoint.sh` | Downloads model/data, converts weights, runs async GRPO training |

## Quick Start

```bash
pip install -U anyscale
anyscale login

cd examples/anyscale_qwen3_8b_h100
anyscale job submit -f job.yaml
```

The entrypoint automatically:
1. Downloads `Qwen/Qwen3-8B` and `zhuzilin/dapo-math-17k` to `/mnt/cluster_storage`
2. Converts HF weights to Megatron torch_dist format (on GPU worker)
3. Runs async GRPO training with `dapo` reward model via `train_async.py`

## Key Differences from the Slime A10G Example (PR #43)

| | Slime A10G (PR #43) | This Example |
|---|---|---|
| GPUs | 2x4 A10G (24GB) | 1x8 H100 (80GB) |
| Model | Qwen3-1.7B | Qwen3-8B |
| Training | `train.py` (sync) | `train_async.py` (pipelined async) |
| Parallelism | TP=2, PP=2 across nodes | TP=2, DP=2, single node |
| A10G patches | sgl_kernel, Triton, multi_platform | Not needed (H100 = SM90) |
| Batch size | 64 (16 prompts x 4 samples) | 256 (32 prompts x 8 samples) |
| Max tokens/GPU | 4096 | 9216 |
| Attention | FA2 only (Ampere) | FA2 (FA3 available with custom image) |

## Verification

A successful run shows:
- SGLang engine startup on rollout GPUs
- Weight conversion completes (first run only)
- Training loss values printed each step
- Reward gradually increasing over rollouts
- Weight sync between training and rollout engines

## If You Hit OOM

**Training GPUs:**
1. `--max-tokens-per-gpu` -> `4096`
2. `--rollout-max-response-len` -> `4096`
3. `--n-samples-per-prompt` -> `4` and `--global-batch-size` -> `128`

**Rollout GPUs:**
1. `--sglang-mem-fraction-static` -> `0.5`
2. Add `--sglang-chunked-prefill-size 4096`

## View the Job

View the job in the [jobs tab](https://console.anyscale.com/jobs) of the Anyscale console.
