# SGLang Multi-Node Inference on Ray (Anyscale)

This example shows how to run [SGLang](https://github.com/sgl-project/sglang) inference across multiple nodes using Ray as the distributed backend, deployed as Anyscale Jobs.

Two modes are provided:

| Mode | Driver | Description |
|------|--------|-------------|
| **Offline (batch)** | `driver_offline.py` | Wraps `sglang.Engine` in a Ray actor for batch generation |
| **Online (HTTP server)** | `driver_online.py` | Launches the SGLang HTTP server inside a Ray task for real-time serving |

In both modes the driver runs on a CPU-only head node while SGLang workers are distributed across GPU worker nodes via Ray placement groups.

## Files

```
sglang_ray_inference/
├── Dockerfile           # Container image (Ray 2.53 + CUDA 12.9 + SGLang)
├── driver_offline.py    # Offline batch inference driver
├── driver_online.py     # Online HTTP server driver
├── job_offline.yaml     # Anyscale job config for offline mode
└── job_online.yaml      # Anyscale job config for online mode
```

## Quick Start

### Prerequisites

- An [Anyscale](https://www.anyscale.com/) account with GPU quota
- The `anyscale` CLI installed and configured

### Run Offline Inference

```bash
cd sglang_ray_inference
anyscale job submit -f job_offline.yaml
```

### Run Online Inference

```bash
cd sglang_ray_inference
anyscale job submit -f job_online.yaml
```

## Configuration

Both job configs default to **TP=8 across 2 nodes** (4x A10G per node via `g5.12xlarge`).

To change the model, tensor-parallelism degree, or node count, edit the `entrypoint` in the YAML file:

```bash
python driver_offline.py \
  --model-path <model> \
  --tp-size <total_gpus> \
  --nnodes <num_nodes>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `Qwen/Qwen3-1.7B` | HuggingFace model ID or path |
| `--tp-size` | `4` | Tensor parallelism degree (total GPUs) |
| `--pp-size` | `1` | Pipeline parallelism degree |
| `--nnodes` | `1` | Number of nodes to spread across |
| `--port` | `30000` | Port for the SGLang server (online mode) |
