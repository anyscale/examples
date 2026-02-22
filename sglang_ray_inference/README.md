# Multi-Node SGLang on Anyscale with Ray Actor Backend

Run SGLang across multiple GPU nodes on Anyscale using Ray placement groups.
The head node needs no GPU — sglang is imported only inside Ray actors.

Two driver scripts:

- **`driver_offline.py`** — batch inference via `sglang.Engine`
- **`driver_online.py`** — HTTP server via `sglang.srt.entrypoints.http_server`

## Files

| File | Purpose |
|------|---------|
| `driver_offline.py` | Batch inference driver |
| `driver_online.py` | HTTP server driver |
| `job_offline.yaml` | Anyscale job config for batch inference |
| `job_online.yaml` | Anyscale job config for HTTP server |
| `Dockerfile` | Base image (`anyscale/ray` + sglang) |

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

### Run Locally (single node, requires GPU)

```bash
python driver_offline.py --model-path Qwen/Qwen3-1.7B --tp-size 1 --nnodes 1
python driver_online.py  --model-path Qwen/Qwen3-1.7B --tp-size 1 --nnodes 1
```

## CLI Arguments

Both scripts accept the same arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `Qwen/Qwen3-1.7B` | HuggingFace model ID or local path |
| `--tp-size` | `4` | Tensor parallelism size |
| `--pp-size` | `1` | Pipeline parallelism size |
| `--nnodes` | `2` | Number of nodes |
| `--port` | `30000` | Server port |

## How It Works

1. **Placement group** — one bundle per node (`{"CPU": 1, "GPU": gpus_per_node}`).
   `STRICT_PACK` for 1 node, `STRICT_SPREAD` for multi-node.
2. **Driver actor** — scheduled on bundle 0 with `num_cpus=1, num_gpus=0`.
   Imports sglang inside the actor to avoid GPU-less head node issues.
3. **SchedulerActors** — created internally by sglang, each claiming `num_gpus=1`
   from the correct node's bundle.

## Troubleshooting

- **Server timeout** — increase the 600s health-check timeout or check NCCL
  connectivity (`NCCL_DEBUG=INFO`).
- **OOM** — reduce model size or use more GPUs.
- **Connection refused** — ensure security groups allow inter-node traffic on
  the server port and NCCL ports.
