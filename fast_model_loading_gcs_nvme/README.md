# Fast Model Loading from GCS with NVMe Caching

This example deploys a model on Anyscale with fast cold starts using a two-phase loading strategy: download model weights from GCS to local NVMe SSDs, then load from NVMe to GPU memory.

**Why this matters:** Loading a 70B model (~145 GB) from a persistent disk takes ~8 minutes. By caching on NVMe and loading with the [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer), cold starts drop to ~36 seconds. Subsequent replicas on the same node skip the download entirely.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Step 1: Upload model weights to GCS

Run once from any machine with `gcloud` and `huggingface_hub` installed:

```bash
gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1

python -c "
from pathlib import Path
from google.cloud.storage import Client, transfer_manager
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    allow_patterns=['*.safetensors', '*.json', '*.txt', 'tokenizer*', '*.model'],
    max_workers=8,
)

bucket = Client().bucket('YOUR_BUCKET')
files = [p.relative_to(local_dir).as_posix() for p in Path(local_dir).rglob('*') if p.is_file()]
transfer_manager.upload_many_from_filenames(
    bucket, files, source_directory=local_dir,
    blob_name_prefix='models/DeepSeek-R1-Distill-Qwen-7B/', max_workers=8,
)
"
```

> **Tip:** If your bucket is under `$ANYSCALE_ARTIFACT_STORAGE`, pods can access it via workload identity — no extra credentials needed.

## Step 2: Deploy the service

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/fast_model_loading_gcs_nvme
```

Two variants are provided. Both use the same two-phase loading strategy; they differ in the GPU-side inference framework.

**vLLM** (OpenAI-compatible API, recommended for production):

```bash
anyscale service deploy -f service_vllm.yaml \
    --env GCS_MODEL_URI=gs://YOUR_BUCKET/models/DeepSeek-R1-Distill-Qwen-7B
```

**HuggingFace Transformers** (custom forward passes, research use):

```bash
anyscale service deploy -f service_hf.yaml \
    --env GCS_MODEL_URI=gs://YOUR_BUCKET/models/DeepSeek-R1-Distill-Qwen-7B
```

## Understanding the example

```
GCS Bucket (durable, shared)
        │
        ▼  Phase 1: runai ObjectStorageModel (file-locked, idempotent)
NVMe SSD (/mnt/local_storage/model)
        │
        ▼  Phase 2: runai_streamer (~4.3 GB/s) or HF/safetensors (~1.5 GB/s)
GPU Memory
```

- **Phase 1** runs once per node. `ObjectStorageModel` uses `fcntl.flock` so only one process downloads while others wait. A `.runai_complete` sentinel marks completion so subsequent processes skip the download.
- **Phase 2** (vLLM only): the `LOAD_FORMAT` env var controls which loader is used:

  | `LOAD_FORMAT` | NVMe → GPU throughput | Notes |
  |---|---|---|
  | `runai_streamer` (default) | ~4.3 GB/s | Concurrent C++ threads, fastest |
  | `auto` | ~1.5 GB/s | HuggingFace/safetensors default loader |

  To use the HuggingFace loader: `--env LOAD_FORMAT=auto`

- **NVMe configuration:** `service_vllm.yaml` attaches 4 NVMe local SSDs (375 GB each, ~1.5 TB total) via `advanced_instance_config`. Anyscale RAIDs and mounts them at `/mnt/local_storage`. `a2-highgpu-4g` supports 0, 4, or 8 local SSDs.
- **Multi-node:** Each node has its own local NVMe. When autoscaling adds a node, it downloads its own copy from GCS. Processes on the same node coordinate via file locks.
- The [Dockerfile](https://github.com/anyscale/examples/blob/main/fast_model_loading_gcs_nvme/Dockerfile) defines service dependencies, built on top of an Anyscale base image when you run `anyscale service deploy`.

## Step 3: Query the service

The `anyscale service deploy` command outputs a line like:

```text
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
```

Query with the OpenAI client (vLLM variant):

```python
from openai import OpenAI

client = OpenAI(base_url="<BASE_URL>/v1", api_key="<SERVICE_TOKEN>")

for chunk in client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "What's the capital of France?"}],
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

View the service in the [services tab](https://console.anyscale.com/services) of the Anyscale console.

## Shutdown

```bash
anyscale service terminate -n fast-model-loading-vllm
# or
anyscale service terminate -n fast-model-loading-hf
```

## Benchmark

Hardware: A100-SXM4-40GB · Model: DeepSeek-R1-Distill-Qwen-7B (14.2 GB safetensors) · Cold cache

### Phase 2: [storage source] → GPU

Columns show where the model weights are loaded **from**; the destination is always GPU memory.
Speedups are relative to the PD baseline (`safetensors.load_file` from persistent disk).

| Method | Persistent Disk → GPU | NVMe → GPU | GCS → GPU |
|---|---|---|---|
| `safetensors.load_file` | 47.3s · 0.32 GB/s · 1.0× | 10.4s · 1.47 GB/s · 4.6× | — |
| `runai_streamer` | 47.7s · 0.32 GB/s · 1.0× | **3.5s · 4.31 GB/s · 13.4×** | **4.7s · 3.21 GB/s · 10.0×** |
| `ray.anyscale.safetensors` | — | — | 8.2s · 1.87 GB/s · 5.8× |

This example uses **NVMe → GPU via `runai_streamer`** (13.4× faster than the PD baseline).

### End-to-end cold start: GCS → NVMe → GPU (70B model, ~145 GB)

| Phase | Flow | Time | Frequency |
|---|---|---|---|
| Phase 1: download | GCS → NVMe | ~48s | First replica per node only |
| Phase 2: load (runai_streamer) | NVMe → GPU | ~36s | Every replica |
| Phase 2: load (HF/safetensors) | NVMe → GPU | ~100s | Every replica |
| **Total (runai_streamer)** | GCS → NVMe → GPU | **~84s** | First replica; subsequent skip Phase 1 |
| **Total (HF/safetensors)** | GCS → NVMe → GPU | **~148s** | First replica; subsequent skip Phase 1 |
