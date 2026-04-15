# Fast Model Loading with GCS and NVMe

Deploy a 70B model on Anyscale with fast cold starts using a two-phase loading strategy: download model weights from GCS to local NVMe SSDs, then load from NVMe to GPU memory.

**Why this matters:** Loading a 70B model (~145 GB) from a persistent disk takes ~8 minutes. By caching on NVMe and loading with the [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer), cold starts drop to ~36 seconds. Subsequent replicas on the same node skip the download entirely.

## Architecture

```
GCS Bucket (durable, shared)
        │
        ▼  Phase 1: runai ObjectStorageModel (file-locked, idempotent)
NVMe SSD (/mnt/local_storage/model)
        │
        ▼  Phase 2: runai_streamer (~4.3 GB/s) or HF/safetensors (~1.5 GB/s)
GPU Memory
```

**Phase 1** runs once per node via a Ray Serve callback. File locking (`fcntl.flock`) ensures only one process downloads while others wait. A `.runai_complete` sentinel file marks completion so subsequent processes skip the download.

**Phase 2** is handled by vLLM. The `LOAD_FORMAT` env var controls which loader is used:

| `LOAD_FORMAT` | NVMe → GPU throughput | Notes |
|---|---|---|
| `runai_streamer` (default) | ~4.3 GB/s | Concurrent C++ threads, fastest |
| `auto` | ~1.5 GB/s | HuggingFace/safetensors default loader |

## Prerequisites

- [Anyscale CLI](https://docs.anyscale.com/reference/quickstart) installed and logged in
- An Anyscale cloud on GCP
- `gcloud` CLI authenticated locally (for the one-time upload step)

## Step 1: Upload model weights to GCS

Run once from any machine with `gcloud` and `huggingface_hub` installed:

```bash
gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1

python -c "
from pathlib import Path
from google.cloud.storage import Client, transfer_manager
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    allow_patterns=['*.safetensors', '*.json', '*.txt', 'tokenizer*', '*.model'],
    max_workers=8,
)

bucket = Client().bucket('YOUR_BUCKET')
files = [p.relative_to(local_dir).as_posix() for p in Path(local_dir).rglob('*') if p.is_file()]
transfer_manager.upload_many_from_filenames(
    bucket, files, source_directory=local_dir,
    blob_name_prefix='models/DeepSeek-R1-Distill-Llama-70B/', max_workers=8,
)
"
```

> **Tip:** If your bucket is under `$ANYSCALE_ARTIFACT_STORAGE`, pods can access it via workload identity — no extra credentials needed.

## Step 2: Deploy the service

```bash
git clone https://github.com/anyscale/examples.git
cd examples/fast_model_loading_gcs_nvme

anyscale service deploy -f service.yaml \
    --env GCS_MODEL_URI=gs://YOUR_BUCKET/models/DeepSeek-R1-Distill-Llama-70B
```

To use the HuggingFace/safetensors loader instead of runai_streamer:

```bash
anyscale service deploy -f service.yaml \
    --env GCS_MODEL_URI=gs://YOUR_BUCKET/models/DeepSeek-R1-Distill-Llama-70B \
    --env LOAD_FORMAT=auto
```

### What happens on startup

1. Worker node starts with 6x 375 GB NVMe local SSDs (~2.2 TB at `/mnt/local_storage`).
2. **Phase 1 (download):** The `NVMeCacheCallback` fires before vLLM initialization. It uses Run:ai's `ObjectStorageModel` to download all model files from GCS to `/mnt/local_storage/model`. File locking ensures only one process per node downloads; others wait then skip.
3. **Phase 2 (load):** vLLM loads the model from the local NVMe path. With `runai_streamer` (default), concurrent C++ threads read safetensors at ~4.3 GB/s. With `auto`, the standard HuggingFace loader reads at ~1.5 GB/s.

### Multi-node behavior

Each node has its own local NVMe — no shared storage. When autoscaling adds a new node, it downloads its own copy from GCS. Multiple processes on the _same_ node coordinate via file locks so only one downloads while the rest wait.

### NVMe configuration

The `service.yaml` attaches 6 NVMe local SSDs via `advanced_instance_config`. Each is 375 GB. Anyscale RAIDs and mounts them at `/mnt/local_storage`. See [Anyscale NVMe docs](https://docs.anyscale.com/configuration/compute/gcp#nvme) for details.

## Step 3: Query the service

The `anyscale service deploy` command outputs a line like:
```
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
```

Query with the OpenAI client:

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

## Shutdown

```bash
anyscale service terminate -n fast-model-loading
```

## Benchmark

Measured on A100-SXM4-40GB with DeepSeek-R1-Distill-Qwen-7B (14.2 GB safetensors), cold cache:

| Method | PD | NVMe | GCS |
|---|---|---|---|
| `safetensors.load_file` | 47.3s / 0.32 GB/s | 10.4s / 1.47 GB/s | n/a |
| **`runai model streamer`** | 47.7s / 0.32 GB/s | **3.5s / 4.31 GB/s** | **4.7s / 3.21 GB/s** |
| `ray.anyscale.safetensors` | n/a | n/a | 8.2s / 1.87 GB/s |

| Speedup vs PD baseline | PD | NVMe | GCS |
|---|---|---|---|
| `safetensors.load_file` | 1.0x | 4.6x | n/a |
| **`runai model streamer`** | 1.0x | **13.4x** | **10.0x** |
| `ray.anyscale.safetensors` | n/a | n/a | 5.8x |

For a 70B model (~145 GB):

| Phase | Time | Notes |
|---|---|---|
| GCS → NVMe download | ~48s | First replica on a node only |
| NVMe → GPU (runai_streamer) | ~36s | Every replica |
| NVMe → GPU (HF/safetensors) | ~100s | Every replica |
| **Total first cold start** | **~84s** (runai) / **~148s** (HF) | Subsequent replicas skip download |
