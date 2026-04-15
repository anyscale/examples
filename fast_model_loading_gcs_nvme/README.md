# Fast Model Loading with GCS and NVMe

Deploy a 70B model on Anyscale with fast startup times by streaming model weights from Google Cloud Storage (GCS) to local NVMe SSDs using `ray.anyscale.safetensors`.

**Why this matters:** Loading a 70B model (~145 GB) from a persistent disk takes 5-10 minutes. Streaming from GCS with the Anyscale safetensors loader runs at ~2 GB/s, and subsequent loads from NVMe hit 3-7 GB/s. This directly reduces replica cold-start time.

## Architecture

```
GCS Bucket (durable, shared)
        │
        ▼  ray.anyscale.safetensors (~2 GB/s, one shard at a time)
Local NVMe (/mnt/local_storage, ~2 TB)
        │
        ▼  mmap (shared across tensor-parallel workers)
GPU Memory
```

**Why per-shard download + save to NVMe:**
- Peak RAM = 1 shard (~5 GB), not the full model — prevents OOM.
- `from_pretrained` mmaps the local files, so 8 TP workers share physical pages (~145 GB total, not 8×145 GB).
- One download per replica — coordinator downloads once, all workers mmap the same local files.

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

### What happens on startup

1. Worker node starts with 6× 375 GB NVMe local SSDs (~2.2 TB at `/mnt/local_storage`).
2. `serve.py` streams each safetensor shard from GCS → saves to NVMe (~2 GB/s per shard).
3. vLLM loads the model from NVMe via mmap — TP workers share physical pages.

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

Measured on A100-40GB with DeepSeek-R1-Distill-Qwen-7B (14.2 GB safetensors):

| Method | Time | Throughput | Speedup |
|---|---|---|---|
| Persistent disk → `safetensors.load_file` → GPU | 49.3s | 0.31 GB/s | baseline |
| **GCS stream → GPU** (`ray.anyscale.safetensors`) | **7.6s** | **2.01 GB/s** | **6.5×** |
| Per-shard GCS → NVMe → mmap → GPU | 43.3s | 0.35 GB/s | 1.1× |

The per-shard pattern is slower end-to-end for a single replica because of the save-to-disk step, but it uses constant memory and enables mmap sharing across TP workers — critical for multi-GPU serving.
