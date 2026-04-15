# Fast Model Loading with GCS and NVMe

Deploy a 70B model on Anyscale with fast cold starts using the [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) to stream safetensor weights from GCS directly to GPU memory.

**Why this matters:** Loading a 70B model (~145 GB) from a persistent disk takes ~8 minutes. The Run:ai streamer uses concurrent C++ threads to stream from GCS at ~3 GB/s or from NVMe at ~4 GB/s — cutting cold starts to under a minute.

## Architecture

```
GCS Bucket (durable, shared)
        │
        ▼  runai model streamer (concurrent C++ threads, ~3 GB/s)
GPU Memory
```

No intermediate disk writes. vLLM's `load_format="runai_streamer"` handles everything — the serve.py is just config.

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
2. vLLM uses the Run:ai Model Streamer (`load_format="runai_streamer"`) to stream safetensor shards from GCS directly to GPU memory using concurrent C++ threads.
3. No disk download step — weights go straight from GCS to GPU.

### NVMe configuration

The `service.yaml` attaches 6 NVMe local SSDs via `advanced_instance_config`. Each is 375 GB. Anyscale RAIDs and mounts them at `/mnt/local_storage`. These are useful for caching if you want to pre-download weights for even faster subsequent loads. See [Anyscale NVMe docs](https://docs.anyscale.com/configuration/compute/gcp#nvme) for details.

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
| `safetensors.load_file` | 1.0× | 4.6× | n/a |
| **`runai model streamer`** | 1.0× | **13.4×** | **10.0×** |
| `ray.anyscale.safetensors` | n/a | n/a | 5.8× |

The Run:ai streamer's concurrent C++ threads saturate both NVMe bandwidth and GCS network throughput far better than single-threaded mmap or Python-based downloaders. On persistent disk, all loaders hit the same ~0.32 GB/s ceiling (disk-bound).

For a 70B model (~145 GB), estimated cold-start: **~48s from GCS**, **~36s from NVMe**.
