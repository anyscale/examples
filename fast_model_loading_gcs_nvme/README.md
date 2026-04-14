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
- `gcloud` CLI authenticated (for the one-time upload step)

## Step 1: Upload model weights to GCS

Run once from any machine with `gcloud` configured:

```bash
pip install huggingface_hub
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT

# Create a bucket in the same region as your Anyscale cloud
gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1

# Download and upload
python upload_to_gcs.py
```

Or manually:

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --include "*.safetensors" "*.json" "*.model" \
    --local-dir /tmp/DeepSeek-R1-Distill-Llama-70B

gcloud storage cp -r /tmp/DeepSeek-R1-Distill-Llama-70B \
    gs://YOUR_BUCKET/models/DeepSeek-R1-Distill-Llama-70B
```

> **Tip:** If your bucket is under `$ANYSCALE_ARTIFACT_STORAGE`, pods can access it automatically via workload identity — no credentials needed.

## Step 2: Deploy the service

```bash
git clone https://github.com/anyscale/examples.git
cd examples/fast_model_loading_gcs_nvme
```

Set your GCS URI in `serve.py` or via environment variable:

```bash
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

The `anyscale service deploy` command outputs:
```
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
```

Edit `query.py` with your token and URL, then:
```bash
pip install openai
python query.py
```

## Shutdown

```bash
anyscale service terminate -n fast-model-loading
```

## Benchmark results

Measured on an A100-40GB node with a 7B model (14.2 GB safetensors):

| Method | Time | Throughput | Speedup |
|---|---|---|---|
| Persistent disk → `safetensors.load_file` → GPU | 49.3s | 0.31 GB/s | baseline |
| **GCS stream → GPU** (`ray.anyscale.safetensors`) | **7.6s** | **2.01 GB/s** | **6.5×** |
| Per-shard GCS → NVMe → mmap → GPU | 43.3s | 0.35 GB/s | 1.1× |

The per-shard pattern is slower end-to-end for a single replica because of the save-to-disk step, but it uses constant memory and enables mmap sharing across TP workers — critical for multi-GPU serving.
