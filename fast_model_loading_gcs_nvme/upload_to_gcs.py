"""One-time script: download a model from Hugging Face and upload to GCS."""

import os
from pathlib import Path

from google.cloud.storage import Client, transfer_manager
from huggingface_hub import snapshot_download

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
GCS_BUCKET = os.environ.get("GCS_BUCKET", os.environ.get("ANYSCALE_CLOUD_STORAGE_BUCKET", ""))
GCS_PREFIX = os.environ.get("GCS_PREFIX", "models")

if not GCS_BUCKET:
    raise ValueError("Set GCS_BUCKET or ANYSCALE_CLOUD_STORAGE_BUCKET")

print(f"=== Downloading {HF_MODEL_ID} from Hugging Face ===")
local_dir = snapshot_download(
    repo_id=HF_MODEL_ID,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"],
    max_workers=8,
)
print(f"Downloaded to: {local_dir}")

model_name = HF_MODEL_ID.split("/")[-1]
dest_prefix = f"{GCS_PREFIX}/{model_name}"

print(f"\n=== Uploading to gs://{GCS_BUCKET}/{dest_prefix} ===")
bucket = Client().bucket(GCS_BUCKET)
files = [p.relative_to(local_dir).as_posix() for p in Path(local_dir).rglob("*") if p.is_file()]
transfer_manager.upload_many_from_filenames(
    bucket, files, source_directory=local_dir, blob_name_prefix=f"{dest_prefix}/", max_workers=8
)

print(f"\n=== Done ===")
print(f"Set in serve.py or as env var:")
print(f'  GCS_MODEL_URI="gs://{GCS_BUCKET}/{dest_prefix}"')
