#!/usr/bin/env python3
"""
Utility to download a LeRobot dataset from Hugging Face and upload it to S3.

Usage:
    python lerobot_to_s3.py --dataset lerobot/pusht --bucket my-bucket
    python lerobot_to_s3.py --dataset lerobot/pusht --bucket my-bucket --s3-prefix lerobot/
    python lerobot_to_s3.py --dataset lerobot/pusht --bucket my-bucket --revision main --resume
"""

import argparse
import hashlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    sys.exit("Missing dependency: pip install boto3")

try:
    from huggingface_hub import HfApi, RepoFile, hf_hub_download, list_repo_tree
except ImportError:
    sys.exit("Missing dependency: pip install huggingface_hub")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Missing dependency: pip install tqdm")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("lerobot_to_s3")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _s3_key(prefix: str, relative_path: str) -> str:
    """Build the full S3 key from the prefix and a file's relative path."""
    return f"{prefix.rstrip('/')}/{relative_path}"


def _s3_object_exists(s3_client, bucket: str, key: str, expected_size: int | None = None) -> bool:
    """Check whether an S3 object already exists (optionally with matching size)."""
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
        if expected_size is not None:
            return resp["ContentLength"] == expected_size
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def _upload_file_to_s3(
    s3_client,
    local_path: str,
    bucket: str,
    key: str,
    file_size: int,
    pbar: tqdm | None = None,
) -> None:
    """Upload a single file to S3 with a progress callback."""

    def _progress_callback(bytes_transferred: int) -> None:
        if pbar is not None:
            pbar.update(bytes_transferred)

    s3_client.upload_file(
        local_path,
        bucket,
        key,
        Callback=_progress_callback,
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def list_dataset_files(api: HfApi, dataset_id: str, revision: str) -> list[dict]:
    """
    List every file in a Hugging Face dataset repo.

    Returns a list of dicts with keys: rfilename, size.
    """
    files = []
    for entry in api.list_repo_tree(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
    ):
        # Only include blobs (files), not trees (directories)
        if isinstance(entry, RepoFile):
            files.append({
                "rfilename": entry.path,
                "size": entry.size or 0,
            })
    return files


def sync_dataset_to_s3(
    dataset_id: str,
    bucket: str,
    s3_prefix: str = "lerobot/",
    revision: str = "main",
    resume: bool = True,
    hf_token: str | None = None,
) -> None:
    """
    Download a LeRobot dataset from Hugging Face and stream it to S3.

    Each file is downloaded to a temporary location, uploaded to S3, then
    deleted locally — keeping disk usage minimal.

    Args:
        dataset_id: HF dataset repo id, e.g. "lerobot/pusht".
        bucket: Target S3 bucket name.
        s3_prefix: Key prefix inside the bucket (default "lerobot/").
        revision: Git revision / branch / tag to download (default "main").
        resume: If True, skip files that already exist in S3 with matching size.
        hf_token: Optional Hugging Face token for private datasets.
    """
    api = HfApi(token=hf_token)
    s3 = boto3.client("s3")

    # Derive a sub-folder from the dataset id so multiple datasets stay
    # organised:  lerobot/<org>/<dataset_name>/...
    dataset_subpath = dataset_id  # e.g. "lerobot/pusht"

    log.info("Listing files in dataset %s (revision=%s) …", dataset_id, revision)
    files = list_dataset_files(api, dataset_id, revision)

    if not files:
        log.warning("No files found in %s — nothing to do.", dataset_id)
        return

    total_size = sum(f["size"] for f in files)
    log.info(
        "Found %d files totalling %.2f GB",
        len(files),
        total_size / (1024 ** 3),
    )

    # ------------------------------------------------------------------
    # Filter out already-uploaded files when resuming
    # ------------------------------------------------------------------
    if resume:
        pending_files = []
        skipped = 0
        for f in tqdm(files, desc="Checking S3 for existing files", unit="file"):
            key = _s3_key(s3_prefix, f"{dataset_subpath}/{f['rfilename']}")
            if _s3_object_exists(s3, bucket, key, expected_size=f["size"]):
                skipped += 1
            else:
                pending_files.append(f)
        if skipped:
            log.info("Resuming — skipped %d files already in S3.", skipped)
    else:
        pending_files = files

    if not pending_files:
        log.info("All files are already in S3. Nothing to upload.")
        return

    pending_size = sum(f["size"] for f in pending_files)
    log.info(
        "Uploading %d files (%.2f GB) to s3://%s/%s%s/ …",
        len(pending_files),
        pending_size / (1024 ** 3),
        bucket,
        s3_prefix.rstrip("/") + "/",
        dataset_subpath,
    )

    # ------------------------------------------------------------------
    # Download each file → upload to S3 → delete local copy
    # ------------------------------------------------------------------
    overall = tqdm(
        total=pending_size,
        desc="Overall progress",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )

    errors: list[str] = []

    for idx, f in enumerate(pending_files, 1):
        rfilename = f["rfilename"]
        file_size = f["size"]
        key = _s3_key(s3_prefix, f"{dataset_subpath}/{rfilename}")

        file_label = f"[{idx}/{len(pending_files)}] {rfilename}"
        log.info("Downloading %s …", file_label)

        try:
            # Download from HF to a temp directory
            local_path = hf_hub_download(
                repo_id=dataset_id,
                filename=rfilename,
                repo_type="dataset",
                revision=revision,
                token=hf_token,
            )

            log.info("Uploading %s → s3://%s/%s", file_label, bucket, key)
            _upload_file_to_s3(s3, local_path, bucket, key, file_size, pbar=overall)
            log.info("Uploaded %s ✓", rfilename)

        except Exception as exc:
            log.error("Failed on %s: %s", rfilename, exc)
            errors.append(rfilename)
            # Still update the progress bar so totals remain consistent
            overall.update(file_size)

    overall.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    succeeded = len(pending_files) - len(errors)
    log.info(
        "Done — %d/%d files uploaded to s3://%s/%s%s/",
        succeeded,
        len(pending_files),
        bucket,
        s3_prefix.rstrip("/") + "/",
        dataset_subpath,
    )
    if errors:
        log.warning("The following files failed (re-run with --resume to retry):")
        for name in errors:
            log.warning("  • %s", name)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a LeRobot dataset from Hugging Face and upload it to S3.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset repo id, e.g. 'lerobot/pusht'.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Target S3 bucket name.",
    )
    parser.add_argument(
        "--s3-prefix",
        default="lerobot/",
        help="Key prefix inside the bucket (default: 'lerobot/').",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision / branch / tag to download (default: 'main').",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip files already in S3 with matching size (default: True).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Re-upload all files even if they already exist in S3.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for private datasets (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    sync_dataset_to_s3(
        dataset_id=args.dataset,
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
        revision=args.revision,
        resume=args.resume,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
