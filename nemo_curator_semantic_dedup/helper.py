# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper functions for downloading and preparing image datasets.

This module provides `parquet_to_webdataset_ray()` for converting parquet files 
(with URLs) to WebDataset format using Ray Data:

- Uses ThreadPoolExecutor + requests.Session for high-throughput downloads
- Connection pooling for efficient HTTP connections
- Scales across all nodes in the Ray cluster
- Best for large datasets (millions of images)
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pandas as pd
import requests
from PIL import Image

if TYPE_CHECKING:
    pass


# =============================================================================
# Image Download Utilities
# =============================================================================

def download_single_image(url: str, session: requests.Session) -> dict[str, Any]:
    """
    Download a single image using an existing session for connection pooling.
    
    Args:
        url: Image URL to download
        session: requests.Session with connection pooling configured
        
    Returns:
        Dict with 'content' (bytes or None), 'status', and 'url'
    """
    try:
        response = session.get(url, timeout=5, stream=True)
        
        if response.status_code == 200:
            return {"content": response.content, "status": "success", "url": url}
        else:
            return {"content": None, "status": f"http_{response.status_code}", "url": url}
    except Exception as e:
        return {"content": None, "status": f"error_{type(e).__name__}", "url": url}


def image_download_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """
    Download a batch of images using ThreadPoolExecutor for parallelism.
    
    Uses connection pooling for efficient HTTP connections - this is much
    faster than ray.data.expressions.download() which processes URLs one at a time.
    
    Args:
        batch: Dict with 'URL' or 'url' column containing URLs
        
    Returns:
        Batch with 'bytes' column added containing downloaded image bytes
    """
    # Get URLs from batch (handle both cases)
    urls = batch.get("URL", batch.get("url", []))
    
    # Create a session with connection pooling for efficient downloads
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=100,
        pool_maxsize=100,
        max_retries=0,  # No automatic retries - fail fast
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Use ThreadPoolExecutor for parallel downloads within this batch
    # 50 threads means 50 concurrent downloads per Ray task
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(lambda url: download_single_image(url, session), urls))
    
    # Add bytes column to batch
    batch["bytes"] = [r["content"] for r in results]
    return batch


# =============================================================================
# Image Validation Utilities
# =============================================================================

def validate_and_convert_to_jpeg(image_bytes: bytes) -> bytes | None:
    """
    Validate image and convert to JPEG format for DALI compatibility.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        JPEG bytes if valid, None if image is invalid/corrupted
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verify it's a valid image
        # Re-open after verify (verify consumes the file)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Robust RGB conversion for ALL image modes (L, LA, P, PA, RGBA, CMYK, etc.)
        # This ensures CLIP gets 3-channel images
        if img.mode != "RGB":
            # For palette images, convert to RGBA first to preserve transparency info
            if img.mode == "P":
                img = img.convert("RGBA")
            # For any mode with alpha, composite onto white background
            if img.mode in ("RGBA", "LA", "PA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                # Use alpha channel as mask
                if img.mode == "LA":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                # Simple conversion for grayscale (L), CMYK, etc.
                img = img.convert("RGB")
        
        # Final safety check - ensure we have exactly 3 channels
        if img.mode != "RGB":
            return None
        
        # Skip images that are too small (CLIP needs at least 3x3 to avoid channel ambiguity)
        if img.size[0] < 3 or img.size[1] < 3:
            return None
        
        # Re-encode as JPEG to ensure DALI compatibility
        jpeg_buffer = io.BytesIO()
        img.save(jpeg_buffer, format="JPEG", quality=95)
        return jpeg_buffer.getvalue()
    except Exception:
        return None


def process_image(row: dict[str, Any]) -> dict[str, Any]:
    """
    Process a single image: validate and convert to JPEG.
    
    This is used as a Ray Data map function after downloading.
    """
    image_bytes = row.get("bytes")
    if image_bytes is None:
        row["jpeg_bytes"] = None
        return row
    
    row["jpeg_bytes"] = validate_and_convert_to_jpeg(image_bytes)
    return row


def write_tar_shard(
    images: list[dict[str, Any]],
    output_path: str,
    shard_id: str,
) -> dict[str, int]:
    """
    Write a tar shard with downloaded images.
    
    Args:
        images: List of dicts with 'url', 'caption', 'jpeg_bytes'
        output_path: Path to write tar file
        shard_id: Unique identifier for this shard
        
    Returns:
        Dict with 'success_count' and 'total_count'
    """
    # Count successful images first to avoid creating empty tar files
    valid_images = [img for img in images if img["jpeg_bytes"] is not None]
    if not valid_images:
        # Don't create empty tar files - DALI will crash on them
        return {"success_count": 0, "total_count": len(images)}
    
    success_count = 0
    metadatas = []
    
    with tarfile.open(output_path, "w") as tar:
        for i, img_data in enumerate(images):
            if img_data["jpeg_bytes"] is None:
                continue
            
            key = f"{shard_id}_{i:06d}"
            jpeg_bytes = img_data["jpeg_bytes"]
            
            # Add image bytes
            jpg_info = tarfile.TarInfo(name=f"{key}.jpg")
            jpg_info.size = len(jpeg_bytes)
            tar.addfile(jpg_info, fileobj=io.BytesIO(jpeg_bytes))
            
            # Add caption text
            caption_bytes = str(img_data["caption"]).encode("utf-8")
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(caption_bytes)
            tar.addfile(txt_info, fileobj=io.BytesIO(caption_bytes))
            
            # Add JSON metadata
            meta = {"url": img_data["url"], "caption": img_data["caption"], "key": key}
            json_bytes = json.dumps(meta).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, fileobj=io.BytesIO(json_bytes))
            
            metadatas.append(meta)
            success_count += 1
    
    # Write parquet sidecar
    if metadatas:
        parquet_path = output_path.replace(".tar", ".parquet")
        pd.DataFrame(metadatas).to_parquet(parquet_path)
    
    return {"success_count": success_count, "total_count": len(images)}


# =============================================================================
# Ray Data Approach (Distributed)
# =============================================================================

def write_tar_batch(batch: dict[str, Any], output_dir: str) -> dict[str, Any]:
    """
    Ray Data map_batches function to write a batch of images to a tar shard.
    
    This function is called after images have been downloaded and processed.
    It writes valid images to a WebDataset tar file.
    
    Args:
        batch: Dict with 'URL', 'TEXT', 'jpeg_bytes' arrays
        output_dir: Directory to write tar files
        
    Returns:
        Dict with statistics about processing
    """
    import ray
    
    # Generate unique shard ID using node ID + UUID to avoid collisions
    node_id = ray.get_runtime_context().get_node_id()[:8]
    shard_id = f"{node_id}_{uuid.uuid4().hex[:8]}"
    tar_path = os.path.join(output_dir, f"{shard_id}.tar")
    
    # Convert batch to list of dicts for write_tar_shard
    images = []
    urls = batch.get("URL", batch.get("url", []))
    texts = batch.get("TEXT", batch.get("text", batch.get("caption", [])))
    jpeg_bytes_list = batch.get("jpeg_bytes", [])
    
    for i in range(len(urls)):
        images.append({
            "url": urls[i],
            "caption": texts[i] if i < len(texts) else "",
            "jpeg_bytes": jpeg_bytes_list[i] if i < len(jpeg_bytes_list) else None,
        })
    
    # Write tar shard
    stats = write_tar_shard(images, tar_path, shard_id)
    
    # Return statistics as a single-row batch
    return {
        "shard_id": [shard_id],
        "success_count": [stats["success_count"]],
        "total_count": [stats["total_count"]],
    }


def parquet_to_webdataset_ray(
    parquet_path: str,
    output_dir: str,
    entries_per_tar: int = 1000,
    max_entries: int | None = None,
    concurrency: int | None = None,
) -> dict[str, int]:
    """
    Convert parquet file with URLs to WebDataset tar files using Ray Data.
    
    Uses ThreadPoolExecutor + requests.Session with connection pooling for
    high-throughput downloads. Each Ray task spawns 50 threads for concurrent
    downloads, making this much faster than sequential per-row approaches.
    
    Args:
        parquet_path: Path to parquet file with URL and TEXT columns
        output_dir: Directory to save tar files
        entries_per_tar: Number of entries per tar shard
        max_entries: Maximum entries to process (for testing)
        concurrency: Number of concurrent download tasks (defaults to num CPUs)
        
    Returns:
        Dict with 'total_success' and 'total_attempted' counts
    """
    import ray
    import ray.data
    from functools import partial
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading parquet from: {parquet_path}")
    
    # Read parquet with Ray Data - this distributes reading across the cluster
    ds = ray.data.read_parquet(parquet_path)
    
    # Get schema and normalize column names
    schema = ds.schema()
    col_names = schema.names if hasattr(schema, 'names') else [f.name for f in schema]
    
    # Find URL column (case-insensitive)
    url_col = None
    text_col = None
    for col in col_names:
        if col.lower() == "url":
            url_col = col
        elif col.lower() in ("text", "caption"):
            text_col = col
    
    if url_col is None:
        raise ValueError(f"No URL column found in parquet. Available: {col_names}")
    
    print(f"Using columns: URL={url_col}, TEXT={text_col}")
    
    # Select only the columns we need
    columns_to_keep = [url_col]
    if text_col:
        columns_to_keep.append(text_col)
    ds = ds.select_columns(columns_to_keep)
    
    # Apply max_entries limit
    if max_entries is not None:
        print(f"Limiting to {max_entries} entries for testing")
        ds = ds.limit(max_entries)
    
    # Count total rows for progress reporting
    total_rows = ds.count()
    print(f"Total entries to process: {total_rows}")
    
    # Determine concurrency based on cluster resources
    if concurrency is None:
        cluster_resources = ray.cluster_resources()
        concurrency = max(4, int(cluster_resources.get("CPU", 4)))
    
    print(f"Processing with concurrency={concurrency}, entries_per_tar={entries_per_tar}")
    
    # Step 1: Download images using ThreadPoolExecutor + connection pooling
    # This is much faster than ray.data.expressions.download() because:
    # - 50 concurrent downloads per Ray task via ThreadPoolExecutor
    # - Connection pooling reuses HTTP connections
    print("Downloading images with ThreadPoolExecutor + connection pooling...")
    ds = ds.map_batches(
        image_download_batch,
        batch_size=100,  # 100 URLs per batch, 50 threads = fast downloads
        batch_format="numpy",
    )
    
    # Step 2: Process images - validate and convert to JPEG
    # Filter out failed downloads first
    ds = ds.filter(lambda row: row.get("bytes") is not None)
    ds = ds.map(process_image)
    
    # Step 3: Filter out images that failed validation
    ds = ds.filter(lambda row: row.get("jpeg_bytes") is not None)
    
    # Normalize column names for tar writing
    def normalize_columns(row):
        return {
            "URL": row.get(url_col, row.get("url", "")),
            "TEXT": row.get(text_col, row.get("text", row.get("caption", ""))) if text_col else "",
            "jpeg_bytes": row.get("jpeg_bytes"),
        }
    ds = ds.map(normalize_columns)
    
    # Step 4: Write tar shards in batches
    # Note: Don't use repartition() here - it's a barrier that blocks streaming
    write_fn = partial(write_tar_batch, output_dir=output_dir)
    
    results_ds = ds.map_batches(
        write_fn,
        batch_size=entries_per_tar,
        batch_format="numpy",
        concurrency=concurrency,
    )
    
    # Materialize results and aggregate statistics
    results = results_ds.take_all()
    
    total_success = sum(r["success_count"] for r in results)
    total_attempted = total_rows
    num_shards = len(results)
    
    # Report results
    success_rate = (total_success / total_attempted * 100) if total_attempted > 0 else 0
    print(f"\n✓ Download complete: {total_success} images in {num_shards} shards ({success_rate:.1f}% success rate)")
    print(f"  Note: LAION datasets have high link rot - many URLs no longer work.")
    
    if total_success == 0:
        print("\n⚠️  WARNING: No images were downloaded successfully!")
        print("  This is likely due to LAION link rot. Try increasing MAX_ENTRIES.")
    
    return {
        "total_success": total_success,
        "total_attempted": total_attempted,
        "num_shards": num_shards,
    }
