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

This module provides two approaches for converting parquet files (with URLs) to WebDataset format:

1. `parquet_to_webdataset_ray()` - Distributed approach using Ray Data (recommended)
   - Scales across all nodes in the cluster
   - Uses Ray Data for parallel reading and processing
   - Best for large datasets (millions of images)

2. `download_webdataset()` - Single-node multiprocessing approach (legacy)
   - Runs on a single machine
   - Uses Python multiprocessing for parallelism
   - Simpler but doesn't scale beyond one node
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tarfile
import uuid
from typing import TYPE_CHECKING, Any

import aiohttp
import pandas as pd
from loguru import logger
from PIL import Image

if TYPE_CHECKING:
    pass

# HTTP status codes
HTTP_OK = 200


# =============================================================================
# Image Download and Validation Utilities
# =============================================================================

async def fetch_image_bytes(session: aiohttp.ClientSession, url: str, retries: int = 3) -> bytes | None:
    """Fetch image bytes from URL with retries."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == HTTP_OK:
                    return await response.read()
                last_error = f"HTTP {response.status}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = str(e)

        if attempt < retries:
            await asyncio.sleep(1)

    return None


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


async def download_batch_images(
    batch: pd.DataFrame,
    url_col: str = "URL",
    text_col: str = "TEXT",
) -> list[dict[str, Any]]:
    """
    Download images for a batch of URLs asynchronously.
    
    Args:
        batch: DataFrame with URL and TEXT columns
        url_col: Name of URL column
        text_col: Name of text/caption column
        
    Returns:
        List of dicts with 'url', 'caption', 'jpeg_bytes' (None if failed)
    """
    timeout = aiohttp.ClientTimeout(total=15)
    connector = aiohttp.TCPConnector(limit=256, limit_per_host=16)
    
    results = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        metadata = []
        
        for _, row in batch.iterrows():
            url = row[url_col]
            caption = row[text_col]
            metadata.append({"url": url, "caption": caption})
            tasks.append(fetch_image_bytes(session, url, retries=3))
        
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for meta, raw_bytes in zip(metadata, raw_results):
            jpeg_bytes = None
            if isinstance(raw_bytes, bytes) and raw_bytes:
                jpeg_bytes = validate_and_convert_to_jpeg(raw_bytes)
            
            results.append({
                "url": meta["url"],
                "caption": meta["caption"],
                "jpeg_bytes": jpeg_bytes,
            })
    
    return results


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

def process_batch_ray(batch: dict[str, Any], output_dir: str) -> dict[str, Any]:
    """
    Ray Data map function to process a batch of URLs.
    
    This function is called by Ray Data's map_batches() and runs distributed
    across all nodes in the cluster.
    
    Args:
        batch: Dict with 'URL' and 'TEXT' arrays (Ray Data batch format)
        output_dir: Directory to write tar files
        
    Returns:
        Dict with statistics about processing
    """
    import ray
    
    # Convert Ray Data batch format to DataFrame
    df = pd.DataFrame({
        "URL": batch["URL"],
        "TEXT": batch["TEXT"],
    })
    
    # Generate unique shard ID using node ID + UUID to avoid collisions
    node_id = ray.get_runtime_context().get_node_id()[:8]
    shard_id = f"{node_id}_{uuid.uuid4().hex[:8]}"
    tar_path = os.path.join(output_dir, f"{shard_id}.tar")
    
    # Download images asynchronously
    images = asyncio.run(download_batch_images(df))
    
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
    
    This distributes the download work across all nodes in the Ray cluster,
    providing much better scalability than single-node processing.
    
    Args:
        parquet_path: Path to parquet file with URL and TEXT columns
        output_dir: Directory to save tar files
        entries_per_tar: Number of entries per tar shard
        max_entries: Maximum entries to process (for testing)
        concurrency: Number of concurrent download tasks (defaults to num CPUs)
        
    Returns:
        Dict with 'total_success' and 'total_attempted' counts
    """
    import ray.data
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading parquet from: {parquet_path}")
    
    # Read parquet with Ray Data - this distributes reading across the cluster
    ds = ray.data.read_parquet(parquet_path)
    
    # Get schema and normalize column names
    schema = ds.schema()
    col_names = schema.names if hasattr(schema, 'names') else [f.name for f in schema]
    col_map = {}
    
    # Handle case-insensitive column matching
    for col in col_names:
        if col.lower() == "url":
            col_map[col] = "URL"
        elif col.lower() in ("text", "caption"):
            col_map[col] = "TEXT"
    
    if col_map:
        # Rename columns to standard names
        def rename_cols(batch):
            result = {}
            for old_name, new_name in col_map.items():
                if old_name in batch:
                    result[new_name] = batch[old_name]
            # Keep any columns that weren't renamed
            for col in batch:
                if col not in col_map and col not in result:
                    result[col] = batch[col]
            return result
        
        ds = ds.map_batches(rename_cols, batch_format="pandas")
    
    # Select only the columns we need
    ds = ds.select_columns(["URL", "TEXT"])
    
    # Apply max_entries limit
    if max_entries is not None:
        print(f"Limiting to {max_entries} entries for testing")
        ds = ds.limit(max_entries)
    
    # Count total rows for progress reporting
    total_rows = ds.count()
    print(f"Total entries to process: {total_rows}")
    
    # Process batches in parallel across the cluster
    # Each batch becomes one tar shard
    from functools import partial
    
    process_fn = partial(process_batch_ray, output_dir=output_dir)
    
    # Determine concurrency based on cluster resources
    if concurrency is None:
        import ray
        cluster_resources = ray.cluster_resources()
        concurrency = max(1, int(cluster_resources.get("CPU", 4) // 2))
    
    print(f"Processing with concurrency={concurrency}, entries_per_tar={entries_per_tar}")
    
    # map_batches distributes work across all nodes
    results_ds = ds.map_batches(
        process_fn,
        batch_size=entries_per_tar,
        batch_format="numpy",
        concurrency=concurrency,
    )
    
    # Materialize results and aggregate statistics
    results = results_ds.take_all()
    
    total_success = sum(r["success_count"] for r in results)
    total_attempted = sum(r["total_count"] for r in results)
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


# =============================================================================
# Single-Node Multiprocessing Approach (Legacy)
# =============================================================================

async def process_batch_single_node(batch: pd.DataFrame, output_dir: str, batch_num: int) -> int:
    """Process a batch of URLs and return the number of successfully downloaded images."""
    tar_filename = os.path.join(output_dir, f"{batch_num:05d}.tar")
    shard_id = f"{batch_num:05d}"
    
    # Download images
    images = await download_batch_images(batch)
    
    # Write tar shard
    stats = write_tar_shard(images, tar_filename, shard_id)
    return stats["success_count"]


def process_parquet_chunk(chunk: tuple[int, pd.DataFrame], output_dir: str) -> int:
    """Process a chunk and return the number of successfully downloaded images."""
    batch_num, batch = chunk
    return asyncio.run(process_batch_single_node(batch, output_dir, batch_num))


def download_webdataset(
    parquet_path: str,
    output_dir: str,
    entries_per_tar: int = 10000,
    num_processes: int = 2,
    max_entries: int | None = None,
) -> None:
    """
    Single-node approach: Stream parquet into WebDataset tar shards using multiprocessing.
    
    This is the legacy approach that runs on a single machine. For distributed
    processing across a Ray cluster, use `parquet_to_webdataset_ray()` instead.
    
    Args:
        parquet_path: Path to the parquet file containing URLs and text
        output_dir: Directory to save the webdataset tar files
        entries_per_tar: Number of entries per tar file
        num_processes: Number of parallel download processes
        max_entries: Maximum number of entries to process (for testing). None = no limit.
    """
    import math
    from functools import partial
    from multiprocessing import Pool
    
    import pyarrow.dataset as pa_ds
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)

    # Stream the Parquet in batches
    dataset = pa_ds.dataset(parquet_path, format="parquet")
    schema = dataset.schema
    available = set(schema.names)

    def resolve_cols() -> list[str]:
        resolved = []
        for col in ["URL", "TEXT"]:
            if col in available:
                resolved.append(col)
                continue
            lower = col.lower()
            if lower in available:
                resolved.append(lower)
                continue
            if col.upper() == "TEXT" and "caption" in available:
                resolved.append("caption")
        if not resolved:
            raise ValueError(f"No URL/TEXT-like columns found in {parquet_path}; available: {sorted(available)}")
        return resolved

    resolved_cols = resolve_cols()
    total_rows = dataset.count_rows()

    # Apply max_entries limit for testing
    if max_entries is not None and total_rows is not None:
        total_rows = min(total_rows, max_entries)
        print(f"Limiting to {max_entries} entries for testing")

    total_chunks = math.ceil(total_rows / entries_per_tar) if total_rows is not None else None

    def batch_iter():
        batch_num = 0
        rows_yielded = 0
        for batch in dataset.to_batches(columns=resolved_cols, batch_size=entries_per_tar):
            df = batch.to_pandas()

            # Apply max_entries limit
            if max_entries is not None:
                remaining = max_entries - rows_yielded
                if remaining <= 0:
                    break
                if len(df) > remaining:
                    df = df.head(remaining)

            # normalize column names to URL/TEXT expected downstream
            col_map: dict[str, str] = {}
            if "url" in df.columns and "URL" not in df.columns:
                col_map["url"] = "URL"
            if "caption" in df.columns and "TEXT" not in df.columns:
                col_map["caption"] = "TEXT"
            df = df.rename(columns=col_map)
            yield (batch_num, df)
            rows_yielded += len(df)
            batch_num += 1

    total_success = 0
    total_attempted = 0
    with Pool(processes=num_processes) as pool:
        func = partial(process_parquet_chunk, output_dir=output_dir)
        for success_count in tqdm(
            pool.imap_unordered(func, batch_iter()),
            total=total_chunks,
            desc="Processing chunks",
            unit="chunk",
        ):
            total_success += success_count
            total_attempted += entries_per_tar  # approximate

    # Report download success rate
    success_rate = (total_success / total_attempted * 100) if total_attempted > 0 else 0
    print(f"\n✓ Download complete: {total_success} images saved ({success_rate:.1f}% success rate)")
    print(f"  Note: LAION datasets have high link rot - many URLs no longer work.")
    
    if total_success == 0:
        print("\n⚠️  WARNING: No images were downloaded successfully!")
        print("  This is likely due to LAION link rot. Try increasing MAX_ENTRIES.")

    # Best-effort cleanup of legacy tmp dir from previous versions
    tmp_dir = os.path.join(output_dir, "tmp")
    try:
        if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)
    except OSError as e:
        logger.debug(f"Failed to remove tmp dir {tmp_dir}: {e}")
