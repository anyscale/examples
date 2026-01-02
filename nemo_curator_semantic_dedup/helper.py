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

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import tarfile
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING

import aiohttp
import pandas as pd
import pyarrow.dataset as pa_ds
from loguru import logger
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from nemo_curator.tasks import ImageObject
    from nemo_curator.tasks.image import ImageBatch

# HTTP status codes
HTTP_OK = 200


async def fetch_image_bytes(session: aiohttp.ClientSession, url: str, retries: int = 3) -> bytes | None:
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

    # Only log final failure (not every retry) to reduce noise
    # logger.debug(f"Failed: {url} ({last_error})")
    return None


async def process_batch(batch: pd.DataFrame, output_dir: str, batch_num: int) -> int:
    """Process a batch of URLs and return the number of successfully downloaded images."""
    tar_filename = os.path.join(output_dir, f"{batch_num:05d}.tar")

    metadatas = []
    # Set timeout and connection limits for the session
    timeout = aiohttp.ClientTimeout(total=15)
    connector = aiohttp.TCPConnector(limit=256, limit_per_host=16)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for i, (_, row) in enumerate(batch.iterrows()):
            caption = row["TEXT"]
            url = row["URL"]

            key = f"{batch_num:05d}{i:04d}"

            meta = {"url": url, "caption": caption, "key": key}
            metadatas.append(meta)

            tasks.append(fetch_image_bytes(session, url, retries=3))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = 0
    with tarfile.open(tar_filename, "w") as tar:
        for i, result in enumerate(results):
            # Only proceed for successful downloads (bytes)
            if isinstance(result, bytes) and result:
                # Validate and convert to JPEG (DALI doesn't support WebP/other formats)
                try:
                    img = Image.open(io.BytesIO(result))
                    img.verify()  # Verify it's a valid image
                    # Re-open after verify (verify consumes the file)
                    img = Image.open(io.BytesIO(result))
                    
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
                        continue  # Skip if conversion somehow failed
                    
                    # Skip images that are too small (CLIP needs at least 3x3 to avoid channel ambiguity)
                    if img.size[0] < 3 or img.size[1] < 3:
                        continue
                    
                    # Re-encode as JPEG to ensure DALI compatibility
                    jpeg_buffer = io.BytesIO()
                    img.save(jpeg_buffer, format="JPEG", quality=95)
                    jpeg_bytes = jpeg_buffer.getvalue()
                except Exception:
                    # Skip invalid/corrupted images (e.g., HTML error pages)
                    continue

                success_count += 1
                key = f"{batch_num:05d}{i:04d}"

                # Add image bytes (now guaranteed to be JPEG)
                jpg_info = tarfile.TarInfo(name=f"{key}.jpg")
                jpg_info.size = len(jpeg_bytes)
                tar.addfile(jpg_info, fileobj=io.BytesIO(jpeg_bytes))

                # Add caption text
                caption_bytes = str(metadatas[i]["caption"]).encode("utf-8")
                txt_info = tarfile.TarInfo(name=f"{key}.txt")
                txt_info.size = len(caption_bytes)
                tar.addfile(txt_info, fileobj=io.BytesIO(caption_bytes))

                # Add JSON metadata
                json_bytes = json.dumps(metadatas[i]).encode("utf-8")
                json_info = tarfile.TarInfo(name=f"{key}.json")
                json_info.size = len(json_bytes)
                tar.addfile(json_info, fileobj=io.BytesIO(json_bytes))

    # Write parquet
    meta_df = pd.DataFrame(metadatas)
    parquet_path = os.path.join(output_dir, f"{batch_num:05d}.parquet")
    meta_df.to_parquet(parquet_path)
    
    return success_count


def process_parquet_chunk(chunk: tuple[int, pd.DataFrame], output_dir: str) -> int:
    """Process a chunk and return the number of successfully downloaded images."""
    batch_num, batch = chunk
    return asyncio.run(process_batch(batch, output_dir, batch_num))


def download_webdataset(
    parquet_path: str,
    output_dir: str,
    entries_per_tar: int = 10000,
    num_processes: int = 2,
    max_entries: int | None = None,
) -> None:
    """Stream a large Parquet of URLs/TEXT into WebDataset tar shards.

    Uses pyarrow dataset streaming to avoid loading the entire Parquet into memory,
    so it can scale to 100M+ rows (e.g., LAION subsets).

    Args:
        parquet_path: Path to the parquet file containing URLs and text
        output_dir: Directory to save the webdataset tar files
        entries_per_tar: Number of entries per tar file
        num_processes: Number of parallel download processes
        max_entries: Maximum number of entries to process (for testing). None = no limit.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stream the Parquet in batches; resolve URL/TEXT in a case-insensitive way and map TEXT->caption if needed
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
        