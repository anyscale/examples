"""
pipeline.py — DROID → BLIP captioning pipeline (Ray Data).

Reads a copy of the DROID Raw 1.0.1 dataset from a public S3 bucket,
generates image captions for each wrist-camera frame using BLIP-large
on GPU, and writes annotated parquet partitioned by episode uuid.

This is a heterogeneous pipeline: CPU workers handle I/O-bound data
loading while GPU workers run model inference. Ray Data manages
backpressure between stages automatically.

See: https://docs.ray.io/en/latest/data/overview.html

Stages
------
  [manifest parquet]
      │
      ▼  flat_map(episode_to_training_rows)  — CPU
         Resolve S3 paths, stream HDF5 + MP4; yield one row per timestep.
      │
      ▼  map_batches(BlipCaptionStage)  — GPU
         BLIP-large fp16: image captioning for each wrist-camera frame.
      │
      ▼  write_parquet, partitioned by uuid

Usage
-----
  uv run python pipeline.py
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import ray
import ray.data
import torch
from PIL import Image

from data import episode_to_training_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MANIFEST       = "episodes_droid_v1.0.1_s3.parquet"
OUTPUT         = "/mnt/cluster_storage/droid-annotated"
NUM_EPISODES: int | None = None  # set to an int to limit for smoke tests

MODEL_NAME: str = "Salesforce/blip-image-captioning-large"
CUSTOM_ENV_VAR: str = "hello"  # example: propagated to all workers via runtime_env

# -- CPU data-loading stage --
# Fractional CPUs per flat_map worker. Episode loading is I/O-bound (S3
# download + video decode), so each worker needs little CPU. Setting < 1
# lets Ray pack more loaders per node, increasing S3 parallelism.
# See: https://docs.ray.io/en/latest/data/transforming-data.html#configuring-parallelism
CPU_LOADER_NUM_CPUS: float = 0.5

# -- GPU inference stage --
GPU_WORKER_NUM_GPUS: float = 1.0
# (min, max) actor pool size. Ray Data autoscales within this range based
# on throughput.
GPU_CONCURRENCY: tuple[int, int] = (24, 32)
# Rows per GPU batch. Increase until VRAM is ~80% utilized; BLIP-large fp16
# uses ~1.5 GB, so 32–64 fits comfortably on most GPUs.
GPU_BATCH_SIZE: int = 32


# ---------------------------------------------------------------------------
# GPU stage: BLIP captioning
# ---------------------------------------------------------------------------


class BlipCaptionStage:
    """Ray Data actor-class callable for GPU inference.

    When passed as a class to map_batches(), Ray Data creates a pool of actors
    (one per GPU).  __init__ runs once per actor to load the model;
    __call__ is invoked per batch.  This pattern avoids reloading the model
    for every batch.

    See: https://docs.ray.io/en/latest/data/transforming-data.html#stateful-setup-with-actors
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # Import here so transformers is only loaded on GPU workers, not the driver.
        from transformers import BlipForConditionalGeneration, BlipProcessor

        log.info("Loading BLIP model: %s …", model_name)
        t0 = time.monotonic()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16,
        ).to(self.device).eval()

        log.info(
            "BLIP loaded on %s in %.1fs  (fp16, %d params).",
            self.device,
            time.monotonic() - t0,
            sum(p.numel() for p in self.model.parameters()),
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process one batch of rows. Ray Data calls this with a dict of
        column-name → list[value], not a single row at a time."""
        images_raw: list[np.ndarray] = batch["wrist_frame"]
        n = len(images_raw)
        t0 = time.monotonic()

        pil_images = [Image.fromarray(arr, mode="RGB") for arr in images_raw]

        with torch.inference_mode():
            inputs = self.processor(
                images=pil_images, return_tensors="pt", padding=True,
            ).to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        batch["caption"] = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        elapsed = time.monotonic() - t0
        log.info(
            "GPU batch: %d frames in %.2fs  (%.1f frames/s)",
            n, elapsed, n / elapsed if elapsed > 0 else 0.0,
        )
        return batch


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Connect to the existing Ray cluster (started by Anyscale or `ray start`).
# runtime_env propagates env vars, pip packages, or working_dir to all workers.
# See: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments
ray.init(
    address="auto",
    ignore_reinit_error=True,
    runtime_env={"env_vars": {"CUSTOM_ENV_VAR": CUSTOM_ENV_VAR}},
)

# read_parquet returns a lazy Dataset — no data is read until execution.
# See: https://docs.ray.io/en/latest/data/loading-data.html
ds = ray.data.read_parquet(MANIFEST)
if NUM_EPISODES is not None:
    ds = ds.limit(NUM_EPISODES)

ds = (
    ds
    # flat_map: 1 episode → N timestep rows (CPU, I/O-bound).
    # Ray Data runs this on CPU workers and streams results to the GPU stage.
    # num_cpus < 1 lets Ray pack many loaders per node for higher S3 throughput.
    .flat_map(episode_to_training_rows, num_cpus=CPU_LOADER_NUM_CPUS)
    # map_batches with a class + num_gpus: creates a pool of GPU actors.
    # Each actor gets batch_size rows at a time as a dict of columns.
    # See: https://docs.ray.io/en/latest/data/transforming-data.html#computing-over-batches
    .map_batches(
        BlipCaptionStage,
        batch_size=GPU_BATCH_SIZE,
        num_gpus=GPU_WORKER_NUM_GPUS,
        concurrency=GPU_CONCURRENCY,
    )
)

# write_parquet triggers execution of the full pipeline.
# Everything above is lazy — this call materializes the DAG.
ds.write_parquet(OUTPUT, partition_cols=["uuid"])
