# DROID → BLIP Captioning Pipeline

A [Ray Data](https://docs.ray.io/en/latest/data/overview.html) pipeline that reads the [DROID Raw 1.0.1](https://droid-dataset.github.io/) robotics dataset from a public S3 bucket, generates image captions for each wrist-camera frame using [BLIP-large](https://huggingface.co/Salesforce/blip-image-captioning-large) on GPU, and writes annotated parquet partitioned by episode.

Demonstrates **heterogeneous Ray Data pipelines** — CPU workers stream data from S3 while GPU workers run model inference, with Ray Data managing backpressure automatically.

## Files

| File | Description |
|---|---|
| `pipeline.py` | Main entry point — pipeline wiring, GPU stage, and config knobs |
| `data.py` | CPU stage — S3 I/O, HDF5 + MP4 reading, per-timestep row expansion |
| `episodes_droid_v1.0.1_s3.parquet` | Pre-built manifest of DROID episodes and their S3 paths |
| `pyproject.toml` | Dependencies (managed by `uv`) |

## Usage

```bash
# Install dependencies
uv sync

# Run on an existing Ray cluster
uv run python pipeline.py

# Smoke test (limit to N episodes)
# Set NUM_EPISODES in pipeline.py, then run as above
```

## Configuration

Key settings in `pipeline.py`:

| Setting | Default | Description |
|---|---|---|
| `NUM_EPISODES` | `None` (all) | Limit episodes for testing |
| `MODEL_NAME` | `Salesforce/blip-image-captioning-large` | HuggingFace model ID |
| `CPU_LOADER_NUM_CPUS` | `0.5` | Fractional CPUs per data-loading worker |
| `GPU_BATCH_SIZE` | `32` | Frames per GPU batch |
| `GPU_CONCURRENCY` | `(24, 32)` | GPU actor pool autoscale range |
| `OUTPUT` | `/mnt/cluster_storage/droid-annotated` | Parquet output path |
