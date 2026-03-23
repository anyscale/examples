# LeRobot Ray Data Datasource

> **LeRobot v3 only.** Earlier dataset formats (v1, v2) use a different directory layout and episode schema and are not supported.

A [Ray Data](https://docs.ray.io/en/latest/data/data.html) `Datasource` for reading [LeRobot v3](https://github.com/huggingface/lerobot) robotics datasets from local disk or cloud storage (GCS, S3).

Each output row combines low-dimensional data (state, action, etc.) from chunked Parquet files with decoded camera frames from chunked mp4 files, matching LeRobot's `video_backend="pyav"` output pixel-for-pixel.

## Installation

```bash
uv sync
# or: pip install -e .
```

## Quick start

```python
import ray
from lerobot_datasource import read_lerobot, LeRobotDatasource, Partitioning

ray.init()

# Read with default partitioning (one task per video-file group)
ds = read_lerobot("/data/my_dataset")

# Read from cloud storage
ds = read_lerobot("gs://bucket/my_dataset")

# Choose a partitioning explicitly
ds = read_lerobot("/data/my_dataset", partitioning=Partitioning.EPISODE)

# Fixed-size row blocks (useful when episodes are large and uneven)
ds = read_lerobot("/data/my_dataset", partitioning=Partitioning.ROW_BLOCK, block_size=1024)

# Access dataset metadata before reading
source = LeRobotDatasource("/data/my_dataset")
print(source.meta.total_frames, source.meta.video_keys)
ds = ray.data.read_datasource(source)
```

## Output schema

| Column | Type | Description |
|---|---|---|
| `index` | `int64` | Global 0-based row id |
| `episode_index` | `int64` | Episode this row belongs to |
| `frame_index` | `int64` | 0-based position within the episode |
| `timestamp` | `float64` | Elapsed seconds within the episode |
| `<state/action>` | `float32` | Low-dimensional vectors from Parquet |
| `<camera_key>` | `uint8` | Decoded HWC RGB frame (e.g. `observation.image`) |
| `task` | `string` | Natural-language task description |

## Partitioning

| Mode | Tasks | Best for |
|---|---|---|
| `file_group` *(default)* | one per unique video-file set | balanced tasks; each mp4 opened once per task |
| `episode` | one per episode | small local datasets; maximum task count |
| `chain` | one per connected component of file groups | large cloud datasets; minimises total video-file opens |
| `sequential` | one total | cloud datasets where peak memory matters most |
| `row_block` | `ceil(total / block_size)` | fixed-size blocks; set via `block_size` kwarg |

## Running the tests

Tests download datasets from Hugging Face and compare output against LeRobot's own loader.
Datasets are cached under `HF_HOME_DIR` (default `/mnt/cluster_storage/.cache`).

```bash
# Run all tests
uv run pytest test_datasource.py -v

# Custom cache directory
HF_HOME_DIR=/my/cache uv run pytest test_datasource.py -v
```

> **Note:** Tests require a running Ray cluster (or `ray.init()`) and the `lerobot` package.
> Install dev dependencies with `uv sync --group dev`.
