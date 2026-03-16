# LeRobot Ray Data Datasource

A Ray Data `Datasource` for reading [LeRobot v3](https://github.com/huggingface/lerobot) robotics datasets from local disk or cloud storage (GCS/S3).

## Setup

```bash
uv sync
```

## Example

See [example_usage.ipynb](example_usage.ipynb) — downloads `lerobot/berkeley_autolab_ur5` and reads it through the datasource with two parallelism modes (`file_group` and `episode`).

## Blog post

For a deeper dive, see [Streaming LeRobot Dataset v3.0 with Ray Data](TBD).
