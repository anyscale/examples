"""LeRobot datasource usage examples.

Reads four LeRobot v3 datasets from S3, each with a different partitioning mode:

    pusht                         SEQUENTIAL  — one task for the whole dataset
    aloha_sim_transfer_cube_human EPISODE    — one task per episode
    fmb                           CHAIN      — one task per connected component of shared files
    xvla-soft-fold                FILE_GROUP — one task per unique video-file set (default)

For each dataset the read plan is printed before execution, showing task count,
per-task row ranges, estimated size, and the parquet / video files each task
will open.  Schema and row count are printed after the read completes.

Two access patterns are shown:

    read_lerobot(root, partitioning=...)
        Convenience wrapper; returns a Ray Data Dataset directly.

    LeRobotDatasource(root, partitioning=...)  +  ray.data.read_datasource(source)
        Direct datasource construction; exposes source.meta and source.plan()
        before any data is read.

Additional examples at the end of the script:

    Take 1 row — reads one row from pusht and prints each column's shape/value.

    write_parquet — writes pusht as a single parquet file to
        /mnt/cluster_storage/lerobot/pusht.parquet, then inspects the written
        file and prints row count, row groups, column count, size, and schema.

All datasets are read from::

    s3://anyscale-public-robotics-datasets/lerobot/lerobot/<dataset>/
"""

import ray

from lerobot_datasource import LeRobotDatasource, Partitioning, read_lerobot

BUCKET_PREFIX = "s3://anyscale-public-robotics-datasets/lerobot"

ray.init()

# ---------------------------------------------------------------------------
# pusht — read_lerobot convenience API, SEQUENTIAL mode
# ---------------------------------------------------------------------------
print("\n=== pusht (read_lerobot, SEQUENTIAL) ===")
source = LeRobotDatasource(
    f"{BUCKET_PREFIX}/lerobot/pusht",
    partitioning=Partitioning.SEQUENTIAL,
)
plan = source.plan()
print(f"  {len(plan)} tasks")
for t in plan:
    print(f"  task={t['task']}  rows=[{t['start']}, {t['end']})  size={t['size_bytes'] / 1024:.1f} KB")
    for f in t["parquet_files"]:
        print(f"    parquet  {f.split('lerobot/')[-1]}")
    for key, paths in t["video_files"].items():
        for f in paths:
            print(f"    video    {f.split('lerobot/')[-1]}  ({key})")

ds = read_lerobot(f"{BUCKET_PREFIX}/lerobot/pusht", partitioning=Partitioning.SEQUENTIAL)
print("schema:", ds.schema())
print("count: ", ds.count())

# ---------------------------------------------------------------------------
# aloha_sim_transfer_cube_human — EPISODE mode
# ---------------------------------------------------------------------------
print("\n=== aloha_sim_transfer_cube_human (EPISODE) ===")
source = LeRobotDatasource(
    f"{BUCKET_PREFIX}/lerobot/aloha_sim_transfer_cube_human",
    partitioning=Partitioning.EPISODE,
)
plan = source.plan()
print(f"  {source.meta.total_episodes} episodes, {source.meta.total_frames} frames, {len(plan)} tasks")
for t in plan:
    print(f"  task={t['task']}  rows=[{t['start']}, {t['end']})  size={t['size_bytes'] / 1024:.1f} KB")
    for f in t["parquet_files"]:
        print(f"    parquet  {f.split('lerobot/')[-1]}")
    for key, paths in t["video_files"].items():
        for f in paths:
            print(f"    video    {f.split('lerobot/')[-1]}  ({key})")

ds = ray.data.read_datasource(source)
print("schema:", ds.schema())
print("count: ", ds.count())

# ---------------------------------------------------------------------------
# fmb — CHAIN mode
# ---------------------------------------------------------------------------
print("\n=== fmb (CHAIN) ===")
source = LeRobotDatasource(
    f"{BUCKET_PREFIX}/lerobot/fmb",
    partitioning=Partitioning.CHAIN,
)
plan = source.plan()
print(f"  {source.meta.total_episodes} episodes, {source.meta.total_frames} frames, {len(plan)} tasks")
for t in plan:
    print(f"  task={t['task']}  rows=[{t['start']}, {t['end']})  size={t['size_bytes'] / 1024:.1f} KB")
    for f in t["parquet_files"]:
        print(f"    parquet  {f.split('lerobot/')[-1]}")
    for key, paths in t["video_files"].items():
        for f in paths:
            print(f"    video    {f.split('lerobot/')[-1]}  ({key})")

ds = ray.data.read_datasource(source)
print("schema:", ds.schema())
print("count: ", ds.count())

# ---------------------------------------------------------------------------
# xvla-soft-fold — FILE_GROUP mode
# ---------------------------------------------------------------------------
print("\n=== xvla-soft-fold (FILE_GROUP) ===")
source = LeRobotDatasource(
    f"{BUCKET_PREFIX}/lerobot/xvla-soft-fold",
    partitioning=Partitioning.FILE_GROUP,
)
plan = source.plan()
print(f"  {source.meta.total_episodes} episodes, {source.meta.total_frames} frames, {len(plan)} tasks")
for t in plan:
    print(f"  task={t['task']}  rows=[{t['start']}, {t['end']})  size={t['size_bytes'] / 1024:.1f} KB")
    for f in t["parquet_files"]:
        print(f"    parquet  {f.split('lerobot/')[-1]}")
    for key, paths in t["video_files"].items():
        for f in paths:
            print(f"    video    {f.split('lerobot/')[-1]}  ({key})")

ds = ray.data.read_datasource(source)
print("schema:", ds.schema())
print("count: ", ds.count())

# ---------------------------------------------------------------------------
# Take 1 row from pusht and print it
# ---------------------------------------------------------------------------
print("\n=== pusht — take 1 row ===")
ds = read_lerobot(f"{BUCKET_PREFIX}/lerobot/pusht", partitioning=Partitioning.SEQUENTIAL)
row = ds.take(1)[0]
for col, val in row.items():
    # summarise arrays rather than printing raw bytes
    summary = f"shape={val.shape} dtype={val.dtype}" if hasattr(val, "shape") else repr(val)
    print(f"  {col}: {summary}")

# ---------------------------------------------------------------------------
# Write pusht to parquet
# ---------------------------------------------------------------------------
print("\n=== pusht — write_parquet ===")
OUT = "/mnt/cluster_storage/lerobot/pusht.parquet"
ds = read_lerobot(f"{BUCKET_PREFIX}/lerobot/pusht", partitioning=Partitioning.SEQUENTIAL)
ds.repartition(1).write_parquet(OUT)
print(f"  written to {OUT}")

# ---------------------------------------------------------------------------
# Inspect written parquet
# ---------------------------------------------------------------------------
import glob
import pyarrow.parquet as pq

files = sorted(glob.glob(f"{OUT}/*.parquet"))
print(f"\n=== inspect {OUT} ===")
print(f"  files: {len(files)}")
for path in files:
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    print(f"  {path.split('/')[-1]}")
    print(f"    rows        : {meta.num_rows}")
    print(f"    row groups  : {meta.num_row_groups}")
    print(f"    columns     : {meta.num_columns}")
    print(f"    size        : {sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)) / 1024 / 1024:.1f} MB")
    print(f"    schema      : {pf.schema_arrow}")

ray.shutdown()
