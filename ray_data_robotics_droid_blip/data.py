"""data.py — DROID Raw 1.0.1 episode I/O: read HDF5 + MP4 from S3 and expand into per-timestep rows.

This module runs on CPU workers inside Ray Data's flat_map stage.
Each call to episode_to_training_rows handles one episode: it downloads
the HDF5 (robot state) and streams the MP4 (wrist camera), then yields
one dict per timestep that Ray Data will batch and forward to the GPU stage.
"""

from __future__ import annotations

import io
from typing import Any, Generator, Iterator

import av
import h5py
import numpy as np
import smart_open
from botocore import UNSIGNED
from botocore.client import Config

DATASET_BUCKET = "s3://anyscale-public-droid-dataset"
DATASET_PREFIX = "droid/1.0.1"

# Only these HDF5 datasets are needed downstream; the rest are skipped
# to avoid downloading the full file contents into memory.
HDF5_KEYS_NEEDED = (
    "action/cartesian_velocity",
    "action/gripper_velocity",
    "observation/robot_state/cartesian_position",
    "observation/robot_state/gripper_position",
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

# Anonymous (unsigned) S3 access — the DROID bucket is publicly readable.
# We build a dedicated boto3 session with no credentials so that the
# Anyscale node's IAM role is never consulted (its policy may lack
# s3:GetObject on the public bucket).
import boto3

_ANON_SESSION = boto3.Session(
    aws_access_key_id="",
    aws_secret_access_key="",
)
_ANON_CLIENT = _ANON_SESSION.client("s3", config=Config(signature_version=UNSIGNED))
S3_TRANSPORT = {"client": _ANON_CLIENT}


def open_file(path: str, mode: str = "rb"):
    """Open a local or S3 file via smart_open with anonymous S3 access."""
    if path.startswith("s3://"):
        return smart_open.open(path, mode, transport_params=S3_TRANSPORT)
    return smart_open.open(path, mode)


def read_hdf5_keys(hdf5_path: str, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read only the specified datasets from an HDF5 file on S3.

    HDF5 requires random access, so the full file is buffered into memory.
    We then read only the keys we need rather than walking the entire tree.
    """
    with open_file(hdf5_path) as f:
        buf = io.BytesIO(f.read())
    data = {}
    with h5py.File(buf, "r") as hf:
        for key in keys:
            if key in hf:
                data[key.replace("/", ".")] = hf[key][()]
    return data


def iter_video_frames(mp4_path: str) -> Iterator[np.ndarray]:
    """Yield RGB frames (HWC uint8) from an MP4 on S3, streaming.

    Unlike HDF5, video can be decoded progressively — smart_open streams
    bytes from S3 while PyAV decodes frames, so the full MP4 is never
    buffered in memory.
    """
    with open_file(mp4_path) as f:
        with av.open(f, mode="r") as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")


def resolve_path(val: str | None, s3_base: str) -> str:
    """Prepend S3 base to a relative path, or return as-is if already absolute."""
    if not val:
        return ""
    return val if "://" in val else s3_base + val


# ---------------------------------------------------------------------------
# CPU stage: episode → per-timestep rows
# ---------------------------------------------------------------------------


def episode_to_training_rows(episode: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
    """Expand one episode into per-timestep rows with resolved S3 paths.

    This is the flat_map function passed to Ray Data. For each input row
    (one episode from the manifest), it yields N rows (one per timestep),
    each containing the wrist camera frame and robot state vectors.

    Ray Data calls this function on CPU workers and handles serialization
    of the yielded dicts into Arrow format automatically.
    """
    s3_base = f"{DATASET_BUCKET}/{DATASET_PREFIX}/"

    hdf5_path = resolve_path(episode.get("hdf5_path"), s3_base)
    wrist_mp4_path = resolve_path(episode.get("wrist_mp4_path"), s3_base)

    if not hdf5_path:
        return

    hdf5_data = read_hdf5_keys(hdf5_path, HDF5_KEYS_NEEDED)
    frames = iter_video_frames(wrist_mp4_path) if wrist_mp4_path else iter([])

    T = episode.get("trajectory_length")
    base = {k: episode.get(k) for k in ("uuid", "task", "timestamp")}

    for t, frame in enumerate(frames):
        if T is not None and t >= T:
            break
        row = {**base, "timestep": t, "wrist_frame": frame}
        for key, arr in hdf5_data.items():
            row[key] = arr[t].tolist() if arr.ndim > 1 else arr[t].item()
        yield row
