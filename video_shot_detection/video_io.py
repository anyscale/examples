from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from transnetv2 import TransNetV2

LOCAL_STORAGE_ROOT = "/mnt/local_storage"


def get_fps(video_path: str) -> float:
    """Returns frames-per-second for the first video stream in `video_path`."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True,
    )
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def download_video(remote_path: str, local_dir: str) -> str:
    """Downloads a remote video (s3:// or rclone path) into `local_dir` and returns the local path."""
    if remote_path.startswith("s3://"):
        local_path = os.path.join(local_dir, os.path.basename(remote_path))
        subprocess.run(
            ["aws", "s3", "cp", remote_path, local_path, "--only-show-errors"],
            check=True,
            capture_output=True,
        )
        return local_path

    subprocess.run(["rclone", "copy", remote_path, local_dir], check=True, capture_output=True)
    return os.path.join(local_dir, os.path.basename(remote_path))


def upload_clip(local_clip_path: str, remote_clips_base: str) -> None:
    """Uploads `local_clip_path` to `remote_clips_base` via aws s3 cp or rclone."""
    if remote_clips_base.startswith("s3://"):
        remote_path = f"{remote_clips_base.rstrip('/')}/{os.path.basename(local_clip_path)}"
        subprocess.run(
            ["aws", "s3", "cp", local_clip_path, remote_path, "--only-show-errors"],
            check=True,
            capture_output=True,
        )
        return

    subprocess.run(
        ["rclone", "copy", local_clip_path, remote_clips_base],
        check=True,
        capture_output=True,
    )


def mk_local_storage_dir(prefix: str) -> str:
    """Creates a temp directory under `LOCAL_STORAGE_ROOT` with the given prefix."""
    os.makedirs(LOCAL_STORAGE_ROOT, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=LOCAL_STORAGE_ROOT)


def resolve_weights_path(weights_path: str) -> str:
    """Downloads weights from S3 if the path starts with s3://, otherwise returns as-is."""
    if not weights_path.startswith("s3://"):
        return weights_path
    local_dir = mk_local_storage_dir(prefix="transnet_weights_")
    local_path = os.path.join(local_dir, os.path.basename(weights_path) or "transnetv2_weights.pt")
    subprocess.run(
        ["aws", "s3", "cp", weights_path, local_path, "--only-show-errors"],
        check=True,
        capture_output=True,
    )
    return local_path


def list_s3_files(remote_base: str) -> list[str]:
    """Lists filenames (not directories) under an S3 prefix."""
    result = subprocess.run(
        ["aws", "s3", "ls", f"{remote_base.rstrip('/')}/"],
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4 and not parts[3].endswith("/"):
            files.append(parts[3])
    return sorted(files)


def list_rclone_files(remote_base: str) -> list[str]:
    """Lists filenames under a remote path using rclone lsf."""
    result = subprocess.run(
        ["rclone", "lsf", remote_base, "--files-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted([line for line in result.stdout.splitlines() if line and not line.endswith("/")])


def list_remote_video_ids(remote_base: str) -> list[str]:
    """Dispatches to `list_s3_files` or `list_rclone_files` based on scheme."""
    if remote_base.startswith("s3://"):
        return list_s3_files(remote_base)
    return list_rclone_files(remote_base)


def read_video_frames(video_path: str) -> np.ndarray:
    """Decodes all frames from `video_path` and resizes each to (INPUT_W, INPUT_H) for TransNetV2."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.resize takes (width, height)
        frame = cv2.resize(frame, (TransNetV2.INPUT_W, TransNetV2.INPUT_H))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    shape = (0, TransNetV2.INPUT_H, TransNetV2.INPUT_W, 3)
    return np.stack(frames, axis=0) if frames else np.zeros(shape, dtype=np.uint8)
