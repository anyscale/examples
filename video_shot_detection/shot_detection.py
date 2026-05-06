from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from transnetv2 import TransNetV2


@dataclass
class ShotDetectionConfig:
    transition_threshold: float
    weights_path: str
    inference_batch_windows: int | None = None  # None = all windows in one pass; set to limit GPU memory


def frames_to_windows(frames: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Pads frames and slices into overlapping 100-frame windows (stride 50); returns windows and original frame count."""
    n = len(frames)
    pad_end = 25 + 50 - (n % 50 if n % 50 != 0 else 50)  # align to next 50-frame boundary
    padded = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * pad_end, axis=0)
    windows = [padded[p:p + 100] for p in range(0, len(padded) - 99, 50)]
    return windows, n


def run_inference_fused(
    model: TransNetV2,
    all_frames: list[np.ndarray],
    max_batch_windows: int | None = None,
) -> list[np.ndarray]:
    """Runs TransNetV2 on multiple videos in a single batched forward pass; splits scores back per-video.

    Args:
        model: TransNetV2 model already on the target device.
        all_frames: One (T, H, W, 3) uint8 array per video.
        max_batch_windows: Cap on windows per forward pass; guards against GPU OOM on long videos.
    """
    all_windows: list[np.ndarray] = []
    window_counts: list[int] = []
    frame_counts: list[int] = []
    for frames in all_frames:
        windows, n = frames_to_windows(frames)
        all_windows.extend(windows)
        window_counts.append(len(windows))
        frame_counts.append(n)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # autocast runs convolutions and matmuls in FP16 on CUDA, roughly doubling
    # throughput on Ampere+ GPUs. sigmoid stays in FP32 automatically.
    use_autocast = device == "cuda"
    chunk_size = max_batch_windows or len(all_windows)
    flat_scores_parts: list[np.ndarray] = []
    for start in range(0, len(all_windows), chunk_size):
        chunk = all_windows[start:start + chunk_size]
        # pin_memory() lets the DMA engine transfer data asynchronously, overlapping H2D copy with CPU work.
        cpu_tensor = torch.from_numpy(np.stack(chunk))
        if device == "cuda":
            cpu_tensor = cpu_tensor.pin_memory()
        batch = cpu_tensor.to(device=device, non_blocking=True)  # [W, 100, 27, 48, 3]
        with torch.no_grad(), torch.autocast(device_type=device, enabled=use_autocast):
            one_hot, _ = model(batch)  # [W, 100, 1] logits
        flat_scores_parts.append(one_hot[:, 25:75, 0].sigmoid().cpu().float().numpy())

    flat_scores = np.concatenate(flat_scores_parts, axis=0)  # [total_windows, 50]

    per_video_scores: list[np.ndarray] = []
    offset = 0
    for wc, n in zip(window_counts, frame_counts):
        scores = flat_scores[offset:offset + wc].reshape(-1)[:n]
        per_video_scores.append(scores)
        offset += wc
    return per_video_scores


def scores_to_clips(scores: np.ndarray, frame_offset: int, fps: float, threshold: float) -> list[dict]:
    """Converts per-frame transition scores to clip dicts with frame indices and timestamps."""
    cut_positions = np.where(scores > threshold)[0].tolist()
    n = len(scores)
    boundaries = [0] + cut_positions + [n]
    clips = []
    for i in range(len(boundaries) - 1):
        start_local = min(boundaries[i], n - 1)
        end_local = min(boundaries[i + 1] - 1, n - 1)
        if end_local < start_local:
            continue
        start_frame = frame_offset + start_local
        end_frame = frame_offset + end_local
        clips.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_ts": start_frame / fps,
            "end_ts": end_frame / fps,
        })
    return clips
