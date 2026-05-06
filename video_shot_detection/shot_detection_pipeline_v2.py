import argparse

import numpy as np
import ray
import ray.data
import ray_data_patches  # noqa: F401 — patches ray.data.read_videos + Dataset.map_batches
import torch
from ray.data import DataContext

from shot_detection import ShotDetectionConfig, frames_to_windows, scores_to_clips
from transnetv2 import TransNetV2
from video_io import list_remote_video_ids, resolve_weights_path


class TransNetV2Inference:
    """GPU actor: windowing + TransNetV2 inference + clip detection for one video per call."""

    def __init__(self, config: ShotDetectionConfig) -> None:
        self._config = config
        self._model = self._load_model(config.weights_path)

    def __call__(self, batch: dict) -> dict:
        """All frames from one video → one row per detected clip."""
        path, frames, fps = self._unpack_batch(batch)
        scores = self._run_inference(frames)
        clips = scores_to_clips(
            scores,
            frame_offset=0,
            fps=fps,
            threshold=self._config.transition_threshold,
        )
        return self._clips_to_batch(path, clips)

    @staticmethod
    def _load_model(weights_path: str) -> TransNetV2:
        model = TransNetV2()
        state_dict = torch.load(resolve_weights_path(weights_path), map_location="cpu")
        model.load_state_dict(state_dict)
        return model.eval().cuda()

    @staticmethod
    def _unpack_batch(batch: dict) -> tuple[str, np.ndarray, float]:
        """Extract path, RGB frames, and FPS from a Ray Data batch dict."""
        paths = batch["path"]
        assert len(set(paths.tolist())) == 1, (
            f"Expected one video per batch, got paths: {set(paths.tolist())}"
        )
        frames = batch["frame"][..., :3]       # drop alpha if decord returned RGBA
        timestamps = batch["frame_timestamp"]  # [T, 2]: [start_ts, end_ts] in seconds
        # Frame duration = end_ts - start_ts for any frame in a constant frame-rate video.
        fps = 1.0 / float(timestamps[0, 1] - timestamps[0, 0]) if len(timestamps) else 24.0
        return str(paths[0]), frames, fps

    def _run_inference(self, frames: np.ndarray) -> np.ndarray:
        """Run TransNetV2 on all sliding windows and return per-frame transition scores."""
        windows, n_frames = frames_to_windows(frames)
        window_tensor = torch.from_numpy(np.stack(windows)).cuda()  # [W, 100, 27, 48, 3]

        with torch.no_grad(), torch.autocast(device_type="cuda"):
            one_hot, _ = self._model(window_tensor)  # [W, 100, 1]

        # Slice the central 50 frames of each 100-frame window, flatten, then
        # trim to the actual frame count (the last window may be padded).
        scores = one_hot[:, 25:75, 0].sigmoid().cpu().float().numpy()
        return scores.reshape(-1)[:n_frames]

    @staticmethod
    def _clips_to_batch(path: str, clips: list[dict]) -> dict:
        """Convert clip dicts to a Ray Data batch dict (one row per clip)."""
        n = len(clips)

        return {
            "video_path":  np.full(n, path),
            "clip_index":  np.arange(n, dtype=np.int32),
            "start_frame": np.array([c["start_frame"] for c in clips], dtype=np.int32),
            "end_frame":   np.array([c["end_frame"]   for c in clips], dtype=np.int32),
            "start_ts":    np.array([c["start_ts"]    for c in clips], dtype=np.float32),
            "end_ts":      np.array([c["end_ts"]      for c in clips], dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_video_paths(input_s3_path: str, num_videos: int) -> list[str]:
    """List input videos and optionally cap the count."""
    video_ids = list_remote_video_ids(input_s3_path)
    if not video_ids:
        raise RuntimeError(f"No input files found under {input_s3_path}")
    if num_videos > 0:
        video_ids = video_ids[:num_videos]
    return [f"{input_s3_path.rstrip('/')}/{vid}" for vid in video_ids]


def run_pipeline(
    *,
    input_s3_path: str,
    weights_s3_path: str,
    output_s3_path: str,
    transition_threshold: float = 0.5,
    total_gpus: int = 2,
    gpu_resource_per_worker: float = 0.25,
    num_videos: int = 0,
) -> None:
    config = ShotDetectionConfig(
        transition_threshold=transition_threshold,
        weights_path=weights_s3_path,
    )
    video_paths = build_video_paths(input_s3_path, num_videos)
    num_gpu_workers = int(total_gpus / gpu_resource_per_worker)

    # Prevent Ray Data from dynamic block splitting
    DataContext.get_current().target_max_block_size = None

    (
        # Stage 1: Read — decode each video into frame rows resized to the
        # TransNetV2 input resolution inside the decoder.
        ray.data.read_videos(
            video_paths,
            output_size=(TransNetV2.INPUT_W, TransNetV2.INPUT_H),
            include_paths=True,
            include_timestamps=True,
            override_num_blocks=len(video_paths),  # 1 block per file → Stage 2 always sees one full video
        )
        # Stage 2: GPU inference — windowing + forward pass + clip detection.
        # batch_size=None passes the entire block (= one video) to each actor call.
        .map_batches(
            TransNetV2Inference,
            fn_constructor_args=(config,),
            batch_size=None,
            # Fractional GPU lets multiple actors share one device,
            # improving utilization via concurrent kernel execution.
            num_gpus=gpu_resource_per_worker,
            concurrency=num_gpu_workers,
            batch_format="numpy",
            memory=8 * 1024**3,  # 8 GiB reserved per actor for frame buffers
        )
        # Stage 3: Write — consolidate clip rows into Parquet files.
        .write_parquet(output_s3_path, min_rows_per_file=100_000)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shot detection pipeline: reads videos, runs TransNetV2, writes clip timestamps to Parquet."
    )
    parser.add_argument("--input-s3-path",        required=True,            help="S3 prefix containing input video files.")
    parser.add_argument("--weights-s3-path",       required=True,            help="S3 path to TransNetV2 weights (.pt).")
    parser.add_argument("--output-s3-path",        required=True,            help="S3 prefix to write output Parquet files.")
    parser.add_argument("--transition-threshold",  type=float, default=0.5,  help="Per-frame score threshold for shot cuts.")
    parser.add_argument("--total-gpus",              type=int,   default=2,     help="Total GPUs available across the cluster.")
    parser.add_argument("--gpu-resource-per-worker", type=float, default=0.25,  help="Fractional GPU allocated per inference actor.")
    parser.add_argument("--num-videos",              type=int,   default=0,     help="Cap on videos to process (0 = all).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ray.init(ignore_reinit_error=True)
    run_pipeline(
        input_s3_path=args.input_s3_path,
        weights_s3_path=args.weights_s3_path,
        output_s3_path=args.output_s3_path,
        transition_threshold=args.transition_threshold,
        total_gpus=args.total_gpus,
        gpu_resource_per_worker=args.gpu_resource_per_worker,
        num_videos=args.num_videos,
    )
    ray.shutdown()
