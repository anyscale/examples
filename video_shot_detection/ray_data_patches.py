"""Minimal patches to ray.data applied once at import time.

Import this module before any ray.data pipeline code:

    import ray_data_patches  # noqa: F401
    import ray.data

Patch 1 — ray.data.read_videos: adds an ``output_size`` keyword argument so
frames are resized inside the C++/FFmpeg decoder (via decord) and the
native-resolution buffer is never materialised in Python.

Patch 2 — Dataset.map_batches: lifts Ray's hard rejection of
``batch_size=None`` when ``num_gpus > 0``. Ray added that guard to nudge
users toward explicit batch sizes for GPU workloads; for whole-video
processing (1 block = 1 video, enforced by ``override_num_blocks``) the
entire block is exactly the right batch, so the guard is counterproductive.
"""

import ray.data as _rd
from read_videos_with_resize import read_videos_with_resize as _read_videos_with_resize

# ---------------------------------------------------------------------------
# Patch 1: ray.data.read_videos — add output_size parameter
# ---------------------------------------------------------------------------

_orig_read_videos = _rd.read_videos


def read_videos(paths, *, output_size=None, **kwargs):
    """ray.data.read_videos extended with decode-time resize.

    Identical to the stock function when ``output_size`` is omitted.
    When ``output_size=(width, height)`` is given, passes it to decord's
    VideoReader so frames arrive at the target resolution with no extra copy.
    """
    if output_size is not None:
        return _read_videos_with_resize(paths, output_size=output_size, **kwargs)
    return _orig_read_videos(paths, **kwargs)


_rd.read_videos = read_videos


# ---------------------------------------------------------------------------
# Patch 2: Dataset.map_batches — allow batch_size=None with num_gpus > 0
# ---------------------------------------------------------------------------

_orig_map_batches = _rd.Dataset.map_batches


def _map_batches(self, fn, *, batch_size=None, num_gpus=None, **kwargs):
    if num_gpus and batch_size is None:
        return self._map_batches_without_batch_size_validation(
            fn,
            batch_size=None,
            compute=kwargs.pop("compute", None),
            batch_format=kwargs.pop("batch_format", "default"),
            zero_copy_batch=kwargs.pop("zero_copy_batch", True),
            fn_args=kwargs.pop("fn_args", None),
            fn_kwargs=kwargs.pop("fn_kwargs", None),
            fn_constructor_args=kwargs.pop("fn_constructor_args", None),
            fn_constructor_kwargs=kwargs.pop("fn_constructor_kwargs", None),
            num_cpus=kwargs.pop("num_cpus", None),
            num_gpus=num_gpus,
            memory=kwargs.pop("memory", None),
            concurrency=kwargs.pop("concurrency", None),
            udf_modifying_row_count=kwargs.pop("udf_modifying_row_count", True),
            ray_remote_args_fn=kwargs.pop("ray_remote_args_fn", None),
            **kwargs,
        )
    return _orig_map_batches(self, fn, batch_size=batch_size, num_gpus=num_gpus, **kwargs)


_rd.Dataset.map_batches = _map_batches
