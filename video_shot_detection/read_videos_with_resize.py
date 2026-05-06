"""read_videos with decode-time frame resize.

Passes ``output_size`` directly to decord's VideoReader so frames are resized
inside the C++/FFmpeg decoder. The native-resolution buffer is never
materialised in Python.

Automatically detects whether Anyscale Ray is installed. If so, subclasses
the Anyscale-internal VideoReader. Otherwise, falls back to the OSS
VideoDatasource, which already plumbs ``decord_load_args`` through to the
underlying VideoReader with no subclassing required.

Usage:

    from read_videos_with_resize import read_videos_with_resize
    ds = read_videos_with_resize("s3://bucket/videos/", output_size=(48, 27))

``output_size`` is ``(width, height)`` in pixels, matching the convention
used by decord and cv2.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Union

from ray.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anyscale path
# ---------------------------------------------------------------------------

def _read_anyscale(
    paths, *, output_size, include_paths, include_timestamps,
    filesystem, arrow_open_stream_args, partition_filter,
    partitioning, ignore_missing_paths, file_extensions, shuffle,
    concurrency, num_cpus, num_gpus, memory, ray_remote_args,
) -> Dataset:
    from ray.anyscale.data._internal.readers.video_reader import (
        VideoReader as _BaseVideoReader,
    )
    from ray.anyscale.data.api.read_api import read_files

    _width, _height = output_size
    _include_timestamps = include_timestamps

    class _ResizedVideoReader(_BaseVideoReader):
        def read_stream(self, file, path, metadata=None):
            from decord import VideoReader

            reader = VideoReader(file, width=_width, height=_height)
            for frame_index, frame in enumerate(reader):
                item = {
                    "frame": [frame.asnumpy()[:, :, :3]],  # drop alpha — decord returns RGBA for some codecs
                    "frame_index": [frame_index],
                }
                if _include_timestamps:
                    item["frame_timestamp"] = [reader.get_frame_timestamp(frame_index)]
                yield item

    reader = _ResizedVideoReader(
        include_paths=include_paths,
        partitioning=partitioning,
        open_args=arrow_open_stream_args,
    )
    return read_files(
        paths,
        reader,
        filesystem=filesystem,
        columns=None,
        partition_filter=partition_filter,
        ignore_missing_paths=ignore_missing_paths,
        file_extensions=file_extensions,
        shuffle=shuffle,
        concurrency=concurrency,
        ray_remote_args=ray_remote_args,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory=memory,
    )


# ---------------------------------------------------------------------------
# OSS path
# ---------------------------------------------------------------------------

def _read_oss(
    paths, *, output_size, include_paths, include_timestamps,
    filesystem, arrow_open_stream_args, partition_filter,
    partitioning, ignore_missing_paths, file_extensions, shuffle,
    concurrency, override_num_blocks, num_cpus, num_gpus, memory,
    ray_remote_args,
) -> Dataset:
    from ray.data._internal.datasource.video_datasource import (
        VideoDatasource as _BaseVideoDatasource,
    )
    from ray.data.read_api import read_datasource

    if file_extensions is None:
        file_extensions = _BaseVideoDatasource._FILE_EXTENSIONS

    width, height = output_size
    # VideoDatasource already passes decord_load_args to VideoReader in _read_stream,
    # so no subclassing is needed — just inject the resize dimensions here.
    datasource = _BaseVideoDatasource(
        paths,
        include_timestamps=include_timestamps,
        decord_load_args={"width": width, "height": height},
        filesystem=filesystem,
        open_stream_args=arrow_open_stream_args,
        partition_filter=partition_filter,
        partitioning=partitioning,
        ignore_missing_paths=ignore_missing_paths,
        shuffle=shuffle,
        include_paths=include_paths,
        file_extensions=file_extensions,
    )
    return read_datasource(
        datasource,
        ray_remote_args=ray_remote_args,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory=memory,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )


# ---------------------------------------------------------------------------
# Public API — auto-detects Anyscale vs OSS
# ---------------------------------------------------------------------------

def read_videos_with_resize(
    paths: Union[str, List[str]],
    *,
    output_size: tuple[int, int],
    include_paths: bool = False,
    include_timestamps: bool = False,
    filesystem: Optional[Any] = None,
    arrow_open_stream_args: Optional[Dict[str, Any]] = None,
    partition_filter=None,
    partitioning=None,
    ignore_missing_paths: bool = False,
    file_extensions: Optional[List[str]] = None,
    shuffle: Union[Literal["files"], None] = None,
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
    num_cpus: Optional[float] = None,
    num_gpus: Optional[float] = None,
    memory: Optional[float] = None,
    ray_remote_args: Optional[Dict[str, Any]] = None,
) -> Dataset:
    """ray.data.read_videos with decode-time frame resize.

    Passes ``output_size`` directly to decord's VideoReader so frames are
    resized inside the C++/FFmpeg decoder — the native-resolution buffer is
    never materialised in Python.

    Args:
        paths: Video file path(s) or S3/GCS prefix.
        output_size: ``(width, height)`` in pixels.

    Note:
        ``frame_timestamp`` values from decord are 2-element arrays
        ``[start_ts, end_ts]`` in seconds. For constant frame-rate video, FPS
        is ``1.0 / (frame_timestamp[0][1] - frame_timestamp[0][0])``.
    """
    try:
        import ray.anyscale.data  # noqa: F401

        return _read_anyscale(
            paths,
            output_size=output_size,
            include_paths=include_paths,
            include_timestamps=include_timestamps,
            filesystem=filesystem,
            arrow_open_stream_args=arrow_open_stream_args,
            partition_filter=partition_filter,
            partitioning=partitioning,
            ignore_missing_paths=ignore_missing_paths,
            file_extensions=file_extensions,
            shuffle=shuffle,
            concurrency=concurrency,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            ray_remote_args=ray_remote_args,
        )
    except ModuleNotFoundError:
        logger.debug("ray.anyscale not found, falling back to OSS path")
        return _read_oss(
            paths,
            output_size=output_size,
            include_paths=include_paths,
            include_timestamps=include_timestamps,
            filesystem=filesystem,
            arrow_open_stream_args=arrow_open_stream_args,
            partition_filter=partition_filter,
            partitioning=partitioning,
            ignore_missing_paths=ignore_missing_paths,
            file_extensions=file_extensions,
            shuffle=shuffle,
            concurrency=concurrency,
            override_num_blocks=override_num_blocks,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            ray_remote_args=ray_remote_args,
        )
