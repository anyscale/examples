"""Ray Data datasource for LeRobot v3 robotics datasets.

Reads LeRobot-format datasets from local disk or cloud storage (GCS/S3) into
Ray Data.  Supports four parallelism strategies that trade off task-level
parallelism against video-file deduplication and memory.

Typical usage::

    import ray
    from lerobot_datasource import LeRobotDatasource, ParallelismMode

    ds = ray.data.read_datasource(LeRobotDatasource(root="/data/my_dataset"))
    ds = ray.data.read_datasource(LeRobotDatasource(
        root="gs://bucket/dataset",
        parallelism_mode=ParallelismMode.SEQUENTIAL,
    ))

Rationale
---------
``LeRobotDataset`` (the upstream PyTorch dataset) loads one frame at a time
from local disk and has no support for cloud object storage or parallel
ingestion.  Training pipelines that store datasets on GCS or S3 must either
download everything upfront or build custom I/O.  This datasource fills that
gap: it reads parquet and mp4 files directly from any fsspec-compatible
filesystem and exposes them as a Ray Data dataset, enabling parallel,
distributed ingestion without a local copy.

LeRobot v3 stores video frames in mp4 files that each cover a fixed chunk of
consecutive episodes — not individual episodes.  Multiple episodes therefore
share the same video file, and the episodes within a file are laid out
sequentially from the start.  This has a direct impact on parallelisation:
splitting the dataset naïvely by episode causes the same video file to be
opened, seeked, and decoded independently by many workers, multiplying cloud
reads and decode overhead proportionally.

The key insight is that episodes sharing a video file can be grouped into
the same read task.  Because all cameras follow the same chunking schedule, a
*file group* — the set of video files referenced by one episode — is identical
for every episode in the same chunk.  Assigning one task per unique file group
significantly reduces the number of mp4 opens compared to per-episode splitting
while still producing one task per chunk, preserving meaningful parallelism.
This is the default strategy.

Parallelism strategies
----------------------
+---------------+------------------+------------------------------------+
| Mode          | Tasks created    | Best for                           |
+===============+==================+====================================+
| ``episode``   | one per episode  | small local datasets; maximum      |
|               |                  | parallelism regardless of I/O cost |
+---------------+------------------+------------------------------------+
| ``file_group``| one per unique   | **default** — balanced parallelism |
| *(default)*   | video-file set   | with each mp4 opened once per task |
+---------------+------------------+------------------------------------+
| ``chain``     | one per connected| large cloud datasets where         |
|               | component of     | minimising total video-file opens  |
|               | file groups      | across workers matters most        |
+---------------+------------------+------------------------------------+
| ``sequential``| one (total)      | cloud datasets where peak memory   |
|               |                  | must be minimised over throughput  |
+---------------+------------------+------------------------------------+

Output schema (one row per frame)
----------------------------------
Parquet columns (pass-through from data files):
  ``observation.state``   list<float>   — robot proprioception
  ``action``              list<float>   — commanded joint / EE action
  ``timestamp``           float64       — time within episode (seconds)
  ``frame_index``         int64         — 0-based index within episode
  ``episode_index``       int64         — global episode identifier
  ``index``               int64         — global frame identifier
  ``task_index``          int64         — index into the task description table
  *(dataset-specific)*    varies        — e.g. ``next.reward``, ``next.done``

Appended columns:
  ``<video_key>``         fixed_shape_tensor[uint8, (H, W, 3)]
                          — one column per camera, decoded from mp4 via PyAV.
                          Layout is **HWC uint8 [0, 255]** (contrast with
                          ``LeRobotDataset`` which returns CHW float32 [0, 1]).
  ``task``                string — human-readable task description looked up
                          from ``meta/tasks.parquet`` by ``task_index``.
"""

import enum
import json
import logging
from pathlib import Path
from typing import Any

import av
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import ray
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource
from ray.data.datasource.datasource import ReadTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


class LeRobotDatasourceMetadata:
    """Lightweight metadata for the datasource

    Rationale: Lerobot Dataset v3.0 supports hf hub and local datasets but not
    cloud storage (e.g. gs:// or s3://) This class allows that.

    Loads ``meta/info.json`` and the episode-level parquet metadata.  Data
    parquet files are read lazily per task during iteration — not preloaded —
    so this object remains small and pickle-safe for ``ray.put``.

    Examples::

        meta = LeRobotDatasourceMetadata("/data/my_dataset")
        meta = LeRobotDatasourceMetadata("gs://bucket/my_dataset")
        print(meta.total_episodes, meta.video_keys)
    """

    def __init__(self, root: str) -> None:
        root = root.rstrip("/")
        self.root = root
        # url_to_fs is the fsspec entry point that resolves the scheme and
        # returns a filesystem object + the scheme-stripped root path.
        # This is what lets the same code work for local paths, gs://, and s3://.
        fs, fs_root = fsspec.core.url_to_fs(root)

        self.info = self._fetch_info(fs, fs_root)
        self.video_keys: list[str] = [
            k for k, v in self.info["features"].items() if v.get("dtype") == "video"
        ]
        self.stats = self._fetch_stats(fs, fs_root)
        self.episodes = self._fetch_episodes(fs, fs_root)
        self.tasks = self._fetch_tasks(fs, fs_root)

    def _fetch_info(self, fs: Any, fs_root: str) -> dict:
        """Load meta/info.json; return the parsed dict."""
        info_path = f"{fs_root}/meta/info.json"
        if not fs.exists(info_path):
            raise FileNotFoundError(
                f"No LeRobot dataset found at {self.root!r}: meta/info.json is missing. "
                "Make sure the path points to the dataset root."
            )
        with fs.open(info_path, "r") as f:
            info = json.load(f)
        for required in (
            "total_frames",
            "total_episodes",
            "fps",
            "data_path",
            "features",
        ):
            if required not in info:
                raise ValueError(
                    f"{self.root!r}: meta/info.json is missing required key {required!r}"
                )
        return info

    def _fetch_stats(self, fs: Any, fs_root: str) -> dict[str, dict]:
        """Load meta/stats.json; return dataset-wide normalisation statistics.

        Contains min, max, mean, std, and quantiles for each scalar feature column.
        Useful for building normalisation transforms in VLA fine-tuning pipelines.
        """
        with fs.open(f"{fs_root}/meta/stats.json", "r") as f:
            return json.load(f)

    def _fetch_episodes(self, fs: Any, fs_root: str) -> pa.Table:
        """Load meta/episodes/**/*.parquet; return the concatenated table."""
        # Episode metadata is sharded across multiple parquet files for large datasets.
        # Each row describes one episode: its chunk/file indices for data and every
        # video camera, plus its timestamp range within the dataset.
        ep_files = sorted(fs.glob(f"{fs_root}/meta/episodes/**/*.parquet"))
        if not ep_files:
            raise FileNotFoundError(
                f"No episode parquet files found under {self.root!r}/meta/episodes/. "
                "The dataset may be incomplete or use an unsupported layout."
            )
        return pa.concat_tables([pq.read_table(fs.open(f, "rb")) for f in ep_files])

    def _fetch_tasks(self, fs: Any, fs_root: str) -> dict[int, str]:
        """Load meta/tasks.parquet; return a task_index → task string mapping."""
        # tasks.parquet maps integer task_index → human-readable instruction string
        # (e.g. "pick up the red block"). The column name varies by dataset version.
        table = pq.read_table(fs.open(f"{fs_root}/meta/tasks.parquet", "rb"))
        if "task" in table.column_names:
            task_col = "task"
        elif "__index_level_0__" in table.column_names:
            task_col = "__index_level_0__"
        else:
            raise ValueError(
                f"{self.root!r}: meta/tasks.parquet has no recognised task column "
                f"(expected 'task' or '__index_level_0__'); found {table.column_names}"
            )
        return dict(
            zip(
                table.column("task_index").to_pylist(),
                table.column(task_col).to_pylist(),
            )
        )

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def estimated_row_size_bytes(self) -> int:
        """Estimated in-memory size of one fully-decoded frame row (bytes).

        Sums scalar column bytes (``np.prod(shape) * itemsize`` for each
        non-video feature in ``info.json``) and decoded video frame bytes
        (HWC uint8, 1 byte per element, per camera).  String columns have no
        fixed width and are skipped.  Used to populate ``BlockMetadata.size_bytes``
        without any cloud I/O.
        """
        features = self.info.get("features", {})
        total = 0
        for feat in features.values():
            if feat.get("dtype") == "video":
                # Video frames: HWC uint8 — shape is [H, W, C], 1 byte per element
                shape = feat.get("shape")
                if shape:
                    total += int(np.prod(shape))
            else:
                shape = feat.get("shape", [1])
                try:
                    total += int(np.prod(shape)) * np.dtype(feat["dtype"]).itemsize
                except (TypeError, KeyError):
                    continue  # string or unknown dtype
        return total

    @property
    def estimated_inmemory_size_bytes(self) -> int:
        """Rough total in-memory size for the full dataset."""
        return self.total_frames * self.estimated_row_size_bytes

    def video_file_path(self, video_key: str, chunk_index: int, file_index: int) -> str:
        """Resolve the full path to a video file for a given camera."""
        template = self.info.get("video_path", "")
        if not template:
            raise ValueError(
                f"{self.root!r}: dataset has video keys {self.video_keys} "
                "but meta/info.json has no 'video_path' template"
            )
        return f"{self.root}/{template.format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)}"

    def data_file_path(self, chunk_index: int, file_index: int) -> str:
        """Resolve the full path to a data parquet file."""
        return f"{self.root}/{self.info['data_path'].format(chunk_index=chunk_index, file_index=file_index)}"

    def get_episode(self, ep_idx: int) -> dict:
        """Return one episode's metadata dict by its ``episode_index``.

        LeRobot v3 enforces 0-based sequential episode indices, so ep_idx
        equals the row position in the episodes table.
        """
        return self.episodes.slice(ep_idx, 1).to_pylist()[0]


# ---------------------------------------------------------------------------
# Segment iterators
# ---------------------------------------------------------------------------


class DataSegmentIterator:
    """Yields one-row :class:`pa.Table` slices for a sequence of parquet files.

    Each segment is a ``(chunk_index, file_index)`` file key.  Rows outside
    ``[start_episode, end_episode)`` are filtered out; within that range
    episodes are read in natural row order.
    """

    def __init__(
        self, ds_info: LeRobotDatasourceMetadata, start_episode: int, end_episode: int
    ) -> None:
        self._ds_info = ds_info
        self._start = start_episode
        self._end = end_episode
        # Slice the episodes table to this task's range and extract the two key columns.
        # Because episode_index == row position, slice(start, length) is O(1).
        ep_slice = ds_info.episodes.slice(start_episode, end_episode - start_episode)
        chunks = ep_slice.column("data/chunk_index").combine_chunks()
        files = ep_slice.column("data/file_index").combine_chunks()
        n = len(ep_slice)
        # Mark the first row and an y row where either index differs from the previous
        # row as a segment boundary, then filter both columns to those rows.
        is_new = (
            pa.concat_arrays(
                [
                    pa.array([True]),
                    pc.or_(  # type: ignore[attr-defined]
                        pc.not_equal(chunks.slice(1), chunks.slice(0, n - 1)),  # type: ignore[attr-defined]
                        pc.not_equal(files.slice(1), files.slice(0, n - 1)),  # type: ignore[attr-defined]
                    ),
                ]
            )
            if n > 0
            else pa.array([], type=pa.bool_())
        )
        self._segments: list[tuple[int, int]] = list(
            zip(
                pc.filter(chunks, is_new).to_pylist(),  # type: ignore[attr-defined]
                pc.filter(files, is_new).to_pylist(),  # type: ignore[attr-defined]
            )
        )

    def __iter__(self):
        """Yield one row per frame, in episode order across all segments."""
        ds_info = self._ds_info
        for chunk_idx, file_idx in self._segments:
            path = ds_info.data_file_path(chunk_idx, file_idx)
            fs, fs_path = fsspec.core.url_to_fs(path)
            pq_table = pq.read_table(fs.open(fs_path, "rb"))
            # A parquet file can span more episodes than the [start, end) range
            # assigned to this task (e.g. when two tasks share a boundary file).
            # Filter to only the rows belonging to this task's episode range so
            # that frames are never emitted twice across adjacent tasks.
            ep_col = pq_table.column("episode_index")
            pq_table = pq_table.filter(  # type: ignore[attr-defined]
                pc.and_(
                    pc.greater_equal(ep_col, self._start), pc.less(ep_col, self._end)
                )  # type: ignore[attr-defined]
            )
            for i in range(pq_table.num_rows):
                yield pq_table.slice(i, 1)


class VideoSegmentIterator:
    """Yields ``(frame, half_frame_duration)`` pairs for a sequence of video files.

    Opens each file in turn, streaming frames until exhausted before moving to
    the next.  ``start_ts`` is the seek target for the first file only; all
    subsequent files begin at ts=0 by the sequential dataset layout invariant.

    ``half_frame_duration`` is ``0.5 / fps`` in seconds, computed once per
    container from the stream's time base and frame rate.  It is yielded
    alongside each frame so :class:`EpisodeRangeReader` can advance to the
    frame nearest a target timestamp without access to the container.
    """

    def __init__(
        self,
        ds_info: LeRobotDatasourceMetadata,
        vid_key: str,
        start_episode: int,
        end_episode: int,
    ) -> None:
        # Resolve the filesystem once here so we don't pay url_to_fs overhead
        # on every video open.  For cloud paths (gs://, s3://) url_to_fs
        # returns the scheme-stripped path in _fs_root; for local paths the
        # two values are effectively the same.
        self._fs, _fs_root = fsspec.core.url_to_fs(ds_info.root)
        self._is_local = self._fs.protocol == "file"
        _root_prefix = ds_info.root  # strip this prefix to get fs-local path
        self._start_ts = 0.0
        self._current_container: Any = None
        # Same consecutive-dedup logic as DataSegmentIterator but over video
        # chunk/file columns.  The is_new mask finds segment boundaries in one
        # vectorised pass; path construction then runs only over the unique files.
        ep_slice = ds_info.episodes.slice(start_episode, end_episode - start_episode)
        chunks = ep_slice.column(f"videos/{vid_key}/chunk_index").combine_chunks()
        files = ep_slice.column(f"videos/{vid_key}/file_index").combine_chunks()
        n = len(ep_slice)
        is_new = pa.concat_arrays([
            pa.array([True]),
            pc.or_(  # type: ignore[attr-defined]
                pc.not_equal(chunks.slice(1), chunks.slice(0, n - 1)),  # type: ignore[attr-defined]
                pc.not_equal(files.slice(1), files.slice(0, n - 1)),  # type: ignore[attr-defined]
            ),
        ]) if n > 0 else pa.array([], type=pa.bool_())
        if n > 0:
            # from_timestamp of the first episode is the seek target inside the
            # first mp4 file.  Subsequent files always start at ts=0.
            self._start_ts = ep_slice.column(f"videos/{vid_key}/from_timestamp")[0].as_py()
        # Convert the full URI to the fs-local path form the cached filesystem understands.
        self._fs_paths: list[str] = [
            _fs_root + ds_info.video_file_path(vid_key, c, f)[len(_root_prefix):]
            for c, f in zip(
                pc.filter(chunks, is_new).to_pylist(),  # type: ignore[attr-defined]
                pc.filter(files, is_new).to_pylist(),  # type: ignore[attr-defined]
            )
        ]

    def close(self) -> None:
        """Close the currently open video container, if any."""
        if self._current_container is not None:
            self._current_container.close()
            self._current_container = None

    def __iter__(self):
        """Yield all frames in path order; close each container before the next."""
        for i, video_path in enumerate(self._fs_paths):
            self._current_container = self._open_container(video_path)
            try:
                stream = self._current_container.streams.video[0]
                assert stream.time_base is not None and stream.average_rate is not None
                # Only seek within the first file: from_timestamp tells us where
                # the first episode starts within the mp4.  Files beyond the first
                # always begin at the episode boundary that starts the file, so no
                # seek is needed — we just stream from the beginning.
                if i == 0 and self._start_ts > 0:
                    self._current_container.seek(
                        int(self._start_ts / stream.time_base), stream=stream
                    )
                # half_frame is half the inter-frame interval in seconds.  It is used
                # by EpisodeRangeReader as a tolerance window: a decoded frame is
                # considered the match for target_ts when frame.time ≥ target_ts − half_frame.
                half_frame = 0.5 * float(stream.time_base / stream.average_rate)
                for packet in self._current_container.demux(video=0):
                    try:
                        for frame in packet.decode():
                            yield frame, half_frame
                    except av.InvalidDataError:  # type: ignore[attr-defined]
                        # Corrupted or incomplete packets are silently skipped;
                        # EpisodeRangeReader will advance to the next valid frame.
                        continue
            finally:
                self.close()

    def _open_container(self, fs_path: str) -> Any:
        """Open a PyAV InputContainer for *fs_path* using the cached filesystem."""
        if self._is_local:
            return av.open(fs_path)
        return av.open(self._fs.open(fs_path, "rb"))


# ---------------------------------------------------------------------------
# Episode range reader
# ---------------------------------------------------------------------------


class EpisodeRangeReader:
    """Yields fully-decoded frame rows for a contiguous range of episodes.

    At construction, builds a :class:`DataSegmentIterator` and one
    :class:`VideoSegmentIterator` per camera key.  ``__iter__`` drives the
    data iterator row by row, advances each video iterator to the nearest
    frame, and appends video and task columns before yielding.
    """

    def __init__(
        self, ds_info: LeRobotDatasourceMetadata, start_episode: int, end_episode: int
    ) -> None:
        self._ds_info = ds_info
        self._data = DataSegmentIterator(ds_info, start_episode, end_episode)
        self._video: dict[str, VideoSegmentIterator] = {
            vid_key: VideoSegmentIterator(ds_info, vid_key, start_episode, end_episode)
            for vid_key in ds_info.video_keys
        }

    def __iter__(self):
        """Yield one-row Arrow tables with parquet, video, and task columns."""
        ds_info = self._ds_info

        if not ds_info.video_keys:
            for row in self._data:
                task_idx = row.column("task_index")[0].as_py()
                yield row.append_column(
                    "task", pa.array([ds_info.tasks[task_idx]], type=pa.string())
                )
            return

        video_iters = {vid_key: iter(vi) for vid_key, vi in self._video.items()}
        try:
            current_frames: dict[str, tuple[Any, float]] = {}
            current_ep_idx: int | None = None
            current_ep: dict | None = None

            for row in self._data:
                ep_idx: int = row.column("episode_index")[0].as_py()
                if ep_idx != current_ep_idx:
                    current_ep_idx = ep_idx
                    current_ep = ds_info.get_episode(ep_idx)
                row_ts: float = row.column("timestamp")[0].as_py()

                for vid_key in ds_info.video_keys:
                    assert current_ep is not None
                    target_ts = current_ep[f"videos/{vid_key}/from_timestamp"] + row_ts

                    if vid_key not in current_frames:
                        current_frames[vid_key] = next(video_iters[vid_key])
                    frame, half_frame = current_frames[vid_key]
                    prev_time: float | None = None
                    while True:
                        if frame.time is None:
                            logger.warning(
                                "ep=%d ts=%.4f %s: frame.time is None, skipping",
                                ep_idx,
                                target_ts,
                                vid_key,
                            )
                            current_frames[vid_key] = next(video_iters[vid_key])
                            frame, half_frame = current_frames[vid_key]
                            continue
                        if prev_time is not None and frame.time < prev_time:
                            logger.warning(
                                "ep=%d ts=%.4f %s: timestamp went backwards "
                                "(%.4f → %.4f), possible discontinuity at file boundary",
                                ep_idx,
                                target_ts,
                                vid_key,
                                prev_time,
                                frame.time,
                            )
                        if frame.time >= target_ts - half_frame:
                            break
                        prev_time = frame.time
                        current_frames[vid_key] = next(video_iters[vid_key])
                        frame, half_frame = current_frames[vid_key]

                    frame_np = frame.to_ndarray(format="rgb24")
                    ext_type = pa.fixed_shape_tensor(
                        pa.from_numpy_dtype(frame_np.dtype), frame_np.shape
                    )
                    storage = pa.FixedSizeListArray.from_arrays(
                        frame_np.reshape(-1), int(np.prod(frame_np.shape))
                    )
                    row = row.append_column(
                        vid_key, pa.ExtensionArray.from_storage(ext_type, storage)
                    )

                task_idx = row.column("task_index")[0].as_py()
                row = row.append_column(
                    "task", pa.array([ds_info.tasks[task_idx]], type=pa.string())
                )

                yield row
        finally:
            for vi in self._video.values():
                vi.close()


# ---------------------------------------------------------------------------
# Read task
# ---------------------------------------------------------------------------


class LeRobotReadTask(ReadTask):
    """Ray Data read task for a contiguous range of LeRobot episodes.

    Args:
        ds_info_ref: ``ray.put`` reference shared by all tasks in a build
            batch; retrieved both driver-side (for ``num_frames``) and
            worker-side (for iteration) via ``ray.get``.
        start_episode: First episode index (inclusive).
        end_episode: Last episode index (exclusive).
        per_task_row_limit: Optional Ray Data row cap per task.
    """

    def __init__(
        self,
        ds_info_ref: Any,
        start_episode: int,
        end_episode: int,
        per_task_row_limit: int | None = None,
    ) -> None:
        self.ref = ds_info_ref
        self.start = start_episode
        self.end = end_episode

        ds_info: LeRobotDatasourceMetadata = ray.get(ds_info_ref)
        ep_slice = ds_info.episodes.slice(start_episode, end_episode - start_episode)
        num_frames = int(pc.sum(pc.subtract(  # type: ignore[attr-defined]
            ep_slice.column("dataset_to_index"), ep_slice.column("dataset_from_index"),
        )).as_py())

        def read_fn():
            yield from EpisodeRangeReader(ray.get(self.ref), self.start, self.end)

        super().__init__(
            read_fn,
            BlockMetadata(
                num_rows=num_frames,
                size_bytes=num_frames * ds_info.estimated_row_size_bytes,
                input_files=None,
                exec_stats=None,
            ),
            per_task_row_limit=per_task_row_limit,
        )


# ---------------------------------------------------------------------------
# Builder base class and subclasses
# ---------------------------------------------------------------------------


class LeRobotReadTaskBuilder:
    """Base class for LeRobot read-task builders.

    Each subclass implements :meth:`episode_groups` to partition the dataset
    into contiguous episode ranges.  :meth:`build` merges those ranges down to
    the Ray ``parallelism`` target and creates one :class:`LeRobotReadTask` per
    merged range.  Merging happens at the grouping stage — wider episode ranges
    are passed directly to :class:`LeRobotReadTask`, which handles them natively
    via :class:`EpisodeRangeReader`.  No task-chaining or closure tricks needed.
    """

    def __init__(self, ds_info: LeRobotDatasourceMetadata) -> None:
        self.ds_info = ds_info

    def episode_groups(self) -> list[tuple[int, int]]:
        """Return ``(start_episode, end_episode)`` pairs, one per natural group."""
        raise NotImplementedError

    def build(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
    ) -> list[ReadTask]:
        """Merge :meth:`episode_groups` down to *parallelism* and build tasks.

        Consecutive groups are merged by spanning their episode ranges, then one
        :class:`LeRobotReadTask` is created per merged range.  ``ray.put`` is
        called once so all tasks share a single serialised ``LeRobotDatasourceMetadata``.
        """
        ds_info = self.ds_info
        groups = sorted(self.episode_groups())
        if parallelism > 0 and len(groups) > parallelism:
            n = len(groups)
            base, remainder = divmod(n, parallelism)
            merged: list[tuple[int, int]] = []
            i = 0
            for g in range(parallelism):
                chunk = groups[i : i + base + (1 if g < remainder else 0)]
                i += len(chunk)
                merged.append((chunk[0][0], chunk[-1][1]))
            groups = merged
        logger.info(
            "%s: %d tasks, %d frames, %d cameras",
            type(self).__name__,
            len(groups),
            ds_info.total_frames,
            len(ds_info.video_keys),
        )
        ds_info_ref = ray.put(ds_info)
        return [
            LeRobotReadTask(ds_info_ref, start, end, per_task_row_limit)
            for start, end in groups
        ]


class EpisodeReadTaskBuilder(LeRobotReadTaskBuilder):
    """One task per episode — maximum parallelism."""

    def episode_groups(self) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(self.ds_info.total_episodes)]


class FileGroupReadTaskBuilder(LeRobotReadTaskBuilder):
    """One task per unique set of video files.

    Episodes that share the exact same video file for every camera are grouped
    together.  Within a task, the video iterator seeks once per file and
    streams continuously across episodes (seek-skip optimisation).
    """

    def __init__(self, ds_info: LeRobotDatasourceMetadata) -> None:
        super().__init__(ds_info)
        self._groups = self._build_file_groups(ds_info)

    def episode_groups(self) -> list[tuple[int, int]]:
        return [(eps[0], eps[-1] + 1) for eps in self._groups.values()]

    @staticmethod
    def _file_group_key(
        ds_info: LeRobotDatasourceMetadata, ep: dict
    ) -> tuple[str, ...]:
        return tuple(
            ds_info.video_file_path(
                vid_key,
                ep[f"videos/{vid_key}/chunk_index"],
                ep[f"videos/{vid_key}/file_index"],
            )
            for vid_key in ds_info.video_keys
        )

    @staticmethod
    def _build_file_groups(
        ds_info: LeRobotDatasourceMetadata,
    ) -> dict[tuple[str, ...], list[int]]:
        """Map each unique video-file tuple to its sorted list of episode indices."""
        groups: dict[tuple[str, ...], list[int]] = {}
        for i in range(ds_info.total_episodes):
            ep = ds_info.get_episode(i)
            groups.setdefault(
                FileGroupReadTaskBuilder._file_group_key(ds_info, ep), []
            ).append(ep["episode_index"])
        return groups


class ChainReadTaskBuilder(LeRobotReadTaskBuilder):
    """One task per connected component of episodes that share any video file.

    Uses union-find directly over episodes: each episode is a node; two
    episodes are in the same component when they reference the same mp4 file
    for at least one camera.  This ensures each video file is opened at most
    once per task and yields the minimal number of tasks needed to avoid
    redundant cloud reads (often 1–4 for large datasets like DROID).
    """

    def episode_groups(self) -> list[tuple[int, int]]:
        ds_info = self.ds_info
        episode_list = list(range(ds_info.total_episodes))
        parent: dict[int, int] = {ep: ep for ep in episode_list}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        video_file_to_episode: dict[str, int] = {}
        for ep_idx in episode_list:
            ep = ds_info.get_episode(ep_idx)
            for vid_key in ds_info.video_keys:
                path = ds_info.video_file_path(
                    vid_key,
                    ep[f"videos/{vid_key}/chunk_index"],
                    ep[f"videos/{vid_key}/file_index"],
                )
                if path in video_file_to_episode:
                    union(ep_idx, video_file_to_episode[path])
                else:
                    video_file_to_episode[path] = ep_idx

        components: dict[int, list[int]] = {}
        for ep_idx in episode_list:
            components.setdefault(find(ep_idx), []).append(ep_idx)

        return sorted((eps[0], eps[-1] + 1) for eps in components.values())


class SequentialReadTaskBuilder(LeRobotReadTaskBuilder):
    """Single task that streams all episodes sequentially.

    Designed for cloud-hosted datasets (GCS/S3) where minimising peak memory
    matters more than parallelism: only one parquet file and one set of video
    containers are open at a time.
    """

    def episode_groups(self) -> list[tuple[int, int]]:
        return [(0, self.ds_info.total_episodes)]


_BUILDERS: dict[str, type[LeRobotReadTaskBuilder]] = {
    "sequential": SequentialReadTaskBuilder,
    "episode": EpisodeReadTaskBuilder,
    "file_group": FileGroupReadTaskBuilder,
    "chain": ChainReadTaskBuilder,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ParallelismMode(enum.Enum):
    """How the dataset is partitioned into Ray Data read tasks.

    Listed from highest parallelism to lowest:

    EPISODE       One task per episode.  Maximum parallelism but each
                  video file may be opened many times.
    FILE_GROUP    One task per unique set of video files.  Opens each
                  video container once per task; good balance of
                  parallelism and cloud-read efficiency.
    CHAIN         Merges overlapping FILE_GROUP groups via union-find so
                  each video file is opened exactly once.  Fewest cloud
                  reads but very few (often 1-4) tasks.
    SEQUENTIAL    Single task, streams parquet file-by-file from cloud
                  storage without loading the full dataset into memory.
                  Use when the dataset lives on GCS/S3 and parallelism
                  is less important than minimising peak memory.
    """

    EPISODE = "episode"
    FILE_GROUP = "file_group"
    CHAIN = "chain"
    SEQUENTIAL = "sequential"


class LeRobotDatasource(Datasource):
    """Ray Data ``Datasource`` for LeRobot-format robotics datasets.

    Args:
        root: Path to dataset root — local, ``gs://``, or ``s3://``.
        parallelism_mode: Controls how the dataset is split into Ray tasks.
            Accepts a :class:`ParallelismMode` enum value or its string
            equivalent (``"episode"``, ``"file_group"``, ``"chain"``,
            ``"sequential"``).  Defaults to ``"file_group"``.
    """

    def __init__(
        self,
        root: str | Path,
        parallelism_mode: ParallelismMode | str = ParallelismMode.FILE_GROUP,
    ):
        if isinstance(parallelism_mode, ParallelismMode):
            parallelism_mode = parallelism_mode.value
        if parallelism_mode not in _BUILDERS:
            raise ValueError(
                f"Unknown parallelism mode {parallelism_mode!r}. "
                f"Choose from: {', '.join(_BUILDERS)}"
            )

        self._ds_info = LeRobotDatasourceMetadata(str(root))
        self._parallelism_mode = parallelism_mode

        logger.info(
            "LeRobotDatasource ready: %d episodes, %d frames, %d cameras %s, mode=%r, root=%s",
            self._ds_info.total_episodes,
            self._ds_info.total_frames,
            len(self._ds_info.video_keys),
            self._ds_info.video_keys,
            self._parallelism_mode,
            self._ds_info.root,
        )

    def estimate_inmemory_data_size(self) -> int | None:
        """Return estimated in-memory dataset size (parquet + decoded video)."""
        return self._ds_info.estimated_inmemory_size_bytes

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: DataContext | None = None,
    ) -> list[ReadTask]:
        """Build and return read tasks for the configured parallelism mode."""
        builder = _BUILDERS[self._parallelism_mode](self._ds_info)
        return builder.build(parallelism, per_task_row_limit)
