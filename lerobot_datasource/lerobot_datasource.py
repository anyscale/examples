"""Ray Data datasource for LeRobot v3 robotics datasets.

A LeRobot dataset is a set of **parallel streams of equal length**: one data
stream (parquet rows) and one video stream per camera.  Every stream has
``total_frames`` entries.  A **slice** is a ``[start_row, end_row)`` range
applied uniformly to all streams.

Grouping strategies decide how to partition ``[0, total_frames)`` into slices.
Episode boundaries, file-group boundaries, chain components, and arbitrary
N-row blocks are all just slices — the streams resolve which physical files
they need internally.

Data model (relevant columns)
-----------------------------
::

    Row (one per video frame):
        index            int64   global 0-based row id across the entire dataset
        episode_index    int64   which episode this row belongs to
        frame_index      int64   0-based position within its episode
        timestamp        float64 time in seconds within its episode

    Episode metadata (one per episode, in meta/episodes/*.parquet):
        episode_index        int64   episode id (= row position in the table)
        dataset_from_index   int64   first global row index (inclusive)
        dataset_to_index     int64   last global row index (exclusive)
        data/chunk_index     int64   which parquet chunk file holds this episode's data
        data/file_index      int64   file index within the chunk
        videos/<key>/chunk_index   int64   video chunk for camera <key>
        videos/<key>/file_index    int64   video file index for camera <key>
        videos/<key>/from_timestamp float64 seek offset into the video file

    Relationship:
        index == dataset_from_index[episode_index] + frame_index

Grouping strategies
-------------------
+---------------+------------------+------------------------------------+
| Mode          | Groups created   | Best for                           |
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
| ``row_block`` | ceil(total / N)  | fixed-size blocks of N rows;       |
|               |                  | set via ``group_size`` argument    |
+---------------+------------------+------------------------------------+

Architecture overview
---------------------
::

    LeRobotDatasource              (public API — Ray Data Datasource)
        │
        ▼
    SliceStrategy                  (base class: slices() → build())
        │  subclasses: Episode, FileGroup, Chain, Sequential, FixedRowBlock
        │
        │  slices() returns [(start_row, end_row), ...]
        │  build() merges slices down to Ray parallelism target
        │
        ▼
    LeRobotReadTask                (one per merged slice)
        │
        ▼
    SampleStream                   (orchestrates streams for a slice)
        ├── LowDimStream           (reads parquet rows for [start, end))
        └── VideoStream            (reads video frames for [start, end))

Typical usage::

    import ray
    from lerobot_datasource import LeRobotDatasource, GroupingMode

    # Default file_group mode
    ds = ray.data.read_datasource(LeRobotDatasource(root="/data/my_dataset"))

    # Fixed 1024-row blocks
    ds = ray.data.read_datasource(LeRobotDatasource(
        root="gs://bucket/dataset",
        grouping_mode=GroupingMode.ROW_BLOCK,
        group_size=1024,
    ))
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
from ray.data.extensions import ArrowVariableShapedTensorArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class LeRobotDatasourceMetadata:
    """Lightweight metadata for the datasource.

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
        """Load meta/stats.json; return dataset-wide normalisation statistics."""
        with fs.open(f"{fs_root}/meta/stats.json", "r") as f:
            return json.load(f)

    def _fetch_episodes(self, fs: Any, fs_root: str) -> pa.Table:
        """Load meta/episodes/**/*.parquet; return the concatenated table."""
        ep_files = sorted(fs.glob(f"{fs_root}/meta/episodes/**/*.parquet"))
        if not ep_files:
            raise FileNotFoundError(
                f"No episode parquet files found under {self.root!r}/meta/episodes/. "
                "The dataset may be incomplete or use an unsupported layout."
            )
        return pa.concat_tables([pq.read_table(fs.open(f, "rb")) for f in ep_files])

    def _fetch_tasks(self, fs: Any, fs_root: str) -> dict[int, str]:
        """Load meta/tasks.parquet; return a task_index → task string mapping."""
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
        """Estimated in-memory size of one fully-decoded frame row (bytes)."""
        features = self.info.get("features", {})
        total = 0
        for feat in features.values():
            if feat.get("dtype") == "video":
                shape = feat.get("shape")
                if shape:
                    total += int(np.prod(shape))
            else:
                shape = feat.get("shape", [1])
                try:
                    total += int(np.prod(shape)) * np.dtype(feat["dtype"]).itemsize
                except (TypeError, KeyError):
                    continue
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

    def episodes_for_row_range(self, start_row: int, end_row: int) -> tuple[int, int]:
        """Return ``(start_ep, end_ep)`` — the minimal episode range covering ``[start_row, end_row)``.

        An episode overlaps the row range when its row span intersects it::

            dataset_from_index < end_row  AND  dataset_to_index > start_row

        Returns a half-open episode range suitable for slicing the episodes table.
        """
        from_idx = self.episodes.column("dataset_from_index")
        to_idx = self.episodes.column("dataset_to_index")
        mask = pc.and_(  # type: ignore[attr-defined]
            pc.less(from_idx, end_row),  # type: ignore[attr-defined]
            pc.greater(to_idx, start_row),  # type: ignore[attr-defined]
        )
        indices = pc.filter(  # type: ignore[attr-defined]
            self.episodes.column("episode_index"), mask
        ).to_pylist()
        if not indices:
            raise ValueError(
                f"No episodes overlap the row range [{start_row}, {end_row}). "
                f"Dataset has {self.total_frames} total frames across "
                f"{self.total_episodes} episodes."
            )
        return (indices[0], indices[-1] + 1)

    def get_episode(self, ep_idx: int) -> dict:
        """Return one episode's metadata dict by its ``episode_index``."""
        return self.episodes.slice(ep_idx, 1).to_pylist()[0]


# ---------------------------------------------------------------------------
# Video stream
# ---------------------------------------------------------------------------


class VideoStream:
    """Decoded video frames for a single camera over a row range.

    An independently iterable stream: given ``[start_row, end_row)`` it
    resolves which episodes overlap, which mp4 files to open, and the
    per-episode seek offsets — then yields one decoded RGB frame per row
    via ``__next__``.

    Timestamp computation uses the episodes table and ``fps`` from
    ``info.json``::

        frame_index = global_index - dataset_from_index
        timestamp   = frame_index / fps
        target_ts   = from_timestamp + timestamp

    This makes VideoStream fully independent of the parquet data stream,
    enabling SampleStream to be a simple ``zip(LowDimStream, *VideoStreams)``.
    """

    def __init__(
        self,
        ds_meta: LeRobotDatasourceMetadata,
        vid_key: str,
        start_row: int,
        end_row: int,
    ) -> None:
        self._fs, _fs_root = fsspec.core.url_to_fs(ds_meta.root)
        self._is_local = self._fs.protocol == "file"
        _root_prefix = ds_meta.root
        self._current_container: Any = None

        self._fps: float = ds_meta.info["fps"]
        self._current_index = start_row
        self._end_row = end_row

        start_ep, end_ep = ds_meta.episodes_for_row_range(start_row, end_row)
        ep_slice = ds_meta.episodes.slice(start_ep, end_ep - start_ep)
        n = len(ep_slice)

        # Episode schedule: (dataset_from_index, dataset_to_index, from_timestamp).
        # Used by __next__ to compute the absolute video timestamp for each row.
        from_indices = ep_slice.column("dataset_from_index").to_pylist()
        to_indices = ep_slice.column("dataset_to_index").to_pylist()
        from_ts_values = ep_slice.column(f"videos/{vid_key}/from_timestamp").to_pylist()
        self._ep_schedule: list[tuple[int, int, float]] = list(
            zip(from_indices, to_indices, from_ts_values)
        )
        self._ep_cursor = 0

        # Consecutive-dedup over video chunk/file columns.
        chunks = ep_slice.column(f"videos/{vid_key}/chunk_index").combine_chunks()
        files = ep_slice.column(f"videos/{vid_key}/file_index").combine_chunks()
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

        self._start_ts = from_ts_values[0] if n > 0 else 0.0

        self._fs_paths: list[str] = [
            _fs_root + ds_meta.video_file_path(vid_key, c, f)[len(_root_prefix) :]
            for c, f in zip(
                pc.filter(chunks, is_new).to_pylist(),  # type: ignore[attr-defined]
                pc.filter(files, is_new).to_pylist(),  # type: ignore[attr-defined]
            )
        ]

        self._frame_iter = iter(self._raw_frames())
        self._current_frame: Any = None
        self._half_frame: float = 0.0

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        """Return the next decoded RGB frame, advancing by one row index."""
        if self._current_index >= self._end_row:
            raise StopIteration

        idx = self._current_index

        # Advance episode cursor if the current row has moved past the current episode.
        while self._ep_cursor < len(self._ep_schedule) - 1:
            _, ep_to, _ = self._ep_schedule[self._ep_cursor]
            if idx < ep_to:
                break
            self._ep_cursor += 1

        ep_from, _, from_ts = self._ep_schedule[self._ep_cursor]
        frame_index = idx - ep_from
        target_ts = from_ts + frame_index / self._fps

        frame = self._advance_to_ts(target_ts)
        self._current_index += 1
        return frame

    def _advance_to_ts(self, target_ts: float) -> np.ndarray:
        """Advance the raw frame iterator to *target_ts* and return the RGB array."""
        if self._current_frame is None:
            self._current_frame, self._half_frame = next(self._frame_iter)

        frame = self._current_frame
        half_frame = self._half_frame

        while True:
            if frame.time is None:
                logger.warning(
                    "row=%d ts=%.4f: frame.time is None, skipping",
                    self._current_index,
                    target_ts,
                )
                self._current_frame, self._half_frame = next(self._frame_iter)
                frame, half_frame = self._current_frame, self._half_frame
                continue

            if frame.time >= target_ts - half_frame:
                break

            self._current_frame, self._half_frame = next(self._frame_iter)
            frame, half_frame = self._current_frame, self._half_frame

        return frame.to_ndarray(format="rgb24")

    def _raw_frames(self):
        """Yield ``(frame, half_frame)`` pairs from all mp4 files in sequence."""
        for i, video_path in enumerate(self._fs_paths):
            self._current_container = self._open_container(video_path)
            try:
                stream = self._current_container.streams.video[0]
                assert stream.time_base is not None and stream.average_rate is not None
                if i == 0 and self._start_ts > 0:
                    self._current_container.seek(
                        int(self._start_ts / stream.time_base), stream=stream
                    )
                half_frame = 0.5 * float(stream.time_base / stream.average_rate)
                for packet in self._current_container.demux(video=0):
                    try:
                        for frame in packet.decode():
                            yield frame, half_frame
                    except av.InvalidDataError:
                        continue
            finally:
                if self._current_container is not None:
                    self._current_container.close()
                    self._current_container = None

    def close(self) -> None:
        """Close the currently open video container, if any."""
        if self._current_container is not None:
            self._current_container.close()
            self._current_container = None

    def _open_container(self, fs_path: str) -> Any:
        """Open a PyAV InputContainer for *fs_path* using the cached filesystem."""
        if self._is_local:
            return av.open(fs_path)
        return av.open(self._fs.open(fs_path, "rb"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch_limits() -> tuple[int, int]:
    """Return ``(min_batch_bytes, max_batch_bytes)`` from Ray Data's DataContext.

    Uses ``target_min_block_size`` and ``target_max_block_size`` so that
    our batching aligns with the same thresholds Ray Data uses internally
    for block sizing.  Falls back to sensible defaults if the context is
    unavailable (e.g. during unit tests without a Ray runtime).
    """
    try:
        ctx = DataContext.get_current()
        return (
            ctx.target_min_block_size,
            ctx.target_max_block_size or 128 * 1024 * 1024,
        )
    except Exception:
        return 1 * 1024 * 1024, 128 * 1024 * 1024


# ---------------------------------------------------------------------------
# Data stream
# ---------------------------------------------------------------------------

# TODO: For small/medium datasets (most are much smaller than DROID's 27.6M
# frames), optionally preload all data parquet into memory instead of reading
# per-segment.  The episodes table (~9 MB even for DROID) already fits easily;
# the data files will too for typical datasets.


class LowDimStream:
    """Parquet data stream for ``[start_row, end_row)``.

    Yields one-row Arrow tables filtered by the global ``index`` column.
    Resolves which parquet files to read internally via episode metadata.

    Construction:
        1. Find which episodes overlap ``[start_row, end_row)``
        2. Extract those episodes' ``(data/chunk_index, data/file_index)`` pairs
        3. Deduplicate consecutive identical pairs (adjacent episodes often
           share the same parquet file)
        4. Store the unique file keys as ``self._segments``

    Iteration (per segment / parquet file):
        1. Open the file via fsspec (works for local, gs://, s3://)
        2. Push down row-group-level filters via ``pq.read_table(filters=...)``
        3. Yield one row at a time as single-row Arrow tables
    """

    def __init__(
        self, ds_meta: LeRobotDatasourceMetadata, start_row: int, end_row: int
    ) -> None:
        self._ds_meta = ds_meta
        self._start_row = start_row
        self._end_row = end_row

        start_ep, end_ep = ds_meta.episodes_for_row_range(start_row, end_row)
        ep_slice = ds_meta.episodes.slice(start_ep, end_ep - start_ep)
        chunks = ep_slice.column("data/chunk_index").combine_chunks()
        files = ep_slice.column("data/file_index").combine_chunks()
        n = len(ep_slice)

        # Consecutive-dedup: mark segment boundaries where chunk or file changes.
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
        """Yield one row per frame, filtered to ``[start_row, end_row)`` by global index."""
        ds_meta = self._ds_meta
        filters = [
            ("index", ">=", self._start_row),
            ("index", "<", self._end_row),
        ]
        for chunk_idx, file_idx in self._segments:
            path = ds_meta.data_file_path(chunk_idx, file_idx)
            fs, fs_path = fsspec.core.url_to_fs(path)
            with fs.open(fs_path, "rb") as f:
                pq_table = pq.read_table(f, filters=filters)
            for i in range(pq_table.num_rows):
                yield pq_table.slice(i, 1)


# ---------------------------------------------------------------------------
# Sample stream
# ---------------------------------------------------------------------------


class SampleStream:
    """Combines LowDimStream + VideoStreams for ``[start_row, end_row)``.

    Yields dynamically-batched Arrow tables with parquet data, decoded video
    frames, and task strings.  Each stream takes the same ``[start, end)``
    range and resolves its own files internally.

    Video frames are stored as **variable-shaped tensors** using
    ``ArrowVariableShapedTensorArray`` so that episodes with different camera
    resolutions can coexist in the same Arrow column.

    **Batching**: rows are accumulated and flushed when the buffer exceeds
    ``min_batch_bytes`` from Ray Data's DataContext.
    """

    def __init__(
        self, ds_meta: LeRobotDatasourceMetadata, start_row: int, end_row: int
    ) -> None:
        self._ds_meta = ds_meta
        self._data = LowDimStream(ds_meta, start_row, end_row)
        self._video: dict[str, VideoStream] = {
            vid_key: VideoStream(ds_meta, vid_key, start_row, end_row)
            for vid_key in ds_meta.video_keys
        }

    def _enrich_row(
        self,
        row: pa.Table,
        video_columns: dict[str, ArrowVariableShapedTensorArray],
        task_str: str,
    ) -> pa.Table:
        """Append all video columns and the task string in one pass."""
        columns: dict[str, pa.Array | pa.ChunkedArray] = {
            row.schema.field(i).name: row.column(i) for i in range(row.num_columns)
        }
        for vid_key, tensor_arr in video_columns.items():
            columns[vid_key] = tensor_arr
        columns["task"] = pa.array([task_str], type=pa.string())
        return pa.table(columns)

    def __iter__(self):
        """Yield batched Arrow tables with parquet, video, and task columns."""
        ds_meta = self._ds_meta
        vid_keys = list(self._video.keys())
        min_batch_bytes, _ = _batch_limits()

        # zip(LowDimStream, VideoStream_0, VideoStream_1, ...)
        streams: Any = (
            zip(self._data, *(self._video[k] for k in vid_keys))
            if vid_keys
            else ((row,) for row in self._data)
        )

        try:
            buffer: list[pa.Table] = []
            buffer_bytes = 0

            for items in streams:
                row = items[0]
                video_columns: dict[str, ArrowVariableShapedTensorArray] = {
                    k: ArrowVariableShapedTensorArray.from_numpy([items[i + 1]])
                    for i, k in enumerate(vid_keys)
                }

                task_idx = row.column("task_index")[0].as_py()
                enriched = self._enrich_row(row, video_columns, ds_meta.tasks[task_idx])

                row_size = enriched.nbytes
                buffer.append(enriched)
                buffer_bytes += row_size
                if buffer_bytes >= min_batch_bytes:
                    yield pa.concat_tables(buffer)
                    buffer.clear()
                    buffer_bytes = 0

            if buffer:
                yield pa.concat_tables(buffer)
        finally:
            for vs in self._video.values():
                vs.close()


# ---------------------------------------------------------------------------
# Read task
# ---------------------------------------------------------------------------


class LeRobotReadTask(ReadTask):
    """Ray Data read task for a single slice ``[start_row, end_row)``.

    Created by :meth:`SliceStrategy.build`.  Each task holds a ``ray.put``
    reference to the shared metadata and its own row range boundaries.
    """

    def __init__(
        self,
        ds_meta_ref: Any,
        start_row: int,
        end_row: int,
        estimated_row_size_bytes: int,
        per_task_row_limit: int | None = None,
    ) -> None:
        num_frames = end_row - start_row

        # Capture only the values the closure needs — not ``self``.
        ref = ds_meta_ref
        start = start_row
        end = end_row

        def read_fn():
            yield from SampleStream(ray.get(ref), start, end)

        super().__init__(
            read_fn,
            BlockMetadata(
                num_rows=num_frames,
                size_bytes=num_frames * estimated_row_size_bytes,
                input_files=None,
                exec_stats=None,
            ),
            per_task_row_limit=per_task_row_limit,
        )


# ---------------------------------------------------------------------------
# Slice strategies
# ---------------------------------------------------------------------------
#
# Each subclass decides *how* to partition the dataset into slices
# (via slices()), while the base class handles merging to match the Ray
# parallelism target and creating the actual ReadTask objects.
#
# Dataflow:
#   1. Subclass.slices() → [(start_row, end_row), ...]
#   2. Base.build() validates contiguity, sorts and merges slices
#   3. Base.build() creates one LeRobotReadTask per merged slice
# ---------------------------------------------------------------------------


class SliceStrategy:
    """Base class for slice strategies.

    Each subclass implements :meth:`slices` to partition the dataset into
    ``(start_row, end_row)`` pairs.  :meth:`build` merges those down to the
    Ray ``parallelism`` target and creates one :class:`LeRobotReadTask` per
    merged slice.
    """

    def __init__(self, ds_meta: LeRobotDatasourceMetadata, **kwargs: Any) -> None:
        self.ds_meta = ds_meta

    def slices(self) -> list[tuple[int, int]]:
        """Return ``(start_row, end_row)`` pairs, one per natural group.

        **Contract**: the returned ranges MUST be non-overlapping and
        contiguous — each slice's ``start_row`` must equal the previous
        slice's ``end_row``.
        """
        raise NotImplementedError

    def build(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
    ) -> list[ReadTask]:
        """Merge :meth:`slices` down to *parallelism* and build read tasks."""
        ds_meta = self.ds_meta
        groups = sorted(self.slices())

        # Validate contiguity.
        for i in range(1, len(groups)):
            prev_end = groups[i - 1][1]
            curr_start = groups[i][0]
            if prev_end != curr_start:
                raise ValueError(
                    f"Non-contiguous slices: slice {i - 1} ends at row "
                    f"{prev_end} but slice {i} starts at row {curr_start}. "
                    f"slices() must return contiguous, non-overlapping ranges."
                )

        # Merge if we have more slices than the parallelism target allows.
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
            "%s: %d tasks, %d total frames, %d cameras",
            type(self).__name__,
            len(groups),
            ds_meta.total_frames,
            len(ds_meta.video_keys),
        )

        ds_meta_ref = ray.put(ds_meta)
        row_size = ds_meta.estimated_row_size_bytes
        return [
            LeRobotReadTask(ds_meta_ref, start, end, row_size, per_task_row_limit)
            for start, end in groups
        ]


# ---- Concrete strategies ---------------------------------------------------
#
# Each strategy answers one question: "given the dataset, what are the natural
# slices?"  The base class handles everything after that.


class EpisodeSlice(SliceStrategy):
    """One slice per episode — maximum parallelism.

    Each episode's ``[dataset_from_index, dataset_to_index)`` becomes one
    slice.  The same video file may be opened by multiple tasks if episodes
    in the same mp4 chunk end up in different tasks.
    """

    def slices(self) -> list[tuple[int, int]]:
        ds = self.ds_meta
        from_indices = ds.episodes.column("dataset_from_index").to_pylist()
        to_indices = ds.episodes.column("dataset_to_index").to_pylist()
        return list(zip(from_indices, to_indices))


class FileGroupSlice(SliceStrategy):
    """One slice per unique set of video files (default strategy).

    Episodes that reference the *exact same* video file for *every* camera
    are grouped together.  Within a chunk, all episodes share the same set
    of mp4 files (one per camera), so each chunk typically becomes one slice.

    The grouping key is the tuple of ``(chunk_index, file_index)`` pairs
    across all cameras.
    """

    def slices(self) -> list[tuple[int, int]]:
        ds = self.ds_meta
        eps = ds.episodes

        key_columns: list[list[int]] = []
        for vk in ds.video_keys:
            key_columns.append(eps.column(f"videos/{vk}/chunk_index").to_pylist())
            key_columns.append(eps.column(f"videos/{vk}/file_index").to_pylist())

        from_indices = eps.column("dataset_from_index").to_pylist()
        to_indices = eps.column("dataset_to_index").to_pylist()
        n = len(eps)

        groups: dict[tuple[int, ...], tuple[int, int]] = {}
        for i in range(n):
            key = tuple(col[i] for col in key_columns)
            from_idx = from_indices[i]
            to_idx = to_indices[i]
            if key in groups:
                prev_from, prev_to = groups[key]
                groups[key] = (min(prev_from, from_idx), max(prev_to, to_idx))
            else:
                groups[key] = (from_idx, to_idx)

        return list(groups.values())


class ChainSlice(SliceStrategy):
    """One slice per connected component of episodes sharing any video file.

    Uses union-find (disjoint-set) with path compression and union-by-rank
    over episodes: two episodes are in the same component when they reference
    the same mp4 file for *at least one* camera.  This produces the minimal
    number of slices such that each video file appears in exactly one slice.

    Typically yields 1-4 slices even for large datasets (like DROID with
    52k episodes), because most video chunks overlap transitively.
    """

    def slices(self) -> list[tuple[int, int]]:
        ds = self.ds_meta
        eps = ds.episodes
        n = len(eps)

        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        video_file_to_episode: dict[tuple[str, int, int], int] = {}
        for vid_key in ds.video_keys:
            vid_chunks = eps.column(f"videos/{vid_key}/chunk_index").to_pylist()
            vid_files = eps.column(f"videos/{vid_key}/file_index").to_pylist()
            for ep_idx in range(n):
                file_key = (vid_key, vid_chunks[ep_idx], vid_files[ep_idx])
                if file_key in video_file_to_episode:
                    union(ep_idx, video_file_to_episode[file_key])
                else:
                    video_file_to_episode[file_key] = ep_idx

        from_indices = eps.column("dataset_from_index").to_pylist()
        to_indices = eps.column("dataset_to_index").to_pylist()

        component_ranges: dict[int, tuple[int, int]] = {}
        for ep_idx in range(n):
            root = find(ep_idx)
            from_idx = from_indices[ep_idx]
            to_idx = to_indices[ep_idx]
            if root in component_ranges:
                prev_from, prev_to = component_ranges[root]
                component_ranges[root] = (
                    min(prev_from, from_idx),
                    max(prev_to, to_idx),
                )
            else:
                component_ranges[root] = (from_idx, to_idx)

        return sorted(component_ranges.values())


class SequentialSlice(SliceStrategy):
    """Single slice spanning all rows — minimises peak memory."""

    def slices(self) -> list[tuple[int, int]]:
        return [(0, self.ds_meta.total_frames)]


class FixedRowBlockSlice(SliceStrategy):
    """Splits total_frames into fixed-size blocks of ``group_size`` rows.

    Block boundaries can fall in the middle of an episode.  The streams
    handle partial episodes transparently.
    """

    def __init__(
        self, ds_meta: LeRobotDatasourceMetadata, *, group_size: int, **kwargs: Any
    ) -> None:
        super().__init__(ds_meta, **kwargs)
        self._group_size = group_size

    def slices(self) -> list[tuple[int, int]]:
        total = self.ds_meta.total_frames
        size = self._group_size
        return [(i, min(i + size, total)) for i in range(0, total, size)]


# Registry mapping mode strings to strategy classes.
_STRATEGIES: dict[str, type[SliceStrategy]] = {
    "sequential": SequentialSlice,
    "episode": EpisodeSlice,
    "file_group": FileGroupSlice,
    "chain": ChainSlice,
    "row_block": FixedRowBlockSlice,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class GroupingMode(enum.Enum):
    """How the dataset is partitioned into Ray Data read tasks.

    All modes produce ``(start_row, end_row)`` slices over the global
    ``index`` column.  Episode-aligned modes are a special case where
    boundaries coincide with episode boundaries.

    EPISODE       One slice per episode.
    FILE_GROUP    One slice per unique set of video files (default).
    CHAIN         One slice per connected component of shared video files.
    SEQUENTIAL    Single slice spanning the entire dataset.
    ROW_BLOCK     Fixed-size blocks of ``group_size`` rows.
    """

    EPISODE = "episode"
    FILE_GROUP = "file_group"
    CHAIN = "chain"
    SEQUENTIAL = "sequential"
    ROW_BLOCK = "row_block"


class LeRobotDatasource(Datasource):
    """Ray Data ``Datasource`` for LeRobot datasets.

    This is the public entry point.  Usage::

        ds = ray.data.read_datasource(LeRobotDatasource(
            root="/data/my_dataset",
            grouping_mode=GroupingMode.ROW_BLOCK,
            group_size=1024,
        ))

    The constructor loads dataset metadata eagerly on the driver.  The actual
    data and video I/O happens lazily on workers when Ray schedules the read
    tasks.
    """

    def __init__(
        self,
        root: str | Path,
        grouping_mode: GroupingMode | str = GroupingMode.FILE_GROUP,
        group_size: int | None = None,
    ):
        if isinstance(grouping_mode, GroupingMode):
            grouping_mode = grouping_mode.value

        if grouping_mode not in _STRATEGIES:
            raise ValueError(
                f"Unknown grouping mode {grouping_mode!r}. "
                f"Choose from: {', '.join(_STRATEGIES)}"
            )
        if grouping_mode == "row_block" and group_size is None:
            raise ValueError("group_size is required when grouping_mode is 'row_block'")
        if grouping_mode != "row_block" and group_size is not None:
            raise ValueError(
                f"group_size is only valid with 'row_block' mode, not {grouping_mode!r}"
            )

        self._ds_meta = LeRobotDatasourceMetadata(str(root))
        self._grouping_mode = grouping_mode
        self._group_size = group_size

        logger.info(
            "LeRobotDatasource ready: %d episodes, %d frames, %d cameras %s, "
            "mode=%r, group_size=%s, root=%s",
            self._ds_meta.total_episodes,
            self._ds_meta.total_frames,
            len(self._ds_meta.video_keys),
            self._ds_meta.video_keys,
            self._grouping_mode,
            self._group_size,
            self._ds_meta.root,
        )

    def estimate_inmemory_data_size(self) -> int | None:
        """Hint for Ray Data's memory-aware scheduling."""
        return self._ds_meta.estimated_inmemory_size_bytes

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: DataContext | None = None,
    ) -> list[ReadTask]:
        """Build and return read tasks for the configured grouping mode."""
        kwargs: dict[str, Any] = {}
        if self._group_size is not None:
            kwargs["group_size"] = self._group_size
        strategy = _STRATEGIES[self._grouping_mode](self._ds_meta, **kwargs)
        return strategy.build(parallelism, per_task_row_limit)


def read_lerobot(
    root: str | Path,
    grouping_mode: GroupingMode | str = GroupingMode.FILE_GROUP,
    group_size: int | None = None,
    **kwargs: Any,
) -> tuple["ray.data.Dataset", LeRobotDatasourceMetadata]:
    """Read a LeRobot dataset as a Ray Data ``Dataset``.

    Convenience wrapper around ``ray.data.read_datasource(LeRobotDatasource(...))``.

    Args:
        root: Path or URI to the dataset root (local, ``gs://``, ``s3://``).
        grouping_mode: How to partition the dataset into read tasks.
        group_size: Block size in rows; required when *grouping_mode* is ``row_block``.
        **kwargs: Forwarded verbatim to ``ray.data.read_datasource`` (e.g.
            ``override_num_blocks``, ``num_cpus``, ``ray_remote_args``).

    Returns:
        A ``(dataset, meta)`` tuple — the Ray Data ``Dataset`` of decoded frames
        and the ``LeRobotDatasourceMetadata`` for the dataset (episodes table,
        video keys, stats, etc.).

    Example::

        import ray
        from lerobot_datasource import read_lerobot, GroupingMode

        ds, meta = read_lerobot("/data/my_dataset")
        ds, meta = read_lerobot("gs://bucket/dataset", grouping_mode=GroupingMode.EPISODE)
        ds, meta = read_lerobot("/data/my_dataset", grouping_mode=GroupingMode.ROW_BLOCK, group_size=1024)
        ds, meta = read_lerobot("/data/my_dataset", override_num_blocks=8, num_cpus=2)
        print(meta.total_frames, meta.video_keys)
    """
    source = LeRobotDatasource(root, grouping_mode=grouping_mode, group_size=group_size)
    return ray.data.read_datasource(source, **kwargs), source._ds_meta
