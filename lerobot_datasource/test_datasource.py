"""Functional tests for lerobot_datasource.

Primary test: collect frames from the first N episodes via both LeRobotDataset
and EpisodeRangeReader and assert the values match.

Video layout difference:
  LeRobotDataset  → CHW float32 [0, 1]
  EpisodeRangeReader → HWC uint8 [0, 255]
Both use PyAV, so after converting to a common format the values should agree
within ±1 (codec rounding on seek boundaries).
"""

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_datasource import EpisodeRangeReader, LeRobotDatasourceMetadata, ParallelismMode

DATASET_REPO = "lerobot/libero"
N_EPISODES = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lerobot_ds():
    return LeRobotDataset(DATASET_REPO, video_backend="pyav")


@pytest.fixture(scope="module")
def meta(lerobot_ds):
    return LeRobotDatasourceMetadata(str(lerobot_ds.root))


@pytest.fixture(scope="module")
def lerobot_rows(lerobot_ds):
    """All frames from the first N_EPISODES via LeRobotDataset, in order."""
    rows = []
    for i in range(len(lerobot_ds)):
        row = lerobot_ds[i]
        if row["episode_index"] >= N_EPISODES:
            break
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def datasource_rows(meta):
    """All frames from the first N_EPISODES via EpisodeRangeReader, in order."""
    return list(EpisodeRangeReader(meta, 0, N_EPISODES))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ds_video_frame(row, vid_key, shape):
    """Extract a HWC uint8 numpy array from a datasource row for one camera."""
    # storage is a FixedSizeListArray wrapping the flat pixel buffer
    storage = row.column(vid_key).chunk(0).storage
    return np.array(storage[0].as_py(), dtype=np.uint8).reshape(shape)


def lr_video_frame(row, vid_key):
    """Convert a LeRobot CHW float32 frame to HWC uint8."""
    return (row[vid_key].permute(1, 2, 0) * 255).to(torch.uint8).numpy()


# ---------------------------------------------------------------------------
# Scalar column tests
# ---------------------------------------------------------------------------


def test_row_count(lerobot_rows, datasource_rows):
    assert len(lerobot_rows) == len(datasource_rows)


def test_episode_index(lerobot_rows, datasource_rows):
    for lr, ds in zip(lerobot_rows, datasource_rows):
        assert lr["episode_index"] == ds.column("episode_index")[0].as_py()


def test_frame_index(lerobot_rows, datasource_rows):
    for lr, ds in zip(lerobot_rows, datasource_rows):
        assert lr["frame_index"] == ds.column("frame_index")[0].as_py()


def test_timestamp(lerobot_rows, datasource_rows):
    for lr, ds in zip(lerobot_rows, datasource_rows):
        assert abs(lr["timestamp"] - ds.column("timestamp")[0].as_py()) < 1e-6


def test_action(lerobot_rows, datasource_rows):
    for lr, ds in zip(lerobot_rows, datasource_rows):
        np.testing.assert_allclose(
            lr["action"].numpy(),
            np.array(ds.column("action")[0].as_py()),
            rtol=1e-5,
        )


def test_observation_state(lerobot_rows, datasource_rows):
    for lr, ds in zip(lerobot_rows, datasource_rows):
        np.testing.assert_allclose(
            lr["observation.state"].numpy(),
            np.array(ds.column("observation.state")[0].as_py()),
            rtol=1e-5,
        )


def test_task(lerobot_rows, datasource_rows, meta):
    for lr, ds in zip(lerobot_rows, datasource_rows):
        assert lr["task"] == ds.column("task")[0].as_py()


# ---------------------------------------------------------------------------
# Video frame tests
# ---------------------------------------------------------------------------


def test_video_frame_dtype_and_shape(datasource_rows, meta):
    """Datasource frames must be HWC uint8."""
    row = datasource_rows[0]
    for vid_key in meta.video_keys:
        shape = meta.info["features"][vid_key]["shape"]  # [H, W, C]
        frame = ds_video_frame(row, vid_key, shape)
        assert frame.shape == tuple(shape)
        assert frame.dtype == np.uint8


def test_video_frames_match_lerobot(lerobot_rows, datasource_rows, meta):
    """Pixel values must agree within ±1 after converting both to HWC uint8."""
    for vid_key in meta.video_keys:
        shape = meta.info["features"][vid_key]["shape"]
        for lr, ds in zip(lerobot_rows, datasource_rows):
            lr_frame = lr_video_frame(lr, vid_key)
            ds_frame = ds_video_frame(ds, vid_key, shape)
            diff = np.abs(lr_frame.astype(np.int16) - ds_frame.astype(np.int16))
            assert diff.max() <= 1, (
                f"{vid_key} ep={lr['episode_index']} fi={lr['frame_index']}: "
                f"max pixel diff {diff.max()} > 1"
            )


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


def test_stats_keys_match_lerobot(lerobot_ds, meta):
    """meta.stats must contain the same feature keys as ds.meta.stats."""
    assert set(meta.stats.keys()) == set(lerobot_ds.meta.stats.keys())


def test_stats_stat_names_match_lerobot(lerobot_ds, meta):
    """Each feature in meta.stats must have the same stat names as LeRobot."""
    for feat in meta.stats:
        assert set(meta.stats[feat].keys()) == set(lerobot_ds.meta.stats[feat].keys()), (
            f"{feat}: stat name mismatch"
        )


def test_stats_values_match_lerobot(lerobot_ds, meta):
    """Numeric stat values must match LeRobot's parsed arrays within float32 precision."""
    for feat in meta.stats:
        for stat_name, lr_val in lerobot_ds.meta.stats[feat].items():
            ds_val = np.array(meta.stats[feat][stat_name])
            np.testing.assert_allclose(
                ds_val, lr_val, rtol=1e-6,
                err_msg=f"{feat}.{stat_name}: value mismatch",
            )


# ---------------------------------------------------------------------------
# Parallelism mode consistency
# ---------------------------------------------------------------------------


def test_all_modes_same_scalars(meta):
    """All four parallelism modes must produce identical scalar columns."""
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from lerobot_datasource import LeRobotDatasource

    def collect(mode):
        ds = ray.data.read_datasource(LeRobotDatasource(root=meta.root, parallelism_mode=mode))
        rows = []
        for row in ds.iter_rows():
            if row["episode_index"] >= N_EPISODES:
                break
            rows.append(row)
        return sorted(rows, key=lambda r: (r["episode_index"], r["frame_index"]))

    reference = collect(ParallelismMode.SEQUENTIAL)
    for mode in (ParallelismMode.EPISODE, ParallelismMode.FILE_GROUP, ParallelismMode.CHAIN):
        rows = collect(mode)
        assert len(rows) == len(reference), f"{mode}: row count mismatch"
        for ref, row in zip(reference, rows):
            assert ref["episode_index"] == row["episode_index"]
            assert ref["frame_index"] == row["frame_index"]
            assert abs(ref["timestamp"] - row["timestamp"]) < 1e-6, f"{mode}: timestamp mismatch"
