"""Tests verifying that read_lerobot produces output identical to lerobot's LeRobotDataset.

Comparison notes
----------------
LeRobot's ``__getitem__`` returns video frames as CHW float32 tensors in [0, 1].
The datasource stores and returns HWC uint8 numpy arrays in [0, 255].

Conversion: lerobot CHW float32 → HWC uint8::

    hwc = (chw_tensor.permute(1, 2, 0) * 255).byte().numpy()

Video shapes in ``meta/info.json`` are stored as [H, W, C] (HWC order), matching
what the datasource writes.  Frames are compared with ±1 pixel tolerance to
account for codec rounding during seek-and-walk vs direct random access.

Running the tests
-----------------
On a single machine, datasets are cached under HF_HOME_DIR / huggingface / lerobot.
On a multi-node Ray cluster, point HF_HOME_DIR at a shared filesystem so all
worker nodes can read the same files.  Defaults to /mnt/cluster_storage/.cache.
Run with::

    uv run pytest test_datasource.py -v

    # Custom cache root:
    HF_HOME_DIR=/my/shared/cache uv run pytest test_datasource.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import ray
import torch

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_EPISODES = 3  # number of episodes to test (keeps runtime short)

DATASETS = [
    "lerobot/pusht",
    "lerobot/aloha_sim_insertion_human",
    "lerobot/xarm_lift_medium_replay",
]

# Root directory used for HuggingFace caching.  Must be readable by all Ray
# worker nodes.  Override via the HF_HOME_DIR environment variable or by
# editing this constant directly.
HF_HOME_DIR: str = os.environ.get("HF_HOME_DIR", "/mnt/cluster_storage/.cache")

# Apply immediately so that lerobot / huggingface_hub pick it up before any
# dataset is downloaded or loaded.
os.environ["HF_HOME"] = HF_HOME_DIR

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from lerobot_datasource import (  # noqa: E402
    LeRobotDatasource,
    LeRobotDatasourceMetadata,
    Partitioning,
    read_lerobot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lerobot_frame_to_hwc_uint8(arr) -> np.ndarray:
    """CHW float32 [0,1] (torch.Tensor or numpy) → HWC uint8 [0,255]."""
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    # arr is (C, H, W) float32
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def ray_frame_to_hwc_uint8(row: dict, vid_key: str, shape: tuple) -> np.ndarray:
    """Extract HWC uint8 frame from a Ray Data row dict.

    ``shape`` is the [H, W, C] shape from ``meta.info["features"][vid_key]["shape"]``.
    The value stored in the row is already HWC uint8; reshape is a correctness guard.
    """
    return np.asarray(row[vid_key], dtype=np.uint8).reshape(shape)


def collect_ray_rows(ds: "ray.data.Dataset", n_episodes: int) -> list[dict]:
    """Collect and sort rows from a Ray dataset, limited to the first n_episodes."""
    rows = [r for r in ds.iter_rows() if r["episode_index"] < n_episodes]
    rows.sort(key=lambda r: (r["episode_index"], r["frame_index"]))
    return rows


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(scope="session", params=DATASETS)
def lerobot_ds(request) -> LeRobotDataset:
    """Download and return a LeRobotDataset for one repo_id (cached on disk)."""
    return LeRobotDataset(request.param, video_backend="pyav")


@pytest.fixture(scope="session")
def meta(lerobot_ds) -> LeRobotDatasourceMetadata:
    """Datasource metadata built from the lerobot dataset's local HF cache root."""
    return LeRobotDatasourceMetadata(str(lerobot_ds.root))


@pytest.fixture(scope="session")
def lerobot_rows(lerobot_ds) -> list[dict]:
    """All rows for the first N_EPISODES from LeRobotDataset, converted to numpy/python.

    Iterates sequentially and stops at the first row belonging to episode N_EPISODES
    (relies on the dataset being sorted by episode_index, which lerobot guarantees).
    Sorted by (episode_index, frame_index) for deterministic comparison.
    """
    rows: list[dict] = []
    for i in range(len(lerobot_ds)):
        item = lerobot_ds[i]
        ep_idx = item["episode_index"].item()
        if ep_idx >= N_EPISODES:
            break
        row: dict = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                row[k] = v.item() if v.ndim == 0 else v.numpy()
            else:
                row[k] = v
        rows.append(row)
    rows.sort(key=lambda r: (r["episode_index"], r["frame_index"]))
    return rows


@pytest.fixture(scope="session")
def ray_rows(meta) -> list[dict]:
    """All rows for the first N_EPISODES via read_lerobot (SEQUENTIAL mode).

    Sorted by (episode_index, frame_index) for deterministic comparison.
    """
    ds = read_lerobot(meta.root, partitioning=Partitioning.SEQUENTIAL)
    return collect_ray_rows(ds, N_EPISODES)


# ---------------------------------------------------------------------------
# Test 1: Schema parity
# ---------------------------------------------------------------------------


def test_schema_columns(lerobot_rows, ray_rows, meta):
    """Column name sets must match between lerobot and Ray Data output."""
    lerobot_keys = set(lerobot_rows[0].keys())
    ray_keys = set(ray_rows[0].keys())

    # Video keys must be present in both.
    for vid_key in meta.video_keys:
        assert vid_key in lerobot_keys, f"Video key {vid_key!r} missing from lerobot"
        assert vid_key in ray_keys, f"Video key {vid_key!r} missing from Ray Data"

    # Non-video column names must match exactly (excluding datasource-only columns).
    _datasource_only = {"dataset_index"}
    lerobot_non_video = lerobot_keys - set(meta.video_keys)
    ray_non_video = ray_keys - set(meta.video_keys) - _datasource_only
    assert lerobot_non_video == ray_non_video, (
        f"Non-video column mismatch:\n"
        f"  lerobot only: {lerobot_non_video - ray_non_video}\n"
        f"  ray only:     {ray_non_video - lerobot_non_video}"
    )


# ---------------------------------------------------------------------------
# Test 2: Row count
# ---------------------------------------------------------------------------


def test_row_count(lerobot_rows, ray_rows):
    """Row count must match for the first N_EPISODES."""
    assert len(lerobot_rows) == len(ray_rows), (
        f"Row count mismatch: lerobot={len(lerobot_rows)}, ray={len(ray_rows)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Scalar column parity
# ---------------------------------------------------------------------------


def test_scalar_columns(lerobot_rows, ray_rows):
    """Integer indices must match exactly; timestamp within 1e-6; task strings equal."""
    for i, (lr, rr) in enumerate(zip(lerobot_rows, ray_rows)):
        for col in ("index", "episode_index", "frame_index", "task_index"):
            assert int(lr[col]) == int(rr[col]), (
                f"Row {i}: {col} mismatch: lerobot={lr[col]}, ray={rr[col]}"
            )
        assert abs(float(lr["timestamp"]) - float(rr["timestamp"])) < 1e-6, (
            f"Row {i}: timestamp mismatch: lerobot={lr['timestamp']}, ray={rr['timestamp']}"
        )
        assert lr["task"] == rr["task"], (
            f"Row {i}: task mismatch: lerobot={lr['task']!r}, ray={rr['task']!r}"
        )


# ---------------------------------------------------------------------------
# Test 4: Vector column parity
# ---------------------------------------------------------------------------


def test_vector_columns(meta, lerobot_rows, ray_rows):
    """Float vector features (action, state, …) must match within floating-point tolerance."""
    features = meta.info.get("features", {})
    # Pick float features whose total element count > 1 (i.e. not scalar).
    vector_keys = [
        k for k, v in features.items()
        if v.get("dtype") not in ("video", "bool")
        and "float" in v.get("dtype", "")
        and np.prod(v.get("shape", [1])) > 1
    ]
    if not vector_keys:
        pytest.skip("No float vector features in this dataset")

    for i, (lr, rr) in enumerate(zip(lerobot_rows, ray_rows)):
        for k in vector_keys:
            if k not in lr or k not in rr:
                continue
            lr_arr = np.asarray(lr[k], dtype=np.float32).flatten()
            rr_arr = np.asarray(rr[k], dtype=np.float32).flatten()
            np.testing.assert_allclose(
                lr_arr, rr_arr, rtol=1e-5,
                err_msg=f"Row {i}, column {k!r}: mismatch",
            )


# ---------------------------------------------------------------------------
# Test 5: Video frame parity
# ---------------------------------------------------------------------------


def test_video_frames(meta, lerobot_rows, ray_rows):
    """Decoded video frames must match within ±1 pixel value (HWC uint8)."""
    if not meta.video_keys:
        pytest.skip("No video keys in this dataset")

    for i, (lr, rr) in enumerate(zip(lerobot_rows, ray_rows)):
        for vid_key in meta.video_keys:
            shape = tuple(meta.info["features"][vid_key]["shape"])  # (H, W, C)
            lr_frame = lerobot_frame_to_hwc_uint8(lr[vid_key])
            rr_frame = ray_frame_to_hwc_uint8(rr, vid_key, shape)
            diff = np.abs(lr_frame.astype(np.int32) - rr_frame.astype(np.int32))
            assert diff.max() <= 1, (
                f"Row {i}, camera {vid_key!r}: max pixel diff={diff.max()} > 1"
            )


# ---------------------------------------------------------------------------
# Test 6: All partitionings produce identical scalar rows
# ---------------------------------------------------------------------------


def test_all_modes_same_output(meta):
    """All partitionings must produce identical rows for the first N_EPISODES.

    Video frames are not compared here (covered by test_video_frames for SEQUENTIAL).
    """
    root = meta.root
    scalar_cols = ("index", "episode_index", "frame_index", "task_index")

    def rows_for_mode(mode, **kwargs):
        ds = ray.data.read_datasource(
            LeRobotDatasource(root, partitioning=mode, **kwargs)
        )
        return collect_ray_rows(ds, N_EPISODES)

    reference = rows_for_mode(Partitioning.SEQUENTIAL)

    modes_under_test = [
        ("EPISODE",    lambda: rows_for_mode(Partitioning.EPISODE)),
        ("FILE_GROUP", lambda: rows_for_mode(Partitioning.FILE_GROUP)),
        ("CHAIN",      lambda: rows_for_mode(Partitioning.CHAIN)),
        ("ROW_BLOCK",  lambda: rows_for_mode(Partitioning.ROW_BLOCK, block_size=256)),
    ]

    for label, get_rows in modes_under_test:
        rows = get_rows()
        assert len(rows) == len(reference), (
            f"Mode {label}: row count {len(rows)} != reference {len(reference)}"
        )
        for i, (ref, row) in enumerate(zip(reference, rows)):
            for col in scalar_cols:
                assert int(ref[col]) == int(row[col]), (
                    f"Mode {label}, row {i}, col {col!r}: {row[col]} != {ref[col]}"
                )
            assert abs(float(ref["timestamp"]) - float(row["timestamp"])) < 1e-6, (
                f"Mode {label}, row {i}: timestamp mismatch"
            )


# ---------------------------------------------------------------------------
# Test 7: Stats parity
# ---------------------------------------------------------------------------


def test_stats(lerobot_ds, meta):
    """Stats keys and numeric values must match between lerobot and the datasource."""
    lr_stats = lerobot_ds.meta.stats  # dict[str, dict[str, np.ndarray]] | None
    our_stats = meta.stats            # dict[str, dict] — raw JSON values

    if lr_stats is None:
        pytest.skip("This dataset has no stats")

    assert set(lr_stats.keys()) == set(our_stats.keys()), (
        f"Stats key mismatch:\n"
        f"  lerobot only: {set(lr_stats.keys()) - set(our_stats.keys())}\n"
        f"  ours only:    {set(our_stats.keys()) - set(lr_stats.keys())}"
    )

    for feat_key in lr_stats:
        lr_feat = lr_stats[feat_key]
        our_feat = our_stats[feat_key]
        assert set(lr_feat.keys()) == set(our_feat.keys()), (
            f"Stats sub-key mismatch for {feat_key!r}: "
            f"lerobot={set(lr_feat.keys())}, ours={set(our_feat.keys())}"
        )
        for stat_key in lr_feat:
            lr_val = np.asarray(lr_feat[stat_key], dtype=np.float64).flatten()
            our_val = np.asarray(our_feat[stat_key], dtype=np.float64).flatten()
            np.testing.assert_allclose(
                lr_val, our_val, rtol=1e-6,
                err_msg=f"Stats mismatch for {feat_key!r}/{stat_key!r}",
            )


# ---------------------------------------------------------------------------
# Multi-root tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def two_roots(lerobot_ds) -> tuple[str, str]:
    """Return the same dataset root twice — cheapest way to test multi-root without two downloads."""
    root = str(lerobot_ds.root)
    return root, root


def test_multi_root_dataset_index_column(two_roots):
    """dataset_index column must be present and contain only 0 and 1."""
    root_a, root_b = two_roots
    ds = read_lerobot([root_a, root_b], partitioning=Partitioning.SEQUENTIAL)
    rows = list(ds.iter_rows())
    assert len(rows) > 0, "Expected non-empty dataset"
    indices = {r["dataset_index"] for r in rows}
    assert indices == {0, 1}, f"Expected {{0, 1}}, got {indices}"


def test_multi_root_row_count(two_roots):
    """Reading two identical roots must yield exactly twice as many rows as one root."""
    root_a, root_b = two_roots
    single = read_lerobot(root_a, partitioning=Partitioning.SEQUENTIAL)
    multi = read_lerobot([root_a, root_b], partitioning=Partitioning.SEQUENTIAL)
    assert multi.count() == 2 * single.count()


def test_multi_root_episode_index_not_remapped(two_roots):
    """episode_index values must be per-root local (not globally remapped)."""
    root_a, root_b = two_roots
    ds = read_lerobot([root_a, root_b], partitioning=Partitioning.SEQUENTIAL)
    rows_by_ds: dict[int, list] = {0: [], 1: []}
    for r in ds.iter_rows():
        rows_by_ds[r["dataset_index"]].append(r["episode_index"])
    # Both roots are the same dataset, so their episode_index ranges must be equal.
    assert sorted(set(rows_by_ds[0])) == sorted(set(rows_by_ds[1])), (
        "episode_index ranges differ between roots — possible unintended remapping"
    )


def test_multi_root_single_string_unchanged(two_roots):
    """Single-string API must still work and always produce dataset_index == 0."""
    root_a, _ = two_roots
    ds = read_lerobot(root_a, partitioning=Partitioning.SEQUENTIAL)
    rows = list(ds.iter_rows())
    assert all(r["dataset_index"] == 0 for r in rows), (
        "Single-root read produced dataset_index != 0"
    )


def test_multi_root_incompatible_fps_raises():
    """LeRobotDatasource must raise ValueError when roots have different fps."""
    # We can't easily manufacture a real dataset with different fps,
    # so we test that the validation path is reachable by mocking.
    from unittest.mock import MagicMock, patch

    mock_meta_a = MagicMock()
    mock_meta_a.video_keys = ["observation.image"]
    mock_meta_a.info = {"fps": 10, "features": {}}
    mock_meta_a.root = "/fake/ds_a"

    mock_meta_b = MagicMock()
    mock_meta_b.video_keys = ["observation.image"]
    mock_meta_b.info = {"fps": 30, "features": {}}
    mock_meta_b.root = "/fake/ds_b"

    with patch(
        "lerobot_datasource.LeRobotDatasourceMetadata",
        side_effect=[mock_meta_a, mock_meta_b],
    ):
        with pytest.raises(ValueError, match="fps mismatch"):
            LeRobotDatasource(["/fake/ds_a", "/fake/ds_b"])
