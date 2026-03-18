"""Print schema + first 3 rows, then verify parity against lerobot's LeRobotDataset."""

import pathlib

import numpy as np
import ray
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_datasource import GroupingMode, read_lerobot

DATASET_ROOT = pathlib.Path.home() / ".cache/huggingface/lerobot/lerobot/berkeley_autolab_ur5"
REPO_ID = "lerobot/berkeley_autolab_ur5"
N = 3000


def summarise(val):
    if isinstance(val, np.ndarray):
        return f"ndarray{list(val.shape)} dtype={val.dtype}"
    if isinstance(val, list):
        return f"list[{len(val)}] = {val[:4]}{'...' if len(val) > 4 else ''}"
    return repr(val)


def main():
    root = str(DATASET_ROOT)
    ray.init(ignore_reinit_error=True)

    ds, ds_meta = read_lerobot(root, grouping_mode=GroupingMode.FILE_GROUP)
    ref = LeRobotDataset(REPO_ID, root=root, video_backend="pyav")

    print("=== Schema ===")
    schema = ds.schema()
    assert schema is not None
    for name, dtype in zip(schema.names, schema.types):
        print(f"  {name:<45} {dtype}")
    print()

    print(f"=== First {N} rows ===")
    our_rows = list(ds.sort("index").limit(N).iter_rows())
    for i, row in enumerate(our_rows):
        print(f"--- row {i} ---")
        for col, val in row.items():
            print(f"  {col:<45} {summarise(val)}")
        print()

    print(f"=== Parity vs lerobot (pyav, first {N} rows) ===")
    scalar_fields = ["index", "episode_index", "frame_index", "timestamp",
                     "next.done", "next.reward", "task_index"]

    for ours in our_rows:
        global_idx = ours["index"]
        ref_item = ref[global_idx]
        ok = True

        for field in scalar_fields:
            if field not in ref_item:
                continue
            our_val = ours[field]
            ref_val = ref_item[field]
            if hasattr(ref_val, "item"):
                ref_val = ref_val.item()
            if our_val != ref_val:
                print(f"  index {global_idx} [{field}] FAIL  ours={our_val!r}  ref={ref_val!r}")
                ok = False

        for field in ["observation.state", "action"]:
            our_arr = np.asarray(ours[field], dtype=np.float32)
            ref_arr = ref_item[field].numpy().flatten()
            if not np.allclose(our_arr, ref_arr, rtol=1e-5):
                print(f"  index {global_idx} [{field}] FAIL  max_diff={np.abs(our_arr - ref_arr).max():.2e}")
                ok = False

        for vid_key in ds_meta.video_keys:
            # ours: HWC uint8 [0,255]  →  CHW float32 [0,1]
            our_frame = np.asarray(ours[vid_key], dtype=np.float32) / 255.0
            our_chw = our_frame.transpose(2, 0, 1)
            ref_chw = ref_item[vid_key].numpy()
            max_diff = np.abs(our_chw - ref_chw).max()
            if max_diff > 0.05:  # ~13/255 — accounts for pyav seek rounding
                print(f"  index {global_idx} [{vid_key}] FAIL  max_diff={max_diff:.4f}")
                ok = False

        if ok:
            print(f"  index {global_idx}  OK")

    print()


if __name__ == "__main__":
    main()
