import ray
import numpy as np
import os
from huggingface_hub import HfFileSystem
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops import load_ops

NUM_ROWS = 10**7

output_path = f"/mnt/shared_storage/fineweb_processed/{np.random.randint(1000000)}"

# Load FineWeb-edu data from HF with rate limiting to avoid bandwidth quota errors
ds = (
    ray.data.read_parquet(
        "hf://datasets/HuggingFaceFW/fineweb-edu/data/",
        file_extensions=["parquet"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        concurrency=20,
    )
    .limit(NUM_ROWS)
    .repartition(target_num_rows_per_block=500)
)

# Wrap with RayDataset
dj_dataset = RayDataset(ds)

# Define processing pipeline
process_ops = [
    # Text Cleaning
    {"clean_html_mapper": {}},
    {"clean_links_mapper": {}},
    {"clean_email_mapper": {}},
    {"whitespace_normalization_mapper": {}},
    {"fix_unicode_mapper": {}},
    # Text Filtering
    {"text_length_filter": {"min_len": 50, "max_len": 10000}},
    {"alphanumeric_filter": {"min_ratio": 0.6, "max_ratio": 0.95}},
    {"special_characters_filter": {"min_ratio": 0.0, "max_ratio": 0.25}},
    {"character_repetition_filter": {"rep_len": 10, "max_ratio": 0.5}},
    # Deduplication (Ray-based, supports GPU)
    {
        "ray_bts_minhash_deduplicator": {
            "tokenization": "character",
            "window_size": 5,
            "jaccard_threshold": 0.7,
            "num_permutations": 256,
            "union_find_parallel_num": 200,  # --union_find_parallel_num=200
            "work_dir": "/mnt/cluster_storage/tmp",
            "actor_memory": 20_000_000_000,  # 20GB per BTSUnionFind/EdgeBuffer actor
            "task_memory": 2_000_000_000,  # 2GB per map_batches task
            "union_threshold": 256,
            "max_pending_edge_buffer_task": 40,
            "num_edge_buffer_task_returns": 20,
            "max_pending_filter_tasks": 40,
            "num_filter_task_returns": 20,
        }
    },
]

ops = load_ops(process_ops)

# Process data
processed = dj_dataset.process(ops)

# Export results to local cluster path
processed.data.write_parquet(output_path)
print(f"Processing complete. Output written to {output_path}")
print(f"Final sample count: {processed.count()}")

# Load the output and print the first 10 rows.
output_ds = ray.data.read_parquet(output_path)
print("First 10 rows of the output:")
print(output_ds.take(10))