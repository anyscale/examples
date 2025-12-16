import ray
import gcsfs
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops import load_ops
from datetime import datetime, timezone

def main():
    # Initialize Ray (auto-connect on Anyscale)
    ray.init(address="auto")

    # Load FineWeb data from GCS
    filesystem = gcsfs.GCSFileSystem(project="test-vertex")
    ds = (
        ray.data.read_parquet(
            "gs://anyscale-example-datasets/HuggingFaceFW/fineweb-edu/data/",
            filesystem=filesystem,
        )
        .limit(30000)
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
        # Actors are now created lazily in run() after cluster autoscales
        {"ray_document_deduplicator": {}},
        {
            "ray_bts_minhash_deduplicator": {
                "tokenization": "character",
                "jaccard_threshold": 0.7,
                "num_permutations": 256,
                "work_dir": "/mnt/cluster_storage/tmp",
            }
        },
    ]

    ops = load_ops(process_ops)

    # Process data
    processed = dj_dataset.process(ops)

    # Export results to local cluster path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    processed.data.write_parquet(f"/mnt/cluster_storage/fineweb_processed/{timestamp}")

    print(f"Processing complete. Output written to /mnt/cluster_storage/fineweb_processed/{timestamp}")
    print(f"Final sample count: {processed.count()}")


if __name__ == "__main__":
    main()
