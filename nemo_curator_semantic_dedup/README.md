# Image Semantic Deduplication with NeMo Curator

This example uses [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) to perform GPU-accelerated semantic deduplication on image datasets. It reads image URLs from a HuggingFace parquet dataset, downloads them into [WebDataset](https://github.com/webdataset/webdataset) tar shards, generates CLIP embeddings on GPUs, clusters them with K-means + DBSCAN to find near-duplicates, and writes a clean deduplicated dataset.

## Install the Anyscale CLI (required version = 0.26.82+)

```bash
pip install -U anyscale
anyscale login
```

## Run the job

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/nemo_curator_semantic_dedup
```

Submit the job. Use `--env` to forward your HuggingFace token for dataset access.

```bash
anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN
```

## How it works

The entrypoint defined in [job.yaml](./job.yaml) runs [image_dedup_example.py](./image_dedup_example.py), which executes four steps:

```
HuggingFace parquet (URLs + captions)
    │
    ▼
Step 1: Download images → WebDataset tar shards
    │
    ▼
Step 2: Generate CLIP embeddings (GPU)
    │
    ▼
Step 3: Semantic deduplication (K-means + DBSCAN)
    │
    ▼
Step 4: Write deduplicated dataset (new tar shards without duplicates)
```

### Step 1: Parquet to WebDataset

`main()` calls `parquet_to_webdataset_ray()` in [helper.py](./helper.py). Ray Data reads parquet columns (`url`, `caption`) from HuggingFace, distributes HTTP downloads across the cluster with `ThreadPoolExecutor` parallelism within each Ray task, validates and converts each image to RGB JPEG with Pillow, and packs them into WebDataset `.tar` shards on cluster storage. Each shard holds up to `ENTRIES_PER_TAR` images (default 500).

### Step 2: Image embedding pipeline

`create_image_embedding_pipeline()` builds a NeMo Curator `Pipeline` with five stages:

| Stage | What it does |
|-------|-------------|
| `FilePartitioningStage` | Discovers `.tar` files and groups them into partitions (`tar_files_per_partition=1`). Not a bottleneck at this scale — GPU throughput on the embedding stage is. |
| `ImageReaderStage` | Reads images from tar shards. DALI decodes JPEGs on the GPU. |
| `ImageEmbeddingStage` | Runs OpenAI CLIP ViT-L/14 to produce 768-dimensional embeddings for each image. |
| `ConvertImageBatchToDocumentBatchStage` | Converts from `ImageBatch` (numpy pixel arrays) to `DocumentBatch` (DataFrames), keeping only `image_id` and `embedding`. |
| `ParquetWriter` | Saves embeddings to parquet files on cluster storage. |

This pipeline runs on a `RayDataExecutor`, which treats the stages as a streaming dataflow — data flows through one batch at a time in a producer-consumer pattern. It's well suited for stages that process data independently without coordination between workers.

### Step 3: Semantic deduplication workflow

`SemanticDeduplicationWorkflow` is where the actual deduplication happens:

1. **K-means clustering** (`n_clusters=100`) — groups similar embeddings together using RAFT/NCCL across all GPU actors.
2. **DBSCAN within clusters** (`eps=0.01`) — finds near-duplicate pairs based on cosine distance within each cluster.

This step runs on a `RayActorPoolExecutor`, which creates a fixed pool of long-lived Ray actors, each holding a GPU. This is needed because K-means requires state across batches — all GPU actors must coordinate via RAFT/NCCL to fit a single model across the full dataset. A stateless streaming executor can't do that.

### Step 4: Write deduplicated dataset

`create_image_deduplication_pipeline()` re-reads the original tar shards, filters out images whose IDs appear in the removal list, and writes new tar shards. You can't selectively remove files from a tar archive — tar is a flat concatenation of file records with no editable index — so writing new shards without the duplicates is the only option. `ImageWriterStage` re-encodes each image's numpy pixel array back to JPEG with Pillow, packs them into new `.tar` files (up to 1000 images per tar), and writes a companion `.parquet` with metadata (`image_id`, `tar_file`, `member_name`, `original_path`).

## Cluster storage

All intermediate and output data lives under `/mnt/cluster_storage/`, a shared network filesystem (backed by S3) that is automatically mounted on every node in the cluster.

| Step | Directory | Contents |
|------|-----------|----------|
| 1 | `webdataset/` | WebDataset `.tar` files (downloaded images) |
| 2 | `embeddings/` | Parquet files with `image_id` + embedding vectors |
| 3 | `removal_ids/` | K-means results, pairwise results, and `duplicates/` parquet listing IDs to remove |
| 4 | `results/` | Final deduplicated WebDataset `.tar` files |

## Configuration

All configuration is done through environment variables in [job.yaml](./job.yaml). Override them at submit time:

```bash
anyscale job submit -f job.yaml \
  --env INPUT_PARQUET=hf://datasets/your-org/your-dataset/ \
  --env MAX_ENTRIES=10000 \
  --env HF_TOKEN=$HF_TOKEN
```

The [Dockerfile](./Dockerfile) builds a custom image with NeMo Curator's CUDA dependencies (`nemo-curator[image_cuda12]`), including DALI, cuML, and RAFT. The NeMo Curator Python package itself is overridden at runtime via `PYTHONPATH=Curator` in job.yaml, which points to a local `Curator/` directory uploaded with the working directory.

## View the job

View the job in the [jobs tab](https://console.anyscale.com/jobs) of the Anyscale console.

## Learn more

- [NeMo Curator Documentation](https://docs.nvidia.com/nemo/curator/latest/)
- [NeMo Curator Image Tutorials](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/image/getting-started)
- [Anyscale Jobs Documentation](https://docs.anyscale.com/platform/jobs/)
