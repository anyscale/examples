# Semantic Deduplication with NeMo Curator

This example demonstrates how to perform GPU-accelerated semantic deduplication on large text datasets using [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) on Anyscale.

## What is Semantic Deduplication?

Unlike exact or fuzzy deduplication that matches text patterns, **semantic deduplication** identifies documents that are conceptually similar even if they use different words. This is achieved by:

1. **Computing embeddings**: Converting text into dense vector representations using neural network models
2. **Clustering**: Grouping similar embeddings using GPU-accelerated k-means
3. **Similarity matching**: Identifying near-duplicates within clusters based on cosine similarity

This approach is particularly effective for:
- Removing paraphrased content
- Identifying translated duplicates
- Cleaning datasets with rephrased information
- Improving LLM training data quality

## Performance

NeMo Curator leverages NVIDIA RAPIDS™ libraries (cuDF, cuML, cuGraph) for GPU acceleration:

- **16× faster** fuzzy deduplication compared to CPU-based alternatives
- **40% lower** total cost of ownership (TCO)
- **Near-linear scaling** across multiple GPUs

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Submit the Job

Clone the example from GitHub:

```bash
git clone https://github.com/anyscale/examples.git
cd examples/nemo_curator_semantic_dedup
```

Submit the job:

```bash
anyscale job submit -f job.yaml
```

### Using Your Own Data

To process your own dataset, set the `INPUT_DATA_PATH` environment variable:

```bash
anyscale job submit -f job.yaml \
  --env INPUT_DATA_PATH=s3://your-bucket/your-data/ \
  --env OUTPUT_DATA_PATH=s3://your-bucket/output/
```

Your input data should be in Parquet or JSONL format with at least two columns:
- `id`: Unique document identifier
- `text`: The text content to deduplicate

## Configuration Options

You can customize the pipeline behavior via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DATA_PATH` | `/mnt/cluster_storage/semantic_dedup/input` | Path to input dataset |
| `OUTPUT_DATA_PATH` | `/mnt/cluster_storage/semantic_dedup/output` | Path for output data |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model for embeddings |
| `EMBEDDING_BATCH_SIZE` | `64` | Batch size per GPU for embedding computation |
| `NUM_CLUSTERS` | `1000` | Number of k-means clusters |
| `SIMILARITY_THRESHOLD` | `0.8` | Cosine similarity threshold (0.0-1.0) |

### Embedding Model Options

| Model | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | Good | Fast | Quick experiments, large datasets |
| `intfloat/e5-large-v2` | Better | Medium | Production workloads |
| `BAAI/bge-large-en-v1.5` | Best | Slower | High-quality deduplication |

### Tuning the Similarity Threshold

- **Higher threshold (e.g., 0.9)**: Stricter matching, fewer duplicates removed
- **Lower threshold (e.g., 0.7)**: Looser matching, more duplicates removed

Start with 0.8 and adjust based on your quality requirements.

## Understanding the Example

### Pipeline Architecture

```
┌─────────────────┐
│   Input Data    │  Parquet/JSONL files
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Embedding    │  GPU-accelerated transformer inference
│    Creation     │  (sentence-transformers, E5, BGE, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Clustering    │  GPU-accelerated k-means (cuML)
│    (k-means)    │  Groups similar embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Duplicate     │  Pairwise similarity within clusters
│   Extraction    │  Identifies semantic duplicates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deduplicated   │  Original data minus duplicates
│     Output      │
└─────────────────┘
```

### Key Components

- **EmbeddingCreator**: Computes dense vector embeddings using pre-trained models
- **ClusteringModel**: GPU-accelerated k-means clustering with cuML
- **SemanticClusterLevelDedup**: Finds duplicates within clusters using cosine similarity

### Scaling Considerations

- **Number of clusters**: Should be roughly √(n_documents) for balanced cluster sizes
- **Memory**: Each GPU should have enough memory for the embedding model (~4GB for MiniLM, ~8GB for larger models)
- **Batch size**: Increase for faster processing, decrease if running out of GPU memory

## Output

The pipeline produces:

1. **Deduplicated dataset**: Parquet files in `{OUTPUT_DATA_PATH}/{timestamp}/deduplicated/`
2. **Cache files**: Intermediate embeddings and clusters for debugging

Example output log:

```
============================================================
Semantic Deduplication Complete!
============================================================
Original documents: 1,000,000
Duplicates removed: 127,543
Final documents: 872,457
Reduction: 12.75%
Output saved to: /mnt/cluster_storage/semantic_dedup/output/20250115T143022Z/deduplicated
============================================================
```

## Monitoring

View your job progress in the [Anyscale Console](https://console.anyscale.com/jobs). The Dask dashboard link will be printed in the logs for detailed task monitoring.

## Cost Optimization Tips

1. **Use spot instances**: The job.yaml is configured with `market_type: PREFER_SPOT` for cost savings
2. **Start small**: Test with a subset of your data before running on the full dataset
3. **Choose the right GPU**: A10G instances offer a good balance of cost and performance
4. **Tune batch size**: Larger batches = better GPU utilization = faster processing

## Learn More

- [NeMo Curator Documentation](https://docs.nvidia.com/nemo/curator/latest/)
- [Semantic Deduplication Guide](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/index.html)
- [NeMo Curator GitHub](https://github.com/NVIDIA-NeMo/Curator)
- [Anyscale Documentation](https://docs.anyscale.com/)

