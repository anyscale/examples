# Large-Scale Image Processing with Vision Language Models

This example demonstrates how to build a production-ready image processing pipeline that scales to billions of images using Ray Data and vLLM on Anyscale. We process the [ReLAION-2B dataset](https://huggingface.co/datasets/laion/relaion2B-en-research-safe), which contains over 2 billion image URLs with associated metadata.

## What This Pipeline Does

The pipeline performs three main stages on each image:

1. **Parallel Image Download**: Asynchronously downloads images from URLs using aiohttp with 1,000 concurrent connections, handling timeouts and validation gracefully.

2. **Image Preprocessing**: Validates, resizes, and standardizes images to 128×128 JPEG format in RGB color space using PIL, filtering out corrupted or invalid images.

3. **Vision Model Inference**: Runs the Qwen2.5-VL-3B-Instruct vision-language model using vLLM to generate captions or analyze image content, scaling across up to 64 L4 GPU replicas based on workload.

The entire pipeline is orchestrated by Ray Data, which handles distributed execution, fault tolerance, and resource management across your cluster.

## Key Features

- **Massive Scale**: Processes 2B+ images efficiently with automatic resource scaling
- **High Throughput**: Concurrent downloads (1,000 connections) and batched inference (8 images per batch, 16 concurrent batches per GPU)
- **Fault Tolerant**: Gracefully handles network failures, invalid images, and transient errors
- **Cost Optimized**: Automatic GPU autoscaling (up to 64 L4 replicas) based on workload demand
- **Production Ready**: Timestamped outputs, configurable memory limits, and structured error handling

## How to Run

First, make sure you have the [Anyscale CLI](https://docs.anyscale.com/get-started/install-anyscale-cli) installed.

You'll need a HuggingFace token to access the ReLAION-2B dataset. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Submit the job:

```bash
anyscale job submit -f job.yaml --env HF_TOKEN=$HF_TOKEN
```

Or use the convenience script:

```bash
./run.sh
```

Results will be written to `/mnt/shared_storage/process_images_output/{timestamp}/` in Parquet format.

## Configuration

The pipeline is configured for high-throughput processing:

- **Compute**: Up to 530 CPUs and 64 L4 GPUs (g6.xlarge workers) with auto-scaling
- **Vision Model**: Qwen2.5-VL-3B-Instruct on NVIDIA L4 GPUs with vLLM
- **Download**: 1,000 concurrent connections, 5-second timeout per image
- **Batch Processing**: 50 images per download batch, 8 images per inference batch
- **Output**: 100,000 rows per Parquet file for efficient storage

You can adjust these settings in `process_images.py` and `job.yaml` to match your requirements.