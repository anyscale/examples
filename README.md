# Examples

This repository contains examples for deploying and running distributed applications.

## Job Examples

### 1. Hello World Job
**Directory:** `01_job_hello_world/`

A simple "Hello World" example demonstrating how to submit and run basic jobs.

### 2. Image Processing
**Directory:** `image_processing/`

Process large-scale image datasets using Ray Data. This example demonstrates processing the ReLAION-2B dataset with over 2 billion rows.

### 3. Megatron + Ray Fault Tolerant Training
**Directory:** `megatron_ray_fault_tolerant/`

Implements PPO-style distributed training with Megatron and Ray, featuring comprehensive fault tolerance capabilities:
- Automatic actor recovery from failures
- Backup actor groups for seamless replacement
- Distributed checkpoint saving/loading
- Process group re-initialization after failures
- Support for tensor, pipeline, data, and context parallelism

## Service Examples

### 1. Hello World Service
**Directory:** `02_service_hello_world/`

A simple service deployment example demonstrating the basics of Ray Serve.

### 2. Deploy Llama 3.1 8B
**Directory:** `03_deploy_llama_3_8b/`

Deploy Llama 3.1 8B model using Ray Serve and vLLM with autoscaling capabilities.

### 3. Deploy Llama 3.1 70B
**Directory:** `deploy_llama_3_1_70b/`

Deploy the larger Llama 3.1 70B model with optimized serving configuration.

### 4. Tensor Parallel Serving
**Directory:** `serve_tensor_parallel/`

Demonstrates tensor parallelism for serving large language models across multiple GPUs.

### 5. FastVideo Generation
**Directory:** `video_generation_with_fastvideo/`

Deploy a video generation service using the FastVideo framework.

## Reinforcement Learning Examples

### SkyRL
**Directory:** `skyrl/`

Reinforcement learning training example using Ray and distributed computing.

## Getting Started

Most examples include their own README with specific instructions. Generally, you'll need:

1. Install the Anyscale CLI:
```bash
pip install -U anyscale
anyscale login
```

2. Navigate to the example directory:
```bash
cd <example_directory>
```

3. Deploy the service or submit the job:
```bash
# For services
anyscale service deploy -f service.yaml

# For jobs
anyscale job submit -f job.yaml
```

## Requirements

- Anyscale account and CLI access
- Appropriate cloud credentials configured
- GPU resources for ML/LLM examples

## Contributing

When adding new examples:
1. Create a descriptive directory name
2. Include a README.md with setup and usage instructions
3. Add appropriate YAML configuration files
4. Update this main README with your example

## License

See individual example directories for specific licensing information.

