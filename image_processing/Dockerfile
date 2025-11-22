FROM anyscale/ray:2.52.0-slim-py312-cu128

# C compiler for Triton’s runtime build step (vLLM V1 engine)
# https://github.com/vllm-project/vllm/issues/2997
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends build-essential

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system huggingface_hub boto3

RUN uv pip install --system vllm==0.11.2

RUN uv pip install --system transformers==4.57.1
