FROM anyscale/ray:2.48.0-slim-py312-cu128

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system vllm==0.9.2
RUN uv pip install --system transformers==4.53.3