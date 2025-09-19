FROM anyscale/ray:2.49.1-slim-py312-cu128

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system transformers deepspeed
