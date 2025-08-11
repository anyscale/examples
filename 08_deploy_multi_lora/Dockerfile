FROM anyscale/ray:2.48.0-slim-py312-cu128

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system vllm==0.9.2
RUN uv pip install --system transformers==4.52.4
RUN uv pip install --system boto3==1.26.76
RUN uv pip install --system google-cloud-storage==2.14.0