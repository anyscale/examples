FROM anyscale/ray:nightly-py311-cu128

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system vllm==0.10.0
RUN uv pip install --system transformers==4.54.1
RUN uv pip install --system boto3==1.26.76
RUN uv pip install --system google-cloud-storage==2.14.0