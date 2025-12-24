FROM anyscale/ray:2.52.1-slim-py312-cu128

# Install system dependencies for kenlm
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libboost-all-dev \
        git \
    && sudo rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies directly (Anyscale doesn't allow COPY from local directories)
RUN uv pip install --system \
    "py-data-juicer[nlp] @ git+https://github.com/xyuzh/data-juicer.git@feat/ray-minhash-memory-reservation" \
    gcsfs \
    s3fs \
    tiktoken \
    s3fs
