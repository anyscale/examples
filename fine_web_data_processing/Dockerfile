FROM anyscale/ray:2.52.0-slim-py312-cu128

# Install system dependencies for kenlm
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libboost-all-dev \
        git \
    && sudo rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Set UV to use system Python (no venv required)
ENV UV_SYSTEM_PYTHON=1

# Install dependencies directly (Anyscale doesn't allow COPY from local directories)
RUN uv pip install --system \
    "py-data-juicer[nlp]" \
    gcsfs \
    s3fs \
    tiktoken