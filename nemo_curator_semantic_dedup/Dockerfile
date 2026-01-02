# NeMo Curator Image Deduplication Example
# Uses CUDA 12.8 for GPU-accelerated processing
FROM anyscale/ray:2.52.0-slim-py312-cu128

# Note: Cache busting for git clone is done via CURATOR_CACHE_BUST arg below

# Install system dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        unzip \
        wget \
        curl \
        git && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
# Use uv pip install --system to install into the base anaconda environment
# so all Ray workers (not just the driver) have these packages
RUN python -m pip install --upgrade pip setuptools wheel

# IMPORTANT: Uninstall any pre-existing RAPIDS/cuML packages from the base image
# The base image may have incompatible versions that conflict with scikit-learn
RUN python -m pip uninstall -y cuml-cu12 cudf-cu12 cugraph-cu12 pylibraft-cu12 raft-dask-cu12 rmm-cu12 || true && \
    echo "Cleaned up pre-existing RAPIDS packages"

# Clone NeMo-Curator from fork and install in editable mode
# This ensures all Ray workers have the same code with your local edits
ARG CURATOR_REPO=https://github.com/avigyabb/Curator.git
ARG CURATOR_REF=avi-test
# ARG CURATOR_REF=main
# Cache bust for git clone - change this value to force re-clone after pushing to branch
ARG CURATOR_CACHE_BUST=2025-12-29-v3
RUN echo "Cache bust: ${CURATOR_CACHE_BUST}" && \
    git clone --depth 1 -b ${CURATOR_REF} ${CURATOR_REPO} /home/ray/NeMo-Curator && \
    uv pip install --system -e /home/ray/NeMo-Curator[image_cuda12]

# Re-upgrade scikit-learn AFTER nemo-curator in case it was downgraded
# cuML 25.6.* needs sklearn >= 1.5 (has _get_default_requests)
RUN uv pip install --system "scikit-learn>=1.5,<1.6" && \
    python -c "import sklearn; print(f'Final scikit-learn version: {sklearn.__version__}')"

# Additional dependencies for image downloading and processing
RUN uv pip install --system \
    loguru \
    Pillow \
    aiohttp \
    tqdm \
    pandas \
    pyarrow \
    huggingface_hub \
    transformers

# Set environment variable for model directory
ENV MODEL_DIR=/home/ray/model_weights

# Create output directories
RUN mkdir -p /home/ray/data/webdataset \
             /home/ray/data/results \
             /home/ray/data/embeddings \
             /home/ray/data/removal_ids

