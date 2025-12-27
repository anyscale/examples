# NeMo Curator Image Deduplication Example
# Uses CUDA 12.8 for GPU-accelerated processing
FROM anyscale/ray:2.52.0-slim-py312-cu128

# Cache buster - change this to force rebuild
ARG CACHE_BUST=2025-12-27-v5

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

# Upgrade scikit-learn FIRST in the system environment for cuML compatibility
# cuML 25.6.* requires sklearn with _get_default_requests (added in sklearn 1.5)
# This MUST be in the base anaconda env so workers have it
RUN uv pip install --system "scikit-learn>=1.5,<1.6" && \
    python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

# Clone NeMo-Curator from fork and install in editable mode
# This ensures all Ray workers have the same code with your local edits
ARG CURATOR_REPO=https://github.com/avigyabb/Curator.git
ARG CURATOR_REF=avi-test
# ARG CURATOR_REF=main
RUN git clone --depth 1 -b ${CURATOR_REF} ${CURATOR_REPO} /home/ray/NeMo-Curator && \
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

# Pre-download CLIP model weights to avoid runtime downloads
# This makes job startup faster and more reliable
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
model_dir = '/home/ray/model_weights/openai/clip-vit-large-patch14'; \
os.makedirs(model_dir, exist_ok=True); \
snapshot_download('openai/clip-vit-large-patch14', local_dir=model_dir)"

# Set environment variable for model directory
ENV MODEL_DIR=/home/ray/model_weights

# Download and prepare the example dataset from HuggingFace
# Downloads MS COCO parquet, deduplicates URLs, and truncates to 100k rows
RUN mkdir -p /home/ray/data && \
    curl -L https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet \
         -o /home/ray/data/mscoco.parquet && \
    python -c "\
import pandas as pd; \
df = pd.read_parquet('/home/ray/data/mscoco.parquet'); \
deduped = df[~df['URL'].duplicated()]; \
truncated = deduped[:100000]; \
truncated.to_parquet('/home/ray/data/truncated_100k_mscoco.parquet'); \
print(f'Created truncated dataset with {len(truncated)} rows')" && \
    rm /home/ray/data/mscoco.parquet

# Create output directories
RUN mkdir -p /home/ray/data/webdataset \
             /home/ray/data/results \
             /home/ray/data/embeddings \
             /home/ray/data/removal_ids

