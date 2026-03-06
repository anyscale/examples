FROM anyscale/ray:2.53.0-py312-cu129

# System dependencies for building SGLang extensions
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        libnuma-dev \
        curl \
    && sudo rm -rf /var/lib/apt/lists/*

# CUDA toolkit (nvcc) for compiling CUDA kernels
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb && \
    sudo dpkg -i /tmp/cuda-keyring.deb && \
    rm /tmp/cuda-keyring.deb && \
    sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        cuda-nvcc-12-9 \
        cuda-cudart-dev-12-9 \
        cuda-crt-12-9 \
    && sudo rm -rf /var/lib/apt/lists/* && \
    sudo rm -rf /usr/local/cuda && \
    sudo ln -s /usr/local/cuda-12.9 /usr/local/cuda

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Python dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/ray/.local/bin:${PATH}"

# Install SGLang from a feature branch that supports Ray actor scheduling.
# Replace with a released version after this feature is merged into a future SGLang release.
RUN uv pip install --system \
    "sglang[all] @ git+https://github.com/xyuzh/sglang.git@feature/ray-actor-scheduler#subdirectory=python"
