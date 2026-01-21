# Containerfile for Megatron-Bridge with transformer_engine
FROM anyscale/ray:2.53.0-py312-cu128

# Install core dependencies
RUN pip install --no-cache-dir \
    transformers>=4.57.1 \
    omegaconf>=2.3.0 \
    tensorboard>=2.19.0 \
    typing-extensions \
    pyyaml>=6.0.2 \
    tqdm>=4.67.1 \
    "hydra-core>1.3,<=1.3.2" \
    megatron-energon \
    datasets \
    bitstring \
    filetype \
    timm \
    accelerate \
    rich \
    wandb>=0.19.10

# Install NVIDIA packages - transformer_engine is the key dependency
RUN pip install --no-cache-dir nvidia-modelopt
RUN pip install --no-cache-dir nvidia-resiliency-ext
RUN pip install --no-cache-dir --no-build-isolation transformer_engine[pytorch]

# Clone Megatron-Bridge and submodules to /app
WORKDIR /app
RUN git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git && \
    cd Megatron-Bridge && \
    git submodule update --init 3rdparty/Megatron-LM
