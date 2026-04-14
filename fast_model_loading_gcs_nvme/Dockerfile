FROM anyscale/ray:2.55.0-slim-py312-cu129

RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends build-essential && \
    sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# ray.anyscale.safetensors is pre-installed — do not add it.
RUN pip install vllm==0.11.0 google-cloud-storage
