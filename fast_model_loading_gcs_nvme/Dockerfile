FROM anyscale/ray:2.55.0-slim-py312-cu129

RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends build-essential && \
    sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

RUN pip install "ray[llm]" runai-model-streamer runai-model-streamer-gcs
