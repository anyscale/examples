FROM anyscale/ray:2.48.0-slim-py312-cu128

RUN sudo apt-get update -y \
    && sudo apt-get install --no-install-recommends -y build-essential libnuma-dev \
    && sudo rm -f /etc/apt/sources.list.d/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN git clone https://github.com/novasky-ai/SkyRL.git
