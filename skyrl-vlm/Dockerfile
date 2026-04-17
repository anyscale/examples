FROM novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 # TODO: Base image verification

RUN sudo apt-get update -y \
    && sudo apt-get install --no-install-recommends -y build-essential libnuma-dev \
    && sudo rm -f /etc/apt/sources.list.d/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN git clone https://github.com/novasky-ai/SkyRL.git
