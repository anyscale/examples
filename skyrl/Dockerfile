FROM anyscale/ray:2.48.0-slim-py312-cu128

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y build-essential libnuma-dev \
    && rm -f /etc/apt/sources.list.d/* \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN git clone https://github.com/novasky-ai/SkyRL.git
