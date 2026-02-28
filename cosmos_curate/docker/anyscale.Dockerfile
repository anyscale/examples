# syntax=docker/dockerfile:1.3-labs
# Anyscale-compatible image built on top of the Cosmos Curate image.
# Build with: docker build --platform linux/amd64 -f docker/anyscale.Dockerfile -t cosmos-curate:anyscale .

ARG COSMOS_TAG=2
FROM cosmos-curate:${COSMOS_TAG}

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages.
RUN set -euxo pipefail \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        sudo \
        tzdata \
        openssh-client \
        openssh-server \
        rsync \
        zip \
        unzip \
        git \
        gdb \
        curl \
        vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# Rename ubuntu -> ray and align uid/gid with Anyscale requirements.
RUN set -euxo pipefail \
    && groupmod -n ray users \
    && usermod -l ray -d /home/ray -m ubuntu \
    && usermod -u 1000 -g 100 ray \
    && usermod -aG sudo ray \
    && echo 'ray ALL=NOPASSWD: ALL' >> /etc/sudoers \
    && chown -R ray:ray /home/ray


# Install required Python packages into the default Pixi env.
# Note: Jupyter is optional; uncomment if needed for workspaces.
RUN set -euxo pipefail \
    && pixi run -e default pip install --no-cache-dir \
        anyscale \
        packaging \
        boto3 \
        google \
        google-cloud-storage \
        terminado \
    && pixi run -e default pip install --no-cache-dir jupyterlab

# Workspace dependencies (optional but safe for all images).
RUN set -euxo pipefail \
    && echo 'PROMPT_COMMAND="history -a"' >> /home/ray/.bashrc \
    && echo '[ -e ~/.workspacerc ] && source ~/.workspacerc' >> /home/ray/.bashrc \
    && chown ray:ray /home/ray/.bashrc

RUN mkdir -p /cosmos_curate/config /config \
    && chown -R ray /cosmos_curate /config
ENV COSMOS_S3_PROFILE_PATH=/mnt/user_storage/s3_creds_file
ENV PATH=/opt/cosmos-curate/.pixi/envs/default/bin:$PATH
ENV HOME=/home/ray
WORKDIR /home/ray
USER ray

RUN sudo mkdir -p /anyscale/init
RUN sudo chown -R ray /anyscale/init
RUN <<'EOF'
sudo cat >/anyscale/init/init.sh <<'EOC'
ls -halrt /mnt/user_storage/ > /tmp/init_ls.log 2>&1 || true
EOC
EOF

# Do not inherit base entrypoint; default to an interactive shell.
ENTRYPOINT []
CMD ["bash"]
