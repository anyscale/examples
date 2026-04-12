# =============================================================================
# This image provides everything needed to run DeepSeek V3 with UCCL EP
# on AWS EFA, including:
#   - EFA SDK + libfabric
#   - gdrcopy (for GPU Direct RDMA)
#   - aws-ofi-nccl (NCCL over EFA)
#   - UCCL EP extension pre-compiled for SM 9.0 (H100/H200)
#   - deep_ep wrapper (replaces original DeepEP)
#   - Library paths and env vars pre-configured
# =============================================================================

FROM anyscale/ray-llm:nightly-py311-cu128

ENV MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
ENV NCCL_HOME=/usr
ENV EFA_VERSION="1.35.0"
ENV AWS_OFI_NCCL_VERSION="1.17.1"
ENV GDRCOPY_VERSION="2.4.1"

# Target GPU arch: 9.0 = H100/H200, 8.0 = A100
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Step 0: System dependencies + EFA SDK + NCCL OFI
RUN <<EOF
set -euxo pipefail

sudo apt-get update -y
sudo apt-get install -y \
    wget git autoconf automake libtool make gcc g++ libhwloc-dev \
    libibverbs-dev libnl-3-dev libnl-route-3-dev libnuma-dev

# EFA installer
wget -q https://efa-installer.amazonaws.com/aws-efa-installer-$EFA_VERSION.tar.gz -O /tmp/aws-efa-installer-$EFA_VERSION.tar.gz
wget -q https://efa-installer.amazonaws.com/aws-efa-installer.key -O /tmp/aws-efa-installer.key
gpg --import /tmp/aws-efa-installer.key
wget -q https://efa-installer.amazonaws.com/aws-efa-installer-$EFA_VERSION.tar.gz.sig -O /tmp/aws-efa-installer-$EFA_VERSION.tar.gz.sig
gpg --verify /tmp/aws-efa-installer-$EFA_VERSION.tar.gz.sig /tmp/aws-efa-installer-$EFA_VERSION.tar.gz
tar -xzf /tmp/aws-efa-installer-$EFA_VERSION.tar.gz -C /tmp
cd /tmp/aws-efa-installer
sudo bash efa_installer.sh --yes --skip-kmod

# gdrcopy + aws-ofi-nccl (if CUDA present)
if [ -d /usr/local/cuda ]; then
  wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v$GDRCOPY_VERSION.tar.gz -O /tmp/v$GDRCOPY_VERSION.tar.gz
  tar -xzf /tmp/v$GDRCOPY_VERSION.tar.gz -C /tmp
  cd /tmp/gdrcopy-${GDRCOPY_VERSION}
  sudo make -j"$(nproc)" lib_install

  cd /tmp
  git clone --depth=1 --branch v${AWS_OFI_NCCL_VERSION} https://github.com/aws/aws-ofi-nccl.git
  cd /tmp/aws-ofi-nccl
  ./autogen.sh
  ./configure \
    --with-libfabric=/opt/amazon/efa \
    --with-mpi=/opt/amazon/openmpi \
    --with-cuda=/usr/local/cuda \
    --prefix=/usr/local
  make -j"$(nproc)"
  sudo make install
fi

sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/* /tmp/*
EOF

# Step 1: Python build dependencies
RUN pip install --no-cache-dir nanobind intervaltree

# Step 2: Ensure linker finds torch + EFA at build and runtime
# This must happen BEFORE the .so build so the compiler/linker can resolve
# all shared library dependencies during setup.py install.
RUN <<EOF
set -euxo pipefail
TORCH_LIB=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")
echo "$TORCH_LIB" | sudo tee /etc/ld.so.conf.d/torch.conf
echo "/opt/amazon/efa/lib" | sudo tee /etc/ld.so.conf.d/efa.conf
sudo ldconfig
echo "ldconfig updated with torch + EFA lib paths"
EOF

# Step 3: Clone UCCL, prepare sources, and BUILD the .so
RUN <<EOF
set -euxo pipefail

cd /tmp
git clone https://github.com/uccl-project/uccl.git

# Prepare the deep_ep wrapper sources
cd /tmp/uccl/ep/deep_ep_wrapper/deep_ep
for f in buffer.py utils.py test_internode.py; do
    if [ -L "$f" ]; then
        target="../../bench/$(basename "$f")"
        rm "$f"
        cp "$target" "$f"
    fi
done

# Install the deep_ep wrapper (pure Python, no native code)
cd /tmp/uccl/ep/deep_ep_wrapper
pip install --no-cache-dir .

# Build and install the UCCL EP native extension
cd /tmp/uccl/ep
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
PER_EXPERT_BATCHING=1 \
python setup.py install

# Verify the .so was produced and installed
python -c "\
import site, pathlib; \
sp = pathlib.Path(site.getsitepackages()[0]) / 'uccl'; \
so_files = list(sp.glob('ep*.so')); \
assert so_files, f'No ep*.so found in {sp}'; \
print(f'UCCL EP .so installed: {so_files[0].name}')"

echo "UCCL EP compiled and installed to site-packages/uccl/"
EOF

# Step 4: Remove original DeepEP C++ extension (if present)
RUN <<EOF
set -euxo pipefail

SITE=$(python -c "import site; print(site.getsitepackages()[0])")

# Remove original deep_ep_cpp .so
find "$SITE" -name "deep_ep_cpp*.so" -delete 2>/dev/null || true

# Remove original deep_ep dist-info
find "$SITE" -maxdepth 1 -name "deep_ep-*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Original DeepEP removed"
EOF

# Step 5: Clean up
RUN rm -rf /tmp/uccl

# Step 6: Set runtime environment variables
ENV LD_LIBRARY_PATH="/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"
ENV FI_EFA_USE_DEVICE_RDMA="1"
ENV VLLM_RAY_DP_PACK_STRATEGY="strict"
ENV NCCL_DEBUG="WARN"
