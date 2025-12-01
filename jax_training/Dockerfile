FROM anyscale/ray:2.52.1-py312-cu128

# By default, Anyscale swaps in an optimized version of Ray, so we need to avoid that
# in order to use our specific patched version of Ray.
ENV ANYSCALE_DISABLE_OPTIMIZED_RAY=1

RUN pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade numpy pandas flax optax

RUN pip uninstall -y ray
RUN pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl
