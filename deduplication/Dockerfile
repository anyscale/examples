FROM anyscale/ray:2.49.2-py312-cu128

ENV ANYSCALE_DISABLE_OPTIMIZED_RAY=1

RUN pip install scipy pydantic gcsfs

RUN pip uninstall -y ray && pip install https://ray-ci-artifact-pr-public.s3.us-west-2.amazonaws.com/0f79023c05aba84e4766e016cd65a22d039c3275/tmp/artifacts/.whl/.whl/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl
