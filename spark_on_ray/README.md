# Run Spark on Ray

This example demonstrates how to run a simple data processing example with [RayDP](https://github.com/ray-project/raydp), a library for running Spark on Ray.


## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```


## Submit the job.

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/spark_on_ray
```

Submit the job.

```bash
anyscale job submit -f job.yaml
```


## Understanding the example

- This example is extremely simple and just uses basic Spark APIs. More configuration is required to read from blob stores like S3.