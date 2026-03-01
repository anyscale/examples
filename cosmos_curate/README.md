# Cosmos Curate

* can we eliminate code from being in the image at all for clarity?

This repository has example Anyscale Jobs for the `cosmos-curate` Hello World & Reference Video Pipelines.

To run these on Anyscale looks like:
```
anyscale job submit -f hello_world.yaml
```

or:
```
anyscale job submit -f reference_video_pipeline.yaml
```

The `entrypoint:` in each job will run:

1. **python all_nodes_init_script.py**

This runs the same script on all nodes to initialize state of the cluster. In particular 

(a) We have to set the hardcoded **/cosmos_curate/config/cosmos_curate.yaml** on each node from the shared storage `/mnt/user_storage/`. This is how `cosmos-curate` does API and model authentication.

(b) We need to write our S3 creds to `COSMOS_S3_PROFILE_PATH` configurable default path of `/dev/shm/s3_creds_file`. 

(c) We run the `pixi run -e model-download python -m cosmos_curate.core.managers.model_cli download --models gpt2` commands to download models for the job.

2. **pixi run python -m cosmos_curate.pipelines.examples.hello_world_pipeline**

This is the actual pipeline entrypoint command. The `pixi run` depends on `PIXI_PROJECT_MANIFEST` being properly set to match what was built into the image.

Where in turn 

```
name: custom-image-cosmos
image_uri: 367974485317.dkr.ecr.us-west-2.amazonaws.com/wagner-west-2:15 
ray_version: 2.48.0
entrypoint: >
  python all_nodes_init_script.py
  && pixi run python -m cosmos_curate.pipelines.examples.hello_world_pipeline
py_modules: ["/Users/davidwagner/git/davidwagnerkc/cosmos-curate"]
compute_config:
  head_node:
    instance_type: m5.2xlarge 
    resources:
      CPU: 8
      GPU: 0
    flags: {}
  worker_nodes:
    - instance_type: g6e.2xlarge
      flags: {}
      min_nodes: 1
      max_nodes: 1
      market_type: ON_DEMAND
working_dir: "."
max_retries: 0
env_vars:
    PIXI_PROJECT_MANIFEST: /opt/cosmos-curate/pixi.toml
```

Key state of the `cosmos-curate` setup:

**/cosmos_curate/config/cosmos_curate.yaml**:

Where the 

To run `comsmos-curate` on Anyscale you need:

1. Anyscale compatible Docker image. 

2. File based authentication

# Building Anyscale Compatible Docker Image

Can skip steps (0) and (1) as the `cosmos-curate.Dockerfile` is committed (and modified to layer per `pixi` env for faster pulling). This is how other `cosmos-curate` build configurations can be built. If you already have an image built you can start at (4) updating the image name and tag to build an Anyscale compatible image on top.

Inside the `docker/` folder:

0. `pip install -e .` inside of the **cosmos-curate/** repo to make the `cosmos-curate` cli command available

1. `./generate_dockerfile.sh` to create **cosmos-curate.Dockerfile**

2. `./build_cosmos.sh`. to produce `cosmos-curate:1` image to build Anyscale image on.

4. `./build_and_push_anyscale.sh` to build and push `anyscale-cosmos-curate:1` image.
