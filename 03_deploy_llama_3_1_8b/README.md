# Deploy Llama 3.1 8b

This example uses Ray Serve along with vLLM to deploy a Llama model as an Anyscale service.

## Install the Anyscale CLI

```
pip install -U anyscale
anyscale login
```

## Deploy the service

Clone the example from GitHub.

```
git clone https://github.com/anyscale/examples.git
cd examples/03_deploy_llama_3_1_8b
```

Deploy the service. Use `--env` to forward your Hugging Face token if you need authentication for gated models like Llama 3.

```
anyscale service deploy -f service.yaml --env HF_TOKEN=$HF_TOKEN
```

If you don't have a Hugging Face token, you can also use an ungated model (replace `model_source="meta-llama/Llama-3.1-8B-Instruct"` with `model_source="Qwen/Qwen2.5-7B"`) and remove all `HF_TOKEN` from the code as well as from the `anyscale service deploy` command.

## Understanding the example

- The required accelerator type is specified in the [application code](https://github.com/anyscale/examples/blob/main/03_deploy_llama_3_1_8b/serve_llama_3_8b.py) in the line `accelerator_type="L4"`. If you prefer to use a different hardware accelerator, change `L4` to a different accelerator name (you can see most of the supported ones [here](https://docs.ray.io/en/latest/ray-core/accelerator-types.html#accelerator-types)).
- Ray Serve will automatically autoscale the number of model replicas between `min_replicas` and `max_replicas`. Ray Serve will adapt the number of replicas by monitoring queue sizes. For more details on configuring autoscaling, see the documentation [here](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.config.AutoscalingConfig.html).
- This example uses vLLM. The dependencies for this service are specified in [this containerfile](https://github.com/anyscale/examples/blob/main/03_deploy_llama_3_1_8b/Dockerfile). When deploying the service with `anyscale service deploy`, the addition dependencies specified in the containerfile will be built on top of one of the Anyscale-provided base images.
- To configure vLLM, modify the `engine_kwargs` dictionary. For more details and other configuration options, see the documentation for the `LLMConfig` object [here](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LLMConfig.html#ray.serve.llm.LLMConfig).


## Query the service

The `anyscale service deploy` command outputs a line that looks like

```
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
```

From this, you can find the service token and base URL. Fill them in in the appropriate location in [query.py](https://github.com/anyscale/examples/blob/main/03_deploy_llama_3_1_8b/query.py). Then query the model with

```
python query.py
```

View the service in the [services tab](https://console.anyscale.com/services) of the Anyscale console.