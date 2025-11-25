# Deploy DeepSeek R1

This example uses Ray Serve along with vLLM to deploy a DeepSeek R1 model as an Anyscale service.

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Deploy the service

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/deploy_deepseek_r1
```

Deploy the service. 
```bash
anyscale service deploy -f service.yaml
```

## Understanding the example

- The [application code](https://github.com/anyscale/examples/blob/main/deploy_deepseek_r1/serve_deepseek_r1.py) sets the required accelerator type with `accelerator_type="H100"`. To use a different accelerator, replace `"H100"` with the desired name. See the [list of supported accelerators](https://docs.ray.io/en/latest/ray-core/accelerator-types.html#accelerator-types) for available options.
- Ray Serve automatically autoscales the number of model replicas between `min_replicas` and `max_replicas`. Ray Serve adapts the number of replicas by monitoring queue sizes. For more information on configuring autoscaling, see the [AutoscalingConfig documentation](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.config.AutoscalingConfig.html).
- This example uses vLLM, and the [Dockerfile](https://github.com/anyscale/examples/blob/main/deploy_deepseek_r1/Dockerfile) defines the service’s dependencies. When you run `anyscale service deploy`, the build process adds these dependencies on top of an Anyscale-provided base image.
- To configure vLLM, modify the `engine_kwargs` dictionary. See [Ray documentation for the `LLMConfig` object](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LLMConfig.html#ray.serve.llm.LLMConfig).


## Query the service

The `anyscale service deploy` command outputs a line that looks like  
```text
curl -H "Authorization: Bearer <SERVICE_TOKEN>" <BASE_URL>
```

From the output, you can extract the service token and base URL. Open [query.py](https://github.com/anyscale/examples/blob/main/deploy_deepseek_r1/query.py) and add them to the appropriate fields.
```python
token = <SERVICE_TOKEN> 
base_url = <BASE_URL> 
```

Query the model  
```bash
pip install openai
python query.py
```

View the service in the [services tab](https://console.anyscale.com/services) of the Anyscale console.

## Shutdown 
 
Shutdown your Anyscale Service:
```bash
anyscale service terminate -n deploy-deepseek-r1
```