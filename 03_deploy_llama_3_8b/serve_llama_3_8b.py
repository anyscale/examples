from ray import serve
from ray.serve.llm import LLMConfig, LLMServer, LLMRouter, build_openai_app
import os

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-llama-3-8B",
        model_source="meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    accelerator_type="A10G",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=2, max_replicas=2,
        )
    ),
    # We need to share our Hugging Face Token to the workers so they can access the gated Llama 3
    # If your model is not gated, you can skip this
    runtime_env=dict(
        env_vars={
          "HF_TOKEN": os.environ["HF_TOKEN"], 
          "VLLM_USE_V1": "0"
        }
    ),
    engine_kwargs=dict(
        max_model_len=8192
    ),
)
deployment = LLMServer.as_deployment(llm_config.get_serve_options(name_prefix="vLLM:")).bind(llm_config)
app = LLMRouter.as_deployment().bind([deployment])

## Alternatively, use the builder method directly on the LLMConfig
# app = build_openai_app({"llm_configs": [llm_config]})

# serve.run(app, blocking=True)