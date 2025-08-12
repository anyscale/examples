# serve_my_llama.py
from ray import serve
from ray.serve.llm import LLMConfig, LLMServer, LLMRouter, build_openai_app
import os

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-llama",
        # Or Qwen/Qwen2.5-7B for an ungated model
        model_source="meta-llama/Llama-3.1-8B-Instruct",
    ),
    accelerator_type="L4",
    lora_config=dict(
        dynamic_lora_loading_path=<YOUR-S3-OR-GCS-URI>,
        max_num_adapters_per_replica=16,
    ),
    runtime_env=dict(
        env_vars={
          ### If your model is not gated, you can skip `HF_TOKEN`
          # Else, we need to share our Hugging Face Token to the workers so they can access the gated Llama 3
          "HF_TOKEN": os.environ["HF_TOKEN"],
          "AWS_REGION": <YOUR-AWS-S3-REGION>
        }
    ),
    engine_kwargs=dict(
        max_model_len=8192,
        enable_lora=True,
        max_lora_rank=32 # Set to the largest rank you use
    ),
)
deployment = LLMServer.as_deployment(llm_config.get_serve_options(name_prefix="vLLM:")).bind(llm_config)
app = LLMRouter.as_deployment().bind([deployment])

## Alternatively, use the builder method directly on the LLMConfig
# app = build_openai_app({"llm_configs": [llm_config]})

serve.run(app, blocking=True)