from ray import serve
from ray.serve.llm import LLMConfig, LLMServer, LLMRouter, build_openai_app
import os

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-llama",
        model_source="meta-llama/Llama-3.1-70B-Instruct",
    ),
    accelerator_type="A100",
    # We need to share our Hugging Face Token to the workers so they can access the gated Llama 3
    # If your model is not gated, you can skip this
    runtime_env=dict(
        env_vars={
          "HF_TOKEN": os.environ["HF_TOKEN"]
        }
    ),
    engine_kwargs=dict(
        max_model_len=8192,
        enable_auto_tool_choice=True,
        tool_call_parser="llama3_json",
        tensor_parallel_size=8
    ),
)
deployment = LLMServer.as_deployment(llm_config.get_serve_options(name_prefix="vLLM:")).bind(llm_config)
app = LLMRouter.as_deployment().bind([deployment])

# Uncomment the below line to run the service locally with Python.
# serve.run(app, blocking=True)
