#serve_my_llama_3_1_8B.py
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app
import os

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-llama-3.1-8B",
        # Or Qwen/Qwen2.5-7B for an ungated model
        model_source="meta-llama/Llama-3.1-8B-Instruct",
    ),
    accelerator_type="L4",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1, max_replicas=2,
        )
    ),
    engine_kwargs=dict(
        max_model_len=8192,
        hf_token=os.environ.get("HF_TOKEN", "")
    )
)

app = build_openai_app({"llm_configs": [llm_config]})

#serve.run(app, blocking=True)

