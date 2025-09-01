#serve_deepseek_r1.py
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-deepseek-r1",
        model_source="deepseek-ai/DeepSeek-R1",
    ),
    accelerator_type="H100",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1, max_replicas=1,
        )
    ),
    engine_kwargs=dict(
        max_model_len=16384,
        # Split weights among 8 GPUs in each node
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        reasoning_parser="deepseek_r1",
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})

# Uncomment the below line to run the service locally with Python.
# serve.run(app, blocking=True)
