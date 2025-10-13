from ray.serve.llm import LLMConfig, build_openai_app
import os

# model_name = "meta-llama/Llama-3.1-70B-Instruct"
# model_name = "meta-llama/Llama-3.3-70B-Instruct"
# model_name = "unsloth/Meta-Llama-3.1-70B-Instruct"  # Ungated, no token required
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  # Ungated, no token required

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-70b-model",
        model_source=model_name,
    ),
    # Valid types (depending on what GPUs are available on the cloud) include "L40S", "A100", and "H100".
    # If you use a cloud other than AWS, in addition to changing the accelerator type, you also need to
    # change the compute_config in service.yaml.
    accelerator_type="L40S",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1,
            max_replicas=4,
        )
    ),
    ### If your model is not gated, you can skip `HF_TOKEN`
    # Share your Hugging Face token with the vllm engine so it can access the gated Llama 3.
    # Type `export HF_TOKEN=<YOUR-HUGGINGFACE-TOKEN>` in a terminal
    engine_kwargs=dict(
        max_model_len=32768,
        # Split weights among 8 GPUs in the node
        tensor_parallel_size=8,
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})

# Uncomment the below line to run the service locally with Python.
# serve.run(app, blocking=True)
