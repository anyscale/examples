from fastapi import FastAPI
import ray
from ray import serve
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer



# Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@ray.remote(num_gpus=1)
class InferenceReplica:
    def __init__(self, rank):
        self.rank = rank

        model_name = "gpt2"  # 124M params - works on small GPUs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Initialize DeepSpeed inference with tensor parallelism
        self.model = deepspeed.init_inference(
            model,
            tensor_parallel={"tp_size": torch.cuda.device_count()},  # Use all available GPUs
            dtype=torch.float16,
            replace_with_kernel_inject=True
        )

    def inference(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.module.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(outputs[0])


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    placement_group_bundles=[{"CPU": 3, "GPU": 2}],
)
@serve.ingress(app)
class InferenceDeployment:
    def __init__(self):
        self.replicas = [InferenceReplica.remote(i) for i in range(2)]

    # FastAPI will automatically parse the HTTP request for us.
    @app.get("/infer")
    def inference(self, prompt: str) -> str:
        result0 = self.replicas[0].inference.remote(prompt)
        result1 = self.replicas[1].inference.remote(prompt)

        return sum(ray.get([result0, result1]))

# Create deployment.
app_deploy = InferenceDeployment.bind()
