import os
import random
import sys
from fastapi import FastAPI
import ray
from ray import serve
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

tensor_parallelism_size = 2
model_name = "gpt2"  # 124M params - works on small GPUs

# Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


# Using max_restarts=3 because occasionally there will be a port conflict (we are choosing a random port)
# and the nccl process group will fail to initialize. We want to retry in those cases.
@ray.remote(num_gpus=1, max_restarts=3)
class InferenceWorker:
    def __init__(self, rank, tensor_parallelism_size, master_address, master_port):
        self.rank = rank
        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(tensor_parallelism_size)
        dist.init_process_group("nccl", rank=self.rank, world_size=tensor_parallelism_size)

        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # Initialize DeepSpeed inference with tensor parallelism
        self.model = deepspeed.init_inference(
            model,
            tensor_parallel={"tp_size": tensor_parallelism_size},  # Use all available GPUs
            dtype=torch.float16,
            replace_with_kernel_inject=True
        )

    def inference(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        if self.rank == 0:
            return self.tokenizer.decode(outputs[0])


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1},
    placement_group_bundles=[{
        "CPU": tensor_parallelism_size + 1,  # One additional CPU for the coordinator actor.
        "GPU": tensor_parallelism_size
    }],
)
@serve.ingress(app)
class InferenceDeployment:
    def __init__(self, tensor_parallelism_size):
        master_address = "localhost"  # This is fine as long as the model fits on a single node.
        master_port = str(random.randint(10000, 65535))
        self.workers = [InferenceWorker.remote(i, tensor_parallelism_size, master_address, master_port) for i in range(tensor_parallelism_size)]

    # FastAPI will automatically parse the HTTP request for us.
    @app.get("/infer")
    def inference(self, text: str) -> str:
        results = ray.get([worker.inference.remote(text) for worker in self.workers])
        return results[0]


# Create deployment.
app = InferenceDeployment.bind(tensor_parallelism_size)
