from typing import Dict
import os
import random
from fastapi import FastAPI
import ray
from ray import serve
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

tensor_parallelism_size = 2
model_name = "gpt2"  # 124M params - works on small GPUs


@ray.remote(num_gpus=1)
class InferenceWorker:
    def initialize_model(self, rank, tensor_parallelism_size, master_address, master_port):
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

    def get_ip_address(self):
        return ray._private.services.get_node_ip_address()

    def inference(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        if self.rank == 0:
            return self.tokenizer.decode(outputs[0])



workers = [InferenceWorker.remote() for i in range(tensor_parallelism_size)]
master_address = ray.get(workers[0].get_ip_address.remote())
master_port = str(random.randint(10000, 65535))
ray.get([workers[i].initialize_model.remote(i, tensor_parallelism_size, master_address, master_port) for i in range(tensor_parallelism_size)])


def inference(text: str) -> str:
    results = ray.get([worker.inference.remote(text) for worker in workers])
    return results[0]


for i in range(5):
    print(inference("What is the capital of France? "))
