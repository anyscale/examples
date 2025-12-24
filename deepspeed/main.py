import os, json, math, tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm
import ray

from datasets import load_dataset
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

from ray.train import Checkpoint, RunConfig, ScalingConfig, DataConfig
from ray.train.torch import TorchTrainer

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
HF_DATASET_ID = "evanfrick/human-pref-debug-dataset"

# ---- Quick-run knobs (tweak for cluster size/speed)
NUM_WORKERS = 8                 # num of GPUs 
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 1
EPOCHS = 1
MAX_LENGTH = 1024               # prompt context length
LEARNING_RATE = 5e-5
LIMIT_ROWS = 4000               # limit HF rows for a fast run; set 0 for full dataset

# ---- Storage on Anyscale
STORAGE_PATH = "/mnt/cluster_storage/lmarena_qwen32b_rm"

ds_cfg = {
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
        "sub_group_size": 1e9,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        }
    },
    "bf16": {"enabled": True},
    "fp16": {"enabled": False},
    "gradient_clipping": 1.0
}


def make_collate_fn(tokenizer, max_len, device):
    def _to_py(x):
        if isinstance(x, np.ndarray):
            try:
                return x.tolist()
            except Exception:
                return [ _to_py(e) for e in x ]
        return x

    def _apply_chat(messages):
        # messages: list[{"role","content"}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # minimal fallback
            return "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)

    def _pair_from_messages_and_labels(msgs, labels):
        """Return (chosen_text, rejected_text) or None if we can't."""
        convs = _to_py(msgs)
        # normalize: convs should be a list of candidate conversations
        if isinstance(convs, list) and convs and isinstance(convs[0], dict):
            convs = [convs]  # wrap single conversation

        # build candidate texts
        cand_texts = []
        if isinstance(convs, list):
            for conv in convs:
                conv_py = _to_py(conv)
                if isinstance(conv_py, list) and conv_py and isinstance(conv_py[0], dict):
                    cand_texts.append(_apply_chat(conv_py))

        if len(cand_texts) < 2:
            return None

        # choose indices from one-hot-ish labels if present
        chosen_idx, rejected_idx = 0, 1
        L = _to_py(labels)
        if isinstance(L, list) and len(L) >= 2 and 1 in L:
            chosen_idx = L.index(1)
            rejected_idx = 1 - chosen_idx if len(cand_texts) > 1 else chosen_idx

        if chosen_idx == rejected_idx or chosen_idx >= len(cand_texts) or rejected_idx >= len(cand_texts):
            return None
        return cand_texts[chosen_idx], cand_texts[rejected_idx]

    def _pair_from_prompt_answers(prompt, answers):
        """Return (chosen_text, rejected_text) built from prompt + answers[rank]."""
        p = str(prompt)
        L = _to_py(answers) or []
        if not isinstance(L, list) or len(L) < 2:
            return None

        # rank==1 is best; pick a contrast as rank==2 if present, else any non-top
        chosen = min(L, key=lambda a: a.get("rank", float("inf")))
        rejected = next((a for a in L if a is not chosen and a.get("rank", None) == 2), None)
        if rejected is None:
            rejected = next((a for a in L if a is not chosen), None)
        if not chosen or not rejected:
            return None

        def _build(ans_text):
            msgs = [{"role": "user", "content": p},
                    {"role": "assistant", "content": str(ans_text)}]
            return _apply_chat(msgs)

        return _build(chosen.get("answer", "")), _build(rejected.get("answer", ""))

    def collate(batch):
        chosen_texts, rejected_texts = [], []

        # Case A: messages + labels (classic RM pair format)
        if "messages" in batch and "labels" in batch:
            msgs_col = _to_py(batch["messages"])
            labels_col = _to_py(batch["labels"])
            for i in range(len(msgs_col)):
                pair = _pair_from_messages_and_labels(msgs_col[i],
                                                      labels_col[i] if i < len(labels_col) else [])
                if pair:
                    c, r = pair
                    chosen_texts.append(c); rejected_texts.append(r)

        # Case B: prompt + answers[{answer, rank}]
        if not chosen_texts and "prompt" in batch and "answers" in batch:
            prompts = _to_py(batch["prompt"])
            answers = _to_py(batch["answers"])
            for p, a in zip(prompts, answers):
                pair = _pair_from_prompt_answers(p, a)
                if pair:
                    c, r = pair
                    chosen_texts.append(c); rejected_texts.append(r)

        if not chosen_texts:
            # Return empty tensors so the training loop can skip this batch safely.
            empty = torch.empty((0, 1), dtype=torch.long, device=device)
            return {
                "chosen_input_ids": empty, "chosen_attn": empty,
                "rejected_input_ids": empty, "rejected_attn": empty,
            }

        chosen = tokenizer(chosen_texts, padding="longest", truncation=True,
                           max_length=max_len, return_tensors="pt")
        rejected = tokenizer(rejected_texts, padding="longest", truncation=True,
                             max_length=max_len, return_tensors="pt")

        return {
            "chosen_input_ids": chosen["input_ids"].to(device),
            "chosen_attn": chosen["attention_mask"].to(device),
            "rejected_input_ids": rejected["input_ids"].to(device),
            "rejected_attn": rejected["attention_mask"].to(device),
        }

    return collate



def training_loop(cfg: dict):
    """
    Ray Train worker function.
    - Loads Qwen/Qwen2.5-32B-Instruct
    - Builds a frozen base + RewardModel head
    - Uses DeepSpeed via Accelerate (ZeRO-3), with Ray Data iterator
    - Computes Bradley–Terry loss on (chosen vs. rejected) pairs
    """

    # --- Setup & config ---
    set_seed(int(cfg["seed"]))

    accelerator = Accelerator(
        deepspeed_plugin=cfg["deepspeed_plugin"],
        gradient_accumulation_steps=int(cfg["grad_accum_steps"]),
        mixed_precision=cfg.get("mixed_precision", "bf16"),
    )

    # Optional debug: print DS micro-batch settings once
    if ray.train.get_context().get_world_rank() == 0:
        try:
            plug = accelerator.state.deepspeed_plugin
            ds_cfg = plug.hf_ds_config.config if hasattr(plug, "hf_ds_config") else plug.deepspeed_config
            print("[ds] train_micro_batch_size_per_gpu =", ds_cfg.get("train_micro_batch_size_per_gpu"))
            print("[ds] gradient_accumulation_steps     =", ds_cfg.get("gradient_accumulation_steps"))
        except Exception as e:
            print("[ds] could not inspect DS config:", e)

    model_id = cfg["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ray Data iterator (collate defined in Cell 7)
    train_ds = ray.train.get_dataset_shard("train")
    collate = make_collate_fn(tokenizer, cfg["max_length"], accelerator.device)
    data_iter = train_ds.iter_torch_batches(
        batch_size=int(cfg["per_device_batch_size"]),
        collate_fn=collate,
        prefetch_batches=2,
    )

    # --- Load base model; handle FlashAttention2 flag safely ---
    use_flash = bool(cfg.get("flash_attn", False))
    base_kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if use_flash:
        try:
            import flash_attn  # noqa: F401
            base_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            if accelerator.is_main_process:
                print("[info] FlashAttention2 not available; using PyTorch SDPA fallback.")

    base = AutoModelForCausalLM.from_pretrained(model_id, **base_kwargs)

    # --- Reward Model (freeze base; train only RM head) ---
    class RMHead(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.value_head = torch.nn.Linear(hidden_size, 1)

        def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            idx = attention_mask.to(torch.int64).sum(dim=1) - 1
            idx = idx.clamp(min=0)
            b = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            last_token = last_hidden_state[b, idx]
            return self.value_head(last_token).squeeze(-1)

    class RewardModel(torch.nn.Module):
        def __init__(self, base_model, hidden_size: int, freeze_base: bool = True):
            super().__init__()
            self.base = base_model
            self.rm_head = RMHead(hidden_size)
            if freeze_base:
                for p in self.base.parameters():
                    p.requires_grad = False

        def forward(self, input_ids, attention_mask):
            outputs = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1]
            rewards = self.rm_head(last_hidden, attention_mask)
            return rewards

    hidden_size = base.config.hidden_size
    rm = RewardModel(base_model=base, hidden_size=hidden_size, freeze_base=True)

    # Optimizer on trainable params only (RM head)
    trainable_params = [p for p in rm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg["lr"]), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    # *** DeepSpeed requirement: joint prepare(model, optimizer) ***
    rm, optimizer = accelerator.prepare(rm, optimizer)

    # --- Training ---
    # Compute approx steps/epoch robustly for iter_torch_batches + grad accumulation
    per_device_bs = int(cfg["per_device_batch_size"])
    world_size = max(1, int(cfg.get("world_size", 1)))
    grad_accum = max(1, int(cfg["grad_accum_steps"]))
    steps_per_epoch = max(
        1,
        (int(cfg["num_samples"]) // (world_size * per_device_bs)) // grad_accum
    )

    if accelerator.is_main_process:
        print(f"Starting training for {cfg['epochs']} epoch(s). ~{steps_per_epoch} steps/epoch")

    for epoch in range(int(cfg["epochs"])):
        rm.train()
        running = 0.0
        prog = tqdm.tqdm(range(steps_per_epoch), disable=not accelerator.is_main_process)
        it = iter(data_iter)

        for _ in prog:
            with accelerator.accumulate(rm):
                try:
                    batch = next(it)
                except StopIteration:
                    break

                r_c = rm(batch["chosen_input_ids"], batch["chosen_attn"])
                r_r = rm(batch["rejected_input_ids"], batch["rejected_attn"])

                # Bradley–Terry loss
                loss = -torch.nn.functional.logsigmoid(r_c - r_r).mean()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                running += loss.item()
                if accelerator.is_main_process:
                    prog.set_description(f"epoch {epoch} loss {running / (prog.n + 1):.4f}")

                ray.train.report({"epoch": epoch, "loss": float(loss.item())})

        # Save tiny checkpoint (tokenizer + rm_head) each epoch
        with tempfile.TemporaryDirectory() as tmp:
            if accelerator.is_main_process:
                tokenizer.save_pretrained(tmp)
                torch.save(rm.module.rm_head.state_dict(), os.path.join(tmp, "rm_head.pt"))
            accelerator.wait_for_everyone()
            ckpt = Checkpoint.from_directory(tmp) if accelerator.is_main_process else None
            ray.train.report({"epoch": epoch, "avg_loss": running / max(1, steps_per_epoch)}, checkpoint=ckpt)



hf = load_dataset(HF_DATASET_ID, split="train")
if LIMIT_ROWS and LIMIT_ROWS > 0:
    hf = hf.select(range(min(LIMIT_ROWS, len(hf))))
num_samples = len(hf)

# Convert to Ray Data
train_df = hf.to_pandas()
train_ds = ray.data.from_pandas(train_df)

num_samples, train_ds



ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_cfg)
world_size = NUM_WORKERS

train_loop_cfg = {
    "seed": 42,
    "epochs": EPOCHS,
    "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "max_length": MAX_LENGTH,
    "model_id": MODEL_ID,
    "lr": LEARNING_RATE,
    "deepspeed_plugin": ds_plugin,
    "mixed_precision": "bf16",
    "num_samples": num_samples,
    "world_size": world_size,
    "flash_attn": False,   # set False if FlashAttention2 isn't available
}

trainer = TorchTrainer(
    training_loop,
    train_loop_config=train_loop_cfg,
    run_config=RunConfig(storage_path=STORAGE_PATH),
    scaling_config=ScalingConfig(num_workers=world_size, use_gpu=True, resources_per_worker={"GPU": 1}),
    datasets={"train": train_ds},
    dataset_config=DataConfig(datasets_to_split=["train"]),
)

result = trainer.fit()
print(result)
