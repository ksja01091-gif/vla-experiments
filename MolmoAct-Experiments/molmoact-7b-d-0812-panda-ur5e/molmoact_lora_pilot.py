"""Small single-A100 action-only LoRA feasibility pilot for public PickPlace_D0.

This adapts the public MolmoAct checkpoint; it is not the paper's full
depth-and-visual-trace post-training recipe. Held-out episodes are never used
for parameter updates. The base checkpoint stays read-only.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from molmoact_train_preflight import ROOT, CHECKPOINT, PROMPT, action_text, action_tokens, center_crop, read_frame


def make_batch(record, processor, stats, device):
    images = [center_crop(read_frame(record[f"{view}_video"])) for view in ("agentview", "wrist")]
    q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)
    q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)
    target = action_text(action_tokens(record["next_8_actions"], processor.tokenizer, q01, q99), processor.tokenizer)
    user = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(user + [{"role": "assistant", "content": [{"type": "text", "text":
        " The action that the robot should take is " + target}]}], tokenize=False, add_generation_prompt=False)
    prompt_ids = processor(images=[images], text=prompt, padding=True, return_tensors="pt")["input_ids"]
    batch = processor(images=[images], text=full, padding=True, return_tensors="pt")
    assert torch.equal(prompt_ids[0], batch["input_ids"][0, :prompt_ids.shape[1]])
    labels = batch["input_ids"].clone()
    labels[:, :prompt_ids.shape[1]] = -100
    batch["labels"] = labels
    return {key: value.to(device) for key, value in batch.items()}


def load_records(robot, split):
    path = ROOT / "sample_manifest" / f"{robot}_{split}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f]


@torch.inference_mode()
def validation_loss(model, processor, stats, records):
    model.eval()
    total, count = 0.0, 0
    for record in records:
        batch = make_batch(record, processor, stats, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss
        total += float(loss.detach())
        count += 1
    model.train()
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "ur5e"], required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--run-name")
    parser.add_argument("--save-interval", type=int, default=1000)
    args = parser.parse_args()
    assert torch.cuda.is_available() and args.steps > 0
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    output = ROOT / "lora_pilot" / (args.run_name or args.robot)
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(CHECKPOINT), trust_remote_code=True, local_files_only=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(str(CHECKPOINT), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["att_proj"], bias="none")
    model = get_peft_model(model, lora_config)
    # MolmoAct adds vision features in-place to word embeddings. PEFT's
    # checkpointing hook marks the embedding output as a grad-requiring leaf;
    # clone it so the subsequent in-place image insertion is autograd-safe.
    model.get_input_embeddings().register_forward_hook(
        lambda _module, _inputs, output: output.clone()
    )
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < 0.01 * total
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    stats = json.loads((ROOT / "train_action_stats.json").read_text())[args.robot]
    records = load_records(args.robot, "train")
    validation_records = load_records(args.robot, "validation")
    rng = random.Random(7)
    rng.shuffle(records)
    torch.cuda.reset_peak_memory_stats()
    history = []
    initial_validation = validation_loss(model, processor, stats, validation_records)
    print(json.dumps({"initial_validation_loss": initial_validation,
                      "validation_frames": len(validation_records)}), flush=True)
    started = time.monotonic()
    for step in range(args.steps):
        batch = make_batch(records[step % len(records)], processor, stats, "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(**batch, use_cache=False)
            loss = result.loss
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        row = {"step": step + 1, "training_loss": float(loss), "grad_norm": float(grad_norm),
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30, "seconds": time.monotonic() - started,
               "episode_index": records[step % len(records)]["episode_index"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if (step + 1) % args.save_interval == 0 or step + 1 == args.steps:
            (output / "metrics.json").write_text(json.dumps(history, indent=2))
            model.save_pretrained(str(output / f"adapter_step_{step + 1:05d}"))
            print(f"Saved step {step + 1} adapter", flush=True)
    final_validation = validation_loss(model, processor, stats, validation_records)
    (output / "validation.json").write_text(json.dumps({
        "initial_loss": initial_validation, "final_loss": final_validation,
        "validation_frames": len(validation_records), "weights_updated_with_validation": False
    }, indent=2))
    (output / "config.json").write_text(json.dumps({"robot": args.robot, "steps": args.steps, "lr": args.lr,
         "seed": 7, "rank": 8, "alpha": 16, "batch_size": 1, "target_modules": ["att_proj"],
         "training": "action-only adaptation; not full official MolmoAct reasoning recipe",
         "train_samples_available": len(records), "validation_samples_available": len(validation_records),
         "checkpoint": str(CHECKPOINT), "trainable_parameters": trainable,
         "total_parameters": total}, indent=2))
    model.save_pretrained(str(output / "adapter"))
    print(f"Saved adapter to {output / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
