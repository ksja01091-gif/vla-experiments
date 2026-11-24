"""Short, auditable LeRobot/OpenVLA LoRA training connection test.
Default: one midpoint frame per training episode, 20 optimizer updates.
Not a complete fine-tuning experiment or a simulator success evaluation.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from peft import PeftModel
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from lerobot.policies.openvla.training_openvla import TrainableOpenVLAPolicy

PROJECT = Path("/home/jovyan/projects/lerobot-openvla")
DATA = Path("/home/jovyan/projects/datasets/libero_bowl_25/20261101T120613_454595Z")

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def frozen_sample_hash(policy):
    chunks = []
    for _, param in policy.model.named_parameters():
        if not param.requires_grad:
            flat = param.detach().flatten()
            n = flat.numel()
            for start in [0, max(0, n//2-8), max(0, n-16)]:
                chunks.append(flat[start:start+16].float())
    values = torch.cat(chunks).cpu().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()

def lora_values(policy):
    return {name: param.detach().float().cpu().clone()
            for name, param in policy.model.named_parameters() if "lora_" in name}

@torch.no_grad()
def measure(policy, batches):
    policy.eval()
    rows = [policy(batch)[1] for batch in batches]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

def episode_midpoints(records):
    result = []
    offset = 0
    for record in records:
        result.append(offset + record["frames"]//2)
        offset += record["frames"]
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        raise ValueError("This connection test is limited to 1..100 updates")
    output = args.output or PROJECT/"outputs/openvla_training"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    print("RUN_OUTPUT", output, flush=True)
    random.seed(7); np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    torch.set_num_threads(4)
    split = json.loads((DATA/"split_manifest.json").read_text())
    assert not {r["content_id"] for r in split["train"]} & {r["content_id"] for r in split["validation"]}
    stats = json.loads((DATA/"action_statistics_train_only.json").read_text())
    assert stats["episodes"] == 20 and stats["frames"] == 1928
    existing_files = list((PROJECT/"src/lerobot/policies/openvla").glob("*.py"))
    existing_files += [PROJECT/"src/lerobot/policies"/name/"__init__.py" for name in ["groot","pi0","pi05"]]
    before_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in existing_files}
    freeze = subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True).stdout
    (output/"packages_before.txt").write_text(freeze)
    config = OpenVLAConfig(model_id="openvla/openvla-7b", unnorm_key="bridge_orig", device="cuda")
    policy = TrainableOpenVLAPolicy(config, stats)
    print("POLICY_TRAINING_IMPORT_AND_LOAD_OK", flush=True)
    train_data = LeRobotDataset("local/openvla-libero-bowl-train",root=DATA/"train")
    val_data = LeRobotDataset("local/openvla-libero-bowl-validation",root=DATA/"validation")
    train_indices = episode_midpoints(split["train"])
    val_indices = episode_midpoints(split["validation"])
    train_batches = [policy.prepare_training_batch(train_data[i]) for i in train_indices]
    val_batches = [policy.prepare_training_batch(val_data[i]) for i in val_indices]
    assert len(train_batches)==20 and len(val_batches)==5
    save_json(output/"run_config.json", {"base_model":config.model_id,"base_snapshot":policy.checkpoint_path,
        "data_root":str(DATA),"steps":args.steps,"batch_size":1,"learning_rate":args.lr,"seed":7,
        "lora_rank":8,"targets":["language_model q_proj","language_model v_proj"],
        "train_frame_indices":train_indices,"validation_frame_indices":val_indices,
        "frame_selection":"one midpoint per episode; short connection test only",
        "image_preprocessing":"native Hugging Face processor on source 256px images; no augmentation; not full official RLDS preprocessing reproduction",
        "normalization":"training-20-episode q01/q99; arm mask only; constant dimensions zero",
        "loss":"cross entropy on seven action tokens plus EOS; prompt masked",
        "validation":"teacher-forced one midpoint per validation episode; no simulator rollout",
        "reference_hashes":policy.reference_hashes})
    before_lora = lora_values(policy)
    frozen_before = frozen_sample_hash(policy)
    before_metrics = {"train_probe":measure(policy,train_batches[:2]),"validation_probe":measure(policy,val_batches)}
    print("BEFORE",json.dumps(before_metrics),flush=True)
    optimizer = torch.optim.AdamW(policy.get_optim_params(),lr=args.lr,weight_decay=0.0)
    rng=np.random.default_rng(7)
    order=list(map(int,rng.permutation(len(train_batches))))
    rows=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        policy.train()
        policy.model.vision_backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = policy(train_batches[order[step%len(order)]])
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(policy.get_optim_params(),1.0)
        if not torch.isfinite(gradient_norm): raise RuntimeError("Non-finite gradient")
        optimizer.step()
        row={"step":step+1,"train_episode_index":order[step%len(order)],**metrics,"gradient_norm":float(gradient_norm)}
        rows.append(row)
        with (output/"metrics.jsonl").open("a") as handle: handle.write(json.dumps(row)+"\n")
        print("UPDATE",json.dumps(row),flush=True)
        del loss
    optimizer.zero_grad(set_to_none=True)
    after_metrics={"train_probe":measure(policy,train_batches[:2]),"validation_probe":measure(policy,val_batches)}
    trained_lora=lora_values(policy)
    deltas={n:float((trained_lora[n]-before_lora[n]).abs().max()) for n in before_lora}
    changed=sum(delta>0 for delta in deltas.values())
    assert changed>0, "No LoRA parameters changed"
    frozen_after=frozen_sample_hash(policy)
    assert frozen_after==frozen_before, "Frozen parameter samples changed"
    policy.save_training_adapter(output/"adapter")
    save_json(output/"optimizer_parameter_delta.json",deltas)
    del optimizer
    gc.collect();torch.cuda.empty_cache()
    policy.eval()
    expected_loss=measure(policy,[val_batches[0]])["loss"]
    base=policy.model.unload()
    policy.model=PeftModel.from_pretrained(base,output/"adapter",is_trainable=False)
    policy.eval()
    reloaded=lora_values(policy)
    assert set(reloaded)==set(trained_lora)
    assert all(torch.equal(reloaded[n],trained_lora[n]) for n in trained_lora), "Reloaded LoRA weights differ"
    reloaded_loss=measure(policy,[val_batches[0]])["loss"]
    assert abs(expected_loss-reloaded_loss)<1e-4,(expected_loss,reloaded_loss)
    freeze_after=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True).stdout
    (output/"packages_after.txt").write_text(freeze_after)
    assert sorted(freeze.splitlines())==sorted(freeze_after.splitlines())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before_hashes.items())
    report={"training_connection_passed":True,"optimizer_steps":args.steps,"trainable_parameters":sum(p.numel() for p in before_lora.values()),
            "changed_lora_tensors":changed,"total_lora_tensors":len(deltas),"max_parameter_delta":max(deltas.values()),
            "before":before_metrics,"after":after_metrics,"adapter_reload_weights_exact":True,
            "adapter_reload_loss_before":expected_loss,"adapter_reload_loss_after":reloaded_loss,
            "frozen_parameter_samples_unchanged":True,"existing_sources_unchanged":True,"packages_unchanged":True,
            "peak_allocated_gpu_gib":torch.cuda.max_memory_allocated()/1024**3,
            "full_training_experiment_completed":False,"simulator_evaluation_performed":False,
            "wandb_connected":False,"output":str(output)}
    save_json(output/"result.json",report)
    print("TRAINING_CONNECTION_COMPLETE",json.dumps(report),flush=True)

if __name__ == "__main__":
    main()
