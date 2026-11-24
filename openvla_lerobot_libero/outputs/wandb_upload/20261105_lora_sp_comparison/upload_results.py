"""Upload recorded experiment metrics only; no training or model-weight access."""
from pathlib import Path
from collections import defaultdict
import argparse, hashlib, json, os, time
import wandb

SOURCE = Path("/home/jovyan/projects/lerobot-openvla/outputs/lora_sp_compare/20261104T091818_826396Z")
OLD5 = Path("/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z")
ORIGIN = Path("/home/jovyan/projects/lerobot-openvla/outputs/bowl40_experiment/20261103T051914_711739Z")
PROJECT = "openvla-libero-bowl-baseline"
GROUP = "libero-bowl40-five-epochs-20261104"
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)

def verify_remote(run_path, expected, eval_only):
    for attempt in range(12):
        check = wandb.Api(timeout=30).run(run_path)
        if check.state == "finished":
            break
        time.sleep(2)
    assert check.state == "finished", check.state
    assert abs(float(check.summary["simulation/success_rate"]) - expected["simulation/success_rate"]) < 1e-9
    assert int(check.summary["simulation/attempts"]) == 5
    assert bool(check.summary["verification/finite_actions"])
    if not eval_only:
        assert abs(float(check.summary["validation/final_task_loss"]) - expected["validation/final_task_loss"]) < 1e-8
    return check

def train_history(method):
    roots = [ORIGIN, OLD5] if method == "lora8" else [SOURCE / method]
    rows = [row for root in roots for row in lines(root / "training_metrics.jsonl")]
    rows.sort(key=lambda r: r["optimizer_step"])
    assert [r["optimizer_step"] for r in rows] == list(range(1,5036))
    for epoch in range(1,6):
        assert sum(r["batch_frames"] for r in rows if r["epoch"] == epoch) == 4025
    val_root = OLD5 if method == "lora8" else SOURCE / method
    vals = read(val_root / "validation_history.json")
    assert len(vals) == 5
    return rows, vals

def metric_summary(summary, method, eta, sim, infer):
    main = next(x for x in summary["methods"] if x["method"] == method)
    result = {
        "simulation/successes": sim["successes"],
        "simulation/attempts": 5,
        "simulation/success_rate": sim["success_rate"],
        "simulation/mean_episode_steps": sim["mean_policy_steps"],
        "inference/mean_action_ms": infer["mean_action_seconds"] * 1000,
        "inference/peak_gpu_gib": infer["peak_allocated_gib"],
        "storage/adapter_mb": infer["adapter_file_bytes"] / 1e6,
        "model/trainable_parameters": main["trainable_parameters"],
        "sp/mean_active_rank": infer["rank_statistics"]["mean_active_rank"] if method == "sp128" else main["nominal_initial_rank"],
        "verification/fresh_process_reload": infer["fresh_process_reload"],
        "verification/finite_actions": all(r["finite"] and r["shape"] == [1,7] for r in infer["rows"]),
        "verification/adapter_hash_unchanged": infer["adapter_hash_exact"],
    }
    if method != "sp128" or eta == .9:
        result.update({
            "validation/final_task_loss": main["validation_loss_task_only"],
            "validation/final_action_token_accuracy": main["action_token_accuracy_teacher_forced"],
            "validation/final_normalized_action_l1": main["normalized_action_l1_teacher_forced"],
            "benchmark/train_peak_gpu_gib": main["isolated_train_peak_gib"],
            "benchmark/train_frames_per_second": main["isolated_train_frames_per_second"],
            "benchmark/train_128_frame_seconds": main["isolated_train_128_frame_seconds"],
        })
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upload", action="store_true")
    args = p.parse_args()
    summary = read(SOURCE / "summary.json")
    cfg = read(SOURCE / "experiment_config.json")
    snapshots = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        SOURCE / "summary.json", SOURCE / "experiment_config.json",
        SOURCE / "lora128/training_result.json", SOURCE / "sp128/training_result.json"
    ]}
    histories = {m: train_history(m) for m in ["lora8", "lora128", "sp128"]}
    specs = [
        ("lora8", .9, "LoRA-r8-5epoch"),
        ("lora128", .9, "LoRA-r128-5epoch"),
        ("sp128", .9, "LoRA-SP-r128-eta0.9-5epoch"),
        ("sp128", .8, "LoRA-SP-r128-eta0.8-inference-only"),
        ("sp128", .7, "LoRA-SP-r128-eta0.7-inference-only"),
    ]
    inputs = []
    for method, eta, name in specs:
        tag = method if method != "sp128" else f"sp128_eta_{round(eta*100):03d}"
        sim = read(SOURCE / ("simulation_" + tag + ".json"))
        infer = read(SOURCE / "inference" / (tag + ".json"))
        assert len(sim["trials"]) == 5
        assert len(infer["rows"]) == 32
        inputs.append((method, eta, name, tag, sim, infer))
    if not args.upload:
        print("VALIDATED", len(inputs), "runs; 3 full training histories; 25 comparison trials, 20 newly executed and 5 reused.")
        return
    api = wandb.Api(timeout=30)
    entity = api.default_entity
    assert entity, "No default entity; login first."
    manifest_path = OUT / "upload_manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else {
        "project": PROJECT, "entity": entity, "source": str(SOURCE), "runs": []
    }
    assert manifest["entity"] == entity
    for method, eta, name, tag, sim, infer in inputs:
        run_id = hashlib.sha256((str(SOURCE) + ":metrics-upload-v1:" + tag).encode()).hexdigest()[:12]
        run_path = f"{entity}/{PROJECT}/{run_id}"
        existing = next((r for r in manifest["runs"] if r["id"] == run_id and r.get("verified")), None)
        eval_only = method == "sp128" and eta != .9
        metrics = metric_summary(summary, method, eta, sim, infer)
        if existing:
            remote = verify_remote(run_path, metrics, eval_only)
            print("ALREADY_UPLOADED", name, existing["url"], flush=True)
            continue
        # A prior process may have finished uploading before its local verification.
        try:
            remote = wandb.Api(timeout=30).run(run_path)
        except (ValueError, wandb.errors.CommError) as error:
            if "Could not find run" not in str(error):
                raise
            remote = None
        if remote is not None and remote.state == "finished":
            check = verify_remote(run_path, metrics, eval_only)
            item={"name":name,"tag":tag,"id":run_id,"path":run_path,"url":check.url,"verified":True,
                  "successes":metrics["simulation/successes"],"attempts":5,"training_history_uploaded":not eval_only}
            manifest["runs"].append(item)
            write(manifest_path,manifest)
            print("RECOVERED_EXISTING_UPLOAD",name,check.url,flush=True)
            continue
        config = {
            "base_model": cfg["base_model"], "method": method,
            "rank": 8 if method == "lora8" else 128,
            "trained_epochs": 5, "batch_size": 4, "learning_rate": .0001,
            "training_demonstrations": 40, "training_frames": 4025,
            "validation_demonstrations": 5, "validation_frames": 462,
            "seed": 7, "optimizer": "AdamW", "weight_decay": 0,
            "target_modules": cfg["target_modules"], "base_dtype": "BF16", "adapter_dtype": "FP32",
            "image_preprocessing": cfg["image_preprocessing"], "normalization": cfg["normalization"],
            "checkpoint_selection": "fixed final epoch 5", "posthoc_metrics_upload": True,
            "retraining_during_upload": False, "inference_only_eta_change": eval_only,
            "training_eta": .9 if method == "sp128" else None,
            "inference_eta": eta if method == "sp128" else None,
            "simulation_initial_states": [0,1,2,3,4], "simulation_horizon": 220,
            "new_state_generalization_test": False,
            "baseline_results_reused": method == "lora8",
            "speed_protocol": summary["speed_measurement"],
            "source_run": str(SOURCE), "checkpoint": infer["checkpoint"],
            "adapter_sha256": infer["adapter_hash"],
            "paper_adaptation": cfg["sp"]["adaptation_not_exact_paper_replication"] if method == "sp128" else None,
            "dense_full_bank": method == "sp128",
        }
        settings = wandb.Settings(
            disable_git=True, disable_code=True, x_disable_stats=True, x_disable_meta=True,
            x_disable_machine_info=True, x_save_requirements=False,
            console="off", init_timeout=90, finish_timeout=120,
        )
        run = wandb.init(
            entity=entity, project=PROJECT, id=run_id, name=name, group=GROUP,
            job_type="eta-evaluation" if eval_only else "recorded-training-and-evaluation",
            tags=["posthoc-upload", "LIBERO", "same-five-states", method] + (["inference-only"] if eval_only else []),
            notes="Recorded results, not a new training run. Same previously used five initial states; one training seed. SP is an OpenVLA adaptation with a dense full bank. Smaller active rank does not imply lower memory or compute.",
            config=config, dir=str(OUT), mode="online", resume="allow", save_code=False, settings=settings,
        )
        run.define_metric("optimizer_step")
        run.define_metric("train/*", step_metric="optimizer_step")
        run.define_metric("validation/*", step_metric="optimizer_step")
        run.define_metric("gpu/*", step_metric="optimizer_step")
        run.define_metric("epoch")
        run.define_metric("epoch_metrics/*", step_metric="epoch")
        if not eval_only:
            train, vals = histories[method]
            val_by_step = {v["optimizer_step"]: v for v in vals}
            for row in train:
                n = row["optimizer_step"]
                if n % 10 and n not in val_by_step and n != 1:
                    continue
                data = {
                    "optimizer_step": n,
                    "train/loss": row["loss"],
                    "train/action_token_accuracy": row["action_token_accuracy"],
                    "train/normalized_action_l1": row["normalized_action_l1"],
                    "train/gradient_norm": row["gradient_norm"],
                    "train/frames_seen": row["frames_seen"],
                    "gpu/allocated_gib": row["gpu_allocated_gib"],
                    "gpu/peak_allocated_gib": row["gpu_peak_allocated_gib"],
                }
                for src,dst in [("total_loss","total_loss"),("spectral_loss","spectral_loss"),("router_loss","router_loss")]:
                    if src in row:data["train/"+dst] = row[src]
                if n in val_by_step:
                    v=val_by_step[n]
                    e=v["epoch"]
                    erows=[x for x in train if x["epoch"] == e]
                    data.update({
                        "epoch": e,
                        "validation/loss": v["loss"],
                        "validation/action_token_accuracy": v["action_token_accuracy"],
                        "validation/normalized_action_l1": v["normalized_action_l1"],
                        "epoch_metrics/training_loss": sum(x["loss"]*x["batch_frames"] for x in erows)/4025,
                        "epoch_metrics/validation_loss": v["loss"],
                    })
                    if "mean_active_rank" in v:data["epoch_metrics/mean_active_rank"] = v["mean_active_rank"]
                run.log(data)
        run.log(metrics)
        for key,value in metrics.items():run.summary[key] = value
        trial_rows = [[t["init_state_index"], bool(t["success"]), t["policy_steps"], t["termination_reason"]] for t in sim["trials"]]
        run.log({"simulation/trials": wandb.Table(columns=["initial_state","success","steps","termination_reason"], data=trial_rows)})
        if method == "sp128" and eta == .9:
            mtable = wandb.Table(
                columns=["method","success_rate","inference_ms","inference_gpu_gib","training_gpu_gib","adapter_mb"],
                data=[[m["method"],m["success_rate"],m["mean_inference_ms"],m["inference_peak_gib"],m["isolated_train_peak_gib"],m["adapter_file_mb"]] for m in summary["methods"]]
            )
            etable = wandb.Table(columns=["eta","active_rank","success_rate","inference_ms","inference_gpu_gib"],
                data=[[e["eta"],e["mean_active_rank"],e["success_rate"],e["mean_inference_ms"],e["inference_peak_gib"]] for e in summary["eta_inference_ablation"]])
            run.log({
                "comparison/methods": mtable, "comparison/eta": etable,
                "comparison/success_rate": wandb.plot.bar(mtable,"method","success_rate",title="Same five initial states: successes"),
                "comparison/inference_ms": wandb.plot.bar(mtable,"method","inference_ms",title="Isolated action prediction latency"),
            })
            bundle=SOURCE/"viewing_bundle"
            run.log({key:wandb.Image(str(bundle/file)) for key,file in [
                ("figures/resources","05_method_resources.png"),
                ("figures/eta","06_eta_effects.png"),
                ("figures/learning_curves","09_learning_curves.png"),
            ]})
        run.finish()
        check = verify_remote(run_path, metrics, eval_only)
        item={"name":name,"tag":tag,"id":run_id,"path":run_path,"url":check.url,"verified":True,
            "successes":metrics["simulation/successes"],"attempts":5,"training_history_uploaded":not eval_only}
        manifest["runs"]=[r for r in manifest["runs"] if r["id"]!=run_id] + [item]
        write(manifest_path,manifest)
        print("UPLOADED_AND_VERIFIED",json.dumps(item),flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in snapshots.items())
    manifest["complete"] = len(manifest["runs"]) == 5
    manifest["original_summary_and_results_unchanged"] = True
    manifest["project_url"] = api.project(PROJECT,entity=entity).url
    write(manifest_path,manifest)
    print("UPLOAD_COMPLETE",manifest["project_url"],flush=True)

if __name__ == "__main__":
    main()

