"""Record completed physical pruning evaluations; never load or train a model."""
from pathlib import Path
import hashlib, json, time
import wandb

R = Path(__file__).resolve().parent
PROJECT = "openvla-libero-bowl-baseline"
ENTITY = "ksja01091-gwangju-institute-of-science-and-technology"

def read(path):
    return json.loads(Path(path).read_text())

def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)

def verify(path, metrics):
    for _ in range(12):
        remote = wandb.Api(timeout=30).run(path)
        if remote.state == "finished":
            break
        time.sleep(2)
    assert remote.state == "finished", remote.state
    for key, value in metrics.items():
        assert abs(float(remote.summary[key]) - float(value)) < 1e-7, key
    return remote

def main():
    assert read(R / "report_complete.json")["complete"]
    summary = read(R / "summary.json")
    assert not summary["preservation"]["changed_files"]
    assert all(summary["preservation"]["environments_unchanged"].values())
    protected = {p: hashlib.sha256((R/p).read_bytes()).hexdigest()
                 for p in ["summary.json", "experiment_config.json"]}
    api = wandb.Api(timeout=30)
    entity = api.default_entity
    assert entity == ENTITY, "Expected the existing experiment workspace account."
    access = api._service_api.execute_graphql(
        "query ProjectAccess($entity: String!, $project: String!) { project(name: $project, entityName: $entity) { name access } }",
        {"entity": entity, "project": PROJECT})
    assert access["project"]["access"] == "PRIVATE", access
    manifest_path = R / "wandb_upload_manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else {
        "project": PROJECT, "entity": entity, "runs": []}
    assert manifest["entity"] == entity
    names = {"sp_full128": "SP128-physical-pruning-control",
             "sp_pruned96": "SP96-physical-pruning-no-retraining",
             "sp_pruned64": "SP64-physical-pruning-no-retraining"}
    for row in summary["results"]:
        tag = row["variant"]
        bench = read(R / "benchmarks" / tag / "result.json")
        sim = read(R / ("simulation_"+tag+".json"))
        finite = all(x["finite"] and x["shape"] == [1, 7] for x in bench["rows"])
        assert finite and row["fresh_process_reload"]
        metrics = {
            "simulation/successes": row["successes"],
            "simulation/attempts": 5,
            "simulation/success_rate": row["success_rate"],
            "simulation/mean_episode_steps": row["mean_episode_steps_including_failures"],
            "inference/mean_action_ms": row["inference_ms"],
            "inference/round_mean_std_ms": row["round_mean_std_ms"],
            "inference/peak_gpu_gib": row["inference_gpu_gib"],
            "storage/adapter_mb": row["adapter_mb"],
            "storage/adapter_reduction_percent": row["adapter_reduction_percent"],
            "model/adapter_parameters": row["adapter_parameters"],
            "sp/physical_bank_rank": row["physical_rank"],
            "sp/router_width": 128,
            "sp/mean_active_rank": row["mean_selected_kept_directions"],
            "validation/final_task_loss": row["validation_loss"],
            "validation/final_action_token_accuracy": row["validation_action_token_accuracy"],
            "validation/final_normalized_action_l1": row["validation_normalized_action_l1"],
            "verification/finite_actions": finite,
            "verification/fresh_process_reload": row["fresh_process_reload"],
        }
        run_id = hashlib.sha256((str(R)+":prune-eval-v1:"+tag).encode()).hexdigest()[:12]
        path = f"{entity}/{PROJECT}/{run_id}"
        existing = next((x for x in manifest["runs"] if x["id"] == run_id), None)
        if existing:
            verify(path, metrics)
            continue
        config = dict(summary["config"])
        config.update({"variant": tag, "physical_bank_rank": row["physical_rank"],
            "original_router_width": 128, "retraining": False,
            "calibration_source": "160 frames from 40 training demonstrations only",
            "validation_used_for_direction_selection": False,
            "simulation_results_reused": tag == "sp_full128",
            "new_robot_trials": row["new_robot_trials"],
            "checkpoint": row["checkpoint"], "source_run": str(R),
            "timing_protocol": "One model at a time; same 32 images x 3 rounds; CUDA synchronized",
            "new_state_generalization_test": False})
        run = wandb.init(entity=entity, project=PROJECT, id=run_id,
            name=names[tag], group="sp-physical-pruning-20261105", job_type="pruning-evaluation",
            tags=["physical-pruning", "evaluation-only", "same-five-states", tag],
            notes="Post-training static A/B removal. Full 128-score router and base 7B preserved. Five old states only. Control robot trials reused. No retraining, no learning curves.",
            config=config, dir=str(R), mode="online", resume="allow", save_code=False,
            settings=wandb.Settings(disable_git=True, disable_code=True,
                x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True,
                x_save_requirements=False, console="off", init_timeout=90, finish_timeout=120))
        run.log(metrics)
        for key, value in metrics.items():
            run.summary[key] = value
        run.log({"simulation/trials": wandb.Table(
            columns=["initial_state", "success", "steps", "termination_reason"],
            data=[[t["init_state_index"], t["success"], t["policy_steps"], t["termination_reason"]]
                  for t in sim["trials"]])})
        if tag == "sp_pruned64":
            run.log({"figures/physical_pruning": wandb.Image(str(R/"viewing_bundle/04_results.png"))})
        run.finish()
        remote = verify(path, metrics)
        item = {"id": run_id, "name": names[tag], "path": path,
                "url": remote.url, "verified": True}
        manifest["runs"].append(item)
        save(manifest_path, manifest)
        print("UPLOADED_AND_VERIFIED", json.dumps(item), flush=True)
    assert all(hashlib.sha256((R/p).read_bytes()).hexdigest() == h for p,h in protected.items())
    manifest["complete"] = len(manifest["runs"]) == 3
    save(manifest_path, manifest)
    print("PRUNING_UPLOAD_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
