"""Reproduce the 40-train/5-validation bowl experiment without modifying prior runs.
Usage: python examples/run_openvla_bowl40.py prepare
       python examples/run_openvla_bowl40.py calibrate --run RUN
       python examples/run_openvla_bowl40.py finish --run RUN
"""
import argparse, hashlib, json, math, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path("/home/jovyan/projects/lerobot-openvla")
OLD=ROOT/"outputs/full_bowl_experiment/20261102T091448_811738Z"
OLD_DATA=Path("/home/jovyan/projects/datasets/libero_bowl_25/20261101T120613_454595Z")
MAIN=ROOT/"examples/openvla_bowl40_experiment.py"

def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def invoke(run,phase,command):
    log=run/(phase+".log")
    attempt=1
    while log.exists():
        log=run/(phase+f".retry{attempt}.log")
        attempt+=1
    env=os.environ.copy()
    env.update(OPENVLA_BOWL40_DATA_ROOT=str(run/"data"),WANDB_MODE="offline",
               HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    with log.open("x") as handle:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        write(run/"active_process.json",{"phase":phase,"pid":p.pid,"log":str(log)})
        print("PHASE_STARTED",phase,p.pid,flush=True)
        code=p.wait()
    if code:
        print(log.read_text()[-5000:],flush=True)
        raise RuntimeError(f"{phase} failed: exit {code}; see {log}")
    print("PHASE_COMPLETE",phase,flush=True)

def prepare():
    from prepare_libero_bowl_data import export_split, verify_split
    from run_openvla_bowl_full import capture_environment
    from concurrent.futures import ThreadPoolExecutor
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run=ROOT/"outputs/bowl40_experiment"/stamp
    data=run/"data"
    data.mkdir(parents=True,exist_ok=False)
    old=json.loads((OLD_DATA/"split_manifest.json").read_text())
    inventory=json.loads((OLD_DATA/"inventory.json").read_text())
    old_ids={r["content_id"] for r in old["train"]}
    val_ids={r["content_id"] for r in old["validation"]}
    candidates=inventory["candidates"]
    assert len(candidates)==45
    remaining=sorted([r for r in candidates if r["content_id"] not in old_ids|val_ids],
                     key=lambda r:r["content_id"])
    train=old["train"]+remaining
    validation=old["validation"]
    assert len(train)==40 and len(validation)==5
    assert not {r["content_id"] for r in train}&val_ids
    assert {r["content_id"] for r in train}|val_ids=={r["content_id"] for r in candidates}
    split={**old,"train":train,"validation":validation,"train_episodes":40,
           "train_frames":sum(r["frames"] for r in train),
           "selection":"Original 20 training demonstrations plus all 20 previously unused target demonstrations. Original five validation demonstrations unchanged.",
           "normalization_note":"Freeze the original train20 statistics for comparable labels, metrics and action decoding. No validation-derived calibration."}
    write(data/"split_manifest.json",split)
    (data/"action_statistics_train_only.json").write_bytes((OLD_DATA/"action_statistics_train_only.json").read_bytes())
    write(data/"normalization_provenance.json",{"source":str(OLD_DATA/"action_statistics_train_only.json"),
          "sha256":digest(OLD_DATA/"action_statistics_train_only.json"),"calibration_episodes":20,
          "training_episodes":40,"validation_used":False,"reason":"hold action normalization fixed against the original experiment"})
    repo,output=export_split(data,"train",train,split["task"])
    checks={"train":verify_split(repo,output,train,split["task"])}
    (data/"validation").symlink_to(OLD_DATA/"validation",target_is_directory=True)
    checks["validation"]=verify_split("local/openvla-libero-bowl-validation",data/"validation",validation,split["task"])
    write(data/"verification_report.json",checks)
    sources=[*ROOT.joinpath("src").rglob("*.py"),*ROOT.joinpath("examples").rglob("*.py")]
    write(run/"sources_before.json",{str(p):digest(p) for p in sources})
    interpreters={"openvla310":"/home/jovyan/anaconda3/envs/openvla310/bin/python",
                  "lerobot312":sys.executable,"eval310":"/home/jovyan/projects/venvs/openvla_eval310/bin/python"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        environment=dict(pool.map(lambda item:capture_environment(run,item),interpreters.items()))
    write(run/"environment_audit.json",environment)
    old_data_files=[*OLD_DATA.rglob("*.json"),*OLD_DATA.rglob("*.parquet"),*OLD_DATA.rglob("*.npz")]
    write(run/"old_data_before.json",{str(p):digest(p) for p in old_data_files})
    write(run/"comparison_plan.json",{
        "previous_run":str(OLD),"fresh_start_base":"openvla/openvla-7b",
        "train_demonstrations":40,"train_frames":split["train_frames"],"validation_demonstrations":5,
        "validation_frames":462,"epochs":1,"batch_size":4,"learning_rate":1e-4,
        "lora_rank":8,"lora_alpha":8,"lora_dropout":0,"seed":7,
        "optimizer_steps":math.ceil(split["train_frames"]/4),
        "validation_every_steps":math.ceil(math.ceil(split["train_frames"]/4)/2),
        "same_calibration":True,"same_original_validation_ids":True,
        "same_simulator_states":[0,1,2,3,4],"horizon":220,"warmup_steps":10,
        "prior_results_reused":["base","reference"],
        "interpretation":"One epoch with more data also means more optimizer updates; not an equal-compute data ablation.",
        "lift_definition":"Report continuous maximum bowl height rise; 2cm is only a diagnostic threshold, not proof of stable grasp."})
    print("PREPARED_RUN",run,flush=True)
    print("DATA_COUNTS",len(train),split["train_frames"],len(validation),462,flush=True)
    return run

def phase(run,name,extra=None):
    plan=json.loads((run/"comparison_plan.json").read_text())
    command=[sys.executable,"-u",str(MAIN),name,"--run",str(run),
             "--validate-every",str(plan["validation_every_steps"])]
    invoke(run,name+("_"+extra if extra else ""),command+(["--model",extra] if extra else []))

def finish(run):
    # No duplicates or partial-training overwrite.
    assert not (run/"pipeline_complete.json").exists()
    if not (run/"training_result.json").exists():
        assert not (run/"training_metrics.jsonl").exists(),"Partial training preserved"
        phase(run,"train")
    if not (run/"checkpoint_reload_check.json").exists():phase(run,"check")
    # Reuse actual prior evaluations only when fixed calibration and settings match.
    assert digest(run/"data/action_statistics_train_only.json")==digest(OLD_DATA/"action_statistics_train_only.json")
    for tag in ["base","reference"]:
        cohort=json.loads((OLD/f"simulation_{tag}.json").read_text())
        assert cohort["states"]==[0,1,2,3,4] and cohort["technical_errors"]==0
        assert cohort["horizon"]==220 and cohort["warmup_steps"]==10
        (run/"simulation"/tag).parent.mkdir(exist_ok=True)
        if not (run/"simulation"/tag).exists():(run/"simulation"/tag).symlink_to(OLD/"simulation"/tag,target_is_directory=True)
        write(run/f"simulation_{tag}.json",{**cohort,"reused_from":str(OLD),"new_trials_executed":False})
    if not (run/"simulation_lora.json").exists():phase(run,"simulate","lora")
    sim=json.loads((run/"simulation_lora.json").read_text())
    assert sim["technical_errors"]==0 and sim["attempts"]==5
    env=os.environ.copy()
    env["PYTHONPATH"]="/home/jovyan/projects/LIBERO"+(":"+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MUJOCO_GL"]="egl"
    with (run/"motion_analysis.log").open("w") as log:
        subprocess.run(["/home/jovyan/anaconda3/envs/openvla310/bin/python","-u",
                        str(ROOT/"examples/analyze_openvla_bowl_motion.py"),"--run",str(run),"--models","lora"],
                       env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    phase(run,"audit")
    old_data=json.loads((run/"old_data_before.json").read_text())
    changes=[p for p,h in old_data.items() if digest(p)!=h]
    assert not changes
    write(run/"old_data_preservation.json",{"files_checked":len(old_data),"changed_files":changes})
    summarize(run)

def summarize(run):
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        report_python="/home/jovyan/anaconda3/envs/openvla310/bin/python"
        report_code="import sys;from pathlib import Path;sys.path.insert(0,'/home/jovyan/projects/lerobot-openvla/examples');from run_openvla_bowl40 import summarize;summarize(Path(sys.argv[1]))"
        subprocess.run([report_python,"-c",report_code,str(run)],check=True)
        return
    import csv
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan=json.loads((run/"comparison_plan.json").read_text())
    new=json.loads((run/"training_result.json").read_text())
    previous=json.loads((OLD/"training_result.json").read_text())
    reload=json.loads((run/"checkpoint_reload_check.json").read_text())
    preservation=json.loads((run/"preservation_report.json").read_text())
    motion=json.loads((run/"failure_motion_analysis.json").read_text())
    rows=[]
    for label,folder in [("base",OLD),("train20",OLD),("train40",run)]:
        tag="base" if label=="base" else "lora"
        cohort=json.loads((folder/f"simulation_{tag}.json").read_text())
        for trial in cohort["trials"]:
            rows.append({"model":label,"state":trial["init_state_index"],"success":trial["success"],
                         "policy_steps":trial["policy_steps"],"video":trial["video"]})
    with (run/"simulation_comparison.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=[]
    for line in (run/"training_metrics.jsonl").read_text().splitlines():
        metrics.append(json.loads(line))
    with (run/"training_metrics.csv").open("w",newline="") as handle:
        keys=[k for k in metrics[0] if k!="indices"]
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in metrics)
    validation=[new["baseline_validation"]]+[json.loads(line) for line in (run/"validation_history.jsonl").read_text().splitlines()]
    with (run/"validation_metrics.csv").open("w",newline="") as handle:
        keys=["optimizer_step","frames","loss","action_token_accuracy","normalized_action_l1"]
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in validation)
    videos=[]
    for trial in json.loads((run/"simulation_lora.json").read_text())["trials"]:
        with imageio.get_reader(trial["video"]) as reader:
            n=reader.count_frames();assert n==trial["policy_steps"]+1
            reader.get_data(0);reader.get_data(n-1)
            videos.append({"video":trial["video"],"frames":n,"fps":reader.get_meta_data()["fps"]})
    write(run/"video_verification.json",videos)
    baseline_delta=abs(new["baseline_validation"]["loss"]-previous["baseline_validation"]["loss"])
    assert baseline_delta<1e-4,baseline_delta
    assert new["frames_seen"]==plan["train_frames"]
    coverage=json.loads((run/"training_coverage.json").read_text())
    assert len(coverage)==1 and coverage[0]["unique_frames"]==plan["train_frames"]
    assert reload["all_actions_finite"] and reload["action_shape"]==[1,7]
    assert preservation["existing_sources_unchanged"] and all(v["packages_unchanged"] for v in preservation["environments"].values())
    summary={"plan":plan,"baseline_validation_loss_delta_vs_prior":baseline_delta,
             "prior_train20_validation":previous["selected_validation"],"train40_validation":new["selected_validation"],
             "training_seconds":new["training_loop_wall_seconds"],"gpu_peak_gib":new["gpu_peak_allocated_gib"],
             "selected_checkpoint":new["selected_checkpoint"],"fresh_reload":reload,
             "simulation":{label:{"successes":sum(r["success"] for r in rows if r["model"]==label),
                                  "attempts":5,"mean_steps":float(np.mean([r["policy_steps"] for r in rows if r["model"]==label]))}
                           for label in ["base","train20","train40"]},
             "train40_bowl_motion":motion,"limitations":["five fixed states only","demonstration/state disjointness not established",
                "2cm rise is not proof of grasp","data count and optimizer-update count both increased"],
             "wandb":"offline only; no credentials uploaded"}
    write(run/"comparison_summary.json",summary)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].plot([r["optimizer_step"] for r in metrics],[r["loss"] for r in metrics],alpha=.6)
    axes[0].set(title="Train40 batch loss",xlabel="Optimizer step")
    axes[1].bar(["Base","Train20","Train40"],[new["baseline_validation"]["loss"],previous["selected_validation"]["loss"],new["selected_validation"]["loss"]])
    axes[1].set(title="Same held-out validation loss")
    axes[2].bar(["Base","Train20","Train40"],[summary["simulation"][label]["successes"]/5 for label in ["base","train20","train40"]])
    axes[2].set(title="Simulator success (5 states)",ylim=(0,1))
    fig.tight_layout();fig.savefig(run/"comparison.png",dpi=160);plt.close(fig)
    write(run/"pipeline_complete.json",{"complete":True,"evidence":["full training coverage","fresh reload","five no-error simulator trials","five verified videos","motion replay parity","source and environment preservation"]})
    print("EXPERIMENT_COMPLETE",json.dumps(summary),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["prepare","calibrate","finish"])
    parser.add_argument("--run",type=Path)
    args=parser.parse_args()
    if args.phase=="prepare":prepare();return
    assert args.run and args.run.is_dir()
    try:
        if args.phase=="calibrate":phase(args.run,"calibrate")
        else:finish(args.run)
    except Exception as error:
        import traceback
        write(args.run/(args.phase+"_error.json"),{"error":repr(error),"traceback":traceback.format_exc()})
        raise

if __name__=="__main__":main()
