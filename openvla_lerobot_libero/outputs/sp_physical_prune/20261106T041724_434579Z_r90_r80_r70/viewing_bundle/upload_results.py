"""Append three evaluations and a unified view of every existing experiment."""
from pathlib import Path
import json,hashlib,time,csv
import wandb
R=Path(__file__).resolve().parent
ENTITY="ksja01091-gwangju-institute-of-science-and-technology"
PROJECT="openvla-libero-bowl-baseline"
EXPECTED_OLD={"d6fe0857012e","af4baa672bec","98d6d9c9239a","e300a98fee5f","1ea481872ad4","72b82eb051a0","102d4d0f0267","63b67f9184b2"}
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    t=p.with_suffix(".tmp");t.write_text(json.dumps(d,indent=2,ensure_ascii=False));t.replace(p)
def settings():return wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,x_disable_meta=True,
    x_disable_machine_info=True,x_save_requirements=False,console="off",init_timeout=90,finish_timeout=120)
def verify(path,metrics):
    for _ in range(15):
        remote=wandb.Api(timeout=30).run(path)
        if remote.state=="finished":break
        time.sleep(2)
    assert remote.state=="finished",remote.state
    for k,v in metrics.items():assert abs(float(remote.summary[k])-float(v))<1e-7,k
    return remote
def main():
    assert read(R/"report_complete.json")["complete"]
    data=read(R/"summary.json");api=wandb.Api(timeout=30)
    assert api.default_entity==ENTITY
    access=api._service_api.execute_graphql("query ProjectAccess($entity: String!, $project: String!) { project(name: $project, entityName: $entity) { name access } }",{"entity":ENTITY,"project":PROJECT})
    assert access["project"]["access"]=="PRIVATE"
    before={r.id:r for r in api.runs(f"{ENTITY}/{PROJECT}",per_page=50)}
    assert EXPECTED_OLD.issubset(before),"A previously uploaded experiment is missing"
    assert all(before[x].state=="finished" for x in EXPECTED_OLD)
    manifest_path=R/"wandb_upload_manifest.json"
    manifest=read(manifest_path) if manifest_path.exists() else {"entity":ENTITY,"project":PROJECT,"runs":[]}
    for rank in [90,80,70]:
        row=next(x for x in data["results"] if x["physical_rank"]==rank)
        tag=row["variant"];bench=read(R/"benchmarks"/tag/"result.json")
        assert all(x["finite"] and x["shape"]==[1,7] for x in bench["rows"])
        metrics={"simulation/successes":row["successes"],"simulation/attempts":5,
            "simulation/success_rate":row["success_rate"],"simulation/mean_episode_steps":row["mean_episode_steps"],
            "inference/mean_action_ms":row["inference_ms"],"inference/round_mean_std_ms":row["round_std_ms"],
            "inference/peak_gpu_gib":row["gpu_gib"],"storage/adapter_mb":row["adapter_mb"],
            "sp/physical_bank_rank":rank,"sp/mean_active_rank":row["mean_active_rank"],
            "validation/final_task_loss":row["validation_loss"],"validation/final_action_token_accuracy":row["action_token_accuracy"],
            "validation/final_normalized_action_l1":row["normalized_action_l1"],
            "verification/finite_actions":True,"verification/fresh_process_reload":True}
        run_id=hashlib.sha256((str(R)+":rank-sweep-v1:"+tag).encode()).hexdigest()[:12]
        path=f"{ENTITY}/{PROJECT}/{run_id}"
        if any(x["id"]==run_id for x in manifest["runs"]):verify(path,metrics);continue
        if run_id in before and before[run_id].state=="finished":remote=verify(path,metrics)
        else:
            cfg=dict(data["config"]);cfg.update({"physical_bank_rank":rank,"checkpoint":row["checkpoint"],"retraining":False,"simulation_results_reused":False})
            run=wandb.init(entity=ENTITY,project=PROJECT,id=run_id,name=f"SP{rank}-physical-pruning-no-retraining",
                group="sp-physical-rank-sweep-20261106",job_type="pruning-evaluation",
                tags=["physical-pruning","rank-sweep","evaluation-only","same-five-states",tag],
                config=cfg,dir=str(R),resume="allow",save_code=False,settings=settings(),
                notes="Same original SP128 checkpoint, original 128-score router retained. Reused training-only calibration; no retraining. Five previously used states, not new-state generalization.")
            run.log(metrics)
            for k,v in metrics.items():run.summary[k]=v
            s=read(R/("simulation_"+tag+".json"))
            run.log({"simulation/trials":wandb.Table(columns=["initial_state","success","steps","termination_reason"],
                data=[[t["init_state_index"],t["success"],t["policy_steps"],t["termination_reason"]] for t in s["trials"]])})
            run.finish();remote=verify(path,metrics)
        manifest["runs"].append({"id":run_id,"name":remote.name,"url":remote.url,"verified":True})
        write(manifest_path,manifest);print("UPLOADED_AND_VERIFIED",rank,remote.url,flush=True)
    allruns={r.id:r for r in wandb.Api(timeout=30).runs(f"{ENTITY}/{PROJECT}",per_page=50)}
    ids=EXPECTED_OLD|{x["id"] for x in manifest["runs"]}
    assert len(ids)==11 and all(allruns[x].state=="finished" for x in ids)
    allrows=[]
    for run_id in sorted(ids):
        r=allruns[run_id];s=r.summary
        allrows.append([r.name,s.get("simulation/success_rate"),s.get("inference/mean_action_ms"),
                        s.get("inference/peak_gpu_gib"),s.get("storage/adapter_mb"),r.url])
    overview_id=hashlib.sha256((str(R)+":all-experiments-overview-v1").encode()).hexdigest()[:12]
    overview_path=f"{ENTITY}/{PROJECT}/{overview_id}"
    if not manifest.get("overview"):
        if overview_id in allruns and allruns[overview_id].state=="finished":
            remote=verify(overview_path,{"verification/completed_experiments":11})
        else:
            overview=wandb.init(entity=ENTITY,project=PROJECT,id=overview_id,name="00-ALL-EXPERIMENTS-COMPARISON",
                group="sp-physical-rank-sweep-20261106",job_type="comparison-overview",
                tags=["overview","all-experiments","physical-pruning","rank-sweep"],
                config={"source_run":str(R),"no_training":True,"new_robot_trials":15,"five_old_states_only":True,
                        "prior_96_64_benchmarks_reused":True},dir=str(R),resume="allow",save_code=False,settings=settings())
            ranktable=wandb.Table(columns=["rank","success_rate","successes","inference_ms","gpu_gib","adapter_mb","validation_loss","resources_reused"],
                data=[[x["physical_rank"],x["success_rate"],x["successes"],x["inference_ms"],x["gpu_gib"],x["adapter_mb"],x["validation_loss"],x["benchmark_reused"]] for x in data["results"]])
            everything=wandb.Table(columns=["experiment","success_rate","inference_ms","gpu_gib","adapter_mb","run_url"],data=allrows)
            overview.log({"comparison/all_experiments":everything,"comparison/physical_ranks":ranktable,
                "comparison/success_rate":wandb.plot.bar(ranktable,"rank","success_rate",title="Physical rank: success on five old states"),
                "comparison/adapter_mb":wandb.plot.bar(ranktable,"rank","adapter_mb",title="Physically saved adapter size (MB)"),
                "comparison/inference_ms":wandb.plot.bar(ranktable,"rank","inference_ms",title="Prediction time; 96/64 reused from prior date"),
                "comparison/gpu_gib":wandb.plot.bar(ranktable,"rank","gpu_gib",title="Inference peak allocated GPU memory"),
                "figures/rank_comparison":wandb.Image(str(R/"viewing_bundle/04_rank_results.png"))})
            for state in range(5):
                overview.log({f"videos/state_{state:02d}_SP96_SP90_SP80_SP70":wandb.Video(str(R/f"viewing_bundle/05_compare_state_{state:02d}.mp4"),format="mp4")})
            overview.summary["verification/completed_experiments"]=11
            overview.finish();remote=verify(overview_path,{"verification/completed_experiments":11})
        manifest["overview"]={"id":overview_id,"url":remote.url,"verified":True}
    manifest.update({"complete":True,"verified_experiment_count":11,"previous_eight_verified":True})
    write(manifest_path,manifest)
    with (R/"viewing_bundle/07_all_experiments.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["experiment","success_rate","inference_ms","gpu_gib","adapter_mb","run_url"]);w.writerows(allrows)
    (R/"viewing_bundle/wandb_upload_manifest.json").write_bytes(manifest_path.read_bytes())
    (R/"viewing_bundle/06_WANDB_OPEN.txt").write_text("W&B 전체 비교 화면\n"+manifest["overview"]["url"]+"\n\n이름: 00-ALL-EXPERIMENTS-COMPARISON\n기존 8개 + 새 rank90/80/70 3개, 총 11개 실험 비교표. rank 비교 그래프와 영상 5개 포함.\n",encoding="utf-8")
    bundle=R/"viewing_bundle"
    write(bundle/"manifest.json",{p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in bundle.iterdir() if p.is_file() and p.name!="manifest.json"})
    print("ALL_EXPERIMENTS_VISIBLE",json.dumps(manifest),flush=True)
if __name__=="__main__":main()
