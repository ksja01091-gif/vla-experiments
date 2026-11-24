"""Upload measured historical results without rerunning or modifying experiments."""
from pathlib import Path
import json,hashlib,time,csv,argparse
import wandb
R=Path(__file__).resolve().parent
ROOT=Path("/home/jovyan/projects/lerobot-openvla/outputs")
ENTITY="ksja01091-gwangju-institute-of-science-and-technology"
PROJECT="openvla-libero-bowl-baseline"
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    t=p.with_suffix(".tmp");t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));t.replace(p)
def settings():return wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,x_disable_meta=True,
    x_disable_machine_info=True,x_save_requirements=False,console="off",init_timeout=90,finish_timeout=120)
def verify(path,metrics):
    for _ in range(15):
        obj=wandb.Api(timeout=30).run(path)
        if obj.state=="finished":break
        time.sleep(2)
    assert obj.state=="finished"
    for k,v in metrics.items():assert abs(float(obj.summary[k])-float(v))<1e-7,(path,k)
    return obj
def sim_metrics(s):
    assert s["attempts"]==5 and len(s["trials"])==5
    assert sum(bool(t["success"]) for t in s["trials"])==s["successes"]
    return {"simulation/attempts":5,"simulation/successes":s["successes"],
            "simulation/success_rate":s["success_rate"],"simulation/mean_episode_steps":s["mean_policy_steps"]}
def val_metrics(v):
    return {"validation/final_task_loss":v["loss"],"validation/final_action_token_accuracy":v["action_token_accuracy"],
            "validation/final_normalized_action_l1":v["normalized_action_l1"]}
def make_specs():
    specs=[]
    ten=ROOT/"ten_epoch_bowl/20261104T075816_073454Z"
    summary=read(ten/"summary.json")
    for x in summary["rows"]:
        e=x["epoch"]
        if e==5:continue  # Existing LoRA-r8-5epoch run already contains this checkpoint.
        cp=Path(x["checkpoint"]);assert cp.is_dir()
        sim=read(x["evaluation_source"]);metrics=sim_metrics(sim)
        metrics.update({"epoch":e,"validation/final_task_loss":x["validation_loss"],
            "validation/final_action_token_accuracy":x["validation_action_token_accuracy"],
            "validation/final_normalized_action_l1":x["validation_normalized_action_l1_teacher_forced"],
            "storage/adapter_mb":(cp/"adapter_model.safetensors").stat().st_size/1e6})
        if x["training_loss"] is not None:metrics["epoch_metrics/training_loss"]=x["training_loss"]
        if e==10:
            b=read(ten/"epoch10_inference_benchmark.json")
            assert b["samples"]==32 and all(t["finite"] and t["shape"]==[1,7] for t in b["rows"])
            metrics.update({"inference/mean_action_ms":b["mean_action_seconds"]*1000,"inference/peak_gpu_gib":b["peak_allocated_gib"]})
        cfg={"family":"epoch-checkpoint","epoch":e,"rank":8,"training_demonstrations":40,"training_frames":4025,
             "validation_frames":462,"checkpoint":str(cp),"source_result":x["evaluation_source"],
             "same_training_continuation_not_independent_training_runs":True,"recording_only_no_training":True,
             "preprocessing":"matched upright native 256 RGB","same_five_old_states":True}
        specs.append((f"epoch{e}-train40",f"LoRA-r8-{e}epoch-train40",cfg,metrics,sim))
    quant=ROOT/"batch_quantization/20261103T121901_577159Z"
    qc=read(quant/"experiment_config.json");qs=read(quant/"summary.json")
    batch=[read(quant/f"batch_{n}.json") for n in [4,8]]
    assert batch[0]["same_measured_frame_indices"]==batch[1]["same_measured_frame_indices"]
    for b in batch:
        assert b["measured_frames"]==128 and b["warmup_frames"]==16 and not b["checkpoint_saved"]
        n=b["batch_size"]
        cfg={"family":"batch-microbenchmark","batch_size":n,"measured_frames":128,"warmup_frames":16,
             "checkpoint":b["starting_checkpoint"],"quality_assessed":False,"full_retraining":False,
             "single_measurement":True,"source_result":str(quant/f"batch_{n}.json")}
        metrics={"benchmark/train_frames_per_second":b["frames_per_second"],"benchmark/train_128_frame_seconds":b["wall_seconds"],
                 "benchmark/train_peak_gpu_gib":b["peak_allocated_gib"]}
        specs.append((f"batch{n}-micro128",f"Batch{n}-128frames-speed-only",cfg,metrics,None))
    for mode in ["bf16","int8"]:
        b=read(quant/f"inference_{mode}.json")
        assert b["sample_count"]==32 and b["all_actions_finite"]
        sim=read(quant/"simulation_int8.json") if mode=="int8" else read(ROOT/"five_epoch_bowl/20261103T064207_586519Z/simulation_epoch5.json")
        metrics=sim_metrics(sim);metrics.update({"inference/mean_action_ms":b["mean_action_seconds"]*1000,
            "inference/peak_gpu_gib":b["peak_allocated_gib"],"model/in_memory_tensor_bytes":b["model_tensor_footprint_bytes"],
            "model/quantized_linear_count":b["quantized_linear_count"]})
        cfg=dict(qc);cfg.update({"family":"quantization","mode":mode,"source_result":str(quant/f"inference_{mode}.json"),
            "recording_only_no_training":True,"bf16_simulation_reused":mode=="bf16","parameter_count_reduction":False,
            "on_disk_base_model_compressed":False,"single_speed_measurement_date":"2026-11-03"})
        specs.append(("quant-"+mode,"LoRA-r8-5epoch-"+mode.upper()+"-quant-comparison",cfg,metrics,sim))
    full=ROOT/"full_bowl_experiment/20261102T091448_811738Z"
    f=read(full/"experiment_summary.json")
    names={"base":"OpenVLA-base-no-finetuning","lora":"LoRA-r8-1epoch-train20","reference":"REFERENCE-published-LIBERO-Spatial"}
    for key in ["base","lora","reference"]:
        sim=read(full/("simulation_"+key+".json"));metrics=sim_metrics(sim)
        if key=="base":metrics.update(val_metrics(f["training"]["baseline_validation"]))
        if key=="lora":metrics.update(val_metrics(f["training"]["final_validation"]))
        cfg={"family":"initial-train20-study","role":key,"model_id":sim["model_id"],"checkpoint":sim["checkpoint"],
             "source_result":str(full/("simulation_"+key+".json")),"training_demonstrations":20 if key=="lora" else None,
             "reference_not_the_untrained_base":key=="reference","action_normalization":sim["action_normalization"],
             "same_five_old_states":True,"recording_only_no_training":True,
             "preprocessing":sim["trials"][0].get("actual_policy_preprocessing","see saved trial metadata"),
             "not_automatically_identical_to_later_matched_train40_protocol":True}
        specs.append(("initial20-"+key,names[key],cfg,metrics,sim))
    assert len(specs)==16
    return specs

def upload():
    specs=make_specs();api=wandb.Api(timeout=30);assert api.default_entity==ENTITY
    manifest_path=R/"historical_upload_manifest.json"
    m=read(manifest_path) if manifest_path.exists() else {"runs":[],"recording_only_no_new_experiments":True}
    existing={r.id:r for r in api.runs(f"{ENTITY}/{PROJECT}",per_page=50)}
    for key,name,cfg,metrics,sim in specs:
        run_id=hashlib.sha256(("openvla-bowl-historical-results-v1:"+key).encode()).hexdigest()[:12]
        path=f"{ENTITY}/{PROJECT}/{run_id}"
        if any(x["id"]==run_id for x in m["runs"]):verify(path,metrics);continue
        if run_id in existing and existing[run_id].state=="finished":remote=verify(path,metrics)
        else:
            run=wandb.init(entity=ENTITY,project=PROJECT,id=run_id,name=name,group="historical-"+cfg["family"],
                job_type="recorded-evaluation",tags=["historical-upload",cfg["family"]],config=cfg,
                notes="Previously executed recorded experiment. This upload does not train or rerun models. Distinguish experiment families and missing measurements.",
                dir=str(R),resume="allow",mode="online",save_code=False,settings=settings())
            run.log(metrics)
            for k,v in metrics.items():run.summary[k]=v
            if sim:
                run.log({"simulation/trials":wandb.Table(columns=["initial_state","success","steps","termination_reason"],
                    data=[[t["init_state_index"],t["success"],t["policy_steps"],t["termination_reason"]] for t in sim["trials"]])})
            run.finish();remote=verify(path,metrics)
        m["runs"].append({"id":run_id,"name":name,"family":cfg["family"],"url":remote.url,"verified":True,"metrics":metrics})
        write(manifest_path,m);print("HISTORY_VERIFIED",len(m["runs"]),"/16",name,flush=True)
    m["complete"]=len(m["runs"])==16;write(manifest_path,m)

def overview():
    m=read(R/"historical_upload_manifest.json");main=read(R/"wandb_upload_manifest.json")
    assert m["complete"] and main["complete"]
    import upload_results as primary
    ids=primary.EXPECTED_OLD|{x["id"] for x in main["runs"]}|{x["id"] for x in m["runs"]}
    assert len(ids)==27
    allruns={r.id:r for r in wandb.Api(timeout=30).runs(f"{ENTITY}/{PROJECT}",per_page=60)}
    assert all(x in allruns and allruns[x].state=="finished" for x in ids)
    cols=["experiment","family","success_rate","inference_ms","gpu_gib","adapter_mb","run_url"]
    rows=[]
    for key in sorted(ids):
        r=allruns[key];s=r.summary;family=r.config.get("family", "physical-rank" if "physical-pruning" in r.tags else "LoRA-or-eta")
        rows.append([r.name,family,s.get("simulation/success_rate"),s.get("inference/mean_action_ms"),s.get("inference/peak_gpu_gib"),s.get("storage/adapter_mb"),r.url])
    specs=make_specs();epochs=read(ROOT/"ten_epoch_bowl/20261104T075816_073454Z/summary.json")["rows"]
    et=wandb.Table(columns=["epoch","success_rate","training_loss","validation_loss"],data=[[x["epoch"],x["success_rate"],x["training_loss"],x["validation_loss"]] for x in epochs])
    bs=read(ROOT/"batch_quantization/20261103T121901_577159Z/summary.json")
    bt=wandb.Table(columns=["batch","frames_per_second","peak_gpu_gib"],data=[[str(n),bs[f"batch_size_{n}"]["frames_per_second"],bs[f"batch_size_{n}"]["peak_allocated_gib"]] for n in [4,8]])
    qt=wandb.Table(columns=["mode","success_rate","inference_ms","peak_gpu_gib"],data=[[mode,bs["simulation"][mode]["success_rate"],bs["inference"][mode]["mean_action_seconds"]*1000,bs["inference"][mode]["peak_allocated_gib"]] for mode in ["bf16","int8"]])
    run=wandb.init(entity=ENTITY,project=PROJECT,id=main["overview"]["id"],resume="must",dir=str(R),mode="online",save_code=False,settings=settings())
    run.config.update({"completed_experiments":27,"includes_historical_epoch_batch_quantization_train20":True},allow_val_change=True)
    run.log({"comparison/all_experiments":wandb.Table(columns=cols,data=rows),"history/epoch_table":et,"history/batch_table":bt,"history/quantization_table":qt,
        "history/epoch_success":wandb.plot.line(et,"epoch","success_rate",title="Epoch 1-10: same five old states"),
        "history/epoch_validation_loss":wandb.plot.line(et,"epoch","validation_loss",title="Epoch 1-10: validation loss"),
        "history/batch_throughput":wandb.plot.bar(bt,"batch","frames_per_second",title="Batch 4/8: 128-frame microbenchmark, not quality"),
        "history/quantization_gpu":wandb.plot.bar(qt,"mode","peak_gpu_gib",title="BF16 vs int8: inference GPU memory"),
        "history/quantization_time":wandb.plot.bar(qt,"mode","inference_ms",title="BF16 vs int8: action prediction time")})
    run.summary["verification/completed_experiments"]=27;run.finish()
    remote=verify(f"{ENTITY}/{PROJECT}/{main['overview']['id']}",{"verification/completed_experiments":27})
    bundle=R/"viewing_bundle"
    with (bundle/"07_all_experiments.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(cols);w.writerows(rows)
    (bundle/"historical_upload_manifest.json").write_bytes((R/"historical_upload_manifest.json").read_bytes())
    (bundle/"supplement_history.py").write_bytes(Path(__file__).read_bytes())
    (bundle/"06_WANDB_OPEN.txt").write_text("W&B 전체 비교 화면\n"+remote.url+"\n\n00-ALL-EXPERIMENTS-COMPARISON\n총 27개 실험·측정 기록: 기존8 + 새rank3 + 누락된 과거16.\nrank 비교 그래프·영상5개, epoch1~10, batch4/8, BF16/int8 비교 포함.\n빈칸은 해당 측정을 하지 않았다는 뜻이며 0이 아님. 서로 다른 실험 종류는 개별 비교표로 보기.\n",encoding="utf-8")
    write(R/"all_history_complete.json",{"complete":True,"verified_experiments":27,"overview_url":remote.url,"historical_records_added":16})
    write(bundle/"manifest.json",{p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in bundle.iterdir() if p.is_file() and p.name!="manifest.json"})
    print("ALL_27_EXPERIMENTS_VERIFIED",remote.url,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--overview",action="store_true");a=p.parse_args()
    overview() if a.overview else upload()
