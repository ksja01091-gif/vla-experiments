"""Run the train-only calibration, fresh-process benchmarks and paired robot tests."""
import argparse,json,time,hashlib,gc,types,subprocess,os,traceback
from pathlib import Path
from pruned_policy import *
def calibrate():
    assert not (RUN/"calibration_complete.json").exists(),"Calibration already completed."
    exp.seed();train,val,stats=exp.get_data()
    episodes=np.asarray(train.hf_dataset["episode_index"],dtype=np.int64).reshape(-1)
    assert len(episodes)==4025 and len(np.unique(episodes))==40
    indices=[]
    episode_rows=[]
    for e in sorted(np.unique(episodes).tolist()):
        rows=np.flatnonzero(episodes==e)
        chosen=rows[np.linspace(0,len(rows)-1,4,dtype=int)].tolist()
        assert len(set(chosen))==4
        indices.extend(chosen);episode_rows.append({"episode_index":e,"indices":chosen})
    assert len(indices)==160 and len(set(indices))==160
    write(RUN/"calibration_indices.json",{"split":"train","validation_used":False,"frames":160,"episodes":episode_rows})
    policy=common.make_policy("sp128",stats);policy.load_saved_adapter(CHECKPOINT)
    policy=common.prepare_inference(policy)
    before=policy.adapter_hash()
    importance={n:torch.zeros(128,device="cuda",dtype=torch.float64) for n,_ in policy.sp_modules()}
    column_norms={n:m.sp_B.weight.detach().float().square().sum(0) for n,m in policy.sp_modules()}
    token_counts={n:0 for n,_ in policy.sp_modules()}
    handles=[]
    def hook_for(name):
        def hook(module,args,output):
            x=args[0]
            with torch.no_grad(),torch.autocast(device_type=x.device.type,enabled=False):
                logits=module.router2(F.silu(module.router1(x.float())))
                scores=F.softplus(logits).clamp_min(1e-6)
                mask,_,_=energy_mask(scores,module.eta)
                coefficients=F.linear(x.float(),module.sp_A.weight.float())*(scores*mask)
                energy=coefficients.square()*column_norms[name]
                axes=tuple(range(energy.ndim-1))
                importance[name].add_(energy.double().sum(axes))
                token_counts[name]+=energy.numel()//128
        return hook
    for name,module in policy.sp_modules():handles.append(module.register_forward_hook(hook_for(name)))
    started=time.monotonic()
    try:
        for i,index in enumerate(indices,1):
            action=common.sample_action(policy,train,index)
            assert action.shape==(1,7) and bool(torch.isfinite(action).all())
            if i%5==0 or i==1:
                write(RUN/"calibration_progress.json",{"completed_frames":i,"total_frames":160,"elapsed_seconds":time.monotonic()-started})
                print("CALIBRATION",i,"/160",flush=True)
    finally:
        for handle in handles:handle.remove()
    assert policy.adapter_hash()==before
    cpu_scores={n:t.cpu() for n,t in importance.items()}
    assert all(torch.isfinite(t).all() and bool(t.sum()>0) for t in cpu_scores.values())
    write(RUN/"calibration_importance.json",{"importance":{n:t.tolist() for n,t in cpu_scores.items()},"token_counts":token_counts})
    exports=[export_compact(policy,k,cpu_scores) for k in [96,64]]
    write(RUN/"calibration_complete.json",{"completed":True,"train_frames":160,"train_episodes":40,"validation_used":False,
        "source_adapter_unchanged":True,"source_adapter_hash":before,"exports":exports,"seconds":time.monotonic()-started})
    print("CALIBRATION_COMPLETE",json.dumps(exports),flush=True)

def benchmark(variant):
    out=RUN/"benchmarks"/variant;out.mkdir(parents=True,exist_ok=True)
    assert not (out/"result.json").exists(),"Completed benchmark preserved."
    torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    policy,val,cp=load_policy(variant)
    load_peak=torch.cuda.max_memory_allocated()/1024**3
    before=policy.adapter_hash()
    shapes=physical_shapes(policy)
    rank=VARIANTS[variant]
    assert all(s["A"][0]==rank and s["B"][1]==rank and s["router_out"]==128 for s in shapes)
    assert len(shapes)==64
    with torch.no_grad():
        full_validation=exp.evaluate(policy,val,4,out/"validation.jsonl",5035)
    if variant=="sp_full128":
        expected=json.loads((SOURCE/"sp128/training_result.json").read_text())["final_validation"]["loss"]
        assert abs(full_validation["loss"]-expected)<1e-4,(full_validation["loss"],expected)
    indices=list(np.linspace(0,len(val)-1,32,dtype=int))
    for i in indices[:3]:common.sample_action(policy,val,int(i))
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    rows=[]
    old=json.loads((SOURCE/"inference/sp128_eta_090.json").read_text())
    prior={r["index"]:np.asarray(r["action"]) for r in old["rows"]}
    baseline_exact=True
    for repeat in range(3):
        for j,i in enumerate(indices):
            torch.cuda.synchronize();start=time.perf_counter()
            action=common.sample_action(policy,val,int(i))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            assert action.shape==(1,7) and bool(torch.isfinite(action).all())
            value=action.cpu().tolist()
            if variant=="sp_full128":baseline_exact=baseline_exact and bool(np.allclose(np.asarray(value),prior[int(i)],rtol=0,atol=1e-6))
            rows.append({"round":repeat,"index":int(i),"seconds":elapsed,"shape":[1,7],"finite":True,"action":value})
        write(out/"progress.json",{"completed_rounds":repeat+1,"total_rounds":3,"mean_ms_so_far":1000*float(np.mean([r["seconds"] for r in rows]))})
        print("BENCHMARK_ROUND",variant,repeat+1,flush=True)
    if variant=="sp_full128":assert baseline_exact,"Full-SP fresh reload differs from preserved baseline actions."
    memory={"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
            "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,
            "allocated_after_benchmark_gib":torch.cuda.memory_allocated()/1024**3,
            "model_tensor_bytes":policy.model.get_memory_footprint(),"load_peak_allocated_gib":load_peak}
    policy.record_rank_stats(True)
    for i in indices:common.sample_action(policy,val,int(i))
    ranks=policy.rank_stats();policy.record_rank_stats(False)
    assert policy.adapter_hash()==before
    mean_rounds=[float(np.mean([r["seconds"] for r in rows if r["round"]==k])) for k in range(3)]
    result={"variant":variant,"physical_bank_rank":rank,"original_router_width":128,"eta":.9,"checkpoint":str(cp),
        "fresh_process_reload":True,"adapter_hash_unchanged":True,"adapter_hash":before,
        "source_baseline_actions_match":baseline_exact if variant=="sp_full128" else None,
        "adapter_file_bytes":(cp/"adapter_model.safetensors").stat().st_size,
        "adapter_parameters":sum(p.numel() for p in policy.adapter_state().values()),
        "mean_action_seconds":float(np.mean([r["seconds"] for r in rows])),
        "median_action_seconds":float(np.median([r["seconds"] for r in rows])),
        "round_means_seconds":mean_rounds,"round_mean_std_seconds":float(np.std(mean_rounds)),
        "validation":full_validation,"rank_statistics":ranks,"matrix_shapes":shapes,"samples_per_round":32,"rounds":3,
        "warmup_calls":3,"timing_excludes_validation_and_rank_instrumentation":True,
        "inference_autocast":"unchanged prior matched_select_action","rows":rows,**memory}
    write(out/"result.json",result)
    print("BENCHMARK_COMPLETE",variant,result["mean_action_seconds"],memory["peak_allocated_gib"],flush=True)

def simulate(variant):
    assert variant!="sp_full128","Baseline simulation is reused; do not create extra unplanned trials."
    policy,val,cp=load_policy(variant);before=policy.adapter_hash()
    base=RUN/"simulation/base";base.parent.mkdir(parents=True,exist_ok=True)
    if not base.exists():base.symlink_to(SOURCE/"simulation/sp128_eta_090",target_is_directory=True)
    ns=dict(vars(exp))
    code=(common.INPUT/"matched_rollout_function.py").read_text()
    exec(compile(code,str(common.INPUT/"matched_rollout_function.py"),"exec"),ns)
    trials=[]
    for state in range(5):
        p=RUN/"simulation"/variant/f"state_{state:02d}"/"result.json"
        t=json.loads(p.read_text()) if p.exists() else ns["simulate_trial"](policy,variant,state,types.SimpleNamespace(run=RUN))
        assert t["trial_executed"],t
        expected=json.loads((SOURCE/"simulation/sp128_eta_090"/f"state_{state:02d}"/"result.json").read_text())
        for key in ["initial_image_sha256","selected_init_state_sha256","init_state_file_sha256","reference_sha256",
                    "simulator_versions","max_policy_steps","warmup_steps"]:
            assert t[key]==expected[key],key
        trials.append(t)
        write(RUN/(variant+"_simulation_progress.json"),{"completed":len(trials),"attempts":5,"successes":sum(t["success"] for t in trials)})
    assert policy.adapter_hash()==before
    report={"variant":variant,"checkpoint":str(cp),"attempts":5,"successes":sum(t["success"] for t in trials),
        "success_rate":sum(t["success"] for t in trials)/5,"mean_policy_steps":float(np.mean([t["policy_steps"] for t in trials])),
        "adapter_unchanged":True,"same_conditions_as_preserved_SP":True,"trials":trials}
    write(RUN/("simulation_"+variant+".json"),report)
    print("SIMULATION_COMPLETE",variant,report["successes"],"/5",flush=True)

def audit():
    before=json.loads((RUN/"protected_before.json").read_text())
    changed=[p for p,h in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    env={}
    for name,exe in json.loads((RUN/"interpreters.json").read_text()).items():
        result=subprocess.run([exe,"-m","pip","freeze"],capture_output=True,text=True,check=True)
        env[name]=sorted(result.stdout.splitlines())==sorted((RUN/(name+"_before.txt")).read_text().splitlines())
    report={"protected_files":len(before),"changed_files":changed,"environments_unchanged":env}
    write(RUN/"preservation_report.json",report)
    assert not changed and all(env.values()),report
    print("PRESERVATION_COMPLETE",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["calibrate","benchmark","simulate","audit"])
    p.add_argument("--variant",choices=list(VARIANTS),default="sp_full128")
    a=p.parse_args()
    try:
        if a.phase in ["calibrate","audit"]:globals()[a.phase]()
        else:globals()[a.phase](a.variant)
    except Exception as e:
        write(RUN/(a.phase+"_"+a.variant+"_error.json"),{"error":repr(e),"traceback":traceback.format_exc()})
        raise
if __name__=="__main__":main()

