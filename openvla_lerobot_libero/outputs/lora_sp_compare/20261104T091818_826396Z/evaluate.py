"""Fresh reload, isolated inference/rank profiling, then same-state robot trials."""
import argparse,json,time,types,traceback
from comparison_common import *
def tag_for(method,eta):
 return method if method!="sp128" else f"sp128_eta_{round(eta*100):03d}"
def load_final(method,eta):
 _,val,stats=exp.get_data();policy=make_policy(method,stats)
 cp=checkpoint_path(method);policy.load_saved_adapter(cp)
 if method=="sp128":policy.set_eta(eta)
 return prepare_inference(policy),val,cp
def infer(method,eta):
 tag=tag_for(method,eta);policy,val,cp=load_final(method,eta)
 before=policy.adapter_hash()
 first_file=(OLD/"validation_step_005035.jsonl") if method=="lora8" else RUN/method/"validation_step_005035.jsonl"
 expected=json.loads(first_file.read_text().splitlines()[0])["loss"]
 with torch.no_grad():loss,_=policy(exp.batch_from(policy,val,[0,1,2,3]))
 # Changed inference thresholds deliberately need not reproduce training-eta loss.
 if method!="sp128" or eta==.9:assert abs(float(loss)-expected)<1e-4,(float(loss),expected)
 indices=list(np.linspace(0,len(val)-1,32,dtype=int))
 for i in indices[:3]:sample_action(policy,val,i)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();rows=[]
 for i in indices:
  torch.cuda.synchronize();tick=time.perf_counter();action=sample_action(policy,val,i);torch.cuda.synchronize()
  assert action.shape==(1,7) and bool(torch.isfinite(action).all())
  rows.append({"index":int(i),"seconds":time.perf_counter()-tick,"shape":[1,7],"finite":True,"action":action.cpu().tolist()})
 memory={"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
         "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"model_tensor_bytes":policy.model.get_memory_footprint()}
 ranks=None
 if method=="sp128":
  policy.record_rank_stats(True)
  for i in indices:sample_action(policy,val,i)
  ranks=policy.rank_stats();policy.record_rank_stats(False)
  assert all(r["minimum_retained_score_energy"]>=eta-2e-6 for r in ranks["rows"])
 assert policy.adapter_hash()==before
 report={"method":method,"tag":tag,"eta":eta if method=="sp128" else None,"checkpoint":str(cp),"fresh_process_reload":True,
   "adapter_hash_exact":True,"adapter_hash":before,"training_eta_loss_check":float(loss),
   "samples":32,"warmup":3,"rank_statistics_excluded_from_timing":True,
   "mean_action_seconds":float(np.mean([r["seconds"] for r in rows])),"median_action_seconds":float(np.median([r["seconds"] for r in rows])),
   "adapter_file_bytes":(cp/"adapter_model.safetensors").stat().st_size,**memory,"rank_statistics":ranks,"rows":rows}
 write(RUN/"inference"/(tag+".json"),report)
 print("INFERENCE_COMPLETE",tag,report["mean_action_seconds"],memory["peak_allocated_gib"],flush=True)

def simulate(method,eta):
 tag=tag_for(method,eta)
 if method=="lora8":
  write(RUN/"simulation_lora8.json",json.loads((OLD/"simulation_epoch5.json").read_text()))
  return
 policy,val,cp=load_final(method,eta);before=policy.adapter_hash()
 base=RUN/"simulation/base";base.parent.mkdir(parents=True,exist_ok=True)
 # Established before simultaneous workers start by orchestrator.
 if not base.exists():base.symlink_to(OLD/"simulation/epoch5",target_is_directory=True)
 ns=dict(vars(exp))
 source=(INPUT/"matched_rollout_function.py").read_text()
 exec(compile(source,str(INPUT/"matched_rollout_function.py"),"exec"),ns)
 trials=[]
 for state in range(5):
  folder=RUN/"simulation"/tag/f"state_{state:02d}"
  if (folder/"result.json").exists():t=json.loads((folder/"result.json").read_text())
  else:t=ns["simulate_trial"](policy,tag,state,types.SimpleNamespace(run=RUN))
  assert t["trial_executed"],t
  baseline=json.loads((OLD/"simulation/epoch5"/f"state_{state:02d}"/"result.json").read_text())
  for key in ["initial_image_sha256","selected_init_state_sha256","init_state_file_sha256","reference_sha256","simulator_versions","max_policy_steps","warmup_steps"]:
   assert t[key]==baseline[key],key
  trials.append(t)
  write(RUN/(tag+"_simulation_progress.json"),{"completed":len(trials),"total":5,"successes":sum(t["success"] for t in trials)})
 assert policy.adapter_hash()==before
 write(RUN/("simulation_"+tag+".json"),{"method":method,"tag":tag,"eta":eta if method=="sp128" else None,
   "checkpoint":str(cp),"attempts":5,"successes":sum(t["success"] for t in trials),"success_rate":sum(t["success"] for t in trials)/5,
   "mean_policy_steps":float(np.mean([t["policy_steps"] for t in trials])),"same_conditions_as_lora8":True,
   "adapter_unchanged":True,"trials":trials})
 print("SIMULATION_COMPLETE",tag,sum(t["success"] for t in trials),"/5",flush=True)

def audit():
 import hashlib,subprocess
 before=json.loads((RUN/"protected_before.json").read_text())
 changed=[p for p,h in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
 env={}
 for name,python in json.loads((RUN/"interpreters.json").read_text()).items():
  x=subprocess.run([python,"-m","pip","freeze"],text=True,capture_output=True,check=True)
  (RUN/(name+"_after.txt")).write_text(x.stdout)
  env[name]=sorted(x.stdout.splitlines())==sorted((RUN/(name+"_before.txt")).read_text().splitlines())
 write(RUN/"preservation_report.json",{"protected_files":len(before),"changed_files":changed,"environments_unchanged":env})
 assert not changed and all(env.values())
 print("PRESERVATION_COMPLETE",flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("phase",choices=["infer","simulate","audit"])
 p.add_argument("--method",choices=METHODS,default="sp128");p.add_argument("--eta",type=float,default=.9);a=p.parse_args()
 try:
  if a.phase=="audit":audit()
  else:globals()[a.phase](a.method,a.eta)
 except Exception as e:
  write(RUN/(a.phase+"_"+tag_for(a.method,a.eta)+"_error.json"),{"error":repr(e),"traceback":traceback.format_exc()});raise
if __name__=="__main__":main()
