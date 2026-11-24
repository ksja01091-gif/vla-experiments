"""Controlled three-method comparison and inference-only eta sweep."""
from pathlib import Path
import subprocess,json,time,hashlib,sys
R=Path(__file__).parent
PY="/home/jovyan/anaconda3/envs/lerobot312/bin/python"
def write(name,obj):
 p=R/name;t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(obj,indent=2));t.replace(p)
def start(script,args,logname):
 log=(R/logname).open("w")
 p=subprocess.Popen([PY,"-u",str(R/script),*args],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 log.close();return p
def wait_success(process,label):
 code=process.wait()
 if code:raise RuntimeError(label+" failed; output preserved")
try:
 assert json.loads((R/"profile_status.json").read_text())["status"]=="complete"
 profiles={m:json.loads((R/m/"profile.json").read_text()) for m in ["lora8","lora128","sp128"]}
 assert max(p["initial_validation_loss_four_frames"] for p in profiles.values())-min(p["initial_validation_loss_four_frames"] for p in profiles.values())<1e-5
 source_files=list(R.glob("*.py"))
 write("source_hashes_at_training_start.json",{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
 peak_sum=profiles["lora128"]["peak_allocated_gib"]+profiles["sp128"]["peak_allocated_gib"]
 assert peak_sum<35.,peak_sum
 write("execution_schedule.json",{"two_training_processes_on_one_A100":True,"measured_peak_sum_gib":peak_sum,
   "isolated_profile_times_are_primary_speed_comparison":True,
   "isolated_final_inference_before_threshold_rollouts":True})
 write("status.json",{"phase":"training","status":"running"})
 trains={m:start("train_and_profile.py",["train","--method",m],m+"_train.log") for m in ["lora128","sp128"]}
 write("training_pids.json",{m:p.pid for m,p in trains.items()})
 # Once the faster trainer exits, use its freed model memory for its robot evaluation.
 pending=dict(trains);early_eval=None
 while pending:
  for method,p in list(pending.items()):
   code=p.poll()
   if code is None:continue
   if code:raise RuntimeError(method+" training failed")
   del pending[method]
   if method=="lora128" and "sp128" in pending:
    early_eval=start("evaluate.py",["simulate","--method","lora128"],"lora128_simulate.log")
  time.sleep(3)
 if early_eval:wait_success(early_eval,"early LoRA128 simulation")
 write("status.json",{"phase":"isolated_inference","status":"running"})
 for method,eta in [("lora8",.9),("lora128",.9),("sp128",.9),("sp128",.8),("sp128",.7)]:
  p=start("evaluate.py",["infer","--method",method,"--eta",str(eta)],method+"_infer_"+str(eta)+".log")
  wait_success(p,"isolated inference "+method+" "+str(eta))
 wait_success(start("evaluate.py",["simulate","--method","lora8"],"reuse_lora8.log"),"baseline reuse")
 if not (R/"simulation_lora128.json").exists():
  wait_success(start("evaluate.py",["simulate","--method","lora128"],"lora128_simulate.log"),"LoRA128 simulation")
 write("status.json",{"phase":"sp_simulation_and_eta_ablation","status":"running"})
 # Two SP models fit after trainers exit. Never use rollout wall time as latency.
 queue=[.9,.8,.7];active=[]
 while queue or active:
  while queue and len(active)<2:
   eta=queue.pop(0)
   active.append((eta,start("evaluate.py",["simulate","--method","sp128","--eta",str(eta)],"sp128_simulate_"+str(eta)+".log")))
  for eta,p in list(active):
   code=p.poll()
   if code is not None:
    if code:raise RuntimeError("SP simulation eta="+str(eta)+" failed")
    active.remove((eta,p))
  time.sleep(3)
 write("status.json",{"phase":"preservation_audit","status":"running"})
 wait_success(start("evaluate.py",["audit"],"audit.log"),"preservation")
 write("status.json",{"status":"complete"})
except Exception as error:
 import traceback
 write("driver_error.json",{"error":repr(error),"traceback":traceback.format_exc()})
 write("status.json",{"status":"error","error":repr(error)})
 raise
