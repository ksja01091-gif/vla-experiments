"""After training exits, optionally evaluate epoch10 on a second model process.
Ready-file lease prevents duplicate evaluation by the original sequential worker.
No timing from overlapping rollouts is used as an isolated inference benchmark.
"""
from pathlib import Path
import json,time,subprocess,sys,traceback,datetime
R=Path(__file__).parent
ready=R/"epoch_10_ready.json";lease=R/"epoch_10_parallel_ready.json"
def write(name,obj):(R/name).write_text(json.dumps(obj,indent=2))
try:
 deadline=time.monotonic()+10800
 while True:
  status=json.loads((R/"pipeline_status.json").read_text())
  if status["status"] in ["complete","error"]:
   write("parallel_eval_scheduler.json",{"used":False,"reason":"pipeline already ended"});sys.exit(0)
  if status.get("stage")=="finish_evaluation":break
  if time.monotonic()>deadline:raise TimeoutError("training not finished")
  time.sleep(5)
 progress=json.loads((R/"simulation_progress.json").read_text()) if (R/"simulation_progress.json").exists() else {"epoch":6}
 if progress["epoch"]>=9 or (R/"simulation/epoch10").exists():
  write("parallel_eval_scheduler.json",{"used":False,"reason":"sequential evaluator already near epoch10"});sys.exit(0)
 assert ready.exists() and not lease.exists()
 ready.replace(lease)
 write("parallel_eval_scheduler.json",{"used":True,"status":"epoch10 reserved","sequential_progress_at_reservation":progress,
  "reason":"training process has exited; two BF16 inference models fit on A100 40GB; isolated benchmark runs after both exit"})
 child="""from pathlib import Path
import importlib.util,inspect,traceback,json
r=Path(__file__).parent
spec=importlib.util.spec_from_file_location("parallel_epoch10",r/"ten_epoch_experiment.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=inspect.getsource(m.simulate_all)
assert "range(6,11)" in source
source=source.replace("range(6,11)","range(10,11)")
source=source.replace('ready=RUN/f"epoch_{epoch:02d}_ready.json"','ready=RUN/"epoch_10_parallel_ready.json"')
source=source.replace('"simulation_progress.json"','"parallel_simulation_progress.json"')
source=source.replace('"simulation_complete.json"','"parallel_epoch10_complete.json"')
source=source.replace('"epochs":[6,7,8,9,10],"new_trials":25','"epochs":[10],"new_trials":5')
exec(compile(source,str(r/"parallel_epoch10_worker.py"),"exec"),m.__dict__)
try:m.simulate_all()
except Exception:
 (r/"parallel_epoch10_error.txt").write_text(traceback.format_exc())
 raise
"""
 (R/"parallel_epoch10_worker.py").write_text(child)
 with (R/"parallel_epoch10.log").open("w") as log:
  code=subprocess.run([sys.executable,"-u",str(R/"parallel_epoch10_worker.py")],stdout=log,stderr=subprocess.STDOUT).returncode
 # Keep all valid completed trials. Archive only incomplete/technical-error attempts
 # before restoring the ready marker so the original worker can safely finish them.
 for state in range(5):
  folder=R/"simulation/epoch10"/f"state_{state:02d}"
  if not folder.exists():continue
  result=folder/"result.json"
  valid=False
  if result.exists():
   try:valid=bool(json.loads(result.read_text()).get("trial_executed"))
   except json.JSONDecodeError:pass
  if not valid:
   assert folder.resolve().is_relative_to(R.resolve())
   folder.rename(folder.with_name(folder.name+".partial_parallel_"+str(time.time_ns())))
 lease.replace(ready)
 write("parallel_eval_scheduler.json",{"used":True,"status":"done" if code==0 else "fallback_to_original",
   "returncode":code,"no_training_running_during_parallel_rollouts":True,"ready_marker_restored":True})
except Exception:
 write("parallel_eval_scheduler_error.json",{"error":traceback.format_exc()})
 if lease.exists() and not ready.exists():lease.replace(ready)
 raise
