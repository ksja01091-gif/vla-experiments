"""Reproduce or finish the remote OpenVLA bowl experiment.
No existing source or base checkpoint is overwritten. Defaults: 1 epoch, batch 4, lr 1e-4, rank 8.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,netrc,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
MAIN=ROOT/"examples/openvla_bowl_experiment.py"

def write(path,value):Path(path).write_text(json.dumps(value,indent=2))

def capture_environment(run,item):
 name,exe=item
 result=subprocess.run([exe,"-m","pip","freeze"],capture_output=True,text=True,check=True,timeout=120)
 (run/(name+"_before.txt")).write_text(result.stdout)
 return name,{"interpreter":exe,"freeze_returncode":0}

def prepare_new_run(run):
 run.mkdir(parents=True,exist_ok=False)
 sources=[*ROOT.joinpath("src").rglob("*.py"),*ROOT.joinpath("examples").rglob("*.py")]
 write(run/"sources_before.json",{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 interpreters={"openvla310":"/home/jovyan/anaconda3/envs/openvla310/bin/python","lerobot312":"/home/jovyan/anaconda3/envs/lerobot312/bin/python","eval310":"/home/jovyan/projects/venvs/openvla_eval310/bin/python"}
 with ThreadPoolExecutor(max_workers=3) as pool:environment=dict(pool.map(lambda item:capture_environment(run,item),interpreters.items()))
 write(run/"environment_audit.json",environment)
 has_netrc=False
 try:has_netrc=bool(netrc.netrc().authenticators("api.wandb.ai"))
 except (FileNotFoundError,netrc.NetrcParseError,PermissionError):pass
 write(run/"initial_audit.json",{"remote_host":os.uname().nodename,"run":str(run),"wandb_api_key_env_present":bool(os.environ.get("WANDB_API_KEY")),"wandb_netrc_credentials_present":has_netrc})
 print("NEW_RUN",run,flush=True)
 print("SETTINGS: 1 epoch; 1928 frames; batch 4; lr 0.0001; rank 8; 482 updates",flush=True)

def wait_for_existing_phase(run):
 pointer=run/"active_process.json"
 if not pointer.exists():return
 active=json.loads(pointer.read_text());pid=active["pid"];proc=Path("/proc")/str(pid)
 started=time.monotonic()
 while True:
  try:status=(proc/"status").read_text()
  except FileNotFoundError:break
  if any(line.startswith("State:") and "Z" in line for line in status.splitlines()):break
  if time.monotonic()-started>600:raise RuntimeError("Existing phase still running; did not start a duplicate GPU job")
  time.sleep(2)

def execute(run,label,command):
 log=run/(label+".log")
 index=1
 while log.exists():
  log=run/(label+"_"+str(index)+".log");index+=1
 with log.open("x") as output:
  process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
  write(run/"active_process.json",{"phase":label,"pid":process.pid,"run":str(run),"log":str(log)})
  print("PHASE_STARTED",label,process.pid,flush=True)
  try:code=process.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   raise RuntimeError(f"Phase exceeded 30 minutes: {label}; own process group stopped")
 if code:
  print(log.read_text()[-6000:],flush=True)
  raise RuntimeError(f"Phase {label} returned {code}; see {log}")
 print("PHASE_COMPLETE",label,flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path)
 args=parser.parse_args()
 run=args.run or ROOT/"outputs/full_bowl_experiment"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
 if not run.exists():prepare_new_run(run)
 assert (run/"sources_before.json").exists() and (run/"environment_audit.json").exists()
 wait_for_existing_phase(run)
 write(ROOT/"outputs/full_bowl_experiment/current_run.json",{"run":str(run)})
 started=time.monotonic()
 try:
  if not (run/"calibration.json").exists():execute(run,"calibration",[sys.executable,"-u",str(MAIN),"calibrate","--run",str(run)])
  estimate=json.loads((run/"calibration.json").read_text())
  print("MEASURED_ESTIMATE_MINUTES",estimate["estimated_total_minutes_with_30pct_margin"],"PEAK_GPU_GIB",estimate["gpu_peak_allocated_gib"],flush=True)
  if not (run/"training_result.json").exists():
   assert not (run/"training_metrics.jsonl").exists(),"Incomplete training preserved; choose a new --run"
   execute(run,"training",[sys.executable,"-u",str(MAIN),"train","--run",str(run)])
  if not (run/"checkpoint_reload_check.json").exists():execute(run,"check",[sys.executable,"-u",str(MAIN),"check","--run",str(run)])
  for tag in ["base","lora","reference"]:
   result=run/f"simulation_{tag}.json"
   if not result.exists():execute(run,"simulation_"+tag,[sys.executable,"-u",str(MAIN),"simulate","--run",str(run),"--model",tag])
   cohort=json.loads(result.read_text())
   if cohort["technical_errors"]:raise RuntimeError(f"Technical simulator errors in {tag}; do not interpret them as policy failures")
  execute(run,"preservation",[sys.executable,"-u",str(MAIN),"audit","--run",str(run)])
  execute(run,"report",["/home/jovyan/anaconda3/envs/openvla310/bin/python",str(ROOT/"examples/openvla_bowl_report.py"),"--run",str(run)])
  execute(run,"wandb_summary",[sys.executable,str(ROOT/"examples/openvla_bowl_wandb_summary.py"),"--run",str(run)])
  write(run/"pipeline_complete.json",{"complete":True,"run":str(run),"driver_remaining_wall_seconds":time.monotonic()-started})
  print("EXPERIMENT_COMPLETE",run,flush=True)
 except Exception as error:
  import traceback
  write(run/"pipeline_error.json",{"error":repr(error),"traceback":traceback.format_exc()})
  raise

if __name__=="__main__":main()
