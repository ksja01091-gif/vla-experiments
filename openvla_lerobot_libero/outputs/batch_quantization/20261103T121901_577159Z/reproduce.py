"""Create a fresh output directory before reproducing this experiment."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,subprocess,json,hashlib,shutil,sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--batch-only",action="store_true",help="run only the short batch 4/8 benchmarks")
a=p.parse_args()
source=Path(__file__).parent
run=source.parent/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
run.mkdir()
for name in ["experiment.py","build_report.py","build_visuals.py","experiment_config.json","interpreters.json"]:
 shutil.copy2(source/name,run/name)
(run/"deps").symlink_to(source/"deps",target_is_directory=True)
root=Path("/home/jovyan/projects/lerobot-openvla")
cp=Path(json.loads((run/"experiment_config.json").read_text())["checkpoint"])
files=[*root.joinpath("src").rglob("*.py"),*root.joinpath("examples").rglob("*.py"),*cp.iterdir()]
(run/"protected_before.json").write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files if x.is_file()},indent=2))
for name,python in json.loads((run/"interpreters.json").read_text()).items():
 s=subprocess.run([python,"-m","pip","freeze"],text=True,capture_output=True,check=True)
 (run/(name+"_before.txt")).write_text(s.stdout)
print("NEW_OUTPUT",run,flush=True)
phases=[["batch","--batch-size","4"],["batch","--batch-size","8"]]
if not a.batch_only:phases += [["infer","--mode","bf16"],["infer","--mode","int8"]]
phases += [["audit"]]
for phase in phases:
 (run/"pipeline_status.json").write_text(json.dumps({"stage":phase,"status":"running"}))
 with (run/("_".join(phase).replace("--","")+".log")).open("w") as f:
  x=subprocess.run([sys.executable,"-u",str(run/"experiment.py"),*phase],stdout=f,stderr=subprocess.STDOUT)
 if x.returncode:
  (run/"pipeline_status.json").write_text(json.dumps({"stage":phase,"status":"error","returncode":x.returncode}))
  raise SystemExit(x.returncode)
(run/"pipeline_status.json").write_text(json.dumps({"status":"complete","batch_only":a.batch_only}))
if not a.batch_only:
 subprocess.run([sys.executable,str(run/"build_report.py")],check=True)
 subprocess.run(["/home/jovyan/anaconda3/envs/openvla310/bin/python",str(run/"build_visuals.py")],check=True)
print("COMPLETE",run,flush=True)
