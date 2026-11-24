"""Reproduce in a NEW folder; --help never allocates a model."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,subprocess,shutil,json,hashlib,ast
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--run",action="store_true",help="start profiles, two 5-epoch trainers, primary evaluation and inference-only eta sweep")
a=p.parse_args()
if not a.run:p.error("Use --run to start, --help to read usage.")
PY="/home/jovyan/anaconda3/envs/lerobot312/bin/python"
gpu=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True,capture_output=True,check=True)
if int(gpu.stdout.splitlines()[0].strip())>1024:
 raise SystemExit("GPU is busy. This comparison needs two large training models; current experiments are preserved.")
source=Path(__file__).parent
r=source.parent/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
r.mkdir()
names=["sp_policy.py","comparison_common.py","train_and_profile.py","evaluate.py","driver.py","check_sp_math.py",
"run_profiles.py","unit_then_profiles.py","trace_one_sample.py","build_report.py","build_visuals.py","reproduce.py","00_CODE_FLOW_KO.txt"]
for name in names:shutil.copy2(source/name,r/name)
root=Path("/home/jovyan/projects/lerobot-openvla")
old=root/"outputs/five_epoch_bowl/20261103T064207_586519Z/checkpoints/step_005035"
data=root/"outputs/bowl40_experiment/20261103T051914_711739Z/data"
files=[*root.joinpath("src").rglob("*.py"),*root.joinpath("examples").rglob("*.py"),*old.iterdir()]
files.extend(p for p in data.rglob("*") if p.is_file() and p.suffix in [".json",".jsonl",".parquet"])
(r/"protected_before.json").write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2))
envs={"lerobot312":PY,"openvla310":"/home/jovyan/anaconda3/envs/openvla310/bin/python","eval310":"/home/jovyan/projects/venvs/openvla_eval310/bin/python"}
(r/"interpreters.json").write_text(json.dumps(envs,indent=2))
for name,python in envs.items():
 out=subprocess.run([python,"-m","pip","freeze"],capture_output=True,text=True,check=True)
 (r/(name+"_before.txt")).write_text(out.stdout)
locations={}
for name in ["comparison_common.py","train_and_profile.py","evaluate.py","sp_policy.py"]:
 locations[name]={n.name:n.lineno for n in ast.parse((r/name).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
(r/"code_locations.json").write_text(json.dumps(locations,indent=2))
print("NEW_OUTPUT",r,flush=True)
for script in ["unit_then_profiles.py","trace_one_sample.py","driver.py","build_report.py"]:
 subprocess.run([PY,"-u",str(r/script)],check=True)
subprocess.run([envs["openvla310"],str(r/"build_visuals.py")],check=True)
print("COMPLETE",r,flush=True)
