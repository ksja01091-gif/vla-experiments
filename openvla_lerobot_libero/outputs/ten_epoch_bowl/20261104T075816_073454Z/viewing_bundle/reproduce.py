"""Start a fresh epoch5-to-10 continuation; original results are preserved."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,shutil,subprocess,sys,json,hashlib,ast
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--run",action="store_true",help="start the full training and evaluation in a new output folder")
a=p.parse_args()
if not a.run:p.error("Use --run to start, or --help to read usage.")
source=Path(__file__).parent
run=source.parent/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
run.mkdir()
for name in ["ten_epoch_experiment.py","pipeline.py","build_report.py","build_visuals.py","finish_report.py","lora_sp_preparation.json","lora_sp_threshold_boundary_check.json","paper_provenance.json","05_NEXT_LORA_SP_PLAN_KO.txt","parallel_eval_scheduler.py","reproduce.py"]:
 shutil.copy2(source/name,run/name)
shutil.copy2(run/"ten_epoch_experiment.py",run/"training_source_snapshot.py")
code_hash=hashlib.sha256((run/"ten_epoch_experiment.py").read_bytes()).hexdigest()
(run/"phase_source_provenance.json").write_text(json.dumps({"training_and_evaluation_source_at_launch":str(run/"training_source_snapshot.py"),
"launch_source_hash":code_hash,"snapshot_matches_launch_hash":True,"post_training_check_source":str(run/"ten_epoch_experiment.py"),
"post_training_check_source_hash":code_hash,"change_after_launch":"none"},indent=2))
locations={n.name:n.lineno for n in ast.parse((run/"ten_epoch_experiment.py").read_text()).body if isinstance(n,ast.FunctionDef)}
(run/"code_locations.json").write_text(json.dumps({"file":str(run/"ten_epoch_experiment.py"),"functions":locations},indent=2))
print("NEW_OUTPUT",run,flush=True)
subprocess.run([sys.executable,"-u",str(run/"pipeline.py")],check=True)
subprocess.run([sys.executable,str(run/"build_report.py")],check=True)
subprocess.run(["/home/jovyan/anaconda3/envs/openvla310/bin/python",str(run/"build_visuals.py")],check=True)
print("COMPLETE",run,flush=True)
