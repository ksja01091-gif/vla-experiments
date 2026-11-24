from pathlib import Path
import json,subprocess,concurrent.futures,sys
run=Path(sys.argv[1])
interpreters={"openvla310":"/home/jovyan/anaconda3/envs/openvla310/bin/python","lerobot312":"/home/jovyan/anaconda3/envs/lerobot312/bin/python","eval310":"/home/jovyan/projects/venvs/openvla_eval310/bin/python"}
def capture(item):
 name,exe=item
 proc=subprocess.run([exe,"-m","pip","freeze"],capture_output=True,text=True,timeout=120)
 (run/(name+"_before.txt")).write_text(proc.stdout)
 code="import sys,importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ['torch','numpy']},sort_keys=True));print(sys.executable)"
 ver=subprocess.run([exe,"-c",code],capture_output=True,text=True,timeout=30)
 return name,{"freeze_returncode":proc.returncode,"versions":ver.stdout.strip(),"interpreter":exe}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: report=dict(pool.map(capture,interpreters.items()))
(run/"environment_audit.json").write_text(json.dumps(report,indent=2))
print("ENVIRONMENT_AUDIT_COMPLETE",json.dumps(report),flush=True)
