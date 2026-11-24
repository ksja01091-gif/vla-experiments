from pathlib import Path
import subprocess,json
r=Path(__file__).parent
p="/home/jovyan/anaconda3/envs/lerobot312/bin/python"
for a,name in [(["infer","--mode","int8"],"infer_mode_int8_retry1"),(["audit"],"audit")]:
 (r/"pipeline_status.json").write_text(json.dumps({"stage":a,"status":"running"}))
 with (r/(name+".log")).open("w") as log:
  x=subprocess.run([p,"-u",str(r/"experiment.py"),*a],stdout=log,stderr=subprocess.STDOUT)
 if x.returncode:
  (r/"pipeline_status.json").write_text(json.dumps({"stage":a,"status":"error","returncode":x.returncode}))
  raise SystemExit(x.returncode)
(r/"pipeline_status.json").write_text(json.dumps({"status":"complete"}))
