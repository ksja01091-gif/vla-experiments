from pathlib import Path
import subprocess,json
r=Path(__file__).parent
python="/home/jovyan/anaconda3/envs/lerobot312/bin/python"
for method in ["lora8","lora128","sp128"]:
 for phase in ["profile","reload"]:
  (r/"profile_status.json").write_text(json.dumps({"method":method,"phase":phase,"status":"running"}))
  with (r/(method+"_"+phase+".log")).open("w") as log:
   result=subprocess.run([python,"-u",str(r/"train_and_profile.py"),phase,"--method",method],stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:
   (r/"profile_status.json").write_text(json.dumps({"method":method,"phase":phase,"status":"error","returncode":result.returncode}))
   raise SystemExit(result.returncode)
(r/"profile_status.json").write_text(json.dumps({"status":"complete"}))
