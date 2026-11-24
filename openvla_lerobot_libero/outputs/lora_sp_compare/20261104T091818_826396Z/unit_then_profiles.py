from pathlib import Path
import subprocess,json
r=Path(__file__).parent
python="/home/jovyan/anaconda3/envs/lerobot312/bin/python"
with (r/"sp_unit_final.log").open("w") as log:
 x=subprocess.run([python,"-u",str(r/"check_sp_math.py")],stdout=log,stderr=subprocess.STDOUT)
if x.returncode:raise SystemExit(x.returncode)
with (r/"profiles.log").open("w") as log:
 x=subprocess.run([python,"-u",str(r/"run_profiles.py")],stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(x.returncode)
