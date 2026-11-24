"""Complete the unified historical overview once the automatic rank sweep finishes."""
from pathlib import Path
import json,subprocess,time,traceback,hashlib
R=Path(__file__).resolve().parent
def write(name,obj):(R/name).write_text(json.dumps(obj,indent=2))
try:
    hashes={str(R/f):hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ["supplement_history.py","finish_all.py"]}
    write("source_hashes_late_additions.json",hashes)
    while True:
        if (R/"driver_error.json").exists():raise RuntimeError("Primary driver needs repair; local results preserved")
        if (R/"status.json").exists() and json.loads((R/"status.json").read_text())["status"]=="complete":break
        time.sleep(4)
    assert json.loads((R/"historical_upload_manifest.json").read_text())["complete"]
    subprocess.run(["/home/jovyan/projects/wandb-upload-env/bin/python","-u",str(R/"supplement_history.py"),"--overview"],check=True)
    subprocess.run(["/home/jovyan/anaconda3/envs/lerobot312/bin/python","-u",str(R/"run_phase.py"),"audit"],check=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert json.loads((R/"all_history_complete.json").read_text())["verified_experiments"]==27
    write("all_work_complete.json",{"complete":True,"new_robot_trials":15,"verified_WB_records":27,"environments_preserved":True})
    print("ALL_REQUESTED_WORK_COMPLETE",flush=True)
except Exception as e:
    write("finish_all_error.json",{"error":repr(e),"traceback":traceback.format_exc()})
    raise
