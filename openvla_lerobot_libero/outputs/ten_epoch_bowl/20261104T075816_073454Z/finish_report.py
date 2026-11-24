from pathlib import Path
import json,time,subprocess,traceback
r=Path(__file__).parent
try:
 deadline=time.monotonic()+10800
 while True:
  status=json.loads((r/"pipeline_status.json").read_text())
  if status["status"]=="complete":break
  if status["status"]=="error":raise RuntimeError(status)
  if time.monotonic()>deadline:raise TimeoutError("experiment still running; results preserved")
  time.sleep(5)
 subprocess.run(["/home/jovyan/anaconda3/envs/lerobot312/bin/python",str(r/"build_report.py")],check=True)
 subprocess.run(["/home/jovyan/anaconda3/envs/openvla310/bin/python",str(r/"build_visuals.py")],check=True)
except Exception:
 (r/"report_error.txt").write_text(traceback.format_exc())
 raise
