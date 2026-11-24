import contextlib,json,sys
from pathlib import Path
from analyze_openvla_bowl_motion import analyze
run=Path("/home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z")
with contextlib.redirect_stdout(sys.stderr):
    report=analyze(run/"simulation/lora/state_00")
print("LORA_ZERO_MOTION_COMPLETE",json.dumps(report),flush=True)
