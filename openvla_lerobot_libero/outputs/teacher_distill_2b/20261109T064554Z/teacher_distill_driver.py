"""Resume-safe supervisor for the isolated teacher-to-student experiment."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

RUN = Path(__file__).resolve().parent
SCRIPT = RUN / "teacher_distill_2b.py"
PYTHON = "/home/jovyan/anaconda3/envs/lerobot312/bin/python"
PHASES = [
    ("evaluate_before", "student_before_evaluation.json"),
    ("train", "training_result.json"),
    ("evaluate_after", "student_after_evaluation.json"),
]


def write(obj):
    path = RUN / "pipeline_status.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


try:
    started = time.time()
    while not (RUN / "teacher_collection.json").exists():
        if time.time() - started > 6 * 3600:
            raise TimeoutError("Teacher collection did not finish within six hours; inspect collect.log")
        write({"phase": "waiting_for_teacher_collection", "status": "running"})
        time.sleep(20)
    collection = json.loads((RUN / "teacher_collection.json").read_text())
    if not collection["complete"] or collection["new_training_frames"] < 100:
        raise RuntimeError("Too few successful teacher frames; no student update attempted")
    env = os.environ.copy()
    env["OPENVLA_KD_RUN"] = str(RUN)
    for phase, marker in PHASES:
        if (RUN / marker).exists():
            continue
        # Baseline evaluation may have been launched concurrently with collection.
        # Wait for its result instead of starting a second writer for the same 50 states.
        if phase == "evaluate_before" and (RUN / f"{phase}.log").exists():
            wait_started = time.time()
            while not (RUN / marker).exists():
                if time.time() - wait_started > 6 * 3600:
                    raise TimeoutError("Concurrent baseline evaluation did not finish; inspect evaluate_before.log")
                write({"phase": "waiting_for_baseline_evaluation", "status": "running"})
                time.sleep(20)
            continue
        write({"phase": phase, "status": "running", "started": time.time()})
        with (RUN / f"{phase}.log").open("x") as log:
            result = subprocess.run([PYTHON, "-u", str(SCRIPT), phase], env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{phase} failed with exit {result.returncode}; inspect {phase}.log")
        record = json.loads((RUN / marker).read_text())
        if not record["complete"]:
            raise RuntimeError(f"{phase} marker is incomplete")
    write({"phase": "complete", "status": "complete", "completed": time.time()})
except Exception as error:
    write({"phase": "error", "status": "error", "error": repr(error),
           "traceback": traceback.format_exc()})
    raise
