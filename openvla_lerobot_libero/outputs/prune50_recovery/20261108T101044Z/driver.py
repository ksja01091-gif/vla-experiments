"""Remote-only supervised sequence. Each phase has a fresh Python/GPU process."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

RUN = Path(__file__).resolve().parent
PY = "/home/jovyan/anaconda3/envs/lerobot312/bin/python"
EVAL_PY = "/home/jovyan/projects/venvs/openvla_eval310/bin/python"


def write(name, data):
    path = RUN / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def call(label, command):
    budget = json.loads((RUN / "budget.json").read_text())
    if time.time() > budget["deadline_unix"] - 180:
        raise TimeoutError("Three-hour run cap reached before " + label)
    write("status.json", {"status": "running", "phase": label, "updated_unix": time.time()})
    print("BEGIN", label, flush=True)
    started = time.monotonic()
    with (RUN / (label + ".log")).open("w") as log:
        proc = subprocess.run(command, cwd=RUN, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT, timeout=max(60, budget["deadline_unix"] - time.time() - 60))
    if proc.returncode:
        raise RuntimeError(label + " failed; see " + label + ".log")
    write(label + "_done.json", {"complete": True, "seconds": time.monotonic() - started})
    print("DONE", label, round(time.monotonic() - started, 1), flush=True)


try:
    start = time.time()
    write("budget.json", {"started_unix": start, "deadline_unix": start + 10800,
                          "cap_seconds": 10800, "previous_runs_read_only": True})
    write("source_hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in RUN.glob("*.py")})
    for branch in ("pruned", "dense"):
        call("sft_" + branch, [PY, "-u", "run.py", "sft", "--branch", branch])
    for branch in ("pruned", "dense"):
        call("sft_eval_" + branch, [PY, "-u", "run.py", "simulate_sft", "--branch", branch])
    call("replay_after_sft", [EVAL_PY, "-u", "replay.py", "--tags", "pruned_sft3", "dense_sft3"])
    shutil.copy2(RUN / "replay_results.json", RUN / "replay_results_after_sft.json")
    for branch in ("pruned", "dense"):
        call("collect_" + branch, [PY, "-u", "run.py", "collect", "--branch", branch])
    call("matched_budget", [PY, "-u", "run.py", "budget"])
    for branch in ("pruned", "dense"):
        call("ppo_" + branch, [PY, "-u", "run.py", "ppo", "--branch", branch])
    for branch in ("pruned", "dense"):
        call("rl_eval_" + branch, [PY, "-u", "run.py", "evaluate_rl", "--branch", branch])
    call("replay_after_rl", [EVAL_PY, "-u", "replay.py", "--tags", "pruned", "dense"])
    shutil.copy2(RUN / "replay_results.json", RUN / "replay_results_after_rl.json")
    call("summary", [PY, "-u", "run.py", "report"])
    write("status.json", {"status": "complete", "total_seconds": time.time() - start})
except Exception as error:
    write("status.json", {"status": "error", "error": repr(error),
                          "traceback": traceback.format_exc()})
    raise
