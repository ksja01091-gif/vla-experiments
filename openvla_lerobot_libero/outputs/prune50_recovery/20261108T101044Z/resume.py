"""Resume the isolated remote run after SFT evaluation, preserving original cap."""
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path

RUN = Path(__file__).resolve().parent
PY = "/home/jovyan/anaconda3/envs/lerobot312/bin/python"
EVAL_PY = "/home/jovyan/projects/venvs/openvla_eval310/bin/python"


def write(name, value):
    path = RUN / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def call(label, command):
    deadline = json.loads((RUN / "budget.json").read_text())["deadline_unix"]
    if time.time() > deadline - 180:
        raise TimeoutError("Original three-hour cap reached before " + label)
    write("status.json", {"status": "running", "phase": label, "updated_unix": time.time()})
    print("BEGIN", label, flush=True)
    started = time.monotonic()
    with (RUN / (label + ".log")).open("w") as output:
        result = subprocess.run(command, cwd=RUN, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, timeout=max(60, deadline - time.time() - 60))
    if result.returncode:
        raise RuntimeError(label + " failed; see " + label + ".log")
    write(label + "_done.json", {"complete": True, "seconds": time.monotonic() - started})
    print("DONE", label, round(time.monotonic() - started, 1), flush=True)


try:
    assert (RUN / "pruned/simulation_sft3_result.json").is_file()
    assert (RUN / "dense/simulation_sft3_result.json").is_file()
    eval_env = ["env", "PYTHONPATH=/home/jovyan/projects/LIBERO", EVAL_PY, "-u", "replay.py"]
    call("replay_after_sft", eval_env + ["--tags", "pruned_sft3", "dense_sft3"])
    shutil.copy2(RUN / "replay_results.json", RUN / "replay_results_after_sft.json")
    for branch in ("pruned", "dense"):
        call("collect_" + branch, [PY, "-u", "run.py", "collect", "--branch", branch])
    call("matched_budget", [PY, "-u", "run.py", "budget"])
    for branch in ("pruned", "dense"):
        call("ppo_" + branch, [PY, "-u", "run.py", "ppo", "--branch", branch])
    for branch in ("pruned", "dense"):
        call("rl_eval_" + branch, [PY, "-u", "run.py", "evaluate_rl", "--branch", branch])
    call("replay_after_rl", eval_env + ["--tags", "pruned", "dense"])
    shutil.copy2(RUN / "replay_results.json", RUN / "replay_results_after_rl.json")
    call("summary", [PY, "-u", "run.py", "report"])
    write("status.json", {"status": "complete", "resumed_after_import_path_fix": True})
except Exception as error:
    write("status.json", {"status": "error", "error": repr(error),
        "traceback": traceback.format_exc()})
    raise
