"""Five fresh, predeclared LIBERO Spatial task-0 trials (initial states 0..4).
Each child uses the verified single-trial evaluator unchanged.
A valid task failure is different from a technical execution error.
"""
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys
import os
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
INITIAL_STATES = [0, 1, 2, 3, 4]
TASK_ID = 0


def write_summary(output, rows, manifest):
    valid = [row for row in rows if row["status"] in ["success", "task_failure"]]
    errors = [row for row in rows if row["status"] == "technical_error"]
    successes = sum(row["success"] is True for row in valid)
    complete = len(rows) == len(INITIAL_STATES)
    summary = {
        "scope": "one_task_five_predeclared_initial_states",
        "task_id": TASK_ID, "initial_states": INITIAL_STATES,
        "planned_trials": len(INITIAL_STATES), "finished_records": len(rows),
        "valid_trials": len(valid), "technical_errors": len(errors),
        "successful_trials": successes,
        "observed_success_rate": successes/len(INITIAL_STATES) if complete and not errors else None,
        "all_five_trials_finished": complete, "trials": rows, "manifest": manifest,
        "benchmark_wide_success_rate_measured": False, "training_run": False, "wandb_used": False,
    }
    temporary = output/"summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    temporary.replace(output/"summary.json")
    fields = ["init_state_index", "status", "success", "policy_steps", "termination_reason", "returncode", "result_path", "log_path"]
    with (output/"results.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return summary


def draw_summary(output, rows, summary):
    try:
        font = ImageFont.load_default(size=20)
        small = ImageFont.load_default(size=16)
    except TypeError:
        font = small = ImageFont.load_default()
    table = Image.new("RGB",(850,450),"white")
    draw = ImageDraw.Draw(table)
    draw.text((24,16),"LIBERO Spatial / task 0 / fresh trials",font=font,fill="black")
    draw.text((24,58),"Initial state       Result                  Actions",font=font,fill="black")
    for index,row in enumerate(rows):
        color = "#157347" if row["status"]=="success" else "#b02a37"
        text = f"{row['init_state_index']:<20}{row['status']:<26}{row.get('policy_steps')}"
        draw.text((24,105+index*48),text,font=small,fill=color)
    if summary["observed_success_rate"] is not None:
        text = f"Observed: {summary['successful_trials']}/5 = {100*summary['observed_success_rate']:.0f}%"
    else:
        text = "Incomplete or technical error: inspect summary.json"
    draw.text((24,370),text,font=font,fill="black")
    draw.text((24,410),"Five fixed starts of one task; not a benchmark-wide score.",font=small,fill="black")
    table.save(output/"results_table.png")
    comparison = Image.new("RGB",(1280,560),"white")
    draw = ImageDraw.Draw(comparison)
    for column,row in enumerate(rows):
        x = column*256
        draw.text((x+5,5),f"State {row['init_state_index']} / start",font=small,fill="black")
        draw.text((x+5,285),f"{row['status']} / {row.get('policy_steps')} actions",font=small,fill="black")
        if not row.get("result_path"):
            continue
        result = json.loads(Path(row["result_path"]).read_text())
        for y,path in [(24,result["initialization"]["frame"]),(304,result["final_state"]["frame"])]:
            with Image.open(path) as source:
                comparison.paste(source.convert("RGB"),(x,y))
    comparison.save(output/"start_and_finish.png")


def main():
    output = ROOT/"outputs/libero_five_trials"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    evaluator = ROOT/"examples/openvla_libero_single_task.py"
    existing = [evaluator, ROOT/"examples/libero_eval_worker.py",ROOT/"examples/openvla_eval_preprocessing.py",
                ROOT/"src/lerobot/policies/openvla/modeling_openvla.py",ROOT/"src/lerobot/policies/openvla/configuration_openvla.py",
                ROOT/"src/lerobot/policies/groot/__init__.py",ROOT/"src/lerobot/policies/pi0/__init__.py",ROOT/"src/lerobot/policies/pi05/__init__.py"]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in existing}
    environments = {"openvla310":Path.home()/"anaconda3/envs/openvla310/bin/python",
                    "lerobot312":Path(sys.executable),
                    "eval310":Path.home()/"projects/venvs/openvla_eval310/bin/python"}
    before = {}
    for name,executable in environments.items():
        before[name] = subprocess.check_output([str(executable),"-m","pip","freeze"],text=True)
        (output/(name+"_packages_before.txt")).write_text(before[name])
    manifest = {"declared_before_trials":True,"fresh_trials_only":True,"task_id":TASK_ID,
                "initial_states":INITIAL_STATES,"global_seed":7,"environment_seed":0,
                "max_policy_steps":220,"warmup_steps":10,
                "model_id":"openvla/openvla-7b-finetuned-libero-spatial",
                "existing_source_sha256":hashes,
                "batch_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inference_versions":{n:version(n) for n in ["lerobot","torch","transformers","timm","peft","numpy"]}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
    rows = []
    write_summary(output,rows,manifest)
    print("FIVE_TRIAL_OUTPUT:",output,flush=True)
    print("PREDECLARED_STATES:",INITIAL_STATES,"TASK:",TASK_ID,flush=True)
    for state in INITIAL_STATES:
        log_path = output/f"state_{state}.log"
        child_output = None
        print(f"TRIAL_START: state={state}",flush=True)
        started = time.monotonic()
        command = [sys.executable,"-u",str(evaluator),"--task-id",str(TASK_ID),"--init-state-index",str(state)]
        with log_path.open("w") as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush()
                if line.startswith("OUTPUT: "):
                    child_output = Path(line.removeprefix("OUTPUT: ").strip())
                match = re.match(r"STEP (\d+)/",line)
                if line.startswith(("OFFICIAL_STATE_READY:","PREPROCESSING_PARITY:","TRIAL_COMPLETE:")) or (match and int(match.group(1))%20==0):
                    print(f"state {state} | {line.strip()}",flush=True)
            returncode = process.wait()
        row = {"init_state_index":state,"returncode":returncode,"log_path":str(log_path),
               "elapsed_wall_seconds":time.monotonic()-started,"result_path":None,
               "policy_steps":None,"termination_reason":None,"success":None,"status":"technical_error"}
        try:
            if returncode != 0 or child_output is None:
                raise RuntimeError(f"Evaluator exited with return code {returncode}; see {log_path}")
            child_output = child_output.resolve()
            if not child_output.is_relative_to(ROOT/"outputs/libero_single_task"):
                raise RuntimeError("Unexpected child output path.")
            result_path = child_output/"result.json"
            result = json.loads(result_path.read_text())
            assert result["task_id"]==TASK_ID and result["init_state_index"]==state
            assert result["model_id"]==manifest["model_id"]
            assert result["max_policy_steps"]==220 and result["warmup_steps"]==10
            assert result["official_initial_state_used"] and result["reference_preprocessing_pixel_parity"]
            assert result["initialization"]["actual_initial_image_reference_check"]["max_pixel_error"]==0
            assert result["all_actions_finite"] and result["all_tensor_roundtrips_exact"]
            assert result["initialization"]["task_success"] is False
            assert result["success"]==result["final_state"]["task_success"]
            assert 1<=result["policy_steps"]<=220
            assert result["success"] or result["policy_steps"]==220
            if "checkpoint_path" not in manifest:
                manifest["checkpoint_path"]=result["checkpoint_path"]
                manifest["reference_eval_sha256"]=result["reference_eval_sha256"]
            assert result["checkpoint_path"]==manifest["checkpoint_path"]
            assert result["reference_eval_sha256"]==manifest["reference_eval_sha256"]
            row.update({"selected_init_state_sha256":result["initialization"]["selected_init_state_sha256"], "status":"success" if result["success"] else "task_failure", "success":result["success"],
                        "policy_steps":result["policy_steps"],"termination_reason":result["termination_reason"],
                        "result_path":str(result_path)})
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)
        write_summary(output,rows,manifest)
        print(f"TRIAL_RECORDED: state={state} status={row['status']} actions={row['policy_steps']}",flush=True)
    for name,digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Existing source changed during trials: {name}")
    for name,executable in environments.items():
        after = subprocess.check_output([str(executable),"-m","pip","freeze"],text=True)
        (output/(name+"_packages_after.txt")).write_text(after)
        if after!=before[name]:
            raise RuntimeError(f"Environment packages changed: {name}")
    manifest["existing_sources_unchanged"] = True
    manifest["environment_packages_unchanged"] = True
    summary = write_summary(output,rows,manifest)
    draw_summary(output,rows,summary)
    print("FIVE_TRIAL_COMPLETE:",summary["successful_trials"],"successes;",summary["valid_trials"],"valid trials;",summary["technical_errors"],"technical errors",flush=True)
    print("SUMMARY:",output/"summary.json",flush=True)
    print("TABLE:",output/"results_table.png",flush=True)
    print("COMPARISON:",output/"start_and_finish.png",flush=True)
    if summary["technical_errors"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
