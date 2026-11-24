"""One LIBERO Spatial task/initial-state trial with reference-matched conditions.
This is one trial, not a benchmark-wide success rate.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from lerobot.policies.openvla.modeling_openvla import OpenVLAPolicy
from openvla_libero_motion import controller_action, send, read_observation

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path("/home/jovyan/projects/openvla")
EVAL_REFERENCE = REFERENCE_ROOT/"experiments/robot/libero/run_libero_eval.py"


def reference_settings():
    tree = ast.parse(EVAL_REFERENCE.read_text())
    horizon = None
    wait = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(isinstance(n,ast.Constant) and n.value=="libero_spatial" for n in ast.walk(node.test)):
            for statement in node.body:
                if isinstance(statement,ast.Assign) and any(isinstance(n,ast.Name) and n.id=="max_steps" for n in statement.targets):
                    horizon = ast.literal_eval(statement.value)
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=="num_steps_wait":
            wait = ast.literal_eval(node.value)
    if (horizon,wait) != (220,10):
        raise RuntimeError(f"Reference settings changed: horizon={horizon}, wait={wait}")
    return horizon,wait


def set_reference_seed():
    path = REFERENCE_ROOT/"experiments/robot/robot_utils.py"
    tree = ast.parse(path.read_text())
    function = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="set_seed_everywhere"]
    if len(function)!=1:
        raise RuntimeError("Could not locate official seed function.")
    namespace={"torch":torch,"np":np,"random":random,"os":os}
    exec(compile(ast.Module(body=function,type_ignores=[]),str(path),"exec"),namespace)
    namespace["set_seed_everywhere"](7)


def save_media(paths, output, success):
    images=[]
    for index,path in enumerate(paths):
        with Image.open(path) as source:
            image=source.convert("RGB")
        label=f"step {index}"
        if index==len(paths)-1:
            label+=" SUCCESS" if success else " HORIZON REACHED"
        ImageDraw.Draw(image).text((6,6),label,fill="white",stroke_width=2,stroke_fill="black")
        images.append(image)
    with imageio.get_writer(str(output/"episode.mp4"),fps=20,codec="libx264",pixelformat="yuv420p") as writer:
        for image in images:
            writer.append_data(np.asarray(image))
    images[0].save(output/"episode.gif",save_all=True,append_images=images[1:],duration=50,loop=0)
    sheet=Image.new("RGB",(768,560),"white")
    indices=np.linspace(0,len(images)-1,6,dtype=int)
    for cell,index in enumerate(indices):
        x,y=(cell%3)*256,(cell//3)*280
        sheet.paste(images[index],(x,y+24))
        ImageDraw.Draw(sheet).text((x+5,y+6),f"step {index}",fill="black")
    sheet.save(output/"episode_overview.png")
    with Image.open(paths[0]) as before, Image.open(paths[-1]) as after:
        comparison=Image.new("RGB",(512,280),"white")
        comparison.paste(before.convert("RGB"),(0,24))
        comparison.paste(after.convert("RGB"),(256,24))
        draw=ImageDraw.Draw(comparison)
        draw.text((6,6),"Initial state 0",fill="black")
        draw.text((262,6),"Task SUCCESS" if success else "Horizon exhausted",fill="black")
        comparison.save(output/"before_after.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id",type=int,default=0)
    parser.add_argument("--init-state-index",type=int,default=0)
    parser.add_argument("--sim-python",type=Path,default=Path.home()/"projects/venvs/openvla_eval310/bin/python")
    args=parser.parse_args()
    # Do not resolve this symlink: the path identifies the isolated virtualenv.
    sim_python=args.sim_python.expanduser().absolute()
    if not sim_python.is_file():
        raise FileNotFoundError(sim_python)
    horizon,wait=reference_settings()
    set_reference_seed()
    output=ROOT/"outputs/libero_single_task"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    env=os.environ.copy()
    env.update({"MUJOCO_GL":"egl","TF_CPP_MIN_LOG_LEVEL":"2","TF_NUM_INTRAOP_THREADS":"2","TF_NUM_INTEROP_THREADS":"1","TOKENIZERS_PARALLELISM":"false"})
    env["PYTHONPATH"]=str(Path.home()/"projects/LIBERO")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    worker=ROOT/"examples/libero_eval_worker.py"
    records=[]
    process=None
    log=(output/"simulator.log").open("w")
    try:
        print("OUTPUT:",output,flush=True)
        print("REFERENCE_SETTINGS:",horizon,"policy actions;",wait,"warmup steps; state index",args.init_state_index,flush=True)
        process=subprocess.Popen([str(sim_python),"-u",str(worker),"--output",str(output),"--task-id",str(args.task_id),"--init-state-index",str(args.init_state_index)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
        initial=current=read_observation(process,timeout=240)
        if initial["type"]!="ready":
            raise RuntimeError("Simulator did not initialize correctly.")
        (output/"initialization.json").write_text(json.dumps(initial,indent=2))
        print("OFFICIAL_STATE_READY:",initial["init_state_index"],"of",initial["available_init_states"],flush=True)
        print("PREPROCESSING_PARITY:",initial["actual_initial_image_reference_check"]["max_pixel_error"],"maximum pixel error",flush=True)
        print("TASK:",initial["task"],flush=True)
        low,high=np.array(initial["action_low"]),np.array(initial["action_high"])
        config=OpenVLAConfig(device="cuda:0")
        policy=OpenVLAPolicy(config)
        policy.reset()
        frames=[initial["frame"]]
        for step in range(1,horizon+1):
            with Image.open(current["policy_frame"]) as source:
                image=source.convert("RGB")
            tensor=pil_to_tensor(image).float().div(255)
            if not np.array_equal(np.asarray(to_pil_image(tensor)),np.asarray(image)):
                raise RuntimeError("LeRobot image tensor roundtrip changed processed pixels.")
            batch={config.image_key:tensor.unsqueeze(0),"task":[initial["task"]]}
            action=policy.select_action(batch)
            if action.shape!=(1,7) or not bool(torch.isfinite(action).all()):
                raise RuntimeError("Invalid policy action.")
            raw=action[0].detach().cpu().numpy()
            converted,command=controller_action(raw,low,high)
            if not np.array_equal(converted,command):
                raise RuntimeError("Action requires clipping, which would differ from reference evaluation.")
            before=current
            send(process,{"type":"step","action":command.tolist()})
            current=read_observation(process)
            if current["type"]!="observation" or current["step"]!=step:
                raise RuntimeError("Unexpected simulator response.")
            record={"step":step,"raw_action":raw.tolist(),"controller_action":command.tolist(),"before":before,"after":current}
            records.append(record)
            with (output/"steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record,allow_nan=False)+"\n")
            frames.append(current["frame"])
            if step==1 or step%10==0 or current["task_success"]:
                print(f"STEP {step:03d}/{horizon}: task_success={current['task_success']} finite=True",flush=True)
            if current["task_success"]:
                break
        success=bool(current["task_success"])
        save_media(frames,output,success)
        result={"trial_executed":True,"scope":"one_task_one_official_initial_state",
                "task_suite":"libero_spatial","task_id":args.task_id,"task":initial["task"],
                "init_state_index":args.init_state_index,"success":success,
                "termination_reason":"task_success" if success else "horizon_exhausted",
                "policy_steps":len(records),"max_policy_steps":horizon,"warmup_steps":wait,
                "official_initial_state_used":True,"reference_preprocessing_pixel_parity":True,
                "all_tensor_roundtrips_exact":True,"all_actions_finite":True,"action_shape":[1,7],
                "action_clipping_used":False,"environment_seed":0,"global_seed":7,
                "initialization":initial,"final_state":current,
                "model_id":config.model_id,"checkpoint_path":policy.checkpoint_path,
                "attention_backend":getattr(policy.model.config,"_attn_implementation",None),
                "model_dtype":str(next(policy.model.parameters()).dtype),
                "inference_versions":{n:version(n) for n in ["lerobot","torch","transformers","timm","peft","numpy"]},
                "reference_eval_sha256":hashlib.sha256(EVAL_REFERENCE.read_bytes()).hexdigest(),
                "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),worker,ROOT/"examples/openvla_eval_preprocessing.py"]},
                "simulated_seconds_after_warmup":current["sim_time"]-initial["sim_time"],
                "video_fps":20,"benchmark_wide_success_rate_measured":False,
                "training_run":False,"wandb_used":False}
        (output/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        print("TRIAL_COMPLETE: success=",success,"steps=",len(records),"reason=",result["termination_reason"],flush=True)
        print("VIDEO:",output/"episode.mp4",flush=True)
        print("OVERVIEW:",output/"episode_overview.png",flush=True)
        print("RESULT:",output/"result.json",flush=True)
    except Exception as error:
        (output/"error.json").write_text(json.dumps({"error":str(error),"completed_steps":len(records)},indent=2))
        log.flush()
        print((output/"simulator.log").read_text()[-7000:],file=sys.stderr)
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                send(process,{"type":"close"})
                process.wait(timeout=15)
            except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
        log.close()


if __name__=="__main__":
    main()
