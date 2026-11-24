"""Short closed-loop motion demo, NOT a LIBERO benchmark evaluation.
Uses lerobot312 for the policy and openvla310 for the simulator.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import pil_to_tensor

from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from lerobot.policies.openvla.modeling_openvla import OpenVLAPolicy

ROOT = Path(__file__).resolve().parents[1]


def controller_action(raw, low, high):
    """Official OpenVLA LIBERO gripper normalization, binarization, inversion."""
    converted = np.asarray(raw, dtype=np.float64).copy()
    converted[-1] = -np.sign(2.0 * converted[-1] - 1.0)
    return converted, np.clip(converted, low, high)


def read_observation(process, timeout=180):
    readable, _, _ = select.select([process.stdout], [], [], timeout)
    if not readable:
        raise TimeoutError("Simulator did not return an observation in time.")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("Simulator exited; inspect simulator.log.")
    return json.loads(line)


def send(process, payload):
    process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
    process.stdin.flush()


def save_media(frame_paths, output):
    images = []
    for index, path in enumerate(frame_paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
        ImageDraw.Draw(image).text((6, 6), f"step {index}", fill="white", stroke_width=2, stroke_fill="black")
        images.append(image)
    images[0].save(output / "motion.gif", save_all=True, append_images=images[1:], duration=200, loop=0)
    with imageio.get_writer(str(output / "motion.mp4"), fps=5, codec="libx264", pixelformat="yuv420p") as writer:
        for image in images:
            writer.append_data(np.asarray(image))
    comparison = Image.new("RGB", (512, 280), "white")
    for x, path, label in [(0, frame_paths[0], "Before model actions"), (256, frame_paths[-1], f"After {len(frame_paths)-1} model actions")]:
        with Image.open(path) as source:
            comparison.paste(source.convert("RGB"), (x, 24))
        ImageDraw.Draw(comparison).text((x+6, 6), label, fill="black")
    comparison.save(output / "before_after.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--sim-python", type=Path, default=Path.home()/"anaconda3/envs/openvla310/bin/python")
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("Use 1 to 100 steps for this connection demo.")
    sim_python = args.sim_python.expanduser().resolve(strict=True)
    output = ROOT / "outputs/libero_motion" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYTHONPATH"] = str(Path.home()/"projects/LIBERO") + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    worker = ROOT/"examples/libero_motion_worker.py"
    process = None
    records = []
    log = (output/"simulator.log").open("w")
    try:
        print("OUTPUT:", output, flush=True)
        process = subprocess.Popen(
            [str(sim_python), "-u", str(worker), "--output", str(output), "--seed", str(args.seed), "--task-id", str(args.task_id)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=env,
        )
        initial = current = read_observation(process)
        if initial["type"] != "ready":
            raise RuntimeError("Expected initial simulator observation.")
        print("SIMULATOR_READY:", initial["task"], flush=True)
        print("SIMULATOR_PYTHON:", sim_python, flush=True)
        low, high = np.array(initial["action_low"]), np.array(initial["action_high"])
        config = OpenVLAConfig(device="cuda:0")
        torch.manual_seed(args.seed)
        policy = OpenVLAPolicy(config)
        policy.reset()
        frames = [initial["frame"]]
        print("POLICY_READY: closed-loop movement starts", flush=True)
        for step in range(1, args.steps+1):
            with Image.open(current["frame"]) as source:
                image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            batch = {config.image_key: pil_to_tensor(image).float().div(255).unsqueeze(0), "task": [initial["task"]]}
            action = policy.select_action(batch)
            if action.shape != (1, 7) or not bool(torch.isfinite(action).all()):
                raise RuntimeError("Invalid policy action.")
            raw = action[0].detach().cpu().numpy()
            converted, command = controller_action(raw, low, high)
            before = current
            send(process, {"type": "step", "action": command.tolist()})
            current = read_observation(process)
            if current["type"] != "observation" or current["step"] != step:
                raise RuntimeError("Unexpected simulator response.")
            record = {"step": step, "raw_action": raw.tolist(), "converted_action": converted.tolist(),
                      "controller_action": command.tolist(), "clipped": bool(np.any(command != converted)),
                      "before": before, "after": current}
            records.append(record)
            with (output/"steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
            frames.append(current["frame"])
            delta = np.linalg.norm(np.array(current["eef_position"])-np.array(before["eef_position"]))
            print(f"STEP {step:02d}: shape=(1,7) finite=True eef_step_delta={delta:.6f}", flush=True)
            if current["done_signal"]:
                print("Environment done signal received; stopping this demo.", flush=True)
                break
        positions = np.array([initial["eef_position"]]+[r["after"]["eef_position"] for r in records])
        joints = np.array([initial["joint_positions"]]+[r["after"]["joint_positions"] for r in records])
        max_eef = float(np.linalg.norm(positions-positions[0], axis=1).max())
        max_joint = float(np.abs(joints-joints[0]).max())
        if max_eef <= 1e-6 or max_joint <= 1e-6:
            raise RuntimeError("No measurable robot movement; inspect recorded steps.")
        save_media(frames, output)
        sim_versions = subprocess.check_output([str(sim_python), "-c", "import json; from importlib.metadata import version; print(json.dumps({n:version(n) for n in ['mujoco','robosuite','numpy']}))"], text=True)
        result = {
            "status": "motion_connection_passed", "scope": "short_closed_loop_demo_not_benchmark",
            "steps": len(records), "task": initial["task"], "task_id": args.task_id, "seed": args.seed,
            "initial_state": initial, "final_state": current,
            "max_eef_displacement_m": max_eef, "max_joint_change_rad": max_joint,
            "simulated_seconds_after_warmup": current["sim_time"]-initial["sim_time"],
            "control_frequency": initial["control_frequency"], "video_playback_fps": 5,
            "all_action_shapes": [1,7], "all_actions_finite": True,
            "clipped_step_count": sum(r["clipped"] for r in records),
            "gripper_conversion": "-sign(2 * raw_gripper - 1), matching official LIBERO conversion",
            "policy_class": type(policy).__name__, "checkpoint_path": policy.checkpoint_path,
            "preprocessing": "180-degree camera rotation in worker; PIL bilinear resize to 224; RGB CHW float /255",
            "official_benchmark_preprocessing": False, "benchmark_initial_states_used": False,
            "training_run": False, "benchmark_success_rate_measured": False, "wandb_used": False,
            "inference_versions": {n:version(n) for n in ['lerobot','torch','transformers','timm','peft','diffusers','numpy']},
            "simulator_versions": json.loads(sim_versions),
            "source_sha256": {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),worker,ROOT/'src/lerobot/policies/openvla/modeling_openvla.py']},
        }
        (output/"result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
        print("MOTION_CONNECTION_OK:", len(records), "steps", flush=True)
        print("MAX_EEF_DISPLACEMENT_M:", max_eef, "MAX_JOINT_CHANGE_RAD:", max_joint, flush=True)
        print("VIDEO:", output/"motion.mp4", flush=True)
        print("GIF:", output/"motion.gif", flush=True)
        print("RESULT:", output/"result.json", flush=True)
    except Exception as error:
        (output/"error.json").write_text(json.dumps({"error":str(error),"completed_steps":len(records)},indent=2))
        log.flush()
        print((output/"simulator.log").read_text()[-5000:], file=sys.stderr)
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                send(process, {"type":"close"})
                process.wait(timeout=15)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        log.close()


if __name__ == "__main__":
    main()
