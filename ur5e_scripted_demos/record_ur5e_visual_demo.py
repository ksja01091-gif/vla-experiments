"""Record one scripted UR5e grasp with aligned images and actuator commands.

This records a scripted teacher demonstration, not an OpenVLA rollout.
The 7 stored commands are six joint target angles plus a gripper actuator
command; they are NOT yet OpenVLA's Cartesian action representation.
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from ur5e_auto_grasp import make_scene, solve_pose


def record(output: Path, seed: int = 7) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    frames = output / "frames"
    frames.mkdir(exist_ok=True)

    model, data, block_addr = make_scene()
    pinch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator")
    home = data.qpos[:6].copy()
    orientation = data.site_xmat[pinch].reshape(3, 3).copy()
    above, _, _ = solve_pose(model, data, pinch, np.array([0.45, 0, 0.19]), orientation, home)
    grasp, _, _ = solve_pose(model, data, pinch, np.array([0.45, 0, 0.035]), orientation, above)
    lift, _, _ = solve_pose(model, data, pinch, np.array([0.45, 0, 0.27]), orientation, grasp)

    # IK calculations move qpos temporarily; reset before the actual episode.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[block_addr : block_addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.2, 0, 0.28]
    camera.distance = 1.8
    camera.azimuth = 130
    camera.elevation = -35

    stages = [
        ("approach_above", above, 0, 2.0),
        ("descend", grasp, 0, 2.0),
        ("close", grasp, 255, 1.5),
        ("lift", lift, 255, 2.5),
        ("hold", lift, 255, 1.0),
    ]
    dt = float(model.opt.timestep)
    steps_per_frame = round(0.1 / dt)
    if abs(steps_per_frame * dt - 0.1) > 1e-9:
        raise ValueError("0.1 seconds must be an integer number of physics steps")
    rows = []
    initial_z = float(data.qpos[block_addr + 2])
    peak_z = initial_z
    start = data.qpos[:6].copy()
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        for stage, target, finger_command, duration in stages:
            steps = round(duration / dt)
            if steps % steps_per_frame:
                raise ValueError(f"Stage duration is not divisible by 0.1s: {stage}")
            for sample in range(steps // steps_per_frame):
                # Save the observation BEFORE applying this interval's commands.
                frame_name = f"frame_{len(rows):04d}.png"
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(frames / frame_name)
                fraction = (sample * steps_per_frame + 1) / steps
                fraction = fraction * fraction * (3 - 2 * fraction)
                arm_command = start + fraction * (target - start)
                rows.append({
                    "frame": f"frames/{frame_name}",
                    "time_s": round(float(data.time), 4),
                    "stage": stage,
                    "instruction": "pick up the orange block",
                    "block_z_m": float(data.qpos[block_addr + 2]),
                    "pinch_z_m": float(data.site_xpos[pinch, 2]),
                    **{f"joint_target_{i+1}_rad": float(arm_command[i]) for i in range(6)},
                    "gripper_target": finger_command,
                })
                for substep in range(steps_per_frame):
                    fraction = (sample * steps_per_frame + substep + 1) / steps
                    fraction = fraction * fraction * (3 - 2 * fraction)
                    data.ctrl[:6] = start + fraction * (target - start)
                    data.ctrl[finger] = finger_command
                    mujoco.mj_step(model, data)
                    peak_z = max(peak_z, float(data.qpos[block_addr + 2]))
            start = target

        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / "final.png")
    finally:
        renderer.close()

    with (output / "demonstration.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    final_z = float(data.qpos[block_addr + 2])
    hold_rows = [row for row in rows if row["stage"] == "hold"]
    held_above_5cm = bool(hold_rows and min(row["block_z_m"] for row in hold_rows) - initial_z >= 0.05)
    summary = {
        "result": "scripted_grasp_lift_success" if held_above_5cm else "not_verified_success",
        "controller": "scripted inverse kinematics, not OpenVLA",
        "seed": seed,
        "randomization_enabled": False,
        "instruction": "pick up the orange block",
        "recording_hz": 10,
        "physics_timestep_s": dt,
        "frames_with_aligned_actions": len(rows),
        "initial_block_z_m": initial_z,
        "peak_block_z_m": peak_z,
        "final_block_z_m": final_z,
        "held_above_initial_plus_5cm_through_hold_stage": held_above_5cm,
        "action_format": "six joint target angles (radians) plus gripper actuator target (0 open, 255 close)",
        "openvla_training_ready": False,
        "training_gap": "Convert joint targets to consistent 6D end-effector deltas + gripper; set action normalization and verify varied successful demonstrations",
        "files": ["demonstration.csv", "frames/", "final.png"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(record(args.output, args.seed), indent=2))
