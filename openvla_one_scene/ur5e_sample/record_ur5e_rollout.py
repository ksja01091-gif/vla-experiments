"""Record one scripted UR5e grasp in MuJoCo; no OpenVLA is used here.

Run with the existing /home/jw/ur5e-venv. The original controller is copied
beside this file and remains unchanged.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import mujoco
import numpy as np

from ur5e_auto_grasp import make_scene, solve_pose


def record_one_rollout(output_dir: Path, seed: int = 7) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    model, data, block_qpos_addr = make_scene()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator")
    driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper/left_driver_joint")
    driver_qpos_addr = model.jnt_qposadr[driver_id]

    home = data.qpos[:6].copy()
    orientation = data.site_xmat[site_id].reshape(3, 3).copy()
    above, _, _ = solve_pose(model, data, site_id, np.array([0.45, 0, 0.19]), orientation, home)
    grasp, _, _ = solve_pose(model, data, site_id, np.array([0.45, 0, 0.035]), orientation, above)
    lift, _, _ = solve_pose(model, data, site_id, np.array([0.45, 0, 0.27]), orientation, grasp)

    # Pose solving temporarily moves the arm; start the physical rollout afresh.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[block_qpos_addr : block_qpos_addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    stages = [
        ("above", above, 0, 2.0),
        ("down", grasp, 0, 2.0),
        ("close", grasp, 255, 1.5),
        ("lift", lift, 255, 2.5),
        ("hold", lift, 255, 1.0),
    ]
    dt = float(model.opt.timestep)
    sample_every_steps = round(0.1 / dt)
    if abs(sample_every_steps * dt - 0.1) > 1e-9:
        raise ValueError("Physics timestep does not divide the 0.1 s recording interval")

    def capture(stage: str) -> dict:
        pos = data.site_xpos[site_id]
        rotation = data.site_xmat[site_id].reshape(3, 3)
        row = {
            "time_s": round(float(data.time), 6),
            "stage": stage,
            "ee_x_m": float(pos[0]),
            "ee_y_m": float(pos[1]),
            "ee_z_m": float(pos[2]),
            "gripper_command": float(data.ctrl[gripper_id]),
            "gripper_driver_rad": float(data.qpos[driver_qpos_addr]),
            "block_x_m": float(data.qpos[block_qpos_addr]),
            "block_y_m": float(data.qpos[block_qpos_addr + 1]),
            "block_z_m": float(data.qpos[block_qpos_addr + 2]),
        }
        row.update({f"ee_rot_{i}{j}": float(rotation[i, j]) for i in range(3) for j in range(3)})
        row.update({f"arm_joint_{i + 1}_rad": float(data.qpos[i]) for i in range(6)})
        row.update({f"arm_command_{i + 1}_rad": float(data.ctrl[i]) for i in range(6)})
        return row

    rows = [capture("start")]
    start = data.qpos[:6].copy()
    peak_block_z = float(data.qpos[block_qpos_addr + 2])
    physics_step = 0
    for name, target, finger, duration in stages:
        steps = round(duration / dt)
        for step in range(steps):
            fraction = min(1, (step + 1) / steps)
            fraction = fraction * fraction * (3 - 2 * fraction)
            data.ctrl[:6] = start + fraction * (target - start)
            data.ctrl[gripper_id] = finger
            mujoco.mj_step(model, data)
            physics_step += 1
            peak_block_z = max(peak_block_z, float(data.qpos[block_qpos_addr + 2]))
            if physics_step % sample_every_steps == 0:
                rows.append(capture(name))
        start = target

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "controller": str(Path(__file__).with_name("ur5e_auto_grasp.py")),
        "controller_type": "scripted inverse kinematics; not OpenVLA",
        "seed": seed,
        "randomization_enabled": False,
        "physics_timestep_s": dt,
        "recording_interval_s": sample_every_steps * dt,
        "physics_steps": physics_step,
        "recorded_rows_including_start": len(rows),
        "duration_s": float(data.time),
        "initial_block_z_m": rows[0]["block_z_m"],
        "final_block_z_m": float(data.qpos[block_qpos_addr + 2]),
        "peak_block_z_m": peak_block_z,
        "stages": [name for name, _, _, _ in stages],
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "csv_path": str(csv_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "trajectory")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(record_one_rollout(args.output_dir, args.seed), indent=2))
