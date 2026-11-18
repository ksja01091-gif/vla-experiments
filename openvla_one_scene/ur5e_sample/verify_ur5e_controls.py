"""Check UR5e world-axis directions and Robotiq gripper commands in MuJoCo.

No OpenVLA inference, teacher model, training, or physical robot is involved.
Each test starts from a fresh, identical scene. All distances are measured from
MuJoCo state, not inferred from camera pixels.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import mujoco
import numpy as np

from ur5e_auto_grasp import make_scene, solve_pose


def ids(model):
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator")
    return site, actuator


def run_arm(axis: int | None, sign: int, distance_m: float, duration_s: float):
    model, data, block_addr = make_scene()
    site, actuator = ids(model)
    initial = data.site_xpos[site].copy()
    home = data.qpos[:6].copy()
    target_joints = home.copy()
    ik_error = None
    if axis is not None:
        target = initial.copy()
        target[axis] += sign * distance_m
        orientation = data.site_xmat[site].reshape(3, 3).copy()
        target_joints, ik_error, _ = solve_pose(model, data, site, target, orientation, home)
        # IK temporarily changes qpos; put the robot and block back exactly.
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[block_addr : block_addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
    steps = round(duration_s / model.opt.timestep)
    for step in range(steps):
        fraction = (step + 1) / steps
        fraction = fraction * fraction * (3 - 2 * fraction)
        data.ctrl[:6] = home + fraction * (target_joints - home)
        data.ctrl[actuator] = 0  # Keep the gripper open in the arm tests.
        mujoco.mj_step(model, data)
    return {
        "initial_xyz_m": initial.tolist(),
        "final_xyz_m": data.site_xpos[site].tolist(),
        "raw_delta_mm": ((data.site_xpos[site] - initial) * 1000).tolist(),
        "ik_position_error_m": None if ik_error is None else float(ik_error),
        "physics_steps": steps,
        "physics_dt_s": float(model.opt.timestep),
    }


def run_gripper(command: int, duration_s: float):
    model, data, _ = make_scene()
    _, actuator = ids(model)
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/left_pad1")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/right_pad1")
    driver = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper/left_driver_joint")
    home = data.qpos[:6].copy()
    steps = round(duration_s / model.opt.timestep)
    for _ in range(steps):
        data.ctrl[:6] = home
        data.ctrl[actuator] = command
        mujoco.mj_step(model, data)
    return {
        "command": command,
        "pad_center_distance_mm": float(np.linalg.norm(data.geom_xpos[left] - data.geom_xpos[right]) * 1000),
        "left_driver_joint_rad": float(data.qpos[model.jnt_qposadr[driver]]),
        "physics_steps": steps,
    }


def verify(output_dir: Path, seed: int = 7) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    distance_m = 0.01
    duration_s = 1.0
    baseline = run_arm(None, 0, distance_m, duration_s)
    baseline_delta = np.asarray(baseline["raw_delta_mm"])
    rows = []
    for axis, name in enumerate("xyz"):
        for sign in (1, -1):
            trial = run_arm(axis, sign, distance_m, duration_s)
            raw = np.asarray(trial["raw_delta_mm"])
            corrected = raw - baseline_delta
            rows.append({
                "command": ("+" if sign > 0 else "-") + name,
                "requested_axis_mm": sign * distance_m * 1000,
                "raw_x_mm": float(raw[0]),
                "raw_y_mm": float(raw[1]),
                "raw_z_mm": float(raw[2]),
                "baseline_corrected_x_mm": float(corrected[0]),
                "baseline_corrected_y_mm": float(corrected[1]),
                "baseline_corrected_z_mm": float(corrected[2]),
                "commanded_axis_went_expected_way": bool(corrected[axis] * sign > 0),
                "ik_position_error_m": trial["ik_position_error_m"],
            })
    gripper = [run_gripper(command, duration_s) for command in (0, 255)]
    if not all(row["commanded_axis_went_expected_way"] for row in rows):
        raise RuntimeError("At least one arm direction did not match its command")
    if not gripper[1]["pad_center_distance_mm"] < gripper[0]["pad_center_distance_mm"]:
        raise RuntimeError("Gripper 255 did not close the pads relative to 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "arm_directions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "purpose": "Verify MuJoCo UR5e control semantics; no OpenVLA involved",
        "seed": seed,
        "randomization_enabled": False,
        "coordinate_frame": "MuJoCo world xyz at gripper/pinch site, not screen pixels",
        "requested_arm_distance_mm": distance_m * 1000,
        "duration_per_case_s": duration_s,
        "zero_command_baseline_raw_delta_mm": baseline["raw_delta_mm"],
        "arm_trials": rows,
        "gripper_trials": gripper,
        "all_six_arm_signs_correct": True,
        "gripper_zero_is_open_and_255_is_closed": True,
        "task_success_measured": False,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
    }
    (output_dir / "control_check.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "control_check")
    args = parser.parse_args()
    result = verify(args.output_dir, args.seed)
    print("ARM DIRECTIONS (baseline-corrected mm)")
    for row in result["arm_trials"]:
        print(row["command"], *(round(row[f"baseline_corrected_{axis}_mm"], 2) for axis in "xyz"), row["commanded_axis_went_expected_way"])
    print("GRIPPER (pad-center distance mm)")
    for row in result["gripper_trials"]:
        print(row["command"], round(row["pad_center_distance_mm"], 2))
    print("SAVED", args.output_dir)
