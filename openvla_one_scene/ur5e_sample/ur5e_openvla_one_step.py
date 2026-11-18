"""Smoke-test one *candidate* OpenVLA-to-UR5e action mapping in MuJoCo.

This reuses a previously recorded normalized OpenVLA action. The base model
has no UR5e action statistics, so the position/rotation gains below are
explicit engineering assumptions, not a validated OpenVLA robot controller.
The gripper stays open because the old seventh token required clipping.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from ur5e_auto_grasp import make_scene, solve_pose


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def run_one_step(prediction_path: Path, output_path: Path) -> dict:
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    action = np.asarray(prediction["normalized_action"], dtype=float)
    if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0001):
        raise ValueError("Expected seven finite normalized action values within [-1, 1]")
    if prediction["tokens_clipped_to_action_bins"] != 1:
        raise ValueError("This smoke test expects the previously reported one clipped gripper token")

    # Candidate limits for a single *simulated* command; not trained action scales.
    position_gain_m = 0.01
    rotation_gain_rad = math.radians(5)
    duration_s = 0.1

    model, data, block_addr = make_scene()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator")
    initial_position = data.site_xpos[site_id].copy()
    initial_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    initial_joints = data.qpos[:6].copy()
    initial_block_z = float(data.qpos[block_addr + 2])

    target_position = initial_position + action[:3] * position_gain_m
    target_rotation = rotation_from_vector(action[3:6] * rotation_gain_rad) @ initial_rotation
    target_joints, ik_error_m, _ = solve_pose(
        model, data, site_id, target_position, target_rotation, initial_joints
    )

    # IK used temporary qpos changes. Restore precisely the observation state.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[block_addr : block_addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    steps = round(duration_s / model.opt.timestep)
    for i in range(steps):
        fraction = (i + 1) / steps
        fraction = fraction * fraction * (3 - 2 * fraction)
        data.ctrl[:6] = initial_joints + fraction * (target_joints - initial_joints)
        data.ctrl[gripper_id] = 0  # Keep open; the clipped seventh token is untrusted.
        mujoco.mj_step(model, data)

    final_position = data.site_xpos[site_id].copy()
    final_block_z = float(data.qpos[block_addr + 2])

    # Same initial scene and duration, but hold the arm at its starting joints.
    control_model, control_data, control_block_addr = make_scene()
    control_site_id = mujoco.mj_name2id(
        control_model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch"
    )
    control_gripper_id = mujoco.mj_name2id(
        control_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator"
    )
    control_initial_position = control_data.site_xpos[control_site_id].copy()
    control_home = control_data.qpos[:6].copy()
    for _ in range(steps):
        control_data.ctrl[:6] = control_home
        control_data.ctrl[control_gripper_id] = 0
        mujoco.mj_step(control_model, control_data)
    control_delta = control_data.site_xpos[control_site_id] - control_initial_position

    result = {
        "input_prediction": str(prediction_path),
        "prediction_provenance": prediction["provenance"],
        "model": prediction["model"],
        "image": prediction["image"],
        "instruction": prediction["instruction"],
        "normalized_action": action.tolist(),
        "mapping_status": "candidate engineering smoke test; no UR5e action normalization",
        "position_gain_m_per_normalized_unit": position_gain_m,
        "rotation_gain_rad_per_normalized_unit": rotation_gain_rad,
        "gripper": "held open; clipped token not executed",
        "duration_s": float(data.time),
        "physics_steps": steps,
        "initial_ee_xyz_m": initial_position.tolist(),
        "requested_ee_xyz_m": target_position.tolist(),
        "actual_ee_xyz_m": final_position.tolist(),
        "actual_ee_delta_m": (final_position - initial_position).tolist(),
        "zero_action_control_ee_delta_m": control_delta.tolist(),
        "difference_vs_zero_action_control_m": (
            final_position - initial_position - control_delta
        ).tolist(),
        "requested_ee_delta_m": (target_position - initial_position).tolist(),
        "ik_position_error_m": float(ik_error_m),
        "initial_block_z_m": initial_block_z,
        "final_block_z_m": final_block_z,
        "zero_action_control_final_block_z_m": float(
            control_data.qpos[control_block_addr + 2]
        ),
        "block_lifted": final_block_z - initial_block_z > 0.05,
        "task_success_measured": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_one_step(args.prediction, args.output), indent=2, allow_nan=False))
