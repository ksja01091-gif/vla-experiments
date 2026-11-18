"""Verify candidate +/- world-axis end-effector rotation controls in MuJoCo."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from ur5e_colab_sim_worker import make_scene, rotation_from_vector, solve_pose


def trial(root: Path, axis: int | None, sign: int) -> dict:
    model, data, block_addr, pinch, finger = make_scene(root)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[block_addr:block_addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    initial_position = data.site_xpos[pinch].copy()
    initial_rotation = data.site_xmat[pinch].reshape(3, 3).copy()
    home = data.qpos[:6].copy()
    target_joints = home.copy()
    if axis is not None:
        vector = np.zeros(3)
        vector[axis] = sign * math.radians(5)
        target_rotation = rotation_from_vector(vector) @ initial_rotation
        target_joints, _, _ = solve_pose(
            model, data, pinch, initial_position, target_rotation, home
        )
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[block_addr:block_addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
    count = round(1.0 / model.opt.timestep)
    for index in range(count):
        fraction = (index + 1) / count
        fraction = fraction * fraction * (3 - 2 * fraction)
        data.ctrl[:6] = home + fraction * (target_joints - home)
        data.ctrl[finger] = 0
        mujoco.mj_step(model, data)
    final_rotation = data.site_xmat[pinch].reshape(3, 3).copy()
    relative = final_rotation @ initial_rotation.T
    small_angle = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ]) / 2
    return {"axis": axis, "sign": sign, "rotation_proxy_deg":
            np.degrees(small_angle).tolist(), "physics_steps": count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = np.array(trial(args.root, None, 0)["rotation_proxy_deg"])
    results = []
    for axis, name in enumerate("xyz"):
        for sign in (-1, 1):
            item = trial(args.root, axis, sign)
            corrected = np.array(item["rotation_proxy_deg"]) - baseline
            item["command"] = ("+" if sign > 0 else "-") + name
            item["baseline_corrected_deg"] = corrected.tolist()
            item["commanded_axis_went_expected_way"] = bool(corrected[axis] * sign > 0)
            results.append(item)
            print(item["command"], np.round(corrected, 3),
                  item["commanded_axis_went_expected_way"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"baseline_deg": baseline.tolist(),
        "trials": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
