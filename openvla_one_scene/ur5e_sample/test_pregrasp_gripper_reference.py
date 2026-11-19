"""Compare open and close from one saved pregrasp state in MuJoCo.

Both branches start from the same state and follow the same scripted lift.
The OpenVLA model is not called here; this establishes a physical reference.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from ur5e_auto_grasp import make_scene, solve_pose


def trial(state_path: Path, command: int) -> dict:
    model, data, block_addr = make_scene()
    with np.load(state_path) as saved:
        data.qpos[:] = saved["qpos"]
        data.qvel[:] = saved["qvel"]
        data.act[:] = saved["act"]
        data.time = float(saved["time"])
    mujoco.mj_forward(model, data)
    pinch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator")
    orientation = data.site_xmat[pinch].reshape(3, 3).copy()
    grasp = data.qpos[:6].copy()
    lift, _, _ = solve_pose(model, data, pinch, np.array([0.45, 0, 0.27]), orientation, grasp)
    # Inverse kinematics above changes qpos; restore the identical initial state.
    with np.load(state_path) as saved:
        data.qpos[:] = saved["qpos"]
        data.qvel[:] = saved["qvel"]
        data.act[:] = saved["act"]
        data.time = float(saved["time"])
    mujoco.mj_forward(model, data)

    initial_height = float(data.qpos[block_addr + 2])
    peak_height = initial_height
    for name, target, duration in (("wait", grasp, 1.5), ("lift", lift, 2.5)):
        start = data.qpos[:6].copy() if name == "wait" else grasp
        steps = round(duration / model.opt.timestep)
        for step in range(steps):
            fraction = (step + 1) / steps
            fraction = fraction * fraction * (3 - 2 * fraction)
            data.ctrl[:6] = start + fraction * (target - start)
            data.ctrl[finger] = command
            mujoco.mj_step(model, data)
            peak_height = max(peak_height, float(data.qpos[block_addr + 2]))
    return {
        "gripper_command": command,
        "initial_block_height_m": initial_height,
        "peak_block_height_m": peak_height,
        "final_block_height_m": float(data.qpos[block_addr + 2]),
        "lifted_5cm": peak_height - initial_height >= 0.05,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"open": trial(args.state, 0), "close": trial(args.state, 255)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
