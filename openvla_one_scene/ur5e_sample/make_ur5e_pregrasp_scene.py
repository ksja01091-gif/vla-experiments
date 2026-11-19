"""Stage an open UR5e gripper at the block, before any model decision.

The arm motion is scripted. This scene is a conditional gripper diagnostic,
not evidence that OpenVLA approached the block by itself.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from ur5e_auto_grasp import make_scene, solve_pose


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model, data, block_addr = make_scene()
    pinch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator")
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/left_pad1")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/right_pad1")
    home = data.qpos[:6].copy()
    orientation = data.site_xmat[pinch].reshape(3, 3).copy()
    above, _, _ = solve_pose(model, data, pinch, np.array([0.45, 0, 0.19]), orientation, home)
    grasp, _, _ = solve_pose(model, data, pinch, np.array([0.45, 0, 0.035]), orientation, above)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[block_addr : block_addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    start = home
    for target in (above, grasp):
        steps = round(2.0 / model.opt.timestep)
        for step in range(steps):
            fraction = (step + 1) / steps
            fraction = fraction * fraction * (3 - 2 * fraction)
            data.ctrl[:6] = start + fraction * (target - start)
            data.ctrl[finger] = 0
            mujoco.mj_step(model, data)
        start = target

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.2, 0, 0.28]
    camera.distance = 1.8
    camera.azimuth = 130
    camera.elevation = -35
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / "pregrasp.png")
    finally:
        renderer.close()

    np.savez(output / "pregrasp_state.npz", qpos=data.qpos, qvel=data.qvel,
             act=data.act, time=data.time)
    block_pos = data.qpos[block_addr : block_addr + 3].copy()
    result = {
        "scene": "UR5e + Robotiq + orange block; scripted pregrasp",
        "model_inference_performed": False,
        "gripper_command": 0,
        "pinch_xyz_m": data.site_xpos[pinch].tolist(),
        "block_xyz_m": block_pos.tolist(),
        "pinch_to_block_center_m": float(np.linalg.norm(data.site_xpos[pinch] - block_pos)),
        "pad_center_distance_mm": float(np.linalg.norm(data.geom_xpos[left] - data.geom_xpos[right]) * 1000),
        "block_lifted_5cm": bool(block_pos[2] > 0.1),
        "simulation_time_s": float(data.time),
        "image": str(output / "pregrasp.png"),
    }
    (output / "pregrasp.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
