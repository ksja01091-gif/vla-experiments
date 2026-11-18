"""MuJoCo half of a bounded UR5e/OpenVLA closed-loop diagnostic.

Run as a separate process from Torch/OpenVLA. This is a candidate motion
mapping, not a calibrated UR5e policy. The gripper stays open because the
base OpenVLA checkpoint has no verified UR5e gripper mapping.
"""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


def make_scene(root: Path):
    scene = mujoco.MjSpec.from_file(str(root / "universal_robots_ur5e/scene.xml"))
    gripper = mujoco.MjSpec.from_file(str(root / "robotiq_2f85/2f85.xml"))
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    scene.option.impratio = 10
    scene.attach(gripper, site=scene.site("attachment_site"), prefix="gripper/")
    block = scene.worldbody.add_body(name="test_block", pos=[0.45, 0, 0.05])
    block.add_freejoint(name="test_block_joint")
    block.add_geom(
        name="block_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.05], mass=0.05,
        rgba=[1, 0.4, 0.1, 1], friction=[1.5, 0.005, 0.0001],
    )
    model = scene.compile()
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_block_joint")
    block_addr = model.jnt_qposadr[joint]
    pinch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator")
    return model, data, block_addr, pinch, finger


def rotation_from_vector(vector):
    angle = float(np.linalg.norm(vector))
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def solve_pose(model, data, site_id, target, orientation, initial):
    q = initial.copy()
    for iteration in range(600):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        current = data.site_xmat[site_id].reshape(3, 3)
        pos_error = target - data.site_xpos[site_id]
        rot_error = 0.5 * sum(
            (np.cross(current[:, i], orientation[:, i]) for i in range(3)),
            np.zeros(3),
        )
        error = np.concatenate([pos_error, rot_error])
        if np.linalg.norm(pos_error) < 0.002 and np.linalg.norm(rot_error) < 0.02:
            return q, float(np.linalg.norm(pos_error)), iteration
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        jac = np.vstack([jac_pos[:, :6], jac_rot[:, :6]])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.02 * np.eye(6), error)
        q += np.clip(dq, -0.08, 0.08)
        q = np.clip(q, model.jnt_range[:6, 0] + 0.02, model.jnt_range[:6, 1] - 0.02)
    raise RuntimeError(f"IK did not converge: position error {np.linalg.norm(pos_error):.3f} m")


def run(root: Path, output: Path, step: int, action_file: Path | None):
    output.mkdir(parents=True, exist_ok=True)
    model, data, block_addr, pinch, finger = make_scene(root)
    state_path = output / "state.npz"
    if step == 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[block_addr : block_addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
    else:
        with np.load(state_path) as saved:
            data.qpos[:] = saved["qpos"]
            data.qvel[:] = saved["qvel"]
            data.act[:] = saved["act"]
            data.time = float(saved["time"])
    mujoco.mj_forward(model, data)
    before = data.site_xpos[pinch].copy()
    block_before = data.qpos[block_addr : block_addr + 3].copy()
    requested = None
    ik_error = None
    physics_steps = 0
    if step > 0:
        action = np.asarray(json.loads(action_file.read_text())["normalized_action"], dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected seven finite normalized action values")
        if np.any(np.abs(action[:6]) > 1.0001):
            raise ValueError("Arm command outside normalized bounds")
        home = data.qpos[:6].copy()
        orientation = data.site_xmat[pinch].reshape(3, 3).copy()
        requested = before + action[:3] * 0.01
        rotation = rotation_from_vector(action[3:6] * math.radians(5)) @ orientation
        saved_qpos, saved_qvel, saved_act, saved_time = (
            data.qpos.copy(), data.qvel.copy(), data.act.copy(), data.time
        )
        target_joints, ik_error, _ = solve_pose(
            model, data, pinch, requested, rotation, home
        )
        data.qpos[:], data.qvel[:], data.act[:], data.time = (
            saved_qpos, saved_qvel, saved_act, saved_time
        )
        mujoco.mj_forward(model, data)
        physics_steps = round(1.0 / model.opt.timestep)
        for i in range(physics_steps):
            fraction = (i + 1) / physics_steps
            fraction = fraction * fraction * (3 - 2 * fraction)
            data.ctrl[:6] = home + fraction * (target_joints - home)
            data.ctrl[finger] = 0
            mujoco.mj_step(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.2, 0, 0.28]
    camera.distance = 1.8
    camera.azimuth = 130
    camera.elevation = -35
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera=camera)
        image_path = output / f"step_{step:02d}.png"
        Image.fromarray(renderer.render()).save(image_path)
    finally:
        renderer.close()
    np.savez(state_path, qpos=data.qpos, qvel=data.qvel, act=data.act, time=data.time)
    block_after = data.qpos[block_addr : block_addr + 3].copy()
    result = {
        "step": step,
        "image": str(image_path),
        "simulation_time_s": float(data.time),
        "physics_steps_this_action": physics_steps,
        "initial_pinch_xyz_m": before.tolist(),
        "final_pinch_xyz_m": data.site_xpos[pinch].tolist(),
        "requested_pinch_xyz_m": None if requested is None else requested.tolist(),
        "ik_position_error_m": ik_error,
        "block_xyz_m_before": block_before.tolist(),
        "block_xyz_m_after": block_after.tolist(),
        "pinch_to_block_center_m": float(np.linalg.norm(data.site_xpos[pinch] - block_after)),
        "block_lifted_5cm": bool(block_after[2] - 0.05 > 0.05),
        "gripper_control": 0,
        "mapping_status": "unvalidated OpenVLA-to-UR5e candidate; gripper held open",
    }
    (output / f"state_{step:02d}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--action-file", type=Path)
    args = parser.parse_args()
    run(args.root, args.output, args.step, args.action_file)
