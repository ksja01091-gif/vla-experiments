"""Small standalone UR5e + Robotiq block grasp test in MuJoCo.

Run inside the existing ~/ur5e-venv. This is scripted control, not OpenVLA.
"""

from pathlib import Path
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path.home() / "mujoco_menagerie"


def make_scene():
    scene = mujoco.MjSpec.from_file(str(ROOT / "universal_robots_ur5e/scene.xml"))
    gripper = mujoco.MjSpec.from_file(str(ROOT / "robotiq_2f85/2f85.xml"))
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    scene.option.impratio = 10
    scene.attach(gripper, site=scene.site("attachment_site"), prefix="gripper/")
    block = scene.worldbody.add_body(name="test_block", pos=[0.45, 0, 0.05])
    block.add_freejoint(name="test_block_joint")
    block.add_geom(
        name="block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.05],
        mass=0.05,
        rgba=[1, 0.4, 0.1, 1],
        friction=[1.5, 0.005, 0.0001],
    )
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_block_joint")
    addr = model.jnt_qposadr[joint]
    data.qpos[addr : addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    return model, data, addr


def solve_pose(model, data, site_id, target, orientation, initial):
    """Damped least squares for a fixed gripper orientation."""
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
            return q, np.linalg.norm(pos_error), iteration
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        jac = np.vstack([jac_pos[:, :6], jac_rot[:, :6]])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.02 * np.eye(6), error)
        q += np.clip(dq, -0.08, 0.08)
        q = np.clip(q, model.jnt_range[:6, 0] + 0.02, model.jnt_range[:6, 1] - 0.02)
    raise RuntimeError(f"IK did not converge: position error {np.linalg.norm(pos_error):.3f} m")


def main(view):
    model, data, addr = make_scene()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    home = data.qpos[:6].copy()
    orientation = data.site_xmat[site_id].reshape(3, 3).copy()
    above, _, _ = solve_pose(model, data, site_id, np.array([0.45, 0, 0.19]), orientation, home)
    grasp, _, _ = solve_pose(model, data, site_id, np.array([0.45, 0, 0.035]), orientation, above)
    lift, _, _ = solve_pose(model, data, site_id, np.array([0.45, 0, 0.27]), orientation, grasp)
    print("IK ready: home -> above -> grasp -> lift", flush=True)

    # Restore the scene after solving poses.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[addr : addr + 7] = [0.45, 0, 0.05, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator")

    def run():
        stages = [
            ("above", above, 0, 2.0),
            ("down", grasp, 0, 2.0),
            ("close", grasp, 255, 1.5),
            ("lift", lift, 255, 2.5),
            ("hold", lift, 255, 1.0),
        ]
        start = data.qpos[:6].copy()
        peak = data.qpos[addr + 2]
        dt = model.opt.timestep
        for name, target, finger, duration in stages:
            steps = round(duration / dt)
            for step in range(steps):
                fraction = min(1, (step + 1) / steps)
                fraction = fraction * fraction * (3 - 2 * fraction)
                data.ctrl[:6] = start + fraction * (target - start)
                data.ctrl[gripper_id] = finger
                mujoco.mj_step(model, data)
                if view:
                    viewer.sync()
                    time.sleep(dt)
                peak = max(peak, data.qpos[addr + 2])
            start = target
            print(f"{name}: block height {data.qpos[addr + 2]:.3f} m", flush=True)
        print(f"Peak block height: {peak:.3f} m; initial: 0.050 m", flush=True)
        if view:
            print("Close the viewer window to exit.", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)

    if view:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run()
    else:
        viewer = None
        run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="show animated MuJoCo viewer")
    main(parser.parse_args().view)
