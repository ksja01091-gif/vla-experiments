"""Collect varied scripted UR5e grasp demonstrations in the same MuJoCo scene.

The scripted controller supplies the answers. This is not OpenVLA inference.
Run --probe first to test all 40 positions without rendering images.
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from ur5e_auto_grasp import make_scene, solve_pose


POSITIONS = [(round(float(x), 4), round(float(y), 4))
             for x in np.linspace(0.405, 0.495, 8)
             for y in np.linspace(-0.04, 0.04, 5)]
STAGES = [("approach", 0, 2.0), ("descend", 0, 2.0),
          ("close", 255, 1.5), ("lift", 255, 2.5), ("hold", 255, 1.0)]
INSTRUCTION = "pick up the orange block"


def run_episode(x: float, y: float, output: Path | None) -> dict:
    model, data, block_addr = make_scene()
    pinch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator")
    home = data.qpos[:6].copy()
    orientation = data.site_xmat[pinch].reshape(3, 3).copy()

    # The target follows the sampled block location, but the arm always starts at home.
    above, _, _ = solve_pose(model, data, pinch, np.array([x, y, 0.19]), orientation, home)
    grasp, _, _ = solve_pose(model, data, pinch, np.array([x, y, 0.035]), orientation, above)
    lift, _, _ = solve_pose(model, data, pinch, np.array([x, y, 0.27]), orientation, grasp)
    targets = [above, grasp, grasp, lift, lift]
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[block_addr : block_addr + 7] = [x, y, 0.05, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    renderer = None
    camera = None
    frame_dir = None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        frame_dir = output / "frames"
        frame_dir.mkdir()
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.2, 0, 0.28]
        camera.distance = 1.8
        camera.azimuth = 130
        camera.elevation = -35
        renderer = mujoco.Renderer(model, height=256, width=256)

    dt = float(model.opt.timestep)
    steps_per_sample = round(0.1 / dt)
    if abs(steps_per_sample * dt - 0.1) > 1e-9:
        raise ValueError("Recording interval is not divisible by physics timestep")
    initial_z = float(data.qpos[block_addr + 2])
    peak_z = initial_z
    hold_min_z = float("inf")
    rows = []
    stage_ends = {}
    start = data.qpos[:6].copy()
    try:
        for (stage, grip, duration), target in zip(STAGES, targets):
            steps = round(duration / dt)
            for tick in range(steps):
                if output is not None and tick % steps_per_sample == 0:
                    idx = len(rows)
                    name = f"frame_{idx:04d}.jpg"
                    renderer.update_scene(data, camera=camera)
                    Image.fromarray(renderer.render()).save(frame_dir / name, quality=88)
                    fraction = (tick + 1) / steps
                    fraction = fraction * fraction * (3 - 2 * fraction)
                    arm = start + fraction * (target - start)
                    rows.append({
                        "frame": f"frames/{name}",
                        "time_s": round(float(data.time), 4),
                        "stage": stage,
                        "instruction": INSTRUCTION,
                        "block_x_m": float(data.qpos[block_addr]),
                        "block_y_m": float(data.qpos[block_addr + 1]),
                        "block_z_m": float(data.qpos[block_addr + 2]),
                        "ee_x_m": float(data.site_xpos[pinch, 0]),
                        "ee_y_m": float(data.site_xpos[pinch, 1]),
                        "ee_z_m": float(data.site_xpos[pinch, 2]),
                        **{f"joint_target_{i+1}_rad": float(arm[i]) for i in range(6)},
                        "gripper_target": grip,
                    })
                f = (tick + 1) / steps
                f = f * f * (3 - 2 * f)
                data.ctrl[:6] = start + f * (target - start)
                data.ctrl[finger] = grip
                mujoco.mj_step(model, data)
                z = float(data.qpos[block_addr + 2])
                peak_z = max(peak_z, z)
                if stage == "hold":
                    hold_min_z = min(hold_min_z, z)
            stage_ends[stage] = float(data.qpos[block_addr + 2])
            start = target
        if renderer is not None:
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(output / "final.jpg", quality=88)
    finally:
        if renderer is not None:
            renderer.close()

    final_z = float(data.qpos[block_addr + 2])
    success = bool(hold_min_z - initial_z >= 0.05 and final_z - initial_z >= 0.05)
    result = {
        "block_start_xy_m": [x, y],
        "success_grasp_lift_hold": success,
        "initial_block_z_m": initial_z,
        "peak_block_z_m": peak_z,
        "hold_min_block_z_m": hold_min_z,
        "final_block_z_m": final_z,
        "stage_end_block_z_m": stage_ends,
        "frames": len(rows),
    }
    if output is not None:
        with (output / "demonstration.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_indexed(index: int, x: float, y: float, root: Path, probe: bool) -> dict:
    target = None if probe else root / f"episode_{index:03d}"
    try:
        row = run_episode(x, y, target)
    except Exception as error:
        row = {"block_start_xy_m": [x, y], "success_grasp_lift_hold": False,
               "error": repr(error), "frames": 0}
    row["episode"] = index
    return row


def main(root: Path, probe: bool, workers: int) -> None:
    if not probe:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.glob("episode_*")) or (root / "collection_summary.json").exists():
            raise FileExistsError(f"Collection already exists under {root}; choose a new output directory")
    results = []
    start = time.monotonic()
    if workers == 1:
        completed = (run_indexed(i, x, y, root, probe) for i, (x, y) in enumerate(POSITIONS))
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        futures = [pool.submit(run_indexed, i, x, y, root, probe)
                   for i, (x, y) in enumerate(POSITIONS)]
        completed = (future.result() for future in as_completed(futures))
    try:
        for row in completed:
            results.append(row)
            x, y = row["block_start_xy_m"]
            print(f"{len(results)}/40 episode={row['episode']} x={x:.3f} y={y:.3f} "
                  f"success={row['success_grasp_lift_hold']} "
                  f"final_z={row.get('final_block_z_m')} {row.get('error', '')}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    results.sort(key=lambda row: row["episode"])

    summary = {
        "probe": probe,
        "controller": "scripted IK expert, not OpenVLA",
        "task": INSTRUCTION,
        "seed": None,
        "random_seed_used": False,
        "position_design": "fixed 8 by 5 grid; 40 unique block start positions; one trajectory per position",
        "episodes": len(results),
        "successful_episodes": sum(row["success_grasp_lift_hold"] for row in results),
        "frames": sum(row["frames"] for row in results),
        "elapsed_wall_s": time.monotonic() - start,
        "action_format": "raw six UR5e joint actuator target angles plus Robotiq gripper target",
        "openvla_training_ready": False,
        "limitations": "scripted single-task demonstrations; no OpenVLA training or policy evaluation; action conversion and normalization still required",
        "results": results,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / ("probe_summary.json" if probe else "collection_summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    main(args.output, args.probe, args.workers)
