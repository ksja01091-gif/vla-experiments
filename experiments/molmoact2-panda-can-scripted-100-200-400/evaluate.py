"""Evaluate a trained MolmoAct2 adapter on the reserved Panda layouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from collect import CanMetrics, INSTRUCTION, SCENE
from collect_manifest import write_json
from xenoarm.envs.robosuite_multi import RobosuiteMultiRobotEnv
from xenoarm.policies.openpi_remote import OpenpiRemotePolicy


MAX_STEPS = 1000
REPLAN = 5


def save_video(path, frames, row):
    outcome = "SUCCESS" if row["placed"] else "FAILURE"
    with imageio.get_writer(path, fps=5, codec="libx264", macro_block_size=1) as writer:
        for step, frame in frames:
            image = Image.new("RGB", (256, 160), (16, 21, 29))
            image.paste(Image.fromarray(frame), (0, 16))
            draw = ImageDraw.Draw(image)
            draw.text((3, 2), "Exterior camera", fill="white")
            draw.text((131, 2), "Wrist camera", fill="white")
            draw.text((3, 146), f"Seed {row['seed']} | {outcome} | Step {step}/{row['steps']}",
                      fill=(150, 255, 180) if row["placed"] else (255, 175, 160))
            writer.append_data(np.asarray(image))


def trial(task):
    layout, output, port = task
    seed = layout["seed"]
    path = output / f"seed-{seed}.json"
    if path.exists():
        row = json.loads(path.read_text())
        assert row["initial_state_hash"] == layout["initial_state_hash"]
        return row
    env = RobosuiteMultiRobotEnv(SCENE)
    policy = OpenpiRemotePolicy({"port": port, "action_space": "ee_delta_si", "wait_s": 60})
    try:
        env.reset(seed)
        assert env.initial_state_hash == layout["initial_state_hash"], (seed, env.initial_state_hash)
        policy.reset(seed)
        metrics = CanMetrics(env)
        actions, index = [], 0
        frames, latencies = [], []
        for step in range(1, MAX_STEPS + 1):
            if index >= min(REPLAN, len(actions)):
                before = time.monotonic()
                actions, index = policy.predict(env.observe(), INSTRUCTION), 0
                latencies.append((time.monotonic() - before) * 1000)
                assert actions.ndim == 2 and actions.shape[1] == 7 and len(actions) and np.isfinite(actions).all()
            if (step - 1) % 5 == 0:
                frames.append((step - 1, np.concatenate(list(env.render().values()), axis=1)))
            env.step(actions[index])
            index += 1
            metrics.update(step)
            if metrics.first_place is not None:
                break
        frames.append((step, np.concatenate(list(env.render().values()), axis=1)))
        row = {**layout, "steps": step, "lifted": metrics.first_lift is not None,
               "placed": metrics.first_place is not None, "lift_step": metrics.first_lift,
               "place_step": metrics.first_place, "clipped_actions": policy.clipped_actions,
               "median_request_ms": float(np.median(latencies))}
        label = "success" if row["placed"] else "failure"
        video = output / "videos" / f"Panda_Can_PickPlace_seed{seed}_{label}_{step}steps.mp4"
        save_video(video, frames, row)
        row["video"] = str(video.relative_to(output))
        write_json(path, row)
        return row
    finally:
        policy.client._ws.close()
        env.close()


def evaluate(layouts, output, port, workers):
    starts = json.loads(layouts.read_text())
    assert len(starts) == 50
    (output / "videos").mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = []
        for row in pool.map(trial, ((start, output, port) for start in starts)):
            rows.append(row)
            print(f"Evaluated {len(rows)}/50 | seed={row['seed']} | lift={row['lifted']} | place={row['placed']}",
                  flush=True)
    summary = {"trials": len(rows), "lifted": sum(row["lifted"] for row in rows),
               "placed": sum(row["placed"] for row in rows), "max_steps": MAX_STEPS,
               "replan_steps": REPLAN, "elapsed_seconds": time.monotonic() - started_at, "episodes": rows}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.layouts, args.output, args.port, args.workers)
