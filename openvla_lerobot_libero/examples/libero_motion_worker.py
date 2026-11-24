"""LIBERO worker: run ONLY with the existing openvla310 interpreter.
JSON lines on stdin/stdout connect it to the LeRobot inference process.
Warnings go to stderr, not to the JSON protocol.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys

PROTOCOL_OUT = sys.stdout


def emit(payload):
    PROTOCOL_OUT.write(json.dumps(payload, allow_nan=False) + "\n")
    PROTOCOL_OUT.flush()


def main():
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    frames = args.output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
        camera_heights=256, camera_widths=256,
    )

    def capture(obs, step, done=False, reward=0.0):
        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        frame = frames / f"{step:04d}.png"
        Image.fromarray(image).save(frame)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        joints = np.asarray(obs["robot0_joint_pos"], dtype=float)
        if not np.isfinite(eef).all() or not np.isfinite(joints).all():
            raise RuntimeError("Nonfinite simulated robot state.")
        return {
            "step": step, "frame": str(frame),
            "eef_position": eef.tolist(), "joint_positions": joints.tolist(),
            "sim_time": float(env.env.sim.data.time),
            "done_signal": bool(done), "reward_signal": float(reward),
        }

    try:
        env.seed(args.seed)
        obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        low, high = env.env.action_spec
        low, high = np.asarray(low), np.asarray(high)
        if low.shape != (7,) or high.shape != (7,):
            raise RuntimeError("Expected a seven-dimensional controller.")
        ready = capture(obs, 0)
        ready.update({"type": "ready", "task": task.language,
                      "action_low": low.tolist(), "action_high": high.tolist(),
                      "control_frequency": float(env.env.control_freq),
                      "initialization": "seeded reset, then 10 no-op steps; not benchmark initial states"})
        emit(ready)
        step = 0
        for line in sys.stdin:
            request = json.loads(line)
            if request["type"] == "close":
                break
            if request["type"] != "step":
                raise ValueError("Unknown worker command.")
            action = np.asarray(request["action"], dtype=np.float64)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Expected seven finite controller values.")
            if np.any(action < low - 1e-7) or np.any(action > high + 1e-7):
                raise ValueError("Controller action outside environment bounds.")
            obs, reward, done, _ = env.step(action.tolist())
            step += 1
            response = capture(obs, step, done, reward)
            response["type"] = "observation"
            emit(response)
    finally:
        env.close()


if __name__ == "__main__":
    with contextlib.redirect_stdout(sys.stderr):
        main()
