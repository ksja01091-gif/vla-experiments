"""Re-render existing MolmoAct 7B simulator actions from exterior and wrist cameras."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np
import robosuite


CLIPS = (
    ("", "01", "base_model_timeout"),
    ("_trained_smoke", "02", "trained_LoRA_1step_smoke_test"),
    ("_trained", "03", "trained_LoRA_timeout"),
    ("_trained_full", "04", "trained_LoRA_full_eval"),
)
SEED = 1007


def replay(env, source, destination, title, label):
    result = json.loads((source / "result.json").read_text())
    records = json.loads((source / "actions.json").read_text())
    assert result["seed"] == SEED and result["task"] == "PickPlace_D0"
    assert result["steps"] == len(records) and records
    assert bool(result["success"]) == bool(records[-1]["success"])
    np.random.seed(SEED)
    env.reset()
    for _ in range(10):
        env.step(np.zeros(7, dtype=np.float32))

    old_video = cv2.VideoCapture(str(source / "agentview.mp4"))
    fps = old_video.get(cv2.CAP_PROP_FPS)
    assert fps > 0, source
    pending = destination.with_suffix(".partial.mp4")
    render = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "512x288",
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-threads", "1", "-preset", "veryfast", "-crf", "22", "-pix_fmt",
            "yuv420p", "-movflags", "+faststart", str(pending),
        ],
        stdin=subprocess.PIPE,
    )
    frame_errors = []
    outcome = "SUCCESS" if result["success"] else "FAIL"
    try:
        for step, item in enumerate(records, 1):
            assert item["step"] == step
            obs, _, _, _ = env.step(np.asarray(item["sim_action"], dtype=np.float32))
            assert bool(env._check_success()) == bool(item["success"]), (source, step)
            exterior = obs["agentview_image"][::-1]
            wrist = obs["robot0_eye_in_hand_image"][::-1]
            ok, old_frame = old_video.read()
            assert ok and exterior.shape == wrist.shape == (256, 256, 3), (source, step)
            error = float(np.abs(old_frame[:, :, ::-1].astype(np.int16) - exterior).mean())
            frame_errors.append(error)
            if error > 15:
                raise ValueError(f"{source.name} frame {step} differs from original: {error:.2f}")
            panel = np.zeros((288, 512, 3), dtype=np.uint8)
            panel[:32] = (28, 36, 47)
            panel[32:] = np.concatenate((exterior, wrist), axis=1)
            color = (125, 235, 159) if result["success"] else (255, 155, 155)
            cv2.putText(panel, f"{title} | {label} | {outcome} | Step {step}/{len(records)}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            for x, camera in ((6, "Exterior camera"), (262, "Wrist camera")):
                panel[34:53, x:x + 115] = (28, 36, 47)
                cv2.putText(panel, camera, (x + 3, 48), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (245, 245, 245), 1, cv2.LINE_AA)
            render.stdin.write(panel.tobytes())
        assert not old_video.read()[0], source
        render.stdin.close()
        if render.wait() or not pending.stat().st_size:
            raise RuntimeError(f"Failed to encode {destination.name}")
        pending.replace(destination)
    finally:
        old_video.release()
        if render.poll() is None:
            render.kill()
            render.wait()
        if not render.stdin.closed:
            render.stdin.close()
    return {
        "file": destination.name, "source_rollout": source.name, "seed": SEED,
        "steps": len(records), "success": result["success"], "termination": result["reason"],
        "original_exterior_max_rgb_mae": round(max(frame_errors), 3),
        "fps": fps, "resolution": [512, 288], "codec": "H.264",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "mimicgen"))
    import mimicgen.envs.robosuite.pick_place  # noqa: F401
    labels = ("Base", "LoRA smoke test", "LoRA timed run", "LoRA full eval")
    summary = {}
    for robot, title in (("panda", "Panda"), ("ur5e", "UR5e")):
        robot_output = args.output / robot
        robot_output.mkdir(parents=True, exist_ok=True)
        env_args = json.loads((args.source / "datasets" / robot / "env_args.json").read_text())
        assert env_args["env_name"] == "PickPlace_D0"
        kwargs = {**env_args["env_kwargs"], "has_renderer": False, "has_offscreen_renderer": True}
        summary[robot] = []
        for (suffix, order, tag), label in zip(CLIPS, labels, strict=True):
            source = args.source / "rollouts" / f"{robot}{suffix}_seed{SEED}"
            result = json.loads((source / "result.json").read_text())
            step_suffix = "" if order == "02" else f"_{result['steps']}steps"
            destination = robot_output / f"{order}_{title}_{tag}{step_suffix}_seed{SEED}_dual_camera.mp4"
            print(f"Replaying {source.name}", flush=True)
            env = robosuite.make("PickPlace_D0", **kwargs)
            try:
                record = replay(env, source, destination, title, label)
            finally:
                env.close()
            summary[robot].append(record)
            print(json.dumps(record), flush=True)
    (args.output / "replay_verified.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
