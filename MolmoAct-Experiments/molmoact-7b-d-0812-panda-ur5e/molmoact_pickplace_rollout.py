"""Bounded closed-loop PickPlace_D0 trials for MolmoAct base or a LoRA adapter."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np
import robosuite
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "mimicgen"))
import mimicgen.envs.robosuite.pick_place  # noqa: F401: register PickPlace_D0
CHECKPOINT = ROOT / "model" / "MolmoAct-7B-D-0812"
TASK = "pick up the objects and place them into the bins"
PROMPT = (
    f"The task is {TASK}. What is the action that the robot should take. "
    f"To figure out the action that the robot should take to {TASK}, let's think through it step by step. "
    "First, what is the depth map for the first image? Second, what is the trajectory of the end effector in the first image? "
    "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
    "along with other images from different camera views as additional information, what is the action that the robot should take?"
)


def center_crop(image):
    width, height = image.size
    scale = 0.9 ** 0.5
    crop_width, crop_height = int(width * scale), int(height * scale)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize((width, height), Image.BILINEAR)


def rgb_image(obs, key):
    return Image.fromarray(obs[key][::-1].copy())


def predict(obs, model, processor, text, robot):
    images = [center_crop(rgb_image(obs, key)) for key in ("agentview_image", "robot0_eye_in_hand_image")]
    inputs = processor(images=[images], text=text, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    generated_text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    parser_model = model.base_model.model if isinstance(model, PeftModel) else model
    actions = np.asarray(parser_model.parse_action(generated_text, unnorm_key=robot), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"Invalid model actions: shape={actions.shape}")
    return actions, generated_text


def run_trial(robot, model, processor, text, seed, horizon, limit_seconds, tag, adapter, resume_from=None):
    env_args = json.loads((ROOT / "datasets" / robot / "env_args.json").read_text())
    assert env_args["env_name"] == "PickPlace_D0"
    kwargs = dict(env_args["env_kwargs"])
    kwargs.update(has_renderer=False, has_offscreen_renderer=True)
    env = robosuite.make("PickPlace_D0", **kwargs)
    out = ROOT / "rollouts" / f"{robot}_{tag}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out / "agentview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (256, 256))
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()
    steps, calls, reason, action_log = 0, 0, "horizon", []
    try:
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        if resume_from is not None:
            previous = json.loads((resume_from / "actions.json").read_text())
            for item in previous:
                obs, _, _, _ = env.step(np.asarray(item["sim_action"], dtype=np.float32))
                writer.write(cv2.cvtColor(obs["agentview_image"][::-1], cv2.COLOR_RGB2BGR))
            steps = len(previous)
            calls = max((item["model_call"] for item in previous), default=0)
            action_log = list(previous)
            cap = cv2.VideoCapture(str(resume_from / "agentview.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, steps - 1)
            ok, last_bgr = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError("Could not read final frame of source rollout")
            current_bgr = cv2.cvtColor(obs["agentview_image"][::-1], cv2.COLOR_RGB2BGR)
            frame_difference = float(np.abs(current_bgr.astype(np.float32) - last_bgr).mean())
            print(json.dumps({"resumed_steps": steps, "frame_difference": frame_difference}), flush=True)
            if frame_difference > 15:
                raise RuntimeError("Replayed environment differs from source video")
        start = time.monotonic()
        while steps < horizon:
            if time.monotonic() - start >= limit_seconds:
                reason = "wall_time_limit"
                break
            try:
                action_chunk, generated_text = predict(obs, model, processor, text, robot)
            except Exception as error:
                reason = f"prediction_error: {type(error).__name__}: {error}"
                break
            calls += 1
            for index, action in enumerate(action_chunk):
                raw = action.copy()
                action[:6] = np.clip(action[:6], -1, 1)
                # Official MolmoAct LIBERO evaluation thresholds the decoded
                # gripper in [0, 1] at 0.5, then inverts for robosuite.
                # Decoded 0 = close and 1 = open; robosuite +1 = close.
                action[6] = 1.0 if raw[6] < 0.5 else -1.0
                obs, _, _, _ = env.step(action)
                frame = obs["agentview_image"][::-1].copy()
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                steps += 1
                action_log.append({"step": steps, "model_action": raw.tolist(), "sim_action": action.tolist(),
                                   "model_call": calls, "chunk_index": index, "success": bool(env._check_success())})
                if steps % 20 == 0:
                    print(json.dumps({"robot": robot, "tag": tag, "steps": steps,
                                      "model_calls": calls, "seconds": time.monotonic() - start}), flush=True)
                if env._check_success():
                    reason = "task_success"
                    break
                if steps >= horizon or time.monotonic() - start >= limit_seconds:
                    break
            if reason == "task_success":
                break
        summary = {"robot": robot, "seed": seed, "task": "PickPlace_D0", "success": reason == "task_success",
                   "steps": steps, "model_calls": calls, "reason": reason, "seconds": time.monotonic() - start,
                   "horizon": horizon, "checkpoint": str(CHECKPOINT),
                   "adapter": str(adapter) if adapter else None,
                   "training": tag,
                   "resumed_from": str(resume_from) if resume_from else None,
                   "action_stats_source": "400 training episodes for this robot"}
        (out / "result.json").write_text(json.dumps(summary, indent=2))
        (out / "actions.json").write_text(json.dumps(action_log))
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        writer.release()
        env.close()


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--robot", choices=["panda", "ur5e", "both"], default="both")
    args_parser.add_argument("--adapter", type=Path)
    args_parser.add_argument("--tag", default="base")
    args_parser.add_argument("--seed", type=int, default=1007)
    args_parser.add_argument("--horizon", type=int, default=800)
    args_parser.add_argument("--limit-seconds", type=int, default=900)
    args_parser.add_argument("--resume-from", type=Path)
    args = args_parser.parse_args()
    processor = AutoProcessor.from_pretrained(str(CHECKPOINT), trust_remote_code=True, local_files_only=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        str(CHECKPOINT), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    ).eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False).eval()
    stats = json.loads((ROOT / "train_action_stats.json").read_text())
    parser_model = model.base_model.model if isinstance(model, PeftModel) else model
    for robot in ("panda", "ur5e"):
        parser_model.norm_stats[robot] = {"action": stats[robot]["action"]}
    if hasattr(processor, "tokenizer") and processor.tokenizer.__class__.__name__.startswith("Qwen2"):
        parser_model._qwen_tokenizer = processor.tokenizer
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        tokenize=False, add_generation_prompt=True,
    )
    results = []
    robots = ("panda", "ur5e") if args.robot == "both" else (args.robot,)
    for robot in robots:
        result = run_trial(robot, model, processor, text, seed=args.seed, horizon=args.horizon,
                           limit_seconds=args.limit_seconds, tag=args.tag, adapter=args.adapter,
                           resume_from=args.resume_from)
        results.append(result)
        (ROOT / f"molmoact_rollout_summary_{args.tag}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
