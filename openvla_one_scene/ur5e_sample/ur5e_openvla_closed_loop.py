"""Bounded OpenVLA -> UR5e loop used for the Colab diagnostic.

The first six action dimensions use an *unvalidated* engineering scale in
ur5e_colab_sim_worker.py. The gripper is held open because no OpenVLA-to-UR5e
gripper calibration has been established.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


INSTRUCTION = "In: What action should the robot take to pick up the orange block?\nOut:"


def predict(model, processor, image_path: Path) -> dict:
    image_bytes = image_path.read_bytes()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(INSTRUCTION, image).to("cpu", dtype=torch.bfloat16)
    if inputs["input_ids"][0, -1].item() != 29871:
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.tensor([[29871]])], dim=1
        )
        inputs["attention_mask"] = torch.cat(
            [
                inputs["attention_mask"],
                torch.ones((1, 1), dtype=inputs["attention_mask"].dtype),
            ],
            dim=1,
        )
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=7, do_sample=False)
    ids = generated[0, -7:].detach().cpu().numpy()
    # n_action_bins=256 tokens, while bin_centers has 255 intervals. Token
    # vocab_size-256 is valid and is clipped to the final interval by OpenVLA.
    first_valid = model.vocab_size - model.config.n_action_bins
    valid = (ids >= first_valid) & (ids < model.vocab_size)
    indices = np.clip(model.vocab_size - ids - 1, 0, len(model.bin_centers) - 1)
    action = model.bin_centers[indices].astype(float)
    return {
        "input_image": str(image_path),
        "input_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "instruction": INSTRUCTION,
        "token_ids": ids.tolist(),
        "token_valid": valid.tolist(),
        "normalized_action": action.tolist(),
        "gripper_applied": "held_open_0_manual_safety",
        "mapping": "candidate_0.01m_per_unit_5deg_per_unit",
    }


def run(root: Path, output: Path, worker: Path, max_actions: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    first = subprocess.run(
        [sys.executable, "-u", str(worker), "--root", str(root), "--output", str(output), "--step", "0"],
        capture_output=True, text=True, timeout=120,
    )
    first.check_returncode()

    model_id = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="cpu", trust_remote_code=True,
    )
    model.eval()

    records = []
    stop_reason = "completed_bounded_test"
    for step in range(1, max_actions + 1):
        prediction = predict(model, processor, output / f"step_{step - 1:02d}.png")
        prediction["step"] = step
        prediction_path = output / f"prediction_step_{step:02d}.json"
        prediction_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
        if not all(prediction["token_valid"]):
            stop_reason = f"invalid_action_token_step_{step}"
            records.append({"prediction": prediction, "execution": "skipped"})
            break
        completed = subprocess.run(
            [sys.executable, "-u", str(worker), "--root", str(root),
             "--output", str(output), "--step", str(step),
             "--action-file", str(prediction_path)],
            capture_output=True, text=True, timeout=120,
        )
        if completed.returncode:
            stop_reason = f"simulator_error_step_{step}"
            records.append({
                "prediction": prediction, "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            })
            break
        scene = json.loads((output / f"state_{step:02d}.json").read_text(encoding="utf-8"))
        records.append({"prediction": prediction, "scene": scene})
        print(
            "step", step, "distance_m", scene["pinch_to_block_center_m"],
            "block_lifted_5cm", scene["block_lifted_5cm"], flush=True,
        )
    summary_path = output / "closed_loop_summary.json"
    summary_path.write_text(
        json.dumps({"stop_reason": stop_reason, "records": records}, indent=2),
        encoding="utf-8",
    )
    print("summary", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="mujoco_menagerie directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("ur5e_colab_sim_worker.py"))
    parser.add_argument("--max-actions", type=int, default=5)
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    run(args.root, args.output, args.worker, args.max_actions)


if __name__ == "__main__":
    main()
