"""Run one fresh base OpenVLA prediction on the recorded UR5e scene.

Run in a Colab high-RAM CPU runtime after installing transformers==4.40.1,
timm==0.9.10, and accelerate. This predicts one normalized action; it does not
calibrate the action to UR5e units or control the robot.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = "openvla/openvla-7b"
INSTRUCTION = "pick up the orange block"
IMAGE_URL = (
    "https://raw.githubusercontent.com/ksja01091-gif/vla-experiments/main/"
    "openvla_one_scene/ur5e_sample/scene.png"
)
EXPECTED_IMAGE_SHA256 = (
    "141849f4896c4214c106aa4e06b4b0e06b77a4d9314490c9a0e44e043b11900c"
)


def predict(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "ur5e_scene_fresh.png"
    urllib.request.urlretrieve(IMAGE_URL, image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if image_sha256 != EXPECTED_IMAGE_SHA256:
        raise ValueError("The downloaded scene.png differs from the recorded scene")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")
    prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
    inputs = processor(prompt, image).to("cpu", dtype=torch.bfloat16)
    # OpenVLA's predict_action() appends this empty token when needed.
    if inputs["input_ids"][0, -1].item() != 29871:
        empty = torch.tensor([[29871]], device="cpu")
        inputs["input_ids"] = torch.cat((inputs["input_ids"], empty), dim=1)
        inputs["attention_mask"] = torch.cat(
            (inputs["attention_mask"], torch.ones_like(empty)), dim=1
        )

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=7, do_sample=False)

    token_ids = generated[0, -7:].cpu().numpy()
    first_valid = model.vocab_size - len(model.bin_centers)
    valid = (token_ids >= first_valid) & (token_ids < model.vocab_size)
    indices = np.clip(
        model.vocab_size - token_ids - 1, 0, len(model.bin_centers) - 1
    )
    action = model.bin_centers[indices]
    result = {
        "provenance": "fresh Colab CPU BF16 inference",
        "model": MODEL_ID,
        "quantization": "none",
        "device": "cpu",
        "image": str(image_path),
        "image_sha256": image_sha256,
        "instruction": INSTRUCTION,
        "action_token_ids": token_ids.tolist(),
        "tokens_clipped_to_action_bins": int((~valid).sum()),
        "normalized_action": action.tolist(),
        "normalized_action_shape": [1, int(action.shape[0])],
        "all_values_finite": bool(np.isfinite(action).all()),
        "cpu_inference": True,
        "transformers_version": "4.40.1",
    }
    (output_dir / "ur5e_fresh_prediction.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(predict(Path("/content")), indent=2, allow_nan=False))
