"""Use base OpenVLA to predict one action from the scripted pregrasp image.

Run in Google Colab after placing pregrasp.png in /content. No robot motion
is applied by this script. Base OpenVLA has no UR5e action calibration.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID = "openvla/openvla-7b"
INSTRUCTION = "pick up the orange block"
IMAGE_PATH = Path("/content/pregrasp.png")
OUTPUT_PATH = Path("/content/pregrasp_prediction.json")

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Upload the staged image first: {IMAGE_PATH}")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.eval()
image = Image.open(IMAGE_PATH).convert("RGB")
prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
inputs = processor(prompt, image).to("cpu", dtype=torch.bfloat16)
if inputs["input_ids"][0, -1].item() != 29871:
    empty = torch.tensor([[29871]], device="cpu")
    inputs["input_ids"] = torch.cat((inputs["input_ids"], empty), dim=1)
    inputs["attention_mask"] = torch.cat(
        (inputs["attention_mask"], torch.ones_like(empty)), dim=1
    )

with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=7, do_sample=False)

tokens = generated[0, -7:].cpu().numpy()
first_token = model.vocab_size - model.config.n_action_bins
valid = (tokens >= first_token) & (tokens < model.vocab_size)
raw_bins = model.vocab_size - tokens - 1
bins = np.clip(raw_bins, 0, len(model.bin_centers) - 1)
action = model.bin_centers[bins]
gripper = float(action[-1])
result = {
    "model": MODEL_ID,
    "image_sha256": hashlib.sha256(IMAGE_PATH.read_bytes()).hexdigest(),
    "instruction": INSTRUCTION,
    "token_ids": tokens.tolist(),
    "valid_action_token_range": [int(first_token), int(model.vocab_size - 1)],
    "all_action_tokens_valid": bool(valid.all()),
    "normalized_action": action.tolist(),
    "gripper_token_id": int(tokens[-1]),
    "gripper_bin": int(bins[-1]),
    "gripper_value": gripper,
    "candidate_mapping": "gripper >= 0.5 means open; below 0.5 means close",
    "candidate_command": "open" if gripper >= 0.5 else "close",
    "ur5e_calibration_verified": False,
}
OUTPUT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, indent=2, allow_nan=False))
