"""Run one OpenVLA-7B prediction; report normalized actions only.

This script deliberately does not apply another dataset's action scale to LIBERO.
It does not move a robot or measure task success.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def predict(image_path: Path, instruction: str) -> dict:
    model_id = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    image = Image.open(image_path).convert("RGB")
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    # Match the empty token insertion in OpenVLA's predict_action().
    if inputs["input_ids"][0, -1].item() != 29871:
        empty = torch.tensor([[29871]], device="cuda:0")
        inputs["input_ids"] = torch.cat((inputs["input_ids"], empty), dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                (inputs["attention_mask"], torch.ones_like(empty)), dim=1
            )

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=7, do_sample=False)

    ids = generated[0, -7:].cpu().numpy()
    # 256 action tokens are mapped to 255 interval centers. The 256th token
    # is valid; OpenVLA clips its decoded index to the final center.
    first_valid_id = model.vocab_size - model.config.n_action_bins
    last_valid_id = model.vocab_size - 1
    in_action_token_range = (ids >= first_valid_id) & (ids <= last_valid_id)
    # OpenVLA's own predict_action() clips the decoded bin index to the
    # available centers. Preserve that behavior and report clipping separately.
    raw_indices = model.vocab_size - ids - 1
    indices = np.clip(raw_indices, 0, len(model.bin_centers) - 1)
    actions = model.bin_centers[indices]
    result = {
        "model": model_id,
        "image": str(image_path),
        "instruction": instruction,
        "action_token_ids": ids.tolist(),
        "all_seven_tokens_in_action_range": bool(in_action_token_range.all()),
        "tokens_outside_action_token_range": int((~in_action_token_range).sum()),
        "tokens_clipped_to_bin_centers": int((raw_indices != indices).sum()),
        "normalized_action": actions.tolist(),
        "normalized_action_shape": [1, int(actions.shape[0])],
        "all_values_finite": bool(np.isfinite(actions).all()),
        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, default=Path("base_openvla_result.json"))
    args = parser.parse_args()
    result = predict(args.image, args.instruction)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
