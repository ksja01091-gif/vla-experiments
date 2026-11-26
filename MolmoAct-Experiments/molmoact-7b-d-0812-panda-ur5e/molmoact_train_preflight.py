"""Check that public PickPlace frames and action targets fit MolmoAct inputs.

This does not train or modify model weights.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "model" / "MolmoAct-7B-D-0812"
TASK = "pick up the objects and place them into the bins"
PROMPT = (
    f"The task is {TASK}. What is the action that the robot should take. "
    f"To figure out the action that the robot should take to {TASK}, let's think through it step by step. "
    "First, what is the depth map for the first image? Second, what is the trajectory of the end effector in the first image? "
    "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
    "along with other images from different camera views as additional information, what is the action that the robot should take?"
)


def read_frame(item):
    cap = cv2.VideoCapture(item["path"])
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame_index_in_video"])
        ok, bgr = cap.read()
        if not ok:
            raise ValueError(f"Missing frame: {item}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def center_crop(image):
    width, height = image.size
    scale = 0.9 ** 0.5
    crop_width, crop_height = int(width * scale), int(height * scale)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize((width, height), Image.Resampling.BILINEAR)


def action_tokens(actions, tokenizer, q01, q99):
    actions = np.asarray(actions, dtype=np.float32)
    assert actions.shape == (8, 7) and np.isfinite(actions).all()
    normalized = actions.copy()
    span = q99[:6] - q01[:6]
    normalized[:, :6] = np.where(np.abs(span) < 1e-8, 0.0, np.clip(2 * (actions[:, :6] - q01[:6]) / np.maximum(span, 1e-8) - 1, -1, 1))
    normalized[:, 6] = np.clip(1 - actions[:, 6], -1, 1)
    bins = np.linspace(-1, 1, 256)
    ids = tokenizer.vocab_size - np.digitize(normalized, bins)
    return ids.tolist()


def action_text(rows, tokenizer):
    # The official preprocess writes convert_ids_to_tokens() strings into text.
    # Those strings expand to multiple tokenizer IDs when training, but the
    # model's parser converts the generated strings back to the intended IDs.
    tokens = [[tokenizer.convert_ids_to_tokens(int(i)) for i in row] for row in rows]
    return "[" + ", ".join("[" + ", ".join(row) + "]" for row in tokens) + "]"


def main():
    processor = AutoProcessor.from_pretrained(str(CHECKPOINT), trust_remote_code=True, local_files_only=True, padding_side="left")
    stats = json.loads((ROOT / "train_action_stats.json").read_text())
    output = {}
    for robot in ("panda", "ur5e"):
        with (ROOT / "sample_manifest" / f"{robot}_train.jsonl").open() as f:
            record = json.loads(f.readline())
        images = [center_crop(read_frame(record[f"{view}_video"])) for view in ("agentview", "wrist")]
        q01 = np.asarray(stats[robot]["action"]["q01"], dtype=np.float32)
        q99 = np.asarray(stats[robot]["action"]["q99"], dtype=np.float32)
        token_ids = action_tokens(record["next_8_actions"], processor.tokenizer, q01, q99)
        answer = " The action that the robot should take is " + action_text(token_ids, processor.tokenizer)
        user = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
        prompt = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}], tokenize=False, add_generation_prompt=False)
        prompt_input = processor(images=[images], text=prompt, padding=True, return_tensors="pt")
        full_input = processor(images=[images], text=full, padding=True, return_tensors="pt")
        pids = prompt_input["input_ids"][0].tolist()
        fids = full_input["input_ids"][0].tolist()
        target_ids = fids[len(pids):]
        decoded = processor.batch_decode([target_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        rows = re.findall(r"\[([^\[\]]+)\]", decoded)
        recovered = [[processor.tokenizer.convert_tokens_to_ids(token.strip()) for token in row.split(",")] for row in rows]
        prefix_matches = full_input["input_ids"][0, :len(pids)].tolist() == pids
        output[robot] = {
            "episode": record["episode_index"],
            "prompt_tokens": len(pids),
            "full_tokens": len(fids),
            "target_tokens": len(target_ids),
            "prompt_is_prefix": prefix_matches,
            "action_token_ids": token_ids[0],
            "decoded_action_rows": len(recovered),
            "action_token_roundtrip": recovered == token_ids,
            "image_shapes": [list(im.size) for im in images],
            "model_input_shapes": {k: list(v.shape) for k, v in full_input.items() if hasattr(v, "shape")},
        }
        print(robot, json.dumps(output[robot]), flush=True)
        assert prefix_matches and target_ids and output[robot]["action_token_roundtrip"]
    (ROOT / "molmoact_train_preflight.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
