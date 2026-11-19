"""Recheck saved OpenVLA token IDs using the model's official decode rule.

This is an offline audit of existing predictions. It does not run the model or
change the original prediction files.
"""

import argparse
import json
import math
from pathlib import Path


VOCAB_SIZE = 32000  # Verified from the loaded openvla/openvla-7b model.
N_ACTION_BINS = 256  # OpenVLA config.n_action_bins.


def audit_prediction(label: str, prediction: dict, token_key: str) -> dict:
    token_ids = prediction[token_key]
    normalized = prediction["normalized_action"]
    if len(token_ids) != 7 or len(normalized) != 7:
        raise ValueError(f"{label}: expected seven token IDs and seven actions")
    valid_min = VOCAB_SIZE - N_ACTION_BINS
    valid_max = VOCAB_SIZE - 1
    valid = [valid_min <= token <= valid_max for token in token_ids]
    decoded = []
    for token in token_ids:
        bin_index = max(0, min(VOCAB_SIZE - token - 1, N_ACTION_BINS - 2))
        decoded.append(-1.0 + (bin_index + 0.5) * 2.0 / (N_ACTION_BINS - 1))
    matches = all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                  for a, b in zip(decoded, normalized))
    if not matches:
        raise ValueError(f"{label}: stored normalized values differ from official decode")
    return {
        "label": label,
        "token_ids": token_ids,
        "valid_with_256_action_tokens": valid,
        "old_last_token_flag": prediction.get("token_valid", [None] * 7)[-1],
        "last_action_value": decoded[-1],
        "official_decode_matches_stored_values": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--loop-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit_prediction("fresh_one_scene",
            json.loads(args.fresh.read_text(encoding="utf-8")), "action_token_ids")]
    for step in range(1, 6):
        path = args.loop_dir / f"prediction_step_{step:02d}.json"
        rows.append(audit_prediction(f"loop_step_{step}",
                    json.loads(path.read_text(encoding="utf-8")), "token_ids"))
    report = {
        "model": "openvla/openvla-7b",
        "vocab_size": VOCAB_SIZE,
        "n_action_bins": N_ACTION_BINS,
        "n_bin_centers": N_ACTION_BINS - 1,
        "valid_token_range_inclusive": [VOCAB_SIZE - N_ACTION_BINS, VOCAB_SIZE - 1],
        "conclusion": "Token 31744 is the valid 256th action token, not an out-of-range token; old check used 255 centers as token count.",
        "new_model_inference_performed": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("audited_predictions", len(rows))
    print("valid_seventh_tokens", sum(row["valid_with_256_action_tokens"][-1] for row in rows))
    print("all_official_decodes_match", all(row["official_decode_matches_stored_values"] for row in rows))
    print("report", args.output)


if __name__ == "__main__":
    main()
