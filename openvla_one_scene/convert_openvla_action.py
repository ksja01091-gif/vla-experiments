"""Convert one OpenVLA prediction to the prior LIBERO bowl action scale.

The statistics come from 20 training demonstrations, never validation data.
This only converts numbers; it does not execute an action in MuJoCo.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def convert(prediction: dict, stats: dict) -> dict:
    normalized = np.asarray(prediction["normalized_action"], dtype=np.float64)
    low = np.asarray(stats["q01"], dtype=np.float64)
    high = np.asarray(stats["q99"], dtype=np.float64)
    if normalized.shape != (7,) or low.shape != (7,) or high.shape != (7,):
        raise ValueError("Expected seven action values and seven q01/q99 values")
    if not np.isfinite(normalized).all() or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Actions and statistics must contain finite numbers")
    if np.any(high[:6] <= low[:6]):
        raise ValueError("Each arm-action q99 must exceed q01")

    # OpenVLA normalizes the arm coordinates to [-1, 1]. The gripper uses a
    # separate 0=close, 1=open convention in the LIBERO data wrapper.
    action = np.empty(7, dtype=np.float64)
    action[:6] = low[:6] + (normalized[:6] + 1.0) * (high[:6] - low[:6]) / 2.0
    action[6] = normalized[6]

    # Match OpenVLA's LIBERO evaluation: [0,1] -> [-1,+1], binarize,
    # then flip sign so LIBERO receives -1=open, +1=close.
    controller = action.copy()
    controller[6] = -np.sign(2.0 * action[6] - 1.0)
    result = {
        "model": prediction["model"],
        "statistics_source": "existing train20 action_stats.json; 20 episodes, 1928 frames",
        "validation_used_for_statistics": False,
        "normalized_action": normalized.tolist(),
        "action_on_train20_scale": action.tolist(),
        "libero_controller_action_candidate": controller.tolist(),
        "shape": [1, 7],
        "all_values_finite": bool(np.isfinite(controller).all()),
        "gripper_action_in_0_1_range": bool(0.0 <= action[6] <= 1.0),
        "controller_executed": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    result = convert(prediction, stats)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
