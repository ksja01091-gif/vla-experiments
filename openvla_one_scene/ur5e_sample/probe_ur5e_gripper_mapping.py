"""Probe a candidate OpenVLA/LIBERO gripper-to-UR5e command mapping.

Uses an existing saved action; does not load OpenVLA or measure task success.
The LIBERO convention (0=close, 1=open) is a transfer assumption for UR5e.
"""

import argparse
import json
import math
from pathlib import Path

from verify_ur5e_controls import run_gripper


def gripper_command(value: float) -> int:
    """Candidate: OpenVLA/LIBERO 0=close, 1=open; UR5e 255=close, 0=open."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Expected a finite gripper value in [0, 1]")
    return 0 if value >= 0.5 else 255


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    value = float(prediction["normalized_action"][-1])
    saved_command = gripper_command(value)
    if saved_command != 0:
        raise RuntimeError("This diagnostic expected the saved action to request open")

    # One real saved model value and one synthetic close request in fresh,
    # identical MuJoCo scenes. The synthetic value is not a model prediction.
    open_trial = run_gripper(saved_command, duration_s=1.0)
    close_trial = run_gripper(gripper_command(0.0), duration_s=1.0)
    if close_trial["pad_center_distance_mm"] >= open_trial["pad_center_distance_mm"]:
        raise RuntimeError("Candidate close command did not bring the pads together")

    result = {
        "purpose": "Test gripper command translation only; no grasp or task success",
        "mapping_assumption": "OpenVLA LIBERO gripper 0=close, 1=open; threshold 0.5",
        "ur5e_actuator": "0=open, 255=close",
        "saved_model_gripper_value": value,
        "saved_model_command": saved_command,
        "saved_model_trial": open_trial,
        "synthetic_close_value": 0.0,
        "synthetic_close_command": gripper_command(0.0),
        "synthetic_close_trial": close_trial,
        "saved_model_requested_close": False,
        "model_inference_performed": False,
        "grasp_test_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("saved_model_value", round(value, 6), "command", saved_command)
    print("open_pad_centers_mm", round(open_trial["pad_center_distance_mm"], 2))
    print("synthetic_close_pad_centers_mm", round(close_trial["pad_center_distance_mm"], 2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
