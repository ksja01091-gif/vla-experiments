"""Replay one OpenVLA gripper value with both UR5e sign conventions.

The arm starts at the saved scripted pregrasp state and follows the same
scripted lift in each branch. This tests the interface mapping, not whether
OpenVLA can approach the block on its own.
"""

import json
from pathlib import Path

from test_pregrasp_gripper_reference import trial


ROOT = Path(__file__).parent / "pregrasp_20261119"
prediction = json.loads((ROOT / "pregrasp_prediction.json").read_text())
value = prediction["gripper_value"]
assert prediction["all_action_tokens_valid"]
assert -1.0 <= value <= 1.0

result = {
    "source_model": prediction["model"],
    "gripper_token_id": prediction["gripper_token_id"],
    "normalized_gripper_value": value,
    "same_saved_pregrasp_state": True,
    "arm_lift_is_scripted": True,
    "positive_means_open": trial(ROOT / "pregrasp_state.npz", 0 if value > 0 else 255),
    "positive_means_close": trial(ROOT / "pregrasp_state.npz", 255 if value > 0 else 0),
    "model_intended_ur5e_sign_verified": False,
}
out = ROOT / "predicted_gripper_mapping_trials.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
