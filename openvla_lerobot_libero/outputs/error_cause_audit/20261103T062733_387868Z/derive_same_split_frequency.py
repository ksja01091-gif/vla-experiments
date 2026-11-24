from pathlib import Path
import json
import numpy as np
r=Path(__file__).parent
old=Path("/home/jovyan/projects/lerobot-openvla/outputs/full_validation_actions/20261103T061517_441641Z")
rows=[json.loads(x) for x in (old/"frame_predictions.jsonl").read_text().splitlines()]
baseline=json.loads((r/"mode_and_image_pairing_diagnostics.json").read_text())
modes=np.asarray(baseline["per_dimension_mode_token_ids"])
truth=np.asarray([x["true_action_token_ids"] for x in rows])
prediction=np.asarray([x["predicted_action_token_ids"] for x in rows])
result={"validation_frames":462,"meaning":"same action bin as the modal training value, per dimension",
 "demonstration_frequency":(truth==modes).mean(axis=0).tolist(),
 "model_frequency":(prediction==modes).mean(axis=0).tolist()}
(r/"same_split_frequency_check.json").write_text(json.dumps(result,indent=2))
