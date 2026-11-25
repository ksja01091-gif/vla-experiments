"""Compute MolmoAct action scaling from training episodes only."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
audit = json.loads((ROOT / "data_audit.json").read_text())
result = {"seed": audit["seed"], "gripper_convention": "model 0=close, 1=open; sim +1=close, -1=open"}

for robot in ("panda", "ur5e"):
    train_ids = set(audit[robot]["episode_ids_by_split"]["train"])
    chunks = []
    for file in sorted((ROOT / "datasets" / robot / "data").rglob("*.parquet")):
        table = pq.read_table(file, columns=["episode_index", "action"])
        mask = np.isin(table["episode_index"].to_numpy(), list(train_ids))
        if mask.any():
            chunks.append(np.asarray(table["action"].to_pylist(), dtype=np.float32)[mask])
    raw = np.concatenate(chunks)
    assert raw.shape == (audit[robot]["frames_by_split"]["train"], 7)
    assert np.isfinite(raw).all()
    assert np.all(np.isin(raw[:, 6], [-1, 1]))
    q01 = np.quantile(raw, 0.01, axis=0)
    q99 = np.quantile(raw, 0.99, axis=0)
    result[robot] = {
        "source_train_episodes": len(train_ids),
        "source_train_frames": len(raw),
        "action": {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "mask": [True] * 6 + [False],
        },
        "sim_gripper_counts": {
            "open_minus_one": int(np.count_nonzero(raw[:, 6] == -1)),
            "close_plus_one": int(np.count_nonzero(raw[:, 6] == 1)),
        },
    }

(ROOT / "train_action_stats.json").write_text(json.dumps(result, indent=2))
print(json.dumps({r: {"frames": result[r]["source_train_frames"],
                      "gripper": result[r]["sim_gripper_counts"]}
                  for r in ("panda", "ur5e")}, indent=2))
