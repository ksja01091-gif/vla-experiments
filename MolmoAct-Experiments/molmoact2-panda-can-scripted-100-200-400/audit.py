"""Replay the native and exported actions from files written by the pilot."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from collect import replay
from collect_manifest import read_episodes, write_json
from xenoarm.data.actions import canonical_to_robosuite_osc


def check(item):
    raw, exported, row = item
    with np.load(raw / row["file"], allow_pickle=False) as original:
        actions = original["action"]
    with np.load(exported / f"seed-{row['seed']}.npz", allow_pickle=False) as converted:
        decoded = canonical_to_robosuite_osc(converted["action"])
    np.testing.assert_allclose(decoded, actions, atol=1e-7, rtol=0)
    result = {"seed": row["seed"], "frames": len(actions), "raw": replay(row, actions)[0],
              "converted": replay(row, decoded)[0],
              "max_action_difference": float(np.max(np.abs(actions - decoded)))}
    for mode in ("raw", "converted"):
        assert result[mode]["lift_step"] is not None and result[mode]["final_stable_steps"] >= 10, result
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--exported", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_episodes(args.raw)[:8]
    assert len(rows) == 8
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, ((args.raw, args.exported, row) for row in rows)))
    write_json(args.out, results)
    print(f"Verified saved native and exported replays: {len(results)}/{len(rows)}", flush=True)
