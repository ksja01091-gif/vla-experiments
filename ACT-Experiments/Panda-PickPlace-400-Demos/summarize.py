"""Export measured Panda ACT timing and verified evaluation totals for publication."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
trials = [
    json.loads(line)
    for line in (root / "evaluation" / "episodes.jsonl").read_text().splitlines()
]
assert sorted(row["seed"] for row in trials) == list(range(50))
summary = json.loads((root / "evaluation" / "summary.json").read_text())
successes = sum(bool(row["success"]) for row in trials)
assert summary["successes"] == successes and summary["episodes"] == 50
elapsed = int((root / "training_seconds.txt").read_text())
result = {
    "model": "ACT",
    "robot": "Panda",
    "task": "XenoArm pp4",
    "gpu": "TITAN RTX",
    "training_demonstrations": 400,
    "training_seed": 777,
    "optimizer_steps": 3000,
    "batch_size": 4,
    "training_wall_seconds": elapsed,
    "eval_seeds": list(range(50)),
    "successes": successes,
    "trials": 50,
    "success_rate": successes / 50,
    "episodes": [
        {k: row[k] for k in ("seed", "success", "initial_state_hash", "steps", "task_metrics") if k in row}
        for row in trials
    ],
}
(root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "episodes"}))
