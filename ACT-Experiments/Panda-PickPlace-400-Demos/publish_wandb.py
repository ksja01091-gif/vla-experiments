"""Publish the measured duration and pp4 results to the team's W&B project."""
import json
from pathlib import Path
import sys
import time

import wandb

result = json.loads(Path(sys.argv[1]).read_text())
trials = result["episodes"]
assert sorted(row["seed"] for row in trials) == list(range(50))
assert sum(row["success"] for row in trials) == result["successes"]

run_id = "act-panda-pp4-k400-seed777-step3000-result"
metrics = {
    "train/wall_seconds": result["training_wall_seconds"],
    "train/wall_minutes": result["training_wall_seconds"] / 60,
    "eval/success_rate": result["success_rate"],
    "eval/successes": result["successes"],
    "eval/trials": result["trials"],
}
config_keys = ("model", "robot", "task", "gpu", "training_demonstrations",
               "training_seed", "optimizer_steps", "batch_size", "eval_seeds")
with wandb.init(entity="gist-infonet", project="xenoarm-proposal", id=run_id,
                name=run_id, job_type="evaluation", mode="online", resume="allow",
                config={key: result[key] for key in config_keys}) as run:
    table = wandb.Table(columns=["seed", "success", "steps"])
    for row in trials:
        table.add_data(row["seed"], row["success"], row.get("steps"))
    run.log({**metrics, "eval/episodes": table})
    run.summary.update(metrics)
    url = run.url

for attempt in range(6):
    checked = wandb.Api(timeout=30).run(f"gist-infonet/xenoarm-proposal/{run_id}")
    if all(checked.summary.get(key) == metrics[key] for key in
           ("eval/trials", "eval/successes", "train/wall_seconds")):
        break
    time.sleep(5)
else:
    raise RuntimeError(f"W&B did not return the uploaded metrics: {url}")
print(f"VERIFIED_WANDB {url}")
