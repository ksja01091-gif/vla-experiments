"""Upload the training curve, per-layout metrics, and representative dual-camera videos."""

import argparse
import json
from pathlib import Path

import wandb


def publish(adapter, evaluation, count, elapsed, expert_only=False):
    receipt = evaluation / "wandb_url.txt"
    if receipt.exists():
        return
    config = json.loads((adapter / "xenoarm_run.json").read_text())
    summary = json.loads((evaluation / "summary.json").read_text())
    scope = "Action Expert only" if expert_only else "Language + Action Expert"
    suffix = "-expert-only" if expert_only else ""
    name = f"MolmoAct2 Panda Can Scripted LoRA r32 | {count} demos | seed 777"
    if expert_only:
        name += f" | {scope}"
    rows = summary["episodes"]
    with wandb.init(entity="gist-infonet", project="xenoarm-proposal", name=name,
                    group="MolmoAct2 Panda Can LoRA Scope" if expert_only else "MolmoAct2 Panda Can Scripted Data Scale",
                    id=f"panda-can-k{count}-s777{suffix}", resume="allow",
                    config={**config, "task": "panda_can_to_bin", "batch_size": 4, "lora_scope": scope},
                    dir=str(evaluation), mode="online") as run:
        run.define_metric("optimizer_step")
        run.define_metric("train/*", step_metric="optimizer_step")
        with (adapter / "train_log.jsonl").open() as source:
            for line in source:
                item = json.loads(line)
                step = item.pop("step")
                run.log({"optimizer_step": step, **{f"train/{k}": v for k, v in item.items()}})
        table = wandb.Table(columns=["seed", "lifted", "placed", "steps", "lift_step", "place_step", "video"])
        featured = {row["seed"] for outcome in (False, True) for row in
                    [item for item in rows if item["placed"] == outcome][:2]}
        for row in rows:
            video = wandb.Video(str(evaluation / row["video"])) if row["seed"] in featured else None
            table.add_data(*(row[key] for key in ("seed", "lifted", "placed", "steps", "lift_step", "place_step")), video)
        run.log({"evaluation/trials": table, "evaluation/lift_rate": summary["lifted"] / summary["trials"],
                 "evaluation/place_rate": summary["placed"] / summary["trials"], "train/wall_seconds": elapsed})
        if expert_only:
            comparison = json.loads((evaluation / "comparison.json").read_text())
            columns = ["LoRA scope", "lifted / 50", "placed / 50", "training seconds", "peak GPU memory (GiB)"]
            variants = ("both", "expert_only")
            keys = ("lifted", "placed", "training_wall_seconds", "peak_training_allocated_vram_gib")
            paired = [[label, *(comparison[variant][key] for key in keys)]
                      for label, variant in zip(("Language + Action Expert", scope), variants, strict=True)]
            run.log({"comparison/metrics": wandb.Table(columns=columns, data=paired),
                     "comparison/chart": wandb.Image(str(evaluation / "comparison.png"))})
        run.summary.update({"evaluation/lifted": summary["lifted"], "evaluation/placed": summary["placed"],
                            "evaluation/attempts": summary["trials"], "eval/wall_seconds": summary["elapsed_seconds"]})
        url = run.url
    receipt.write_text(url + "\n")
    print(url, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("count", type=int)
    parser.add_argument("elapsed", type=int)
    parser.add_argument("--expert-only", action="store_true")
    args = parser.parse_args()
    publish(args.adapter, args.evaluation, args.count, args.elapsed, args.expert_only)
