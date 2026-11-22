"""Publish the completed UR5e pi0.5 evaluation and its raw evidence to W&B.

Run with the already authenticated W&B Python environment. This script does not
train or re-evaluate the policy; it only uploads the recorded 90 trials.
"""

import csv
import json
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval_seen40_unseen50_20261122"
PROJECT = "pi05-ur5e-mujoco"
RUN_ID = "pi05ur5e90nov22"


def read_evidence():
    manifest = json.loads((EVAL / "manifest.json").read_text())
    report = json.loads((EVAL / "report.json").read_text())
    with (EVAL / "all_90_trials.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 90
    assert sum(r["group"] == "seen" for r in rows) == 40
    assert sum(r["group"] == "new" for r in rows) == 50
    for group, count in (("seen", 40), ("new", 50)):
        assert report[group]["completed"] == count
        assert report[group]["lift_5cm"] == sum(
            r["lift_5cm"].lower() == "true" for r in rows if r["group"] == group
        )
        for i in range(count):
            episode = EVAL / group / f"{i:02d}"
            for name in ("summary.json", "actions.csv", "start.png", "final.png", "rollout.mp4"):
                assert (episode / name).is_file(), episode / name
    return manifest, report, rows


def main():
    manifest, report, rows = read_evidence()
    entity = wandb.Api(timeout=30).default_entity
    run = wandb.init(
        entity=entity,
        project=PROJECT,
        id=RUN_ID,
        resume="allow",
        name="Pi05 UR5e: 40 seen + 50 new positions",
        job_type="evaluation",
        tags=["pi05", "UR5e", "MuJoCo", "40-seen", "50-new", "lift-5cm"],
        config={
            "checkpoint": manifest["checkpoint"],
            "training_demonstrations": 40,
            "training_image_action_pairs": 3600,
            "validation_demonstrations_excluded": 5,
            "seen_positions": 40,
            "new_positions": 50,
            "new_position_seed": manifest["new_position_seed"],
            "policy_seed_per_episode": manifest["policy_seed_per_episode"],
            "horizon_actions": manifest["horizon_actions"],
            "physics_steps_per_action": manifest["physics_steps_per_action"],
            "score_definition": manifest["score"],
        },
        settings=wandb.Settings(init_timeout=120),
    )
    assert run is not None
    columns = ["group", "index", "x_m", "y_m", "lift_5cm", "final_lift_5cm",
               "min_pinch_block_distance_m", "peak_block_z_m", "gripper_close_steps",
               "first_close_step", "steps", "wall_seconds"]
    numeric = {"index", "x_m", "y_m", "min_pinch_block_distance_m",
               "peak_block_z_m", "gripper_close_steps", "steps", "wall_seconds"}
    table = wandb.Table(columns=columns)
    for row in rows:
        values = []
        for key in columns:
            value = row[key]
            if key in numeric:
                value = float(value) if value else None
            elif key in {"lift_5cm", "final_lift_5cm"}:
                value = value.lower() == "true"
            elif key == "first_close_step":
                value = int(float(value)) if value else None
            values.append(value)
        table.add_data(*values)
    run.log({
        "evaluation/seen_lifts_5cm": report["seen"]["lift_5cm"],
        "evaluation/seen_attempts": report["seen"]["completed"],
        "evaluation/seen_lift_rate": report["seen"]["lift_5cm"] / report["seen"]["completed"],
        "evaluation/new_lifts_5cm": report["new"]["lift_5cm"],
        "evaluation/new_attempts": report["new"]["completed"],
        "evaluation/new_lift_rate": report["new"]["lift_5cm"] / report["new"]["completed"],
        "evaluation/seen_mean_min_distance_m": report["seen"]["mean_min_distance_m"],
        "evaluation/new_mean_min_distance_m": report["new"]["mean_min_distance_m"],
        "evaluation/trials": table,
        "evaluation/seen_success_video": wandb.Video(str(EVAL / "seen" / "34" / "rollout.mp4"), format="mp4"),
        "evaluation/new_failure_video": wandb.Video(str(EVAL / "new" / "00" / "rollout.mp4"), format="mp4"),
    })
    artifact = wandb.Artifact(
        "pi05-ur5e-seen40-new50-20261122", type="evaluation",
        description="All 90 trial summaries, actions, images, and rollout videos. Score is a 5 cm block lift, not full task completion.",
    )
    artifact.add_dir(str(EVAL))
    artifact.add_file(str(ROOT / "evaluate_pi05_ur5e_90.py"), name="evaluate_pi05_ur5e_90.py")
    run.log_artifact(artifact)
    print("WANDB_RUN", run.url, flush=True)
    run.finish()


if __name__ == "__main__":
    main()
