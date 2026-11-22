"""Publish recorded training and prediction diagnostics (no model weights)."""

import json
from pathlib import Path

import wandb


HERE = Path(__file__).resolve().parent


def main():
    diagnostics = HERE / "diagnostics_20261122"
    model_config = HERE / "model_config"
    probe = json.loads((diagnostics / "expert_frame_probe_results.json").read_text())
    action = json.loads((diagnostics / "fresh_action_result.json").read_text())
    training = json.loads((model_config / "train_config.json").read_text())
    rollout = json.loads((HERE / "rollout_state0" / "summary.json").read_text())
    assert probe["count"] == 10 and action["all_finite"]
    assert action["output_shape"] == [1, 7]
    assert training["steps"] == 3600
    entity = wandb.Api(timeout=30).default_entity
    run = wandb.init(
        entity=entity,
        project="pi05-ur5e-mujoco",
        id="pi05ur5ediagnov22",
        resume="allow",
        name="Pi05 UR5e: training and prediction diagnostics",
        job_type="diagnostics",
        tags=["pi05", "UR5e", "MuJoCo", "40-demonstrations"],
        config={
            "base_model": training["policy"]["pretrained_path"],
            "training_steps": training["steps"],
            "batch_size": training["batch_size"],
            "seed": training["seed"],
            "train_expert_only": training["policy"]["train_expert_only"],
            "freeze_vision_encoder": training["policy"]["freeze_vision_encoder"],
            "use_peft": training["policy"]["use_peft"],
            "paired_evaluation_run": "pi05ur5e90nov22",
        },
        settings=wandb.Settings(init_timeout=120),
    )
    assert run is not None
    run.log({
        "diagnostics/probe_frames": probe["count"],
        "diagnostics/mean_joint_l2_rad": probe["mean_joint_l2_rad"],
        "diagnostics/gripper_matches": probe["gripper_matches"],
        "diagnostics/gripper_match_rate": probe["gripper_matches"] / probe["count"],
        "diagnostics/fresh_action_finite": int(action["all_finite"]),
        "diagnostics/initial_rollout_lift_5cm": int(rollout.get("lift_5cm", False)),
        "diagnostics/initial_rollout_video": wandb.Video(
            str(HERE / "rollout_state0" / "pi05_ur5e_state0.mp4"), format="mp4"
        ),
    })
    artifact = wandb.Artifact(
        "pi05-ur5e-training-diagnostics-20261122",
        type="diagnostics",
        description="Training configuration, model configuration, ten-frame expert comparison, and first rollout. The 9.35 GB checkpoint remains in research-lab Google Drive.",
    )
    artifact.add_dir(str(diagnostics), name="diagnostics_20261122")
    artifact.add_dir(str(model_config), name="model_config")
    artifact.add_dir(str(HERE / "rollout_state0"), name="rollout_state0")
    run.log_artifact(artifact)
    print("WANDB_RUN", run.url, flush=True)
    run.finish()


if __name__ == "__main__":
    main()
