"""Log existing UR5e diagnostics to W&B offline for later authenticated sync.

This is a post-hoc upload of saved results. It does not rerun a simulation,
perform OpenVLA inference, train a model, or claim task success.
"""

import argparse
import csv
import json
from pathlib import Path

import wandb
import PIL  # wandb.Image requires Pillow; check before starting a run.


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def log_existing_results(root: Path, wandb_dir: Path, project: str) -> dict:
    files = {
        "scene": root / "scene.png",
        "scripted_csv": root / "trajectory/trajectory.csv",
        "scripted_metadata": root / "trajectory/metadata.json",
        "prior_prediction": root / "prior_colab_prediction.json",
        "one_step": root / "one_step_result.json",
        "control_csv": root / "control_check/arm_directions.csv",
        "control_json": root / "control_check/control_check.json",
    }
    for label, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    scripted = read_json(files["scripted_metadata"])
    trajectory = read_csv(files["scripted_csv"])
    prediction = read_json(files["prior_prediction"])
    one_step = read_json(files["one_step"])
    controls = read_json(files["control_json"])
    arm_rows = read_csv(files["control_csv"])
    if len(trajectory) != 91 or len(arm_rows) != 6:
        raise ValueError("Unexpected number of saved trajectory or axis rows")
    if scripted["seed"] != controls["seed"] or scripted["seed"] != 7:
        raise ValueError("Saved seed metadata does not match")

    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        name="UR5e-control-checks-seed7",
        mode="offline",
        dir=str(wandb_dir),
        tags=["UR5e", "MuJoCo", "diagnostic", "posthoc-upload", "no-task-success"],
        config={
            "seed": scripted["seed"],
            "randomization_enabled": False,
            "model_inference_in_this_run": False,
            "base_model_for_prior_prediction": prediction["model"],
            "prior_prediction_provenance": prediction["provenance"],
            "scripted_grasp_uses_openvla": False,
            "task_success_measured": False,
            "coordinate_frame": controls["coordinate_frame"],
            "requested_axis_distance_mm": controls["requested_arm_distance_mm"],
            "axis_trial_duration_s": controls["duration_per_case_s"],
            "one_step_mapping_status": one_step["mapping_status"],
            "one_step_gripper_status": one_step["gripper"],
            "mujoco_version": controls["mujoco_version"],
        },
    )
    if run is None:
        raise RuntimeError("wandb.init did not create an offline run")
    try:
        for row in trajectory:
            run.log({
                "scripted/time_s": float(row["time_s"]),
                "scripted/block_z_m": float(row["block_z_m"]),
                "scripted/end_effector_z_m": float(row["ee_z_m"]),
                "scripted/gripper_command": float(row["gripper_command"]),
            })

        arm_table = wandb.Table(columns=[
            "command", "requested_axis_mm", "corrected_x_mm", "corrected_y_mm",
            "corrected_z_mm", "expected_direction_observed",
        ])
        for row in arm_rows:
            arm_table.add_data(
                row["command"], float(row["requested_axis_mm"]),
                float(row["baseline_corrected_x_mm"]),
                float(row["baseline_corrected_y_mm"]),
                float(row["baseline_corrected_z_mm"]),
                row["commanded_axis_went_expected_way"] == "True",
            )
        gripper_table = wandb.Table(columns=["command", "pad_center_distance_mm"])
        for row in controls["gripper_trials"]:
            gripper_table.add_data(row["command"], row["pad_center_distance_mm"])
        one_step_table = wandb.Table(columns=[
            "axis", "requested_mm", "actual_mm", "zero_command_control_mm",
            "difference_vs_zero_command_mm",
        ])
        for i, axis in enumerate("xyz"):
            one_step_table.add_data(
                axis,
                one_step["requested_ee_delta_m"][i] * 1000,
                one_step["actual_ee_delta_m"][i] * 1000,
                one_step["zero_action_control_ee_delta_m"][i] * 1000,
                one_step["difference_vs_zero_action_control_m"][i] * 1000,
            )
        run.log({
            "control/arm_directions": arm_table,
            "control/gripper_pad_centers": gripper_table,
            "openvla_prior/one_step_motion": one_step_table,
            "scene/initial_image": wandb.Image(str(files["scene"])),
        })
        run.summary["scripted/initial_block_z_m"] = scripted["initial_block_z_m"]
        run.summary["scripted/final_block_z_m"] = scripted["final_block_z_m"]
        run.summary["scripted/peak_block_z_m"] = scripted["peak_block_z_m"]
        run.summary["scripted/recorded_rows"] = len(trajectory)
        run.summary["control/correct_axis_directions"] = sum(
            row["commanded_axis_went_expected_way"] == "True" for row in arm_rows
        )
        run.summary["control/axis_directions_tested"] = len(arm_rows)
        run.summary["control/gripper_open_pad_center_mm"] = controls["gripper_trials"][0]["pad_center_distance_mm"]
        run.summary["control/gripper_closed_pad_center_mm"] = controls["gripper_trials"][1]["pad_center_distance_mm"]
        run.summary["openvla_prior/one_step_block_lifted"] = one_step["block_lifted"]
        run.summary["task_success_measured"] = False

        artifact = wandb.Artifact("ur5e-control-diagnostic-data", type="diagnostic-data")
        for label, path in files.items():
            artifact.add_file(str(path), name=f"{label}/{path.name}")
        run.log_artifact(artifact)
        manifest = {
            "project": project,
            "run_id": run.id,
            "mode": "offline",
            "run_directory": str(Path(run.dir).parent),
            "status": "recorded_locally_not_yet_visible_online",
            "data_files": [str(path) for path in files.values()],
        }
    finally:
        run.finish()
    (root / "wandb_offline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--wandb-dir", type=Path, default=Path.home() / "wandb_offline_runs" / "ur5e")
    parser.add_argument("--project", default="openvla-ur5e-control-diagnostics")
    args = parser.parse_args()
    print(json.dumps(log_existing_results(args.root, args.wandb_dir, args.project), indent=2))
