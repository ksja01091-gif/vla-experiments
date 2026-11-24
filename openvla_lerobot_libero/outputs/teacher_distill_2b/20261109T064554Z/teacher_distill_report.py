"""Make a compact, explicit report from completed teacher-distillation runs."""
import csv
import json
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent


def read(name):
    value = json.loads((RUN / name).read_text())
    assert value["complete"], name
    return value


def main():
    teacher = read("teacher_collection.json")
    before = read("student_before_evaluation.json")
    training = read("training_result.json")
    after = read("student_after_evaluation.json")
    by_model = {"teacher_7B": teacher["trials"],
                "student_2B_before": before["trials"],
                "student_2B_after": after["trials"]}
    with (RUN / "paired_50_states.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "group", "teacher_success", "student_before_success",
                         "student_after_success", "teacher_steps", "before_steps", "after_steps"])
        for state in range(50):
            rows = [by_model[name][state] for name in by_model]
            assert all(row["init_state_index"] == state for row in rows)
            assert len({row["selected_init_state_sha256"] for row in rows}) == 1
            assert len({row["initial_image_sha256"] for row in rows}) == 1
            writer.writerow([state, "training_5_44" if 5 <= state <= 44 else "evaluation_only",
                             *[int(row["success"]) for row in rows],
                             *[row["policy_steps"] for row in rows]])
    lines = ["# Teacher rollout → 2B student: result", "",
             "All rows use the same official LIBERO Spatial bowl task and the same 50 initial states.",
             "The student trained on successful teacher trajectories from states 5–44 only.",
             "States 0–4 and 45–49 were excluded from the new distillation data.", "",
             "| Model | All 50 | Distillation states 5–44 | Evaluation-only states 0–4, 45–49 |",
             "|---|---:|---:|---:|"]
    for name, trials in by_model.items():
        all_success = sum(row["success"] for row in trials)
        train_success = sum(row["success"] for row in trials[5:45])
        eval_success = all_success - train_success
        lines.append(f"| {name} | {all_success}/50 | {train_success}/40 | {eval_success}/10 |")
    lines += ["", f"Successful teacher training episodes: {teacher['new_training_episodes']}",
              f"New teacher image-action pairs: {teacher['new_training_frames']}",
              f"Existing human image-action pairs mixed in training: {training['human_frames']}",
              f"Student training time: {training['training_seconds'] / 60:.1f} minutes",
              f"Student GPU peak allocated: {training['gpu_peak_gib']:.2f} GiB",
              f"Human validation loss: {training['before_validation']['loss']:.4f} → {training['after_validation']['loss']:.4f}",
              "", "Interpretation: the 50-state total is descriptive, not an independent generalization estimate.",
              "The 10 states excluded from new distillation data were previously inspected in earlier experiments; they are not a blind test.",
              "A lower validation loss alone does not prove higher simulator success.",
              "The teacher collection on fixed official states cannot create unlimited distinct start scenes.", ""]
    (RUN / "RESULTS.md").write_text("\n".join(lines))
    print("REPORT_COMPLETE", RUN / "RESULTS.md")


if __name__ == "__main__":
    deadline = time.time() + 12 * 3600
    while not (RUN / "student_after_evaluation.json").exists():
        status_path = RUN / "pipeline_status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("status") == "error":
            raise RuntimeError("Pipeline failed; inspect pipeline_status.json")
        if time.time() >= deadline:
            raise TimeoutError("Pipeline did not finish within 12 hours")
        time.sleep(30)
    main()
