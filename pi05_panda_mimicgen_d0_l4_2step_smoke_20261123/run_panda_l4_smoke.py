"""Repeat the two-update Panda Pi05 training connection test in Colab."""

from pathlib import Path
import subprocess


DATASET = "OliverHausdoerfer/demo_src_pick_place_task_D0_robot_Panda_gripper_PandaGripper"
OUTPUT_DIR = Path("/content/pi05_mimicgen_panda_smoke")
LOG_FILE = Path("/content/panda_smoke.log")


def main() -> None:
    command = [
        "lerobot-train",
        f"--dataset.repo_id={DATASET}",
        "--dataset.episodes=[0,1]",
        "--policy.type=pi05",
        "--policy.pretrained_path=lerobot/pi05_base",
        "--policy.empty_cameras=1",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        '--rename_map={"observation.eef":"observation.state"}',
        f"--output_dir={OUTPUT_DIR}",
        "--batch_size=1",
        "--steps=2",
        "--num_workers=0",
        "--log_freq=1",
        "--save_checkpoint=false",
        "--seed=7",
    ]
    with LOG_FILE.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Training completed. Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
