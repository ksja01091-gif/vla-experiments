"""Verify and index the 40 raw scripted UR5e demonstrations."""

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image


def main(root: Path) -> None:
    manifest = json.loads((root / "collection_summary.json").read_text(encoding="utf-8"))
    episodes = sorted(root.glob("episode_[0-9][0-9][0-9]"))
    problems = []
    positions = set()
    index = []
    target_columns = [f"joint_target_{i}_rad" for i in range(1, 7)] + ["gripper_target"]
    for episode in episodes:
        result = json.loads((episode / "summary.json").read_text(encoding="utf-8"))
        positions.add(tuple(result["block_start_xy_m"]))
        if not result["success_grasp_lift_hold"]:
            problems.append(f"{episode.name}: lift/hold did not succeed")
        with (episode / "demonstration.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        frames = sorted((episode / "frames").glob("frame_*.jpg"))
        if len(rows) != 90 or len(frames) != 90:
            problems.append(f"{episode.name}: {len(rows)} rows, {len(frames)} images")
        for row in rows:
            image_path = episode / row["frame"]
            if not image_path.is_file():
                problems.append(f"Missing image: {image_path}")
                continue
            try:
                with Image.open(image_path) as image:
                    if image.size != (256, 256) or image.mode != "RGB":
                        problems.append(f"Wrong image format: {image_path}")
                    image.verify()
            except Exception as error:
                problems.append(f"Unreadable image: {image_path}: {error}")
            try:
                numbers = [float(row[key]) for key in target_columns]
                if not all(math.isfinite(value) for value in numbers):
                    problems.append(f"Nonfinite action: {episode.name}/{row['frame']}")
                if numbers[-1] not in (0.0, 255.0):
                    problems.append(f"Unexpected gripper target: {episode.name}/{row['frame']}")
            except (KeyError, ValueError) as error:
                problems.append(f"Invalid action: {episode.name}/{row['frame']}: {error}")
            index.append({"episode": episode.name, "image": str(image_path.relative_to(root)),
                          "instruction": row["instruction"], "stage": row["stage"],
                          "time_s": row["time_s"],
                          **{key: row[key] for key in target_columns}})

    if len(episodes) != 40 or len(positions) != 40 or len(index) != 3600:
        problems.append(f"Expected 40 episodes, 40 positions, 3600 pairs; got "
                        f"{len(episodes)}, {len(positions)}, {len(index)}")
    if manifest["successful_episodes"] != len(episodes):
        problems.append("Collection summary disagrees with episode files")

    report = {
        "complete": not problems,
        "episodes": len(episodes),
        "unique_start_positions": len(positions),
        "photo_action_pairs": len(index),
        "image_size": [256, 256],
        "action_type": "six UR5e joint targets plus one Robotiq gripper target",
        "openvla_training_ready": False,
        "problems": problems,
    }
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if problems:
        raise SystemExit(json.dumps(report, indent=2))
    with (root / "all_samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
