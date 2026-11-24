"""Export a fixed 20/5 whole-demonstration split to local LeRobot v3 datasets.
No model loading, optimizer, training, or Hub upload is performed.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from lerobot.datasets.lerobot_dataset import LeRobotDataset

IMAGE_KEY = "observation.images.image"
ROOT = Path("/home/jovyan/projects/datasets/libero_bowl_25/20261101T120613_454595Z")

@contextmanager
def load_demo(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    yield arrays

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

def features():
    return {
        IMAGE_KEY: {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        "action": {"dtype": "float32", "shape": (7,), "names": [f"action_{i}" for i in range(7)]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": [f"state_{i}" for i in range(8)]},
        "observation.joint_state": {"dtype": "float32", "shape": (7,), "names": [f"joint_{i}" for i in range(7)]},
        "source_step_index": {"dtype": "int64", "shape": (1,), "names": ["source_step"]},
    }

def export_split(root, name, records, task):
    repo_id = f"local/openvla-libero-bowl-{name}"
    output = root / name
    if output.exists():
        raise FileExistsError(f"Preserving existing dataset: {output}")
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=20, features=features(),
                                    root=output, robot_type="libero_panda", use_videos=False)
    mapping = []
    try:
        for episode_index, record in enumerate(records):
            archive = Path(record["archive"])
            if hashlib.sha256(archive.read_bytes()).hexdigest() != record["archive_sha256"]:
                raise RuntimeError("Source archive checksum mismatch")
            with load_demo(archive) as demo:
                for step in range(record["frames"]):
                    dataset.add_frame({
                        IMAGE_KEY: demo["images"][step],
                        "action": demo["actions"][step],
                        "observation.state": demo["states"][step],
                        "observation.joint_state": demo["joints"][step],
                        "source_step_index": np.array([step], dtype=np.int64),
                        "task": task,
                    })
                dataset.save_episode()
            mapping.append({"lerobot_episode_index": episode_index, **record})
            print("EXPORTED", name, episode_index + 1, "/", len(records), "frames", record["frames"], flush=True)
    finally:
        dataset.finalize()
    write_json(root / f"{name}_episode_provenance.json", mapping)
    return repo_id, output

def verify_split(repo_id, output, records, task):
    dataset = LeRobotDataset(repo_id=repo_id, root=output)
    expected_frames = sum(r["frames"] for r in records)
    assert dataset.num_episodes == len(records)
    assert len(dataset) == expected_frames
    cursor = 0
    image_checks = 0
    for episode_index, record in enumerate(records):
        with load_demo(record["archive"]) as demo:
            for step in range(record["frames"]):
                item = dataset[cursor + step]
                action = item["action"].cpu().numpy()
                state = item["observation.state"].cpu().numpy()
                joint = item["observation.joint_state"].cpu().numpy()
                assert action.shape == (7,) and np.isfinite(action).all()
                assert np.array_equal(action, demo["actions"][step])
                assert np.array_equal(state, demo["states"][step])
                assert np.array_equal(joint, demo["joints"][step])
                assert 0 <= float(action[-1]) <= 1
                assert int(item["episode_index"]) == episode_index
                assert int(item["source_step_index"].reshape(-1)[0]) == step
                assert item["task"] == task
                if step in {0, record["frames"] - 1}:
                    pixels = item[IMAGE_KEY].permute(1, 2, 0).cpu().numpy()
                    pixels = np.rint(pixels * 255).astype(np.uint8)
                    assert np.array_equal(pixels, demo["images"][step]), "Image pixels changed"
                    image_checks += 1
        cursor += record["frames"]
    print("VERIFIED", output.name, "episodes", len(records), "frames", cursor, "image_checks", image_checks, flush=True)
    return {"episodes": len(records), "frames": cursor, "all_actions_states_joints_exact": True,
            "first_last_images_exact": image_checks, "finite_actions": True, "all_tasks_match": True}

def make_preview(root, record):
    preview = root / "preview"
    preview.mkdir(exist_ok=False)
    with load_demo(record["archive"]) as demo:
        count = len(demo["images"])
        indices = np.linspace(0, count - 1, 6, dtype=int)
        sheet = Image.new("RGB", (3 * 256, 2 * 286), "white")
        draw = ImageDraw.Draw(sheet)
        for cell, index in enumerate(indices):
            x, y = (cell % 3) * 256, (cell // 3) * 286
            sheet.paste(Image.fromarray(demo["images"][index]), (x, y))
            draw.text((x + 8, y + 260), f"Demonstration step {index + 1}/{count}", fill="black")
        sheet.save(preview / "training_demo_contact_sheet.png")
        frames = [Image.fromarray(demo["images"][i]) for i in range(0, count, 3)]
        frames[0].save(preview / "training_demo.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
        Image.fromarray(demo["images"][0]).save(preview / "first_observation.png")
        example = {"meaning": "expert demonstration, not a model rollout", "content_id": record["content_id"],
                   "instruction": record["instruction"], "source_step_index": 0,
                   "image": str(preview / "first_observation.png"),
                   "action_raw": demo["actions_raw"][0].tolist(),
                   "action_openvla_training": demo["actions"][0].tolist(),
                   "state": demo["states"][0].tolist(), "episode_frames": count}
        write_json(preview / "example.json", example)
    return example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.absolute()
    if not (root / "split_manifest.json").exists():
        env = os.environ.copy()
        env.update({"TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "2"})
        subprocess.run(["/home/jovyan/projects/venvs/openvla_data310/bin/python", "-u",
                        str(Path(__file__).with_name("extract_libero_bowl_episodes.py")),
                        "--raw-dir", "/home/jovyan/projects/datasets/openvla_modified_libero_rlds/libero_spatial_no_noops/1.0.0",
                        "--output", str(root)], env=env, check=True)
    split = json.loads((root / "split_manifest.json").read_text())
    train_ids = {r["content_id"] for r in split["train"]}
    val_ids = {r["content_id"] for r in split["validation"]}
    assert len(train_ids) == 20 and len(val_ids) == 5 and not train_ids & val_ids
    reports = {}
    for name in ["train", "validation"]:
        output = root / name
        if output.exists():
            if not (root / f"{name}_episode_provenance.json").exists():
                raise RuntimeError(f"Incomplete existing export preserved: {output}. Use a fresh --root.")
            repo_id = f"local/openvla-libero-bowl-{name}"
        else:
            repo_id, output = export_split(root, name, split[name], split["task"])
        reports[name] = verify_split(repo_id, output, split[name], split["task"])
    if (root / "preview/example.json").exists():
        example = json.loads((root / "preview/example.json").read_text())
    else:
        example = make_preview(root, split["train"][0])
    report = {"source_repo": "openvla/modified_libero_rlds",
              "source_revision": "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551",
              "source_subset": "libero_spatial_no_noops/1.0.0", "lerobot_format": "v3.0",
              "task": split["task"], "splits": reports, "episode_overlap": False,
              "normalization_statistics": "action_statistics_train_only.json (training split only)",
              "fps_note": "20 Hz is an indexing convention for already-compacted no-noops steps; not a recovered wall-clock trajectory",
              "evaluation_initial_state_disjointness": split["evaluation_initial_state_disjointness"],
              "preview": str(root / "preview/training_demo_contact_sheet.png"),
              "example": example, "training_started": False, "hub_upload": False}
    write_json(root / "verification_report.json", report)
    print("DATA_PREPARATION_COMPLETE", root, flush=True)
    print(json.dumps(reports, indent=2), flush=True)

if __name__ == "__main__":
    main()
