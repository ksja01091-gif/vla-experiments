"""Export verified replays with the XenoArm MolmoAct2 LeRobot v3 layout."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from xenoarm.data.converters.xenoarm_sim import from_npz

from convert_full import VideoWriter
from collect_manifest import read_episodes, write_json


def convert(raw, output, count):
    rows = read_episodes(raw)[:count]
    if len(rows) != count:
        raise ValueError(f"Expected {count} verified demonstrations, found {len(rows)}")
    repo_id = f"xenoarm_panda_scripted/can_k{count}"
    marker = output / "conversion_verified.json"
    if marker.exists():
        assert json.loads(marker.read_text())["seeds"] == [row["seed"] for row in rows]
        return
    partial = output.with_name(output.name + ".partial")
    writer = VideoWriter("molmoact2", repo_id, root=partial, use_videos=True,
                         provenance={"source": "xenoarm_center_alignment", "split": "train"})
    frames = 0
    for index, row in enumerate(rows, 1):
        path = raw / row["file"]
        with np.load(path, allow_pickle=False) as original:
            np.testing.assert_allclose(np.diff(original["simulation_time_s"]),
                                       1 / row["control_hz"], atol=1e-6, rtol=0)
        writer.add(from_npz(path, row=row, fps=row["control_hz"]))
        frames += row["steps"]
        if index <= 10 or index % 20 == 0 or index == count:
            print(f"Converted {index}/{count} episodes, {frames} frames", flush=True)
    writer.close()
    data = LeRobotDataset(repo_id, root=partial, video_backend="pyav")
    assert len(data) == frames
    for index in (0, frames // 2, frames - 1):
        frame = data[index]
        assert frame["observation.state"].shape == (8,) and frame["action"].shape == (7,)
        assert frame["task"] == "pick up the can and place it in its bin"
        for name in ("observation.images.image", "observation.images.wrist_image"):
            assert frame[name].shape[0] == 3 and np.isfinite(frame[name].numpy()).all()
    write_json(partial / marker.name, {"repo_id": repo_id, "episodes": count, "frames": frames,
                                      "seeds": [row["seed"] for row in rows], "fps": rows[0]["control_hz"]})
    partial.rename(output)


def extract_actions(raw, converted, export):
    marker = json.loads((converted / "conversion_verified.json").read_text())
    data = LeRobotDataset(marker["repo_id"], root=converted, video_backend="pyav")
    actions = np.asarray(data.hf_dataset.with_format("numpy")["action"], dtype=np.float32)
    export.mkdir(parents=True, exist_ok=True)
    first = 0
    for row in read_episodes(raw)[:marker["episodes"]]:
        end = first + row["steps"]
        np.savez(export / f"seed-{row['seed']}.npz", action=actions[first:end])
        first = end
    assert first == len(actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--extract-actions", type=Path)
    args = parser.parse_args()
    convert(args.raw, args.output, args.count)
    if args.extract_actions:
        extract_actions(args.raw, args.output, args.extract_actions)
