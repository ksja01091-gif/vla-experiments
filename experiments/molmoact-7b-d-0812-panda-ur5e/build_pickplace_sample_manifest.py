"""Index reproducible action chunks from the two public PickPlace_D0 datasets.

This only selects existing public data. It does not create new demonstrations,
change the source files, or expose the held-out test episodes to training.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
SAMPLES_PER_EPISODE = 10
ACTION_HORIZON = 8


def main():
    audit = json.loads((ROOT / "data_audit.json").read_text())
    output = ROOT / "sample_manifest"
    output.mkdir(exist_ok=True)
    counts = {}
    for robot in ("panda", "ur5e"):
        dataset = ROOT / "datasets" / robot
        metadata = json.loads((dataset / "meta/info.json").read_text())
        assert metadata["total_episodes"] == 500
        episode_meta = pq.read_table(dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
                                     columns=["episode_index", "dataset_from_index",
                                              "videos/observation.images.agentview/file_index",
                                              "videos/observation.images.agentview/from_timestamp",
                                              "videos/observation.images.wrist/file_index",
                                              "videos/observation.images.wrist/from_timestamp"])
        episode_video = {int(row["episode_index"]): row for row in episode_meta.to_pylist()}
        table = pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet",
                              columns=["index", "episode_index", "frame_index", "timestamp", "action",
                                       "observation.eef", "observation.joint", "observation.gripper"])
        indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_ids = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        eef = np.asarray(table["observation.eef"].to_pylist(), dtype=np.float32)
        joints = np.asarray(table["observation.joint"].to_pylist(), dtype=np.float32)
        gripper = np.asarray(table["observation.gripper"].to_pylist(), dtype=np.float32)
        assert indices.shape[0] == metadata["total_frames"]
        assert np.array_equal(indices, np.arange(len(indices)))
        assert np.isfinite(actions).all() and actions.shape[1] == 7
        episode_rows = {int(ep): np.flatnonzero(episode_ids == ep) for ep in np.unique(episode_ids)}
        counts[robot] = {}
        for split in ("train", "validation"):
            wanted = audit[robot]["episode_ids_by_split"][split]
            out_file = output / f"{robot}_{split}.jsonl"
            with out_file.open("w", encoding="utf-8") as out:
                count = 0
                for episode in wanted:
                    rows = episode_rows[episode]
                    video_meta = episode_video[episode]
                    assert int(video_meta["dataset_from_index"]) == int(indices[rows[0]])
                    assert len(rows) > ACTION_HORIZON
                    assert frame_ids[rows[0]] == 0
                    chosen = np.linspace(0, len(rows) - ACTION_HORIZON, SAMPLES_PER_EPISODE, dtype=int)
                    for offset in chosen:
                        row = int(rows[offset])
                        chunk_rows = rows[offset:offset + ACTION_HORIZON]
                        assert len(chunk_rows) == ACTION_HORIZON
                        video = {}
                        for camera in ("agentview", "wrist"):
                            prefix = f"videos/observation.images.{camera}/"
                            file_index = int(video_meta[prefix + "file_index"])
                            path = dataset / "videos" / f"observation.images.{camera}" / "chunk-000" / f"file-{file_index:03d}.mp4"
                            assert path.exists(), path
                            video[camera] = {
                                "path": str(path),
                                "frame_index_in_video": int(round(float(video_meta[prefix + "from_timestamp"]) * metadata["fps"])) + int(frame_ids[row]),
                            }
                        record = {
                            "source": audit[robot]["source"],
                            "robot": robot,
                            "split": split,
                            "task": "PickPlace_D0",
                            "instruction": "pick up the objects and place them into the bins",
                            "episode_index": episode,
                            "global_data_index": int(indices[row]),
                            "frame_index_within_episode": int(frame_ids[row]),
                            "timestamp": float(timestamps[row]),
                            "agentview_video": video["agentview"],
                            "wrist_video": video["wrist"],
                            "eef_xyz_quat": eef[row].tolist(),
                            "joint": joints[row].tolist(),
                            "gripper_state": gripper[row].tolist(),
                            "next_8_actions": actions[chunk_rows].tolist(),
                        }
                        out.write(json.dumps(record) + "\n")
                        count += 1
            counts[robot][split] = count
            assert count == len(wanted) * SAMPLES_PER_EPISODE
    summary = {
        "source": "public OliverHausdoerfer MimicGen PickPlace_D0 LeRobot datasets",
        "seed": 7,
        "selection": "10 evenly spaced frames per complete demonstration; 8 consecutive actions per frame",
        "not_new_demonstrations": True,
        "test_episodes_excluded": True,
        "counts": counts,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
