"""Convert the existing full MimicGen split to XenoArm's MolmoAct2 layout."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from xenoarm.data import schema
from xenoarm.data.actions import robosuite_osc_to_canonical
from xenoarm.data.converters.xenoarm_sim import robot_json
from xenoarm.data.export import LeRobotWriter, layout_features
from xenoarm.data.mimicgen import (
    CAMERAS, COLUMNS, REPOS, TASK, LeRobotV3, action_agreement,
    eef_frame, gripper_agreement, problems,
)
from xenoarm.envs.robosuite_multi import TASKS, gripper_open_fraction


class VideoWriter(LeRobotWriter):
    def _create(self, episode):
        self.dataset = self._cls.create(
            repo_id=self.repo_id, root=self.root, fps=round(episode.fps),
            features=layout_features(self.layout, episode, use_videos=True),
            robot_type=episode.meta["robot_type"], use_videos=True,
            streaming_encoding=True, encoder_threads=2,
        )
        self.fps = episode.fps


def convert_episode(reader, episode, robot, ref):
    index = episode["episode_index"]
    raw = reader.frames(episode, list(COLUMNS.values()))
    action, pose, qpos = (raw[COLUMNS[key]] for key in ("action", "eef", "gripper"))
    joint = raw[COLUMNS["joint"]]
    n = len(action)
    assert action.shape == pose.shape == (episode["length"], 7)
    assert all(len(value) == n and np.isfinite(value).all() for value in raw.values())
    frame = eef_frame(pose[:, :3])
    world = pose[:, :3] + (ref["base_pos"] if frame == "base" else 0)
    qnorm = np.linalg.norm(pose[:, 3:], axis=1, keepdims=True)
    if np.any(abs(qnorm - 1) > 0.02):
        raise ValueError(f"Invalid quaternion in {robot} episode {index}")
    quat = pose[:, 3:] / qnorm
    fingers = len(ref["gripper_ranges"])
    opened = np.array([gripper_open_fraction(q, ref["gripper_ranges"], ref["open_at_reset"])
                       for q in qpos[:, :fingers]])
    padding = {"gripper": qpos[:, fingers:], "joint": joint[:, ref["arm_dof"]:]}
    check = {
        "episode_index": index, "length": n, "eef_frame": frame,
        "quat_norm_range": [float(qnorm.min()), float(qnorm.max())],
        "action_out_of_range_fraction": float(np.mean(np.any(abs(action[:, :6]) > 1.0001, axis=1))),
        "agreement": action_agreement(action, world, quat),
        "gripper": gripper_agreement(action[:, 6], opened),
        "padding_max_abs": {k: float(abs(v).max()) if v.size else 0.0 for k, v in padding.items()},
        "start_vs_sim": {
            "pos_error_m": float(np.linalg.norm(world[0] - ref["eef_pos"])),
            "quat_abs_dot": float(abs(quat[0] @ (ref["eef_quat"] / np.linalg.norm(ref["eef_quat"])))),
        },
    }
    errors, warnings = problems(check)
    if errors:
        raise ValueError(f"{robot} episode {index}: {errors}")
    instruction = TASKS[TASK]["instruction"]
    images = {k: np.ascontiguousarray(reader.video(episode, name)[:, :, ::-1])
              for k, name in CAMERAS.items()}
    frames = {
        schema.EXTERIOR: images["agentview"], schema.WRIST: images["wrist"],
        schema.EE_POSE: np.column_stack((world - ref["base_pos"], quat)).astype(np.float32),
        schema.GRIPPER: (1 - opened[:, None]).astype(np.float32),
        schema.ACTION: robosuite_osc_to_canonical(action),
        schema.ACTION_ORIGINAL: action.astype(np.float32),
        schema.TIMESTAMP: np.arange(n, dtype=np.float64) / reader.fps,
    }
    meta = {
        "task_id": TASK, "instruction": instruction, "robot_type": robot,
        "length": n, "source": REPOS[robot], "source_episode": index,
        "success": True, "success_criterion": "Published MimicGen successful demonstration",
        "initial_layout_id": f"mimicgen-{index}", "camera_transform": "horizontal_flip",
        "initial_state_hash": hashlib.sha256(f"{REPOS[robot]}/{index}".encode()).hexdigest(),
    }
    return schema.Episode(frames=frames, meta=meta,
                          robot=robot_json(ref["metadata"], robot, reader.fps, ref["base_pos"]),
                          fps=reader.fps), warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", choices=REPOS, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    target = args.output / args.robot / args.split
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".conversion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        convert(args, target)


def convert(args, target):
    robot, split = args.robot, args.split
    audit = json.loads((args.source / "data_audit.json").read_text())[robot]
    reader = LeRobotV3(args.source / "datasets" / robot)
    episodes = {e["episode_index"]: e for e in reader.episodes()}
    splits = audit["episode_ids_by_split"]
    membership = [i for ids in splits.values() for i in ids]
    assert audit["source"] == REPOS[robot] and len(membership) == len(set(membership)) == len(episodes) == 500
    assert set(membership) == set(episodes)
    ids = sorted(splits[split])[:args.limit]
    partial = target.with_name(target.name + ".partial")
    if (target / "conversion_verified.json").exists():
        print(f"Already verified: {target}", flush=True)
        return
    if partial.exists():
        raise FileExistsError(f"Previous partial dataset at {partial}")
    ref = json.loads(Path(f"full_data_reference_{robot}.json").read_text())
    for key in ("base_pos", "eef_pos", "eef_quat", "gripper_ranges", "open_at_reset"):
        ref[key] = np.asarray(ref[key])
    writer = VideoWriter("molmoact2", f"xenoarm_molmo_full/{robot}_{split}", root=partial,
                         use_videos=True, provenance={"split": split, "split_seed": 7, "source": REPOS[robot]})
    frames = 0
    for number, index in enumerate(ids, 1):
        episode, warnings = convert_episode(reader, episodes[index], robot, ref)
        writer.add(episode)
        frames += episode.length
        print(f"{robot} {split} {number}/{len(ids)} source={index} frames={frames} warnings={warnings}", flush=True)
    writer.close()
    dataset = LeRobotDataset(writer.repo_id, root=partial, video_backend="pyav")
    assert len(dataset) == frames
    for index in (0, frames - 1):
        row = dataset[index]
        assert tuple(row["action"].shape) == (7,) and tuple(row["observation.state"].shape) == (8,)
        assert all(row[key].shape[0] == 3 for key in ("observation.images.image", "observation.images.wrist_image"))
    (partial / "conversion_verified.json").write_text(json.dumps(
        {"robot": robot, "split": split, "episodes": len(ids), "frames": frames, "source_ids": ids}, indent=2))
    partial.rename(target)


if __name__ == "__main__":
    main()
