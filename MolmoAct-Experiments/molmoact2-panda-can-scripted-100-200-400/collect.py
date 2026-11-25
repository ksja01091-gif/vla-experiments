"""Record successful Panda can placements with the existing centered-grasp controller."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

import center_alignment
from xenoarm.data.actions import canonical_to_robosuite_osc, robosuite_osc_to_canonical
from xenoarm.dataset import EpisodeRecorder
from xenoarm.envs.robosuite_multi import RobosuiteMultiRobotEnv
from collect_manifest import write_json


INSTRUCTION = "pick up the can and place it in its bin"
# Preserve the scene used for the 42/50 diagnostic, including the other three objects.
SCENE = {"backend": "robosuite", "robot": "panda", "task": "pick_place", "resolution": 128}
EVALUATION_SEEDS = list(range(30000, 30050))
START_SEED = 10000
SUCCESS_CRITERION = "Can lifted 4 cm while grasped for 3 steps, then released and stable in its bin for 10 steps"


class CanMetrics:
    def __init__(self, env):
        self.raw = env.raw
        self.sim = self.raw.sim
        self.gripper = self.raw.robots[0].gripper
        self.can = next(obj for obj in self.raw.objects if obj.name == "Can")
        self.body = self.raw.obj_body_id[self.can.name]
        self.initial_position = self.sim.data.body_xpos[self.body].copy()
        self.lift_streak = self.place_streak = 0
        self.first_lift = self.first_place = None

    def update(self, step):
        pos = self.sim.data.body_xpos[self.body]
        held = bool(self.raw._check_grasp(self.gripper, self.can))
        lifted = bool(held and pos[2] - self.initial_position[2] >= .04)
        self.lift_streak = self.lift_streak + 1 if lifted else 0
        if self.first_lift is None and self.lift_streak >= 3:
            self.first_lift = step
        inside = not self.raw.not_in_bin(pos, self.raw.object_to_id["can"])
        speed = np.linalg.norm(self.sim.data.get_body_xvelp(self.can.root_body))
        placed = self.first_lift is not None and inside and not held and speed < .02
        self.place_streak = self.place_streak + 1 if placed else 0
        if self.first_place is None and self.place_streak >= 10:
            self.first_place = step


def layout(env, seed):
    state = env.raw.sim.get_state().flatten().copy()
    can = CanMetrics(env).initial_position
    return {"seed": seed, "initial_state_hash": env.initial_state_hash,
            "can_start_m": can.tolist(), "bin_target_m": env.target_position("can").tolist()}, state


def can_position_key(row):
    return tuple(round(axis, 3) for axis in row["can_start_m"][:2])


def evaluation_layout(output):
    rows = []
    for seed in EVALUATION_SEEDS:
        env = RobosuiteMultiRobotEnv({**SCENE, "cameras": False})
        try:
            env.reset(seed)
            rows.append(layout(env, seed)[0])
        finally:
            env.close()
    assert len({row["initial_state_hash"] for row in rows}) == 50
    path = output / "evaluation_layouts.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != rows:
            write_json(output / "evaluation_layouts_initial_pilot.json", previous)
    write_json(path, rows)
    return {can_position_key(row) for row in rows}


def generate(seed):
    env = RobosuiteMultiRobotEnv({**SCENE, "cameras": False})
    try:
        env.reset(seed)
        row, state = layout(env, seed)
    finally:
        env.close()
    actions = []
    result = center_alignment.simulate_grasp(state, seed, row["initial_state_hash"],
                                             place=True, on_action=actions.append)
    row.update(result)
    return row, np.asarray(actions, dtype=np.float64)


def replay(row, actions, recorder=None):
    env = RobosuiteMultiRobotEnv({**SCENE, "cameras": recorder is not None})
    try:
        env.reset(row["seed"])
        assert env.initial_state_hash == row["initial_state_hash"], (row["seed"], "reset")
        metadata = env.metadata() if recorder is not None else None
        metrics = CanMetrics(env)
        for step, action in enumerate(actions, 1):
            if recorder is not None:
                recorder.append(env.observe(), action, INSTRUCTION, env.simulation_time())
            env.step(action)
            metrics.update(step)
        report = {"lift_step": metrics.first_lift, "place_step": metrics.first_place,
                  "final_stable_steps": metrics.place_streak,
                  "can_final_m": env.object_position("can").tolist()}
        return report, metadata, env.raw.control_freq
    finally:
        env.close()


def record(task):
    output, row, actions = task
    path = output / "dataset" / f"seed-{row['seed']}.npz"
    receipt = path.with_suffix(".record.json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved["initial_state_hash"] == row["initial_state_hash"] and (output / saved["file"]).exists()
        return saved
    decoded = canonical_to_robosuite_osc(robosuite_osc_to_canonical(actions))
    np.testing.assert_allclose(decoded, actions, atol=1e-7, rtol=0)
    converted, _, fps = replay(row, decoded)
    if converted["final_stable_steps"] < 10:
        return {"seed": row["seed"], "converted_replay": converted, "success": False,
                "reason": "converted_replay_did_not_place"}
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["action"], actions)
        result, _, fps = replay(row, actions)
    else:
        recorder = EpisodeRecorder()
        result, metadata, fps = replay(row, actions, recorder)
        if result["final_stable_steps"] >= 10:
            recorder.save(output, row | {"success": True}, {"environment": SCENE,
                          "policy": {"backend": "center_alignment"}}, metadata)
    if result["final_stable_steps"] < 10:
        return {"seed": row["seed"], "raw_replay": result, "success": False,
                "reason": "raw_replay_did_not_place"}
    row.update(success=True, steps=len(actions), control_hz=fps, replay={"raw": result, "converted": converted},
               success_criterion=SUCCESS_CRITERION)
    row["file"] = str(path.relative_to(output))
    row["task_id"] = "panda_can_to_bin_in_pick_place_scene"
    row["initial_layout_id"] = f"sim-seed-{row['seed']}"
    row["source"] = "xenoarm_center_alignment"
    meta_file = path.with_suffix(".json")
    meta = json.loads(meta_file.read_text())
    meta.update({key: row[key] for key in ("task_id", "success_criterion", "source", "control_hz")})
    write_json(meta_file, meta)
    write_json(receipt, row)
    return row


def collect(output, target, workers):
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset").mkdir(exist_ok=True)
    manifest = output / "manifest.json"
    state = json.loads(manifest.read_text()) if manifest.exists() else {"episodes": [], "attempts": []}
    accepted = state["episodes"]
    if target <= 100:
        reserved = evaluation_layout(output)
    else:
        reserved = {can_position_key(row) for row in json.loads((output / "evaluation_layouts.json").read_text())}
    assert not (reserved & {can_position_key(row) for row in accepted}), "Training and evaluation can positions overlap"
    used = {can_position_key(row) for row in accepted} | reserved
    seed = START_SEED + len(state["attempts"])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while len(accepted) < target:
            count = min(workers, target - len(accepted))
            candidates = list(pool.map(generate, range(seed, seed + count)))
            requests = []
            attempt_by_seed = {}
            for row, actions in candidates:
                key = can_position_key(row)
                eligible = row["can_placed_and_stable"] and key not in used
                if eligible:
                    used.add(key)
                    requests.append((output, row, actions))
                attempt_by_seed[row["seed"]] = {"seed": row["seed"], "can_start_m": row["can_start_m"],
                                                "script_success": row["can_placed_and_stable"],
                                                "unique_training_position": key not in reserved and (
                                                    eligible or key not in used), "accepted": False}
            for row in pool.map(record, requests):
                if row["success"]:
                    accepted.append(row)
                attempt = attempt_by_seed[row["seed"]]
                attempt["accepted"] = row["success"]
                if not row["success"]:
                    attempt["rejection"] = row
                print(json.dumps({"stage": "record", "seed": row["seed"], "success": row["success"],
                                  "count": len(accepted), "target": target, "reason": row.get("reason")}), flush=True)
            seed += count
            state["attempts"].extend(attempt_by_seed.values())
            write_json(manifest, state)
            if seed > START_SEED + 2000:
                raise RuntimeError("Collection did not reach its requested size in 2,000 attempts")
    print(json.dumps({"stage": "collection_complete", "episodes": len(accepted),
                      "attempts": len(state["attempts"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    collect(args.output, args.target, args.workers)
