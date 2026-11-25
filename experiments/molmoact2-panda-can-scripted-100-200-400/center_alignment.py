"""Paired Panda can-pick diagnosis using the original ACT starting states."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from xenoarm.envs.robosuite_multi import RobosuiteMultiRobotEnv

DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
ROOT = Path("center_alignment_check")


def simulate_grasp(state, seed, initial_hash, record=False, place=False, on_action=None):
    env = RobosuiteMultiRobotEnv(dict(robot="panda", task="pick_place", cameras=record))
    try:
        env.reset(seed)
        raw = env.raw
        sim = raw.sim
        sim.set_state_from_flattened(state.copy())
        sim.forward()
        actual_hash = hashlib.sha256(sim.get_state().flatten().tobytes()).hexdigest()
        assert actual_hash == initial_hash, seed
        gripper = raw.robots[0].gripper
        site = gripper.important_sites["grip_site"]
        pads = gripper.important_geoms["left_fingerpad"] + gripper.important_geoms["right_fingerpad"]
        can = next(obj for obj in raw.objects if obj.name == "Can")
        body = raw.obj_body_id[can.name]
        initial_can = sim.data.body_xpos[body].copy()
        snapshots = []
        streak = 0
        confirmed = False
        max_lift = 0.0
        max_grasp_lift = 0.0
        step_count = 0
        placing_started = False
        phases = {}
        released = False
        placement_streak = 0
        clear_placement_streak = 0
        reached_bin_while_held = False

        def center():
            return np.mean([sim.data.get_geom_xpos(p) for p in pads], axis=0)

        def frame():
            if record:
                views = env.render()
                snapshots.append(np.concatenate(list(views.values()), axis=1))

        def mark(stage, target):
            phases[stage] = {
                "pad_center": center().tolist(),
                "target": target.tolist(),
                "can_center": sim.data.body_xpos[body].tolist(),
                "gripper_open": float(env.observe()["xenoarm_gripper_open"]),
                "holding": bool(raw._check_grasp(gripper, can)),
                "inside_bin": not raw.not_in_bin(sim.data.body_xpos[body], raw.object_to_id["can"]),
            }

        def move(target, close, count):
            nonlocal step_count, streak, confirmed, max_lift, max_grasp_lift
            nonlocal placement_streak, reached_bin_while_held
            nonlocal clear_placement_streak
            for _ in range(count):
                pos_error = target - center()
                rot_error = Rotation.from_matrix(DOWN @ sim.data.get_site_xmat(site).T).as_rotvec()
                osc = np.r_[np.clip(pos_error / .05, -.45, .45),
                            np.clip(rot_error / .5, -.20, .20), close]
                if on_action is not None:
                    on_action(osc)
                env.step(osc)
                step_count += 1
                held = bool(raw._check_grasp(gripper, can))
                height = float(sim.data.body_xpos[body, 2] - initial_can[2])
                max_lift = max(max_lift, height)
                if held and not placing_started:
                    max_grasp_lift = max(max_grasp_lift, height)
                if not placing_started:
                    streak = streak + 1 if held and height >= .04 else 0
                    confirmed |= streak >= 3
                else:
                    inside = not raw.not_in_bin(sim.data.body_xpos[body], raw.object_to_id["can"])
                    reached_bin_while_held |= inside and held
                    if released:
                        speed = np.linalg.norm(sim.data.get_body_xvelp(can.root_body))
                        clear = np.linalg.norm(center() - sim.data.body_xpos[body]) > .06
                        stable = inside and not held and speed < .02
                        placement_streak = placement_streak + 1 if stable else 0
                        clear_placement_streak = clear_placement_streak + 1 if stable and clear else 0
                if step_count % 4 == 0:
                    frame()

        frame()
        above = initial_can + [0, 0, .16]
        move(above, -1, 115)
        move(initial_can, -1, 115)
        error = float(np.linalg.norm(center() - sim.data.body_xpos[body]))
        move(initial_can, 1, 45)
        held_on_close = bool(raw._check_grasp(gripper, can))
        move(above, 1, 100)
        held_after_lift = bool(raw._check_grasp(gripper, can))
        lift_max = max_lift
        if place and confirmed and held_after_lift:
            placing_started = True
            target = env.target_position("can")
            pad_offset = center() - sim.data.body_xpos[body]
            clearance_z = max(initial_can[2], sim.data.body_xpos[body, 2]) + .16
            raised = center().copy()
            raised[2] = clearance_z + pad_offset[2]
            over_bin = np.r_[target[:2], clearance_z] + pad_offset
            release_z = raw.bin2_pos[2] + (initial_can[2] - raw.bin1_pos[2]) + .02
            release_pose = np.r_[target[:2], release_z] + pad_offset
            move(raised, 1, 90)
            mark("raised", raised)
            move(over_bin, 1, 160)
            mark("over_bin", over_bin)
            move(release_pose, 1, 145)
            mark("lowered", release_pose)
            released = True
            move(release_pose, -1, 45)
            mark("opened", release_pose)
            move(over_bin, -1, 100)
            mark("retracted", over_bin)
        if snapshots:
            import imageio.v2 as imageio

            suffix = "pick-place" if place else "centered"
            imageio.mimsave(ROOT / f"seed-{seed}-{suffix}.mp4", snapshots, fps=10, macro_block_size=1)
        return {
            "seed": seed,
            "initial_state_hash": actual_hash,
            "distance_before_closing_m": error,
            "grasp_after_close": held_on_close,
            "can_grasped_and_lifted": bool(confirmed),
            "maximum_can_lift_m": lift_max,
            "maximum_grasped_can_lift_m": max_grasp_lift,
            "grasp_held_after_lift": held_after_lift,
            "reached_bin_while_held": bool(reached_bin_while_held),
            "can_placed_and_stable": bool(placement_streak >= 10),
            "final_stable_steps": placement_streak,
            "can_placed_and_gripper_clear": bool(clear_placement_streak >= 10),
            "placement_phases": phases,
        }
    finally:
        env.close()


def main():
    ROOT.mkdir(exist_ok=True)
    data = json.loads(Path("diagnosis/replay_diagnosis.json").read_text())
    outcomes = {
        row["seed"]: row["objects"]["can"]["first_grasp_and_lift"] is not None
        for row in data["trials"]
    }
    original = Path("evaluation/parts/run-00/episodes.jsonl")
    limit = int(os.environ.get("CENTER_TRIALS", "10"))
    env = RobosuiteMultiRobotEnv(dict(robot="panda", task="pick_place", cameras=False))
    rows = []
    try:
        with original.open() as source:
            for line in source:
                row = json.loads(line)
                seed = row["seed"]
                if seed >= limit:
                    break
                env.reset(seed)
                assert env.initial_state_hash == row["initial_state_hash"], seed
                state = env.raw.sim.get_state().flatten().copy()
                result = simulate_grasp(state, seed, row["initial_state_hash"], record=seed == 1)
                result["original_act_can_grasped_and_lifted"] = outcomes[seed]
                rows.append(result)
                print(json.dumps(result), flush=True)
                if seed + 1 < limit:
                    for action in row["trace"]:
                        env.step(action["action"])
                    assert env.final_metrics() == row["task_metrics"], seed
    finally:
        env.close()
    assert [row["seed"] for row in rows] == list(range(limit))
    report = {
        "task": "Panda pp4 can-only grasp diagnostic",
        "target": "Can",
        "lift_height_m": .04,
        "lift_hold_steps": 3,
        "act_successes": sum(r["original_act_can_grasped_and_lifted"] for r in rows),
        "aligned_successes": sum(r["can_grasped_and_lifted"] for r in rows),
        "trials": rows,
    }
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}))


if __name__ == "__main__":
    main()
