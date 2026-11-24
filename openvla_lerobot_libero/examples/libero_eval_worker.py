"""Official-state LIBERO worker with reference-matched TensorFlow CPU preprocessing."""
import argparse
import contextlib
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys

PROTOCOL_OUT = sys.stdout


def emit(payload):
    PROTOCOL_OUT.write(json.dumps(payload, allow_nan=False)+"\n")
    PROTOCOL_OUT.flush()


def main():
    import numpy as np
    from PIL import Image
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openvla_eval_preprocessing import prepare_image, check_against_reference, self_test, reference_hashes

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-index", type=int, default=0)
    args = parser.parse_args()
    frames = args.output/"frames"
    policy_frames = args.output/"policy_frames"
    frames.mkdir(parents=True)
    policy_frames.mkdir(parents=True)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    preprocessing_tests = self_test()
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.init_state_index < len(states):
        raise ValueError("Initial-state index outside official state list.")
    state = states[args.init_state_index]
    np.save(args.output/"selected_init_state.npy", np.asarray(state))
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(args.task_id), camera_heights=256, camera_widths=256,
    )

    def capture(obs, step, done=False, reward=0.0):
        raw = obs["agentview_image"]
        frame = frames/f"{step:04d}.png"
        policy_frame = policy_frames/f"{step:04d}.png"
        Image.fromarray(np.ascontiguousarray(raw[::-1,::-1])).save(frame)
        Image.fromarray(prepare_image(raw)).save(policy_frame)
        success = bool(env.check_success())
        if bool(done) != success:
            raise RuntimeError("Environment done and task-success predicate disagree.")
        return {"step":step, "frame":str(frame), "policy_frame":str(policy_frame),
                "eef_position":np.asarray(obs["robot0_eef_pos"],dtype=float).tolist(),
                "joint_positions":np.asarray(obs["robot0_joint_pos"],dtype=float).tolist(),
                "sim_time":float(env.env.sim.data.time), "done_signal":bool(done),
                "task_success":success, "reward_signal":float(reward)}

    try:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(state)
        for _ in range(10):
            obs, reward, done, _ = env.step([0,0,0,0,0,0,-1])
        first_image_check = check_against_reference(obs["agentview_image"])
        ready = capture(obs,0,done,reward)
        if ready["task_success"]:
            raise RuntimeError("Selected state is already successful before model actions.")
        low, high = env.env.action_spec
        state_file = Path(get_libero_path("init_states"))/task.problem_folder/task.init_states_file
        ready.update({"type":"ready", "task":task.language, "task_id":args.task_id,
                      "init_state_index":args.init_state_index, "available_init_states":len(states),
                      "init_state_file":str(state_file),
                      "init_state_file_sha256":hashlib.sha256(state_file.read_bytes()).hexdigest(),
                      "selected_init_state_sha256":hashlib.sha256((args.output/"selected_init_state.npy").read_bytes()).hexdigest(),
                      "action_low":np.asarray(low).tolist(), "action_high":np.asarray(high).tolist(),
                      "control_frequency":float(env.env.control_freq), "warmup_steps":10,
                      "environment_seed":0, "global_seed":7,
                      "reference_preprocessing_tests":preprocessing_tests,
                      "actual_initial_image_reference_check":first_image_check,
                      "reference_sha256":reference_hashes(),
                      "simulator_versions":{n:version(n) for n in ["mujoco","robosuite","numpy","tensorflow-cpu"]}})
        emit(ready)
        step = 0
        for line in sys.stdin:
            request = json.loads(line)
            if request["type"] == "close":
                break
            if request["type"] != "step":
                raise ValueError("Unknown worker command.")
            action = np.asarray(request["action"],dtype=np.float64)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Invalid action.")
            if np.any(action < np.asarray(low)-1e-7) or np.any(action > np.asarray(high)+1e-7):
                raise ValueError("Action outside controller bounds.")
            obs,reward,done,_ = env.step(action.tolist())
            step += 1
            response = capture(obs,step,done,reward)
            response["type"] = "observation"
            emit(response)
    finally:
        env.close()


if __name__ == "__main__":
    with contextlib.redirect_stdout(sys.stderr):
        main()
