"""Run two frozen-policy action swaps at the same prior LIBERO state."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

RUN = Path(__file__).resolve().parent
OUTPUTS = Path('/home/jovyan/projects/lerobot-openvla/outputs')
P50_RUN = OUTPUTS / 'prune50_sft1/20261106T122012_474442Z'
DENSE_RUN = OUTPUTS / 'rlrc_matched_3h/20261106T090120_404808Z'
DIAGNOSIS = OUTPUTS / 'prune50_diagnosis/20261106T132021_191937Z'
CHECKPOINTS = {
    'p50': P50_RUN / 'checkpoints/p50/after_sft',
    'dense_sft': DENSE_RUN / 'checkpoints/dense/after_sft',
}
SIM_PYTHON = '/home/jovyan/projects/venvs/openvla_eval310/bin/python'

sys.path.insert(0, str(P50_RUN))
from policy import common, exp, load_checkpoint, torch


def read(path):
    return json.loads(path.read_text())


def first_step(rows, field):
    return next((row['step'] for row in rows if row[field]), None)


def reference(model):
    folder = P50_RUN / 'simulation' / model / 'state_00'
    with (folder / 'steps.jsonl').open() as steps:
        first_action = json.loads(next(steps))
    return read(folder / 'initialization.json'), read(folder / 'result.json'), first_action


def verify_start(initial):
    initial_hash = hashlib.sha256(Path(initial['frame']).read_bytes()).hexdigest()
    assert initial['task'] == exp.TASK and initial['init_state_index'] == 0
    for model in CHECKPOINTS:
        previous, result, _ = reference(model)
        assert result['initial_image_sha256'] == initial_hash
        for key in ('selected_init_state_sha256', 'init_state_file_sha256', 'reference_sha256',
                    'simulator_versions', 'eef_position', 'joint_positions', 'sim_time',
                    'control_frequency', 'warmup_steps', 'environment_seed', 'global_seed',
                    'action_low', 'action_high'):
            assert initial[key] == previous[key], (model, key)
        measured = read(DIAGNOSIS / model / 'state_00/metadata.json')
        assert initial['bowl_xyz_m'] == measured['initial_target_pos']
        assert initial['plate_xyz_m'] == measured['initial_plate_pos']
    return initial_hash


def predict(policy, photo):
    with torch.inference_mode():
        result = policy.select_action({policy.config.image_key: photo, 'task': [exp.TASK]})
    assert result.shape == (1, 7) and bool(torch.isfinite(result).all())
    return result[0].cpu().numpy()


def rollout(tag, arm, gripper, policies):
    horizon, warmup = exp.reference_settings()
    exp.set_reference_seed()
    folder = RUN / tag
    folder.mkdir(exist_ok=False)
    environment = os.environ.copy()
    environment.update(MUJOCO_GL='egl', TF_CPP_MIN_LOG_LEVEL='2',
                       TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1')
    environment['PYTHONPATH'] = '/home/jovyan/projects/LIBERO' + (
        os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    started = time.perf_counter()
    with (folder / 'simulator.log').open('w') as log, (folder / 'steps.jsonl').open('w', buffering=1) as trace:
        process = subprocess.Popen(
            [SIM_PYTHON, '-u', str(RUN / 'simulator.py'), '--output', str(folder),
             '--task-id', '0', '--init-state-index', '0'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=environment,
        )
        try:
            initial = current = exp.read_observation(process, timeout=90)
            assert initial['type'] == 'ready' and not initial['task_success']
            assert initial['actual_initial_image_reference_check']['max_pixel_error'] == 0
            assert initial['warmup_steps'] == warmup
            assert initial['control_frequency'] == 20
            initial_hash = verify_start(initial)
            common.write(folder / 'initialization.json', initial)
            low, high = np.array(initial['action_low']), np.array(initial['action_high'])
            expected = {model: reference(model)[2]['raw_action'] for model in policies}
            for policy in policies.values():
                policy.reset()
            frames = [initial['frame']]
            rows = []
            streak = 0
            for step in range(1, horizon + 1):
                with Image.open(current['frame']) as image:
                    pixels = np.array(image.convert('RGB'))
                photo = torch.from_numpy(pixels).permute(2, 0, 1).float().div(255).unsqueeze(0)
                actions = {model: predict(policy, photo) for model, policy in policies.items()}
                if step == 1:
                    for model, action in actions.items():
                        assert np.array_equal(action, np.asarray(expected[model], dtype=action.dtype)), model
                raw = np.concatenate([actions[arm][:6], actions[gripper][6:]])
                converted, command = exp.controller_action(raw, low, high)
                assert np.array_equal(converted, command), 'Action would require clipping'
                exp.send(process, {'type': 'step', 'action': command.tolist()})
                current = exp.read_observation(process, timeout=45)
                assert current['type'] == 'observation' and current['step'] == step
                grasp_lift = current['both_fingerpads'] and current['bowl_rise_m'] >= .05
                streak = streak + 1 if grasp_lift else 0
                row = {**current, 'both_policies_raw_actions': {m: a.tolist() for m, a in actions.items()},
                       'controller_action': command.tolist(), 'grasp_lift_streak': streak}
                rows.append(row)
                trace.write(json.dumps(row, allow_nan=False) + '\n')
                frames.append(current['frame'])
                if step == 1 or step % 25 == 0 or streak == 3 or current['task_success']:
                    print('STEP', tag, step, 'stable_lift_streak', streak, 'task_success', current['task_success'], flush=True)
                if current['task_success']:
                    break
        finally:
            if process.poll() is None:
                try:
                    exp.send(process, {'type': 'close'})
                    process.wait(timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    success = current['task_success']
    exp.save_media(frames, folder, success)
    best_streak = max(row['grasp_lift_streak'] for row in rows)
    result = {
        'trial': tag, 'arm': arm, 'gripper': gripper, 'state': 0,
        'initial_image_sha256': initial_hash, 'both_initial_model_actions_match_previous': True,
        'first_both_pads_step': first_step(rows, 'both_fingerpads'),
        'max_bowl_rise_cm': 100 * max(0., *(row['bowl_rise_m'] for row in rows)),
        'max_consecutive_grasp_lift_steps': best_streak,
        'stable_grasp_and_lift': best_streak >= 3,
        'first_stable_grasp_lift_step': next((r['step'] - 2 for r in rows if r['grasp_lift_streak'] == 3), None),
        'libero_task_success': success, 'both_pads_at_final_step': current['both_fingerpads'],
        'final_bowl_plate_xy_cm': 100 * current['bowl_plate_xy_m'],
        'steps': len(rows), 'wall_seconds': time.perf_counter() - started,
    }
    common.write(folder / 'result.json', result)
    print('TRIAL_DONE', json.dumps(result), flush=True)
    return result


def main():
    started = time.perf_counter()
    _, _, stats = exp.get_data()
    policies = {}
    # The pruned loader temporarily builds a full model before applying its saved plan.
    for model, checkpoint in CHECKPOINTS.items():
        exp.seed()
        policy = common.prepare_inference(load_checkpoint(checkpoint, stats))
        assert not any(p.requires_grad for p in policy.parameters())
        policies[model] = policy
        print('LOADED', model, round(torch.cuda.memory_allocated() / 1024**3, 2), 'GiB total allocated', flush=True)
    before = {model: policy.actor_hash() for model, policy in policies.items()}
    results = [rollout('01_gripper_swap', 'p50', 'dense_sft', policies),
               rollout('02_arm_swap', 'dense_sft', 'p50', policies)]
    assert before == {model: policy.actor_hash() for model, policy in policies.items()}
    common.write(RUN / 'results.json', {
        'status': 'complete', 'no_training': True, 'actor_hashes_unchanged': before,
        'same_state_and_first_image_verified': True, 'same_observation_to_both_models_per_step': True,
        'checkpoints': {model: str(path) for model, path in CHECKPOINTS.items()},
        'criteria': {'both_fingerpads': True, 'rise_cm': 5, 'consecutive_control_steps': 3},
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'wall_seconds': time.perf_counter() - started, 'trials': results,
    })


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (RUN / 'error.txt').write_text(traceback.format_exc())
        raise
