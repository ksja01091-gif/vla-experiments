"""Evaluate the loaded Colab pi0.5 UR5e policy at 40 seen and 50 new block locations.

Run in the same Colab notebook session that defines policy, preprocess,
postprocess, model, sim, renderer, camera, block_addr, and run.
The score is a 5 cm block lift, not full task completion.
"""

import csv
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image


def make_positions():
    seen = [(round(float(x), 4), round(float(y), 4))
            for x in np.linspace(0.405, 0.495, 8)
            for y in np.linspace(-0.04, 0.04, 5)]
    validation = [(0.4114, -0.0333), (0.4243, 0.0133),
                  (0.45, -0.0133), (0.4757, 0.0333), (0.4886, -0.0067)]
    rng = np.random.default_rng(20261122)
    unseen = []
    for _ in range(100000):
        if len(unseen) == 50:
            break
        candidate = (round(float(rng.uniform(0.405, 0.495)), 4),
                     round(float(rng.uniform(-0.04, 0.04)), 4))
        if min(np.linalg.norm(np.subtract(candidate, p))
               for p in seen + validation) < 0.004:
            continue
        if unseen and min(np.linalg.norm(np.subtract(candidate, p))
                          for p in unseen) < 0.002:
            continue
        unseen.append(candidate)
    assert len(seen) == 40 and len(unseen) == 50
    assert len(set(seen + unseen + validation)) == 95
    return seen, unseen, validation


def evaluate_one(group, index, x, y, output):
    policy.reset()
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    mujoco.mj_resetDataKeyframe(model, sim, 0)
    sim.qpos[block_addr:block_addr + 7] = [x, y, 0.05, 1, 0, 0, 0]
    mujoco.mj_forward(model, sim)

    finger_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper/right_driver_joint')
    finger_addr = int(model.jnt_qposadr[finger_joint])
    finger_actuator = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper/fingers_actuator')
    pinch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripper/pinch')
    initial_z = float(sim.qpos[block_addr + 2])
    peak_z = initial_z
    rows = []
    frames = []
    started = time.perf_counter()
    for step in range(90):
        renderer.update_scene(sim, camera=camera)
        rgb = renderer.render().copy()
        frames.append(rgb)
        state = np.array([*sim.qpos[:6], sim.qpos[finger_addr]], dtype=np.float32)
        obs = {
            'observation.images.image': torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
            'observation.state': torch.from_numpy(state),
            'task': 'pick up the orange block',
        }
        with torch.inference_mode():
            action = postprocess(policy.select_action(preprocess(obs)))
            action = action.detach().cpu().float().reshape(-1).numpy()
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError(f'{group} {index} step {step}: invalid action {action}')
        ctrl = np.clip(action[:6], model.actuator_ctrlrange[:6, 0],
                       model.actuator_ctrlrange[:6, 1])
        grip_cmd = 0 if action[6] >= 0 else 255
        sim.ctrl[:6] = ctrl
        sim.ctrl[finger_actuator] = grip_cmd
        for _ in range(50):
            mujoco.mj_step(model, sim)
            peak_z = max(peak_z, float(sim.qpos[block_addr + 2]))
        z = float(sim.qpos[block_addr + 2])
        dist = float(np.linalg.norm(sim.site_xpos[pinch] -
                                    sim.qpos[block_addr:block_addr + 3]))
        rows.append({
            'step': step + 1, 'time_s': round(float(sim.time), 3),
            'block_z_m': z, 'pinch_block_distance_m': dist,
            'gripper_action': float(action[6]), 'gripper_target': grip_cmd,
            **{f'action_{i + 1}': float(action[i]) for i in range(7)},
        })
    renderer.update_scene(sim, camera=camera)
    frames.append(renderer.render().copy())
    trial_dir = output / group / f'{index:02d}'
    trial_dir.mkdir(parents=True, exist_ok=True)
    with (trial_dir / 'actions.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Image.fromarray(frames[0]).save(trial_dir / 'start.png')
    Image.fromarray(frames[-1]).save(trial_dir / 'final.png')
    imageio.mimsave(trial_dir / 'rollout.mp4', frames, fps=10)
    result = {
        'group': group, 'index': index, 'x_m': x, 'y_m': y,
        'steps': len(rows), 'wall_seconds': time.perf_counter() - started,
        'initial_block_z_m': initial_z, 'peak_block_z_m': peak_z,
        'final_block_z_m': rows[-1]['block_z_m'],
        'lift_5cm': bool(peak_z - initial_z >= 0.05),
        'final_lift_5cm': bool(rows[-1]['block_z_m'] - initial_z >= 0.05),
        'min_pinch_block_distance_m': min(r['pinch_block_distance_m'] for r in rows),
        'gripper_close_steps': sum(r['gripper_target'] == 255 for r in rows),
        'first_close_step': next((r['step'] for r in rows
                                  if r['gripper_target'] == 255), None),
    }
    (trial_dir / 'summary.json').write_text(json.dumps(result, indent=2))
    return result


def run_evaluation(max_episodes=90):
    seen, unseen, validation = make_positions()
    output = run / 'eval_seen40_unseen50_20261122'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'training_positions': seen, 'new_positions': unseen,
        'excluded_validation_positions': validation,
        'training_count': 40, 'new_count': 50,
        'new_position_seed': 20261122, 'policy_seed_per_episode': 7,
        'horizon_actions': 90, 'physics_steps_per_action': 50,
        'score': 'block lifted at least 0.05 m at any point; not full task success',
        'checkpoint': 'UR5e_Pi05_40demo_1epoch_seed7/pretrained_model',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    jobs = [('seen', i, *p) for i, p in enumerate(seen)] + [
        ('new', i, *p) for i, p in enumerate(unseen)]
    for group, index, x, y in jobs[:max_episodes]:
        path = output / group / f'{index:02d}' / 'summary.json'
        if path.exists():
            print('ALREADY_DONE', group, index, flush=True)
            continue
        try:
            result = evaluate_one(group, index, x, y, output)
            print('DONE', group, index, 'lift', result['lift_5cm'],
                  'min_distance', round(result['min_pinch_block_distance_m'], 3),
                  'seconds', round(result['wall_seconds'], 1), flush=True)
        except Exception as exc:
            error = {'group': group, 'index': index, 'x_m': x, 'y_m': y,
                     'error': repr(exc)}
            (output / 'last_error.json').write_text(json.dumps(error, indent=2))
            raise
    rows = []
    for group, index, _, _ in jobs:
        path = output / group / f'{index:02d}' / 'summary.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
    report = {}
    for group, expected in [('seen', 40), ('new', 50)]:
        group_rows = [r for r in rows if r['group'] == group]
        report[group] = {
            'completed': len(group_rows), 'planned': expected,
            'lift_5cm': sum(r['lift_5cm'] for r in group_rows),
            'final_lift_5cm': sum(r['final_lift_5cm'] for r in group_rows),
            'mean_min_distance_m': (sum(r['min_pinch_block_distance_m']
                                        for r in group_rows) / len(group_rows)
                                    if group_rows else None),
        }
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    return report
