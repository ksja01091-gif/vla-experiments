"""Replay finished rollouts to distinguish grasp/lift from task completion."""
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

R = Path(__file__).resolve().parent
PY = '/home/jovyan/projects/venvs/openvla_eval310/bin/python'
SOURCE = R / 'diagnostic_before'


def put(name, obj):
    path = R / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


try:
    while True:
        state = json.loads((R / 'stage3_status.json').read_text())
        if state['status'] == 'error':
            raise RuntimeError('stage3 failed: ' + state['error'])
        if state['status'] == 'complete':
            break
        put('physics_after_status.json', {'status': 'waiting', 'stage3': state})
        time.sleep(30)
    reports = {}
    for label, tag in [('after_ppo', 'p20_rl'),
                       ('after_translation_distill', 'p20_distill_old')]:
        folder = R / ('diagnostic_' + label)
        assert not folder.exists()
        folder.mkdir()
        shutil.copy2(SOURCE / 'replay.py', folder / 'replay.py')
        config = json.loads((SOURCE / 'diagnostic_config.json').read_text())
        config['tags'] = [tag]
        config['scope'] = 'Exact recorded-action replay for ' + label
        (folder / 'diagnostic_config.json').write_text(json.dumps(config, indent=2))
        env = os.environ.copy()
        env['PYTHONPATH'] = '/home/jovyan/projects/LIBERO'
        env['MUJOCO_GL'] = 'egl'
        put('physics_after_status.json', {'status': 'running', 'phase': label})
        with (R / ('physics_' + label + '.log')).open('w') as log:
            result = subprocess.run([PY, '-u', 'replay.py'], cwd=folder,
                                    stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(label + ' replay failed; inspect log')
        data = json.loads((folder / 'replay_results.json').read_text())
        assert data['complete'] and data['all_replays_match_original_trajectories']
        rows = data['rows']
        reports[label] = {
            'tag': tag, 'attempts': len(rows),
            'stable_grasp_and_lift': sum(bool(row['stable_grasp_and_lift_observed']) for row in rows),
            'official_success': sum(row['first_official_success'] is not None for row in rows),
            'by_state': [{'state': row['state'],
                          'stable_grasp_and_lift': row['stable_grasp_and_lift_observed'],
                          'official_success': row['first_official_success'] is not None,
                          'max_lift_m': row['max_target_lift_m']} for row in rows],
        }
        put('physics_after_progress.json', reports)
    before = json.loads((SOURCE / 'replay_results.json').read_text())['rows']
    put('physics_replay_result.json', {'complete': True, 'no_training': True,
        'before_epoch3': {
            'stable_grasp_and_lift': sum(bool(row['stable_grasp_and_lift_observed']) for row in before),
            'official_success': sum(row['first_official_success'] is not None for row in before)},
        'after': reports,
        'note': 'Same five repeatedly inspected states; grasp+lift is separately measured from official task success.'})
    put('physics_after_status.json', {'status': 'complete', 'time': time.time()})
except Exception as error:
    put('physics_after_status.json', {'status': 'error', 'error': repr(error),
                                     'traceback': traceback.format_exc()})
    raise
