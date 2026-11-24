"""Wait for PPO, then run matched held-out and teacher-guided recovery checks."""
import json
import subprocess
import time
import traceback
from pathlib import Path

R = Path(__file__).resolve().parent
PY = '/home/jovyan/projects/venvs/rlrc-smoke312/bin/python'
STEPS = [
    ('heldout_post', ['heldout.py', 'post'], R / 'heldout_post_result.json'),
    ('translation_distill', ['translation_distill.py'], R / 'translation_distill_result.json'),
    ('translation_eval', ['translation_eval.py'], R / 'translation_eval_new_result.json'),
]


def write(name, obj):
    path = R / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


try:
    while True:
        state = json.loads((R / 'stage2_status.json').read_text())
        if state['status'] == 'error':
            raise RuntimeError('PPO stage failed: ' + state['error'])
        if state['status'] == 'complete':
            break
        write('stage3_status.json', {'status': 'waiting_for_ppo', 'phase': state['phase']})
        time.sleep(30)
    while not (R / 'teacher_cache_v2' / 'report.json').exists():
        write('stage3_status.json', {'status': 'waiting_for_teacher_cache'})
        time.sleep(30)
    for label, command, expected in STEPS:
        assert not expected.exists(), f'{label} result already exists'
        write('stage3_status.json', {'status': 'running', 'phase': label})
        with (R / f'stage3_{label}.log').open('w') as log:
            result = subprocess.run([PY, '-u', *command], cwd=R,
                                    stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT)
        if result.returncode or not expected.exists():
            raise RuntimeError(f'{label} failed: exit {result.returncode}; see log')
        write(f'stage3_{label}_done.json', {'complete': True, 'time': time.time()})
    write('stage3_status.json', {'status': 'complete', 'time': time.time()})
except Exception as error:
    write('stage3_status.json', {'status': 'error', 'error': repr(error),
                                'traceback': traceback.format_exc()})
    raise
