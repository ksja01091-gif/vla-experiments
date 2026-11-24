"""Run the selected 100-episode PPO experiment and save each phase outcome."""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

R = Path(__file__).resolve().parent
PY = '/home/jovyan/projects/venvs/rlrc-smoke312/bin/python'
BRANCH = json.loads((R / 'selected_branch.json').read_text())['branch']
assert BRANCH == 'p20'
CFG = json.loads((R / 'experiment_config.json').read_text())
assert CFG['rl_episodes_each'] == len(CFG['rl_training_states']) == 100


def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def phase(name, command):
    write(R / 'stage2_status.json', {'status': 'running', 'phase': name, 'time': time.time()})
    started = time.perf_counter()
    with (R / f'stage2_{name}.log').open('w') as log:
        result = subprocess.run([PY, '-u', *command], cwd=R, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed with exit {result.returncode}; inspect stage2_{name}.log')
    write(R / f'stage2_{name}_done.json', {'complete': True,
        'seconds': time.perf_counter() - started})


try:
    phase('collect', ['previous_worker.py', 'collect', '--branch', BRANCH])
    phase('budget', ['stage2_budget.py', BRANCH])
    phase('ppo', ['previous_worker.py', 'ppo', '--branch', BRANCH])
    phase('evaluate', ['previous_worker.py', 'evaluate', '--branch', BRANCH])
    phase('components', ['components.py', BRANCH, 'after_ppo'])
    write(R / 'stage2_status.json', {'status': 'complete', 'branch': BRANCH,
        'episodes': 100, 'time': time.time()})
except Exception as error:
    write(R / 'stage2_status.json', {'status': 'error', 'error': repr(error),
        'traceback': traceback.format_exc(), 'time': time.time()})
    raise
