"""Define PPO transition/update budget before PPO begins."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
branch = sys.argv[1]
assert branch in ('p20', 'p35')
source = json.loads((R / branch / 'collection_result.json').read_text())
assert source['complete'] and source['episodes'] == 100
n = source['transitions'] // 4 * 4
assert n >= 4
out = R / 'matched_ppo_budget.json'
assert not out.exists()
out.write_text(json.dumps({'transitions_each': n, 'source_counts': {branch: source['transitions']},
    'passes': 2, 'optimizer_updates_each': 2 * (n // 4),
    'selection': 'all 100 training episodes, dropping at most three final transitions'}, indent=2))
print('PPO_BUDGET', branch, source['episodes'], n, 2 * (n // 4), flush=True)
