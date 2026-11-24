"""Correct phase-2 config to 100 episodes before any collection has started."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent
assert not (R / 'rl_collection').exists()
assert not (R / 'p20' / 'collection_progress.json').exists()
path = R / 'experiment_config.json'
cfg = json.loads(path.read_text())
assert cfg['stage2_selected_branch'] == 'p20' and cfg['rl_episodes_each'] == 20
cfg.update({'run_description': '20/35 structural pruning then 100-episode PPO',
    'rl_training_states': list(range(5, 45)) * 2 + list(range(5, 25)),
    'rl_episodes_each': 100,
    'repeated_training_states_use_distinct_rollout_rng_seeds': True})
path.write_text(json.dumps(cfg, indent=2, allow_nan=False))
print('READY_100_EPISODES', len(cfg['rl_training_states']), flush=True)
