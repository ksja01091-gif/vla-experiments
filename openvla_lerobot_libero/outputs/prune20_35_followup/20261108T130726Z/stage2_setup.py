"""Freeze the phase-2 design after phase-1 results are available."""
import argparse
import json
import time
from pathlib import Path

R = Path(__file__).resolve().parent
SOURCE = Path('/home/jovyan/projects/lerobot-openvla/outputs/prune50_recovery/20261108T101044Z')


def write(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('branch', choices=['p20', 'p35'])
    args = parser.parse_args()
    assert (R / args.branch / 'epoch3_simulation_result.json').exists()
    assert (R / 'p20' / 'epoch3_simulation_result.json').exists()
    assert (R / 'p35' / 'epoch3_simulation_result.json').exists()
    cfg = json.loads((SOURCE / 'experiment_config.json').read_text())
    cfg.update({
        'run_description': '20/35 structural-pruning comparison followed by 100-episode PPO on selected branch',
        'stage2_selected_branch': args.branch,
        'stage2_start_checkpoint': 'epoch3',
        'rl_training_states': list(range(5, 45)) * 2 + list(range(5, 25)),
        'evaluation_states': list(range(5)),
        'rl_episodes_each': 100,
        'max_episode_steps': 220,
        'runtime_cap_seconds': 12 * 3600,
        'phase2_budget_is_safety_cap': True,
        'not_exact_paper_reproduction': True,
        'prior_runs_read_only': True,
        'package_installation': False,
        'evaluation_scope': 'five repeatedly inspected states; cannot support a generalization claim',
    })
    write(R / 'experiment_config.json', cfg)
    now = time.time()
    write(R / 'budget.json', {'started_unix': now, 'deadline_unix': now + 12 * 3600,
        'cap_seconds': 12 * 3600, 'previous_runs_read_only': True})
    write(R / 'selected_branch.json', {'branch': args.branch,
        'selection': 'higher epoch3 five-state task success, then lower validation loss on same 462 frames',
        'warning': 'Selecting using repeatedly inspected states is exploratory, not held-out evidence'})


if __name__ == '__main__':
    main()
