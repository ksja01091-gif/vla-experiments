"""Evaluate selected compressed model on previously unused simulator states 45-49."""
import argparse
import json
import types
from pathlib import Path

from policy import *

R = Path(__file__).resolve().parent
ROLLOUT = Path('/home/jovyan/projects/lerobot-openvla/outputs/input_pipeline_check/20261103T055533_073692Z/matched_rollout_function.py')


def put(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pre', 'post'])
    args = parser.parse_args()
    stage = 'epoch3' if args.stage == 'pre' else 'after_ppo'
    tag = 'p20_heldout_' + args.stage
    _, val, stats = exp.get_data()
    policy = load_checkpoint(R / 'checkpoints' / 'p20' / stage, stats)
    policy.eval()
    actor = policy.actor_hash()
    action = common.sample_action(policy, val, 0).cpu()
    assert action.shape == (1, 7) and bool(torch.isfinite(action).all())
    namespace = dict(vars(exp))
    exec(compile(ROLLOUT.read_text(), str(ROLLOUT), 'exec'), namespace)
    trials = []
    for state in range(45, 50):
        trial = namespace['simulate_trial'](policy, tag, state, types.SimpleNamespace(run=R))
        if args.stage == 'post':
            prior = json.loads((R / 'simulation' / 'p20_heldout_pre' /
                                f'state_{state:02d}' / 'result.json').read_text())
            for key in ('initial_image_sha256', 'selected_init_state_sha256',
                        'init_state_file_sha256', 'reference_sha256', 'simulator_versions',
                        'max_policy_steps', 'warmup_steps'):
                assert trial[key] == prior[key], key
        trials.append(trial)
        put(R / f'heldout_{args.stage}_progress.json', {'completed': len(trials),
            'successes': sum(x['success'] for x in trials)})
        print('HELDOUT', args.stage, state, trial['success'], flush=True)
    assert actor == policy.actor_hash()
    put(R / f'heldout_{args.stage}_result.json', {'complete': True,
        'stage': args.stage, 'tag': tag, 'states': list(range(45, 50)),
        'attempts': 5, 'successes': sum(x['success'] for x in trials),
        'trials': trials, 'checkpoint': str(R / 'checkpoints' / 'p20' / stage),
        'scope': 'New evaluation states for this run; independence from demonstration training states not audited.'})


if __name__ == '__main__':
    main()
