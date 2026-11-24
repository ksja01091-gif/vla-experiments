"""Evaluate translation-distilled checkpoint on old and new simulator states."""
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
    _, validation, stats = exp.get_data()
    cp = R / 'checkpoints' / 'p20' / 'after_translation_distill'
    policy = load_checkpoint(cp, stats)
    policy.eval()
    actor = policy.actor_hash()
    action = common.sample_action(policy, validation, 0).cpu()
    assert action.shape == (1, 7) and bool(torch.isfinite(action).all())
    namespace = dict(vars(exp))
    exec(compile(ROLLOUT.read_text(), str(ROLLOUT), 'exec'), namespace)
    for group, states, prior_tag in [
        ('old', range(5), 'p20_e3'),
        ('new', range(45, 50), 'p20_heldout_pre'),
    ]:
        trials = []
        for state in states:
            trial = namespace['simulate_trial'](policy, 'p20_distill_' + group,
                                                 state, types.SimpleNamespace(run=R))
            prior = json.loads((R / 'simulation' / prior_tag /
                                f'state_{state:02d}' / 'result.json').read_text())
            for key in ('initial_image_sha256', 'selected_init_state_sha256',
                        'init_state_file_sha256', 'reference_sha256', 'simulator_versions',
                        'max_policy_steps', 'warmup_steps'):
                assert trial[key] == prior[key], key
            trials.append(trial)
            put(R / f'translation_eval_{group}_progress.json', {'completed': len(trials),
                'successes': sum(x['success'] for x in trials)})
            print('TRANSLATION_EVAL', group, state, trial['success'], flush=True)
        put(R / f'translation_eval_{group}_result.json', {'complete': True,
            'group': group, 'attempts': 5, 'successes': sum(x['success'] for x in trials),
            'trials': trials, 'checkpoint': str(cp),
            'scope': 'Five fixed states; compare to same-state pre and PPO results.'})
    assert actor == policy.actor_hash()


if __name__ == '__main__':
    main()
