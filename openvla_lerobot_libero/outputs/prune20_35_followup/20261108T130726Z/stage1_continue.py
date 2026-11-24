"""Continue the matched 20/35 percent SFT runs through epoch 3."""
import argparse
import json
import math
import time
import types
from pathlib import Path

from policy import *

R = Path(__file__).resolve().parent
BASE = Path('/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z/simulation/epoch5')
ROLLOUT = Path('/home/jovyan/projects/lerobot-openvla/outputs/input_pipeline_check/20261103T055533_073692Z/matched_rollout_function.py')


def put(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def train(percent):
    tag = f'p{percent}'
    train_data, val_data, stats = exp.get_data()
    exp.seed()
    source = R / 'checkpoints' / tag / 'after_sft'
    assert source.exists()
    policy = load_checkpoint(source, stats)
    frozen = exp.frozen_sample_hash(policy)
    prior_hash = policy.actor_hash()
    old = torch.load(source / 'trainer_state.pt', map_location='cpu', weights_only=False)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=0)
    optimizer.load_state_dict(old['optimizer'])
    for epoch in (2, 3):
        order = torch.randperm(4025, generator=torch.Generator().manual_seed(700 + epoch)).tolist()
        totals = {k: 0.0 for k in exp.METRICS}
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        for offset in range(0, len(order), 4):
            indices = order[offset:offset + 4]
            metrics = common.step(policy, optimizer, train_data, indices)
            for key in totals:
                totals[key] += metrics[key] * len(indices)
            if offset == 0 or (offset // 4 + 1) % 25 == 0:
                put(R / tag / 'progress.json', {'phase': 'continued_sft', 'epoch': epoch,
                    'frames': min(offset + 4, 4025), 'total': 4025, 'loss': metrics['loss']})
                print('SFT', tag, epoch, min(offset + 4, 4025), '/4025', flush=True)
        assert policy.actor_hash() != prior_hash
        assert exp.frozen_sample_hash(policy) == frozen
        prior_hash = policy.actor_hash()
        policy.eval()
        val = exp.evaluate(policy, val_data, 4, R / tag / f'validation_epoch{epoch}.jsonl',
                           epoch * math.ceil(4025 / 4))
        cp = R / 'checkpoints' / tag / f'epoch{epoch}'
        save_checkpoint(policy, cp)
        torch.save({'optimizer': optimizer.state_dict(), 'epoch': epoch}, cp / 'trainer_state.pt')
        put(R / tag / f'epoch{epoch}_result.json', {'complete': True, 'epochs': epoch,
            'frames_per_epoch': 4025, 'validation': val,
            'training_metrics': {k: v / 4025 for k, v in totals.items()},
            'seconds': time.perf_counter() - tick, 'memory': memory(),
            'checkpoint': str(cp), 'frozen_samples_unchanged': True})
    print('CONTINUED_TRAIN_COMPLETE', tag, flush=True)


def evaluate(percent):
    tag = f'p{percent}'
    _, val_data, stats = exp.get_data()
    cp = R / 'checkpoints' / tag / 'epoch3'
    policy = load_checkpoint(cp, stats)
    before = policy.actor_hash()
    action = common.sample_action(policy, val_data, 0).cpu()
    assert action.shape == (1, 7) and bool(torch.isfinite(action).all())
    namespace = dict(vars(exp))
    exec(compile(ROLLOUT.read_text(), str(ROLLOUT), 'exec'), namespace)
    trials = []
    for state in range(5):
        result = namespace['simulate_trial'](policy, tag + '_e3', state,
                                             types.SimpleNamespace(run=R))
        reference = json.loads((BASE / f'state_{state:02d}' / 'result.json').read_text())
        for key in ('initial_image_sha256', 'selected_init_state_sha256',
                    'init_state_file_sha256', 'reference_sha256', 'simulator_versions',
                    'max_policy_steps', 'warmup_steps'):
            assert result[key] == reference[key], key
        trials.append(result)
        put(R / tag / 'epoch3_simulation_progress.json', {'completed': len(trials),
            'successes': sum(x['success'] for x in trials)})
        print('EVAL', tag, 'epoch3', state, result['success'], flush=True)
    assert before == policy.actor_hash()
    put(R / tag / 'epoch3_simulation_result.json', {'complete': True, 'tag': tag + '_e3',
        'attempts': 5, 'successes': sum(x['success'] for x in trials),
        'trials': trials, 'checkpoint': str(cp), 'action_shape': [1, 7],
        'all_actions_finite': all(x['all_actions_finite'] for x in trials)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train', 'evaluate'])
    parser.add_argument('percent', type=int, choices=[20, 35])
    args = parser.parse_args()
    globals()[args.phase](args.percent)
