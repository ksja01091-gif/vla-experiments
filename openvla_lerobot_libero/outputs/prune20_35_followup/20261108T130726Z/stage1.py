"""Matched 20/35 percent structural-pruning follow-up on the remote server."""
import argparse
import hashlib
import json
import math
import time
import types
from pathlib import Path

from policy import *

RUN = Path(__file__).resolve().parent
OLD50 = Path('/home/jovyan/projects/lerobot-openvla/outputs/prune50_sft1/20261106T122012_474442Z')
BASE = Path('/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z/simulation/epoch5')
ROLLOUT = Path('/home/jovyan/projects/lerobot-openvla/outputs/input_pipeline_check/20261103T055533_073692Z/matched_rollout_function.py')


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def protected_digest(policy):
    h = hashlib.sha256()
    layers = policy.model.language_model.model.layers
    for i in (0, 31):
        for name, tensor in layers[i].state_dict().items():
            h.update(str(i).encode())
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def plan_for(percent):
    assert percent in (20, 35)
    source = json.loads((OLD50 / 'importance_scores.json').read_text())
    selected = json.loads((OLD50 / 'calibration_selection.json').read_text())
    head_keep = 26 if percent == 20 else 21
    mlp_keep = round(11008 * (1 - percent / 100))
    plan = {
        'method': 'RLRC-inspired grouped Taylor structural pruning; cached train-only calibration',
        'target_pruning': percent / 100,
        'protected_layers': [0, 31], 'head_dim': 128,
        'calibration_split': 'train', 'calibration_indices': selected['indices'],
        'head_keep': head_keep, 'mlp_keep': mlp_keep,
        'actual_target_head_removal': 1 - head_keep / 32,
        'actual_target_mlp_removal': 1 - mlp_keep / 11008,
        'layers': [],
    }
    for i in range(1, 31):
        row = source[str(i)]
        heads = sorted(range(32), key=lambda j: (-row['heads'][j], j))[:head_keep]
        neurons = sorted(range(11008), key=lambda j: (-row['neurons'][j], j))[:mlp_keep]
        plan['layers'].append({'layer': i, 'heads': sorted(heads), 'neurons': sorted(neurons)})
    return plan


def train(percent):
    tag = f'p{percent}'
    out = RUN / tag
    assert not out.exists(), 'Use a new run directory; never overwrite a result'
    out.mkdir(parents=True)
    train_data, val_data, stats = exp.get_data()
    exp.seed()
    assert len(train_data) == 4025 and len(val_data) == 462
    policy = RecoveryPolicy(stats)
    before = policy.parameter_report()
    before_protected = protected_digest(policy)
    plan = plan_for(percent)
    apply_plan(policy, plan)
    after = policy.parameter_report()
    assert before['model_parameters'] > after['model_parameters']
    assert before_protected == protected_digest(policy)
    before_frozen = exp.frozen_sample_hash(policy)
    put(out / 'plan.json', plan)
    put(out / 'size.json', {'before': before, 'after': after,
        'whole_parameter_reduction': 1 - after['model_parameters'] / before['model_parameters']})
    policy.configure_trainables(False)
    first_hash = policy.actor_hash()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=0)
    order = torch.randperm(4025, generator=torch.Generator().manual_seed(7)).tolist()
    sums = {k: 0.0 for k in exp.METRICS}
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    for offset in range(0, len(order), 4):
        indices = order[offset:offset + 4]
        metrics = common.step(policy, optimizer, train_data, indices)
        for key in sums:
            sums[key] += metrics[key] * len(indices)
        if offset == 0 or (offset // 4 + 1) % 25 == 0:
            put(out / 'progress.json', {'phase': 'sft', 'frames': min(offset + 4, 4025),
                'total': 4025, 'loss': metrics['loss']})
            print('SFT', tag, min(offset + 4, 4025), '/4025', flush=True)
    assert policy.actor_hash() != first_hash
    assert exp.frozen_sample_hash(policy) == before_frozen
    policy.eval()
    val = exp.evaluate(policy, val_data, 4, out / 'validation.jsonl', 1007)
    assert val['frames'] == 462
    cp = RUN / 'checkpoints' / tag / 'after_sft'
    save_checkpoint(policy, cp)
    torch.save({'optimizer': optimizer.state_dict(), 'epoch': 1,
        'coverage_sha256': hashlib.sha256(json.dumps(order).encode()).hexdigest()}, cp / 'trainer_state.pt')
    put(out / 'training_result.json', {'complete': True, 'tag': tag, 'epochs': 1,
        'frames': 4025, 'validation': val, 'seconds': time.perf_counter() - tick,
        'memory': memory(), 'checkpoint': str(cp),
        'training_metrics': {k: v / 4025 for k, v in sums.items()},
        'frozen_samples_unchanged': True})
    print('TRAIN_COMPLETE', tag, flush=True)


def evaluate(percent):
    tag = f'p{percent}'
    out = RUN / tag
    _, val_data, stats = exp.get_data()
    policy = load_checkpoint(RUN / 'checkpoints' / tag / 'after_sft', stats)
    policy.eval()
    before = policy.actor_hash()
    sample = common.sample_action(policy, val_data, 0).cpu()
    assert sample.shape == (1, 7) and bool(torch.isfinite(sample).all())
    namespace = dict(vars(exp))
    exec(compile(ROLLOUT.read_text(), str(ROLLOUT), 'exec'), namespace)
    trials = []
    for state in range(5):
        result = namespace['simulate_trial'](policy, tag, state, types.SimpleNamespace(run=RUN))
        reference = json.loads((BASE / f'state_{state:02d}' / 'result.json').read_text())
        for key in ('initial_image_sha256', 'selected_init_state_sha256',
                    'init_state_file_sha256', 'reference_sha256', 'simulator_versions',
                    'max_policy_steps', 'warmup_steps'):
            assert result[key] == reference[key], key
        trials.append(result)
        put(out / 'simulation_progress.json', {'completed': len(trials),
            'successes': sum(x['success'] for x in trials)})
        print('EVAL', tag, state, result['success'], flush=True)
    assert before == policy.actor_hash()
    put(out / 'simulation_result.json', {'complete': True, 'tag': tag,
        'attempts': 5, 'successes': sum(x['success'] for x in trials),
        'mean_episode_steps': sum(x['policy_steps'] for x in trials) / 5,
        'trials': trials, 'action_shape': [1, 7], 'finite': True,
        'scope': 'same five repeatedly inspected LIBERO initial states'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train', 'evaluate'])
    parser.add_argument('percent', type=int, choices=[20, 35])
    args = parser.parse_args()
    globals()[args.phase](args.percent)
