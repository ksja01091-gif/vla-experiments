"""Isolated 32-action latency and GPU-memory profile for one saved checkpoint."""
import argparse
import json
import time
from pathlib import Path

from policy import *

R = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('branch', choices=['p20', 'p35'])
    parser.add_argument('stage', choices=['epoch3', 'after_ppo', 'after_translation_distill'])
    args = parser.parse_args()
    assert args.stage == 'epoch3' or args.branch == 'p20'
    output = R / args.branch / f'inference_{args.stage}_isolated.json'
    assert not output.exists()
    _, validation, stats = exp.get_data()
    policy = load_checkpoint(R / 'checkpoints' / args.branch / args.stage, stats)
    policy.eval()
    actor_hash = policy.actor_hash()
    indices = list(map(int, np.linspace(0, 461, 32, dtype=int)))
    for index in indices[:3]:
        common.sample_action(policy, validation, index)
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for index in indices:
        torch.cuda.synchronize()
        started = time.perf_counter()
        action = common.sample_action(policy, validation, index)
        torch.cuda.synchronize()
        assert action.shape == (1, 7) and bool(torch.isfinite(action).all())
        timings.append(time.perf_counter() - started)
    assert actor_hash == policy.actor_hash()
    report = {'complete': True, 'branch': args.branch, 'stage': args.stage,
              'samples': 32, 'warmup': 3,
              'mean_seconds': float(np.mean(timings)),
              'median_seconds': float(np.median(timings)),
              'all_finite_1x7': True,
              'memory': memory(), 'checkpoint': str(R / 'checkpoints' / args.branch / args.stage)}
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('ISOLATED_INFERENCE_COMPLETE', args.branch, args.stage,
          report['mean_seconds'], report['memory'], flush=True)


if __name__ == '__main__':
    main()
