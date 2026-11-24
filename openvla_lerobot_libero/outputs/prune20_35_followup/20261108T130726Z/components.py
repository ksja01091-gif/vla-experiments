"""Read-only validation diagnostic: per-action-coordinate prediction errors."""
import argparse
import json
import time
from pathlib import Path

from policy import *

RUN = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('branch', choices=['p20', 'p35'])
    parser.add_argument('stage', choices=['epoch3', 'after_ppo'])
    args = parser.parse_args()
    _, validation, stats = exp.get_data()
    policy = load_checkpoint(RUN / 'checkpoints' / args.branch / args.stage, stats)
    policy.eval()
    before = policy.actor_hash()
    absolute = np.zeros(7, dtype=np.float64)
    signed = np.zeros(7, dtype=np.float64)
    count = 0
    started = time.perf_counter()
    for index in range(len(validation)):
        sample = validation[index]
        target = sample['action'].detach().cpu().numpy().reshape(7)
        prediction = common.sample_action(policy, validation, index).detach().cpu().numpy().reshape(7)
        assert np.isfinite(target).all() and np.isfinite(prediction).all()
        delta = prediction - target
        absolute += np.abs(delta)
        signed += delta
        count += 1
        if count % 50 == 0:
            print('ACTION_COMPONENTS', args.branch, args.stage, count, '/', len(validation), flush=True)
    assert count == 462 and before == policy.actor_hash()
    result = {'complete': True, 'branch': args.branch, 'stage': args.stage,
        'validation_frames': count, 'mean_absolute_error_by_coordinate': (absolute / count).tolist(),
        'mean_signed_error_by_coordinate': (signed / count).tolist(),
        'translation_mae': float(absolute[:3].mean() / count),
        'rotation_mae': float(absolute[3:6].mean() / count),
        'gripper_mae': float(absolute[6] / count),
        'seconds': time.perf_counter() - started,
        'warning': 'Action coordinates use different scales; compare the same coordinate across stages, not translation versus rotation magnitude directly.'}
    path = RUN / args.branch / f'components_{args.stage}.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('COMPONENT_DIAGNOSTIC_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
