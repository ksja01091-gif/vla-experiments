"""One-epoch teacher-guided translation recovery from the selected PPO checkpoint."""
import hashlib
import json
import time
from pathlib import Path

from policy import *

R = Path(__file__).resolve().parent
SOURCE = R / 'checkpoints' / 'p20' / 'after_ppo'
DEST = R / 'checkpoints' / 'p20' / 'after_translation_distill'
CACHE = R / 'teacher_cache_v2'


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


class TranslationTeacherDataset:
    def __init__(self, original, teacher_actions):
        self.original = original
        self.teacher_actions = teacher_actions

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        sample = dict(self.original[int(index)])
        action = sample['action'].clone()
        action[:3] = torch.from_numpy(self.teacher_actions[int(index), :3].copy()).to(action.dtype)
        sample['action'] = action
        return sample


def main():
    assert SOURCE.exists() and not DEST.exists()
    cache_report = json.loads((CACHE / 'report.json').read_text())
    assert cache_report['complete'] and cache_report['validation_used'] is False
    teacher = np.load(CACHE / 'teacher_actions.npy', allow_pickle=False)
    assert teacher.shape == (4025, 7) and np.isfinite(teacher).all()
    original, validation, stats = exp.get_data()
    assert len(original) == 4025 and len(validation) == 462
    q01, q99 = np.asarray(stats['q01']), np.asarray(stats['q99'])
    outside_fraction = float(np.mean((teacher[:, :3] < q01[:3]) | (teacher[:, :3] > q99[:3])))
    data = TranslationTeacherDataset(original, teacher)
    policy = load_checkpoint(SOURCE, stats)
    policy.configure_trainables(False)
    frozen = exp.frozen_sample_hash(policy)
    old_hash = policy.actor_hash()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=0)
    order = torch.randperm(4025, generator=torch.Generator().manual_seed(17008)).tolist()
    totals = {k: 0.0 for k in exp.METRICS}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for offset in range(0, 4025, 4):
        indices = order[offset:offset + 4]
        metrics = common.step(policy, optimizer, data, indices)
        for key in totals:
            totals[key] += metrics[key] * len(indices)
        if offset == 0 or (offset // 4 + 1) % 25 == 0:
            put(R / 'translation_distill_progress.json', {'frames': min(offset + 4, 4025),
                'total': 4025, 'loss': metrics['loss']})
            print('TRANSLATION_DISTILL', min(offset + 4, 4025), '/4025', flush=True)
    assert policy.actor_hash() != old_hash
    assert exp.frozen_sample_hash(policy) == frozen
    policy.eval()
    val = exp.evaluate(policy, validation, 4, R / 'p20' / 'validation_after_translation_distill.jsonl', 0)
    assert val['frames'] == 462
    save_checkpoint(policy, DEST)
    put(R / 'translation_distill_result.json', {'complete': True,
        'teacher': cache_report['teacher'], 'source_checkpoint': str(SOURCE),
        'checkpoint': str(DEST), 'train_frames': 4025, 'validation_frames': 462,
        'target': 'teacher translation coordinates 0,1,2; demonstrated rotation/gripper coordinates 3-6',
        'teacher_translation_outside_student_q01_q99_fraction': outside_fraction,
        'epochs': 1, 'batch_size': 4, 'lr': 1e-5,
        'training_metrics': {k: v / 4025 for k, v in totals.items()},
        'validation': val, 'seconds': time.perf_counter() - started,
        'memory': memory(), 'frozen_samples_unchanged': True,
        'warning': 'This uses an extra LIBERO-Spatial-trained teacher, so outcomes cannot be attributed to PPO alone.'})
    print('TRANSLATION_DISTILL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
