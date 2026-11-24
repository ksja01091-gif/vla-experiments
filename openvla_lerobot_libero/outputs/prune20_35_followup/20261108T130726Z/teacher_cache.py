"""Cache actions from the public LIBERO-Spatial teacher on train frames only."""
import json
import time
from pathlib import Path

from policy import *
from lerobot.policies.openvla.modeling_openvla import OpenVLAPolicy

R = Path(__file__).resolve().parent
OUT = R / 'teacher_cache'
assert not OUT.exists(), 'Teacher cache is immutable; use a new directory'
OUT.mkdir()
train, validation, _ = exp.get_data()
assert len(train) == 4025 and len(validation) == 462
exp.seed()
teacher = OpenVLAPolicy(OpenVLAConfig(device='cuda:0'))
teacher.eval()
actions = np.zeros((4025, 7), dtype=np.float32)
demo_actions = np.zeros((4025, 7), dtype=np.float32)
started = time.perf_counter()
for index in range(4025):
    sample = train[index]
    demo_actions[index] = sample['action'].detach().cpu().numpy().reshape(7)
    actions[index] = common.sample_action(teacher, train, index).detach().cpu().numpy().reshape(7)
    assert np.isfinite(actions[index]).all()
    if index == 0 or (index + 1) % 100 == 0:
        status = {'completed': index + 1, 'total': 4025,
                  'seconds': time.perf_counter() - started}
        (OUT / 'progress.json').write_text(json.dumps(status, indent=2))
        print('TEACHER_CACHE', index + 1, '/4025', flush=True)
np.save(OUT / 'teacher_actions.npy', actions, allow_pickle=False)
scale = json.loads((Path('/home/jovyan/projects/lerobot-openvla/outputs/bowl40_experiment/20261103T051914_711739Z/data') /
                    'action_statistics_train_only.json').read_text())
q01, q99 = np.asarray(scale['q01']), np.asarray(scale['q99'])
normalized_mae = 2 * np.abs(actions - demo_actions).mean(axis=0) / (q99 - q01)
report = {'complete': True, 'teacher': 'openvla-7b-finetuned-libero-spatial',
          'source': 'train split only, 40 demonstrations, 4025 frames',
          'validation_used': False, 'teacher_action_shape': list(actions.shape),
          'teacher_min': actions.min(axis=0).tolist(),
          'teacher_max': actions.max(axis=0).tolist(),
          'demonstration_min': demo_actions.min(axis=0).tolist(),
          'demonstration_max': demo_actions.max(axis=0).tolist(),
          'teacher_vs_demo_normalized_mae_by_coordinate': normalized_mae.tolist(),
          'seconds': time.perf_counter() - started}
(OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print('TEACHER_CACHE_COMPLETE', json.dumps(report), flush=True)
