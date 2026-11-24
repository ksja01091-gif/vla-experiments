"""Publish connection/resource measurements, never a task success-rate result."""
from pathlib import Path
import json, hashlib, os
import wandb

R = Path(__file__).resolve().parents[1]
ENTITY = 'ksja01091-gwangju-institute-of-science-and-technology'
PROJECT = 'openvla-libero-bowl-baseline'
data = json.loads((R / 'smoke_summary.json').read_text())
assert data['connection_passed'] and data['no_full_training_started']
assert data['no_success_rate_evaluated']
run_id = 'rlrcsmoke' + hashlib.sha256(str(R).encode()).hexdigest()[:12]
settings = wandb.Settings(disable_git=True, disable_code=True, x_disable_stats=True,
    x_disable_meta=True, x_disable_machine_info=True, x_save_requirements=False,
    console='off', init_timeout=90, finish_timeout=120)
run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id, resume='allow',
    name='RLRC-90pct-CONNECTION-ONLY', mode='online', dir=str(R / 'publication'),
    tags=['RLRC-adaptation', 'connection-only', 'no-success-rate', 'posthoc-upload'],
    config={'stage': 'bounded connection and resource probe',
        'not_paper_reproduction': True, 'no_full_training': True,
        'starting_checkpoint': data['accepted_choices']['reference_checkpoint'],
        'target_internal_pruning': 0.9, 'calibration_train_frames_smoke': 8,
        'sft_updates_each': 4, 'rollout_steps_each': 8, 'ppo_updates_each': 1},
    settings=settings)
rows = []
for branch, d in data['branches'].items():
    metrics = {'parameters_B': d['parameters']['model_parameters'] / 1e9,
        'sft_peak_GiB': d['sft']['memory']['peak_gib'],
        'ppo_peak_GiB': d['ppo']['peak_gib'],
        'greedy_prediction_ms': d['inference_seconds'] * 1000,
        'sft_update_seconds': d['sft']['mean_update_seconds_excluding_first'],
        'rl_transition_seconds': d['rollout']['mean_transition_wall_seconds'],
        'sampling_logp_max_difference': d['rollout']['max_generation_teacher_logp_difference']}
    run.summary.update({f'connection/{branch}/{k}': v for k, v in metrics.items()})
    rows.append([branch, *metrics.values()])
run.summary['connection/resource_comparison'] = wandb.Table(columns=[
    'branch', 'parameters_B', 'sft_peak_GiB', 'ppo_peak_GiB',
    'greedy_prediction_ms', 'sft_update_seconds', 'rl_transition_seconds',
    'sampling_logp_max_difference'], data=rows)
run.summary['connection/interpretation'] = 'Connection checks only; no whole-episode success rate was evaluated.'
run.summary['connection/preservation_passed'] = not data['preservation']['changed_files']
for name in ['smoke_summary.json', 'preservation_report.json', 'RESULTS_KO.txt']:
    run.save(str(R / name), base_path=str(R), policy='now')
url = run.url
run.finish()
api = wandb.Api(timeout=60)
verified = api.run(f'{ENTITY}/{PROJECT}/{run_id}')
assert verified.summary['connection/preservation_passed']
assert verified.summary['connection/dense/sampling_logp_max_difference'] < 1e-4
assert verified.summary['connection/pruned/sampling_logp_max_difference'] < 1e-4
(R / 'publication' / 'upload_result.json').write_text(json.dumps({
    'verified': True, 'run_id': run_id, 'url': url,
    'no_task_success_rate_published': True}, indent=2))
print('WB_CONNECTION_VERIFIED', url, flush=True)
