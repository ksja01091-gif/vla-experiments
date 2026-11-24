"""Use actual update counts on charts and show the three measured validation stages."""
from pathlib import Path
import json
import wandb
R=Path(__file__).resolve().parents[1]
records=json.loads((R/'wandb_upload_result.json').read_text())
assert records['verified']
data=json.loads((R/'final_results.json').read_text())
ENTITY='ksja01091-gwangju-institute-of-science-and-technology'
PROJECT='openvla-libero-bowl-baseline'
for entry in records['runs']:
    branch=entry['branch']
    settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,x_disable_meta=True,
        x_disable_machine_info=True,x_save_requirements=False,console='off',init_timeout=90,finish_timeout=90)
    run=wandb.init(entity=ENTITY,project=PROJECT,id=entry['id'],resume='must',mode='online',dir=str(R),settings=settings)
    run.define_metric('sft/step')
    run.define_metric('sft/*',step_metric='sft/step')
    run.define_metric('ppo/update')
    run.define_metric('ppo/*',step_metric='ppo/update')
    run.define_metric('validation/stage_index')
    run.define_metric('validation/*',step_metric='validation/stage_index')
    rows=[]
    for index,stage in enumerate(['initial','after_sft','after_ppo']):
        metrics=data['branches'][branch]['validation_history'][stage]
        selected={k:metrics[k] for k in ['loss','action_token_accuracy','normalized_action_l1','predicted_action_token_fraction']}
        run.log({'validation/stage_index':index,**{'validation/'+k:v for k,v in selected.items()}})
        rows.append([stage,metrics['loss'],metrics['action_token_accuracy'],metrics['normalized_action_l1']])
    run.summary['validation/stages']=wandb.Table(columns=['stage','loss','action_token_accuracy','normalized_action_l1'],data=rows)
    run.summary['validation/stage_legend']='0=before extra learning (after structural pruning for pruned model),1=after SFT1epoch,2=after PPO298updates'
    run.summary['notes/metric_scope']='CUDA peaks after loading; custom loader first constructs the original7B; see execution_notes.json.'
    if (R/'comparison.png').exists():
        run.summary['comparison/figure']=wandb.Image(str(R/'comparison.png'))
    for name in ['execution_notes.json','training_condition_audit.json','reproduction_check.json','comparison.png','START_HERE_KO.txt']:
        if (R/name).exists():run.save(str(R/name),base_path=str(R),policy='now')
    run.finish()
api=wandb.Api(timeout=60)
for entry in records['runs']:
    remote=api.run(f"{ENTITY}/{PROJECT}/{entry['id']}")
    assert remote.summary['validation/stage_index']==2
(R/'publication/chart_setup_result.json').write_text(json.dumps({'verified':True,'runs':records['runs']},indent=2))
print('CHART_AXES_VERIFIED',flush=True)
