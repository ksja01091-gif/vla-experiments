"""Publish stage-matched SFT controls and keep PPO references separate."""
from pathlib import Path
import json,hashlib
import wandb
R=Path(__file__).resolve().parent
d=json.loads((R/'final_results.json').read_text());assert d['complete']
ENTITY='ksja01091-gwangju-institute-of-science-and-technology';PROJECT='openvla-libero-bowl-baseline'
records=[];prefix='p50'+hashlib.sha256(str(R).encode()).hexdigest()[:12]
for row in d['primary_comparison']:
    tag=row['method'];runid=prefix+tag
    settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,x_disable_meta=True,x_disable_machine_info=True,x_save_requirements=False,console='off',init_timeout=60,finish_timeout=90)
    run=wandb.init(entity=ENTITY,project=PROJECT,id=runid,resume='allow',mode='online',dir=str(R),settings=settings,
        group=R.name,name='P50-'+tag+'-SFT1-NO-NEW-RL',config={**d['config'],'method':tag},
        tags=['50pct-structure-pilot','SFT1','no-new-RL','matched-SFT-stage','same-five-states','posthoc-upload'])
    run.summary.update({'comparison/'+k:v for k,v in row.items() if isinstance(v,(int,float))})
    if tag=='p50':
        run.define_metric('sft/optimizer_step');run.define_metric('sft/*',step_metric='sft/optimizer_step')
        lines=(R/'p50/sft_metrics.jsonl').read_text().splitlines()
        for i,line in enumerate(lines):
            if i%10==0 or i==len(lines)-1:
                m=json.loads(line);run.log({'sft/'+k:v for k,v in m.items() if isinstance(v,(int,float))})
        stages=[d['new_sft']['before_validation'],d['new_sft']['after_validation']]
        run.define_metric('validation/stage');run.define_metric('validation/*',step_metric='validation/stage')
        for i,v in enumerate(stages):run.log({'validation/stage':i,**{'validation/'+k:v[k] for k in ['loss','action_token_accuracy','normalized_action_l1']}})
        run.summary['diagnostic/prior_p90_after_sft']=wandb.Table(columns=['state','success','steps'],data=[[t['init_state_index'],t['success'],t['policy_steps']] for t in d['diagnostics_two_states']['p90_sft']['trials']])
    trials=json.loads((R/(tag+'_result.json')).read_text())['trials']
    run.summary['simulation/trials']=wandb.Table(columns=['state','success','steps','termination'],data=[[t['init_state_index'],t['success'],t['policy_steps'],t['termination_reason']] for t in trials])
    run.summary['simulation/successes']=row['successes'];run.summary['simulation/attempts']=5;run.summary['simulation/success_rate']=row['success_rate']
    run.summary['simulation/video_state0']=wandb.Video(trials[0]['video'],format='mp4')
    run.summary['comparison/figure']=wandb.Image(str(R/'comparison.png'))
    run.summary['notes/stage']='After extra SFT1, before any newPPO. Do not mix priorPPO2/5vs0/5.'
    for name in ['comparison.csv','comparison.png','RESULTS_KO.txt','experiment_config.json','preservation_report.json']:
        run.save(str(R/name),base_path=str(R),policy='now')
    url=run.url;run.finish();records.append({'method':tag,'id':runid,'url':url})
api=wandb.Api(timeout=60)
for entry in records:
    saved=api.run(f"{ENTITY}/{PROJECT}/{entry['id']}")
    expected=next(x for x in d['primary_comparison'] if x['method']==entry['method'])
    assert saved.summary['simulation/successes']==expected['successes'] and saved.summary['simulation/attempts']==5
(R/'wandb_upload_result.json').write_text(json.dumps({'verified':True,'runs':records},indent=2))
print('P50_WANDB_VERIFIED',json.dumps(records),flush=True)
