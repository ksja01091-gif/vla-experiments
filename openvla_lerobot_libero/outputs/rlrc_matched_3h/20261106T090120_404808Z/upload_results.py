"""Posthoc upload of actual paired logs; model and dataset stay on Linux."""
from pathlib import Path
import json,hashlib
import wandb
R=Path(__file__).resolve().parent
ENTITY='ksja01091-gwangju-institute-of-science-and-technology'
PROJECT='openvla-libero-bowl-baseline'
data=json.loads((R/'final_results.json').read_text());assert data['complete']
prefix='rlrc'+hashlib.sha256(str(R).encode()).hexdigest()[:12]
def settings():return wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,
    x_disable_meta=True,x_disable_machine_info=True,x_save_requirements=False,console='off',init_timeout=90,finish_timeout=90)
records=[]
for branch in ['dense','pruned']:
    run_id=prefix+branch
    run=wandb.init(entity=ENTITY,project=PROJECT,id=run_id,resume='allow',group=R.name,
        name='RLRC90-'+('dense-control' if branch=='dense' else 'pruned-2B')+'-SFT1-PPO4',
        mode='online',dir=str(R),settings=settings(),config={**data['config'],'branch':branch},
        tags=['RLRC-adaptation','matched-control','SFT1','PPO4episodes','same-five-evaluation-states','posthoc-upload'])
    for path,namespace in [(R/branch/'sft_metrics.jsonl','sft'),(R/branch/'ppo_metrics.jsonl','ppo')]:
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        for i,row in enumerate(rows):
            if i%10==0 or i==len(rows)-1:
                run.log({f'{namespace}/{k}':v for k,v in row.items() if isinstance(v,(int,float))})
    details=data['branches'][branch]
    for stage,d in details['validation_history'].items():
        run.summary.update({f'validation/{stage}/{k}':v for k,v in d.items() if isinstance(v,(int,float))})
    comparison=next(r for r in data['comparison'] if r['method']==branch)
    run.summary.update({f'comparison/{k}':v for k,v in comparison.items() if isinstance(v,(int,float))})
    run.summary['evaluation/scope']='One task; five previously used states; not independent LIBERO-wide performance.'
    trials=details['simulation_result']['trials']
    run.summary['evaluation/trials']=wandb.Table(columns=['initial_state','success','steps','termination'],
        data=[[t['init_state_index'],t['success'],t['policy_steps'],t['termination_reason']] for t in trials])
    representative=next((t for t in trials if t['success']),trials[0])
    run.summary['evaluation/representative_video']=wandb.Video(representative['video'],format='mp4')
    for name in ['comparison.csv','comparison.png','RESULTS_KO.txt','experiment_config.json','preservation_report.json']:
        if (R/name).exists():run.save(str(R/name),base_path=str(R),policy='now')
    url=run.url;run.finish()
    records.append({'branch':branch,'id':run_id,'url':url})
api=wandb.Api(timeout=60)
for entry in records:
    remote=api.run(f"{ENTITY}/{PROJECT}/{entry['id']}")
    expected=next(r for r in data['comparison'] if r['method']==entry['branch'])
    assert remote.summary['comparison/successes']==expected['successes']
    assert remote.summary['comparison/attempts']==5
(R/'wandb_upload_result.json').write_text(json.dumps({'verified':True,'runs':records},indent=2))
print('WANDB_RESULTS_VERIFIED',json.dumps(records),flush=True)
