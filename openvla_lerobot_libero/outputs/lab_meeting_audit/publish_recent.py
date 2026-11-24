from pathlib import Path
import hashlib
import json
import csv
import wandb

ROOT=Path('/home/jovyan/projects/lerobot-openvla/outputs')
DEST=ROOT/'lab_meeting_audit'
ENTITY='ksja01091-gwangju-institute-of-science-and-technology'
PROJECT='openvla-libero-bowl-baseline'
settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,x_disable_meta=True,x_disable_machine_info=True,x_save_requirements=False,console='off')
uploaded=[]
for source in [ROOT/'grasp_action_swap/20261108T051214_869671Z', ROOT/'arm_position_angle_swap/20261108T052534_438469Z']:
    data=json.loads((source/'results.json').read_text())
    for trial in data['trials']:
        ident='swap'+hashlib.sha256((str(source)+trial['trial']).encode()).hexdigest()[:12]
        run=wandb.init(entity=ENTITY,project=PROJECT,id=ident,resume='allow',name='DIAG-'+trial['trial']+'-state0',mode='online',dir=str(DEST),settings=settings,
            config={'source':str(source),'new_training':False,'state':0,'hybrid_policy':True,'checkpoints':data['checkpoints'],'criteria':data['criteria']},tags=['diagnosis-only','no-training','hybrid-actions','same-state-0'])
        run.summary.update({'diagnosis/'+k:v for k,v in trial.items() if isinstance(v,(str,int,float,bool,type(None)))})
        run.summary['diagnosis/scope']='One previously used initial state per hybrid condition. Both frozen policies see the current shared image.'
        run.summary['video']=wandb.Video(str(source/trial['trial']/'episode.mp4'),format='mp4')
        run.save(str(source/'results.json'),base_path=str(source),policy='now')
        run.save(str(source/trial['trial']/'steps.jsonl'),base_path=str(source),policy='now')
        uploaded.append({'id':ident,'name':run.name,'url':run.url})
        run.finish()
old=ROOT/'full_bowl_experiment/20261102T091448_811738Z'
train=json.loads((old/'training_result.json').read_text())
for row in csv.DictReader((old/'simulation_summary.csv').read_text().splitlines()):
    if row['model_tag']=='base': continue
    tag=row['model_tag']
    ident='historical20'+tag
    run=wandb.init(entity=ENTITY,project=PROJECT,id=ident,resume='allow',name='Historical-20demo-LoRA1epoch' if tag=='lora' else 'Historical-Public-LIBERO-Spatial-reference',mode='online',dir=str(DEST),settings=settings,
        config={'source':str(old),'historical':True,'preprocessing':'historical evaluation pipeline before later input alignment audit','reference_model':tag=='reference'},tags=['historical','posthoc-upload','early-pipeline'])
    run.summary.update({'simulation/successes':int(row['successes']),'simulation/attempts':int(row['attempts']),'simulation/success_rate':float(row['success_rate'])})
    if tag=='lora':
        run.summary.update({'validation/before_loss':train['baseline_validation']['loss'],'validation/after_loss':train['final_validation']['loss'],'train/demonstrations':20,'train/frames':1928})
        for line in (old/'validation_history.jsonl').read_text().splitlines():
            val=json.loads(line)
            run.log({'validation/'+k:v for k,v in val.items() if isinstance(v,(int,float))})
    run.save(str(old/'simulation_summary.csv'),base_path=str(old),policy='now')
    run.save(str(old/'training_result.json'),base_path=str(old),policy='now')
    uploaded.append({'id':ident,'name':run.name,'url':run.url})
    run.finish()
api=wandb.Api(timeout=60)
for row in uploaded:
    r=api.run(f'{ENTITY}/{PROJECT}/{row["id"]}')
    assert r.state=='finished'
all_runs=[]
for r in api.runs(f'{ENTITY}/{PROJECT}',per_page=100):
    all_runs.append({'id':r.id,'name':r.name,'url':r.url,'state':r.state,'summary':{k:v for k,v in dict(r.summary).items() if not k.startswith('_') and isinstance(v,(str,int,float,bool,type(None)))}})
(DEST/'wandb_runs.json').write_text(json.dumps(all_runs,indent=2))
(DEST/'recent_uploads.json').write_text(json.dumps({'verified':True,'new_entries':uploaded,'total_records':len(all_runs),'scope':'Completed scientific runs and diagnosis results. Setup-only import checks are retained locally.'},indent=2))
print('UPLOAD_COMPLETE',len(uploaded),len(all_runs),flush=True)
