"""Publish diagnostic evidence and package it; never claims a new evaluation cohort."""
from pathlib import Path
import json,time,hashlib,zipfile,traceback,subprocess
R=Path(__file__).resolve().parent
def write(p,d):Path(p).write_text(json.dumps(d,indent=2))
for _ in range(180):
    if (R/'START_HERE_KO.txt').exists() and 'GRASP_DIAGNOSIS_COMPLETE' in (R/'report.log').read_text():break
    time.sleep(2)
else:raise RuntimeError('Diagnostic report did not complete; inspect report.log before proceeding')
data=json.loads((R/'diagnosis.json').read_text());assert data['complete'] and data['no_optimizer_updates']
subprocess.run(['/home/jovyan/anaconda3/envs/openvla310/bin/python',str(R/'fix_keyframes.py')],check=True)
try:
    import wandb
    ENTITY='ksja01091-gwangju-institute-of-science-and-technology';PROJECT='openvla-libero-bowl-baseline'
    run_id='graspdiag'+hashlib.sha256(str(R).encode()).hexdigest()[:12]
    settings=wandb.Settings(disable_git=True,disable_code=True,x_disable_stats=True,x_disable_meta=True,x_disable_machine_info=True,x_save_requirements=False,console='off',init_timeout=60,finish_timeout=90)
    run=wandb.init(entity=ENTITY,project=PROJECT,id=run_id,resume='allow',name='P50-grasp-failure-DIAGNOSIS-NO-TRAINING',
        mode='online',dir=str(R),settings=settings,tags=['diagnosis-only','no-training','existing-trials-replayed','same-image-counterfactual'],
        config={'source_run':str(json.loads((R/'diagnostic_config.json').read_text())['source_run']),
            'new_training':False,'new_optimizer_updates':0,'new_performance_cohort':False,'same_image_weights_frozen':True})
    for tag,c in data['cohort'].items():
        run.summary.update({'diagnosis/'+tag+'/'+k:v for k,v in c.items() if isinstance(v,(int,float))})
    columns=['model','state','failure_stage','first_grasp_step','first_lift_5cm_step','stable_grasp_and_lift','max_lift_cm','official_success','gripper_command_switches']
    run.summary['diagnosis/stage_table']=wandb.Table(columns=columns,data=[[r.get(k) for k in columns] for r in data['per_trial']])
    run.summary['diagnosis/no_training']=True
    run.summary['diagnosis/scope']='Diagnosis of the same existing ten trajectories. Replay is not an independent success-rate experiment.'
    run.summary['diagnosis/main_limit']='Identifies contact/lift failure stage and command differences, not the causal head/neuron or a proven training fix.'
    for name in ['state0_keyframes.png','state0_contact_lift_comparison.png','all_states_grasp_lift.png','state0_gripper_commands.png']:
        run.summary['evidence/'+Path(name).stem]=wandb.Image(str(R/name))
    for name in ['stage_comparison.csv','START_HERE_KO.txt','diagnosis.json','same_image_comparison.json','preservation_report.json']:
        run.save(str(R/name),base_path=str(R),policy='now')
    url=run.url;run.finish()
    saved=wandb.Api(timeout=60).run(f'{ENTITY}/{PROJECT}/{run_id}')
    assert saved.summary['diagnosis/no_training']
    assert saved.summary['diagnosis/p50/stable_grasp_and_lift']==data['cohort']['p50']['stable_grasp_and_lift']
    write(R/'wandb_upload_result.json',{'verified':True,'run_id':run_id,'url':url,'no_new_performance_cohort':True})
except Exception as e:
    write(R/'wandb_upload_result.json',{'verified':False,'error':repr(e),'traceback':traceback.format_exc(),'local_results_preserved':True})
entries={}
for name in ['START_HERE_KO.txt','diagnosis.json','diagnostic_config.json','stage_comparison.csv',
    'same_image_comparison.json','same_image_dense_sft.json','same_image_p50.json','replay_results.json',
    'preservation_report.json','wandb_upload_result.json','state0_keyframes.png','state0_contact_lift_comparison.png',
    'all_states_grasp_lift.png','state0_gripper_commands.png']:
    entries[name]=R/name
for tag in ['dense_sft','p50']:
    for state in range(5):
        for name in ['result.json','trace.jsonl','metadata.json']:
            entries[f'traces/{tag}/state_{state:02d}/{name}']=R/tag/f'state_{state:02d}'/name
for name in ['replay.py','same_image_probe.py','report.py','publish_bundle.py','fix_keyframes.py']:entries['source/'+name]=R/name
src=Path(json.loads((R/'diagnostic_config.json').read_text())['source_run'])
entries['01_original_state0.mp4']=src/'simulation/dense_sft/state_00/episode.mp4'
entries['02_pruned50_state0.mp4']=src/'simulation/p50/state_00/episode.mp4'
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in entries.items()}
write(R/'viewing_bundle_manifest.json',manifest)
with zipfile.ZipFile(R/'viewing_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name,path in entries.items():archive.write(path,name)
    archive.write(R/'viewing_bundle_manifest.json','viewing_bundle_manifest.json')
write(R/'complete.json',{'diagnosis_complete':True,'no_training':True,'no_optimizer_updates':True,
    'all_ten_trajectories_reproduced':True,'same_image_probe_completed':True,'bundle_files':len(entries),
    'wandb_verified':json.loads((R/'wandb_upload_result.json').read_text())['verified']})
print('DIAGNOSIS_BUNDLE_READY',len(entries),(R/'viewing_bundle.zip').stat().st_size,flush=True)
