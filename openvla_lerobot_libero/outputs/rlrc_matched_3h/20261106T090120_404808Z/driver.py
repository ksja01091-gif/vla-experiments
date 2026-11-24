"""Sequential paired experiment with a hard three-hour total runtime bound."""
from pathlib import Path
import os, json, time, subprocess, signal, traceback, hashlib, math
R=Path(__file__).resolve().parent
PY='/home/jovyan/projects/venvs/rlrc-smoke312/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
cfg=read(R/'experiment_config.json')
assert cfg['runtime_cap_seconds']==10800 and cfg['sft_epochs_each']==1
started=time.time();deadline=started+cfg['runtime_cap_seconds']
write(R/'budget.json',{'started_unix':started,'deadline_unix':deadline,'cap_seconds':10800,'user_selected':True})
def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=45)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
def phase(name,script,args,python=PY):
    if time.time()+120>=deadline:raise TimeoutError('Run time cap reached before '+name)
    write(R/'status.json',{'status':'running','phase':name,'started_unix':started,'deadline_unix':deadline})
    with (R/(name+'.log')).open('w') as log:
        p=subprocess.Popen([python,'-u',str(R/script),*args],cwd=R,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(R/'active_process.json',{'phase':name,'pid':p.pid})
        while p.poll() is None:
            if time.time()+75>=deadline:
                stop(p);raise TimeoutError('Three-hour run cap: '+name)
            time.sleep(5)
        if p.returncode:raise RuntimeError(name+' exited '+str(p.returncode)+'; see saved log')
    print('FINISHED',name,'elapsed_minutes',round((time.time()-started)/60,2),flush=True)
try:
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')}
    hashes[str(R/'experiment_config.json')]=hashlib.sha256((R/'experiment_config.json').read_bytes()).hexdigest()
    write(R/'sources_at_start.json',hashes)
    phase('math_checks','check_math.py',[])
    phase('prepare','worker.py',['prepare'])
    for b in ['dense','pruned']:phase('sft_'+b,'worker.py',['sft','--branch',b])
    dense=read(R/'dense/sft_result.json');pruned=read(R/'pruned/sft_result.json')
    assert dense['coverage_sha256']==pruned['coverage_sha256']
    for b in ['dense','pruned']:phase('collect_'+b,'worker.py',['collect','--branch',b])
    counts={b:read(R/b/'collection_result.json')['transitions'] for b in ['dense','pruned']}
    n=min(counts.values())//4*4
    assert n>=4
    write(R/'matched_ppo_budget.json',{'transitions_each':n,'source_counts':counts,
        'passes':2,'accumulated_transitions_per_update':4,'optimizer_updates_each':n//4*2,
        'bc_train_frames_per_update':1,'reason':'Equal actor updates and transition count despite different episode lengths',
        'selection':'deterministic uniformly spaced subset, without replacement; last <=3 unmatched transitions excluded'})
    for b in ['dense','pruned']:phase('ppo_'+b,'worker.py',['ppo','--branch',b])
    for b in ['dense','pruned']:phase('evaluate_'+b,'worker.py',['evaluate','--branch',b])
    phase('report','report.py',[])
    try:
        phase('wandb_upload','upload_results.py',[],python='/home/jovyan/projects/wandb-upload-env/bin/python')
        wb=read(R/'wandb_upload_result.json')
    except Exception as e:
        wb={'verified':False,'error':repr(e),'local_results_preserved':True}
        write(R/'wandb_upload_result.json',wb)
    write(R/'status.json',{'status':'complete','paired_experiment_complete':True,
        'elapsed_seconds':time.time()-started,'cap_seconds':10800,'wandb_verified':wb['verified']})
except Exception as e:
    write(R/'driver_error.json',{'error':repr(e),'traceback':traceback.format_exc()})
    write(R/'status.json',{'status':'interrupted' if isinstance(e,TimeoutError) else 'error',
        'paired_experiment_complete':False,'error':repr(e),'elapsed_seconds':time.time()-started,
        'cap_seconds':10800,'incomplete_is_not_a_task_failure':True})
    raise
