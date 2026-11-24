"""One-hour bounded pruning50 SFT experiment; no new reinforcement learning."""
from pathlib import Path
import json,time,subprocess,os,signal,hashlib,traceback
R=Path(__file__).resolve().parent
PY='/home/jovyan/projects/venvs/rlrc-smoke312/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
cfg=read(R/'experiment_config.json')
assert cfg['runtime_cap_seconds']==3600 and cfg['target_internal_removal']==.5 and not cfg['reinforcement_learning']
assert not (R/'budget.json').exists(),'Never reset an existing run deadline; create a NEW directory to repeat'
start=time.time();deadline=start+3600
write(R/'budget.json',{'started_unix':start,'deadline_unix':deadline,'cap_seconds':3600,'user_selected':True})
def phase(label,script,args,python=PY):
    if time.time()+120>=deadline:raise TimeoutError('One-hour cap before '+label)
    write(R/'status.json',{'status':'running','phase':label,'started_unix':start,'deadline_unix':deadline})
    with (R/(label+'.log')).open('w') as log:
        child=subprocess.Popen([python,'-u',str(R/script),*args],cwd=R,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(R/'active_process.json',{'phase':label,'pid':child.pid})
        while child.poll() is None:
            if time.time()+75>=deadline:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                raise TimeoutError('One-hour cap reached in '+label)
            time.sleep(5)
        if child.returncode:raise RuntimeError(label+' exited '+str(child.returncode)+'; see preserved log')
    print('FINISHED',label,'elapsed_minutes',round((time.time()-start)/60,2),flush=True)
try:
    write(R/'sources_at_start.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*R.glob('*.py'),R/'experiment_config.json']})
    phase('math','check_math.py',[])
    phase('p90_sft_diagnostic','worker.py',['simulate','--tag','p90_sft','--states','5','8','--folder','diagnostic'])
    phase('prepare50','worker.py',['prepare'])
    phase('sft50','worker.py',['sft'])
    phase('p50_diagnostic','worker.py',['simulate','--tag','p50','--states','5','8','--folder','diagnostic'])
    # Evaluation always executes regardless of diagnostics; no selection by success.
    phase('p50_evaluation','worker.py',['simulate','--tag','p50'])
    phase('dense_sft_evaluation','worker.py',['simulate','--tag','dense_sft'])
    phase('report','report.py',[],python='/home/jovyan/anaconda3/envs/openvla310/bin/python')
    try:
        phase('wandb','upload_results.py',[],python='/home/jovyan/projects/wandb-upload-env/bin/python')
        wb=read(R/'wandb_upload_result.json')
    except Exception as e:
        wb={'verified':False,'error':repr(e),'local_results_preserved':True};write(R/'wandb_upload_result.json',wb)
    write(R/'status.json',{'status':'complete','experiment_complete':True,'elapsed_seconds':time.time()-start,
        'cap_seconds':3600,'new_reinforcement_learning':False,'wandb_verified':wb['verified']})
except Exception as e:
    write(R/'driver_error.json',{'error':repr(e),'traceback':traceback.format_exc()})
    write(R/'status.json',{'status':'interrupted' if isinstance(e,TimeoutError) else 'error',
        'experiment_complete':False,'elapsed_seconds':time.time()-start,'error':repr(e),'cap_seconds':3600,
        'partial_is_not_a_task_failure':True})
    raise
