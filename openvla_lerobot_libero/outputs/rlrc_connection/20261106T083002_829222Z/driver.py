"""Only bounded connection probes; never starts long SFT or RL training."""
from pathlib import Path
import json,subprocess,time,traceback,hashlib
R=Path(__file__).resolve().parent
PY='/home/jovyan/projects/venvs/rlrc-smoke312/bin/python'
def write(name,d):
    p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def phase(label,script,args):
    write('status.json',{'status':'running','phase':label})
    with (R/(label+'.log')).open('w') as log:
        subprocess.run([PY,'-u',str(R/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    assert json.loads((R/'prepare_complete.json').read_text())['complete']
    write('connection_source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')})
    for branch in ['dense','pruned']:
        phase('measure_'+branch,'measure.py',['--branch',branch])
        phase('reload_'+branch,'reload.py',['--branch',branch])
    phase('report','report.py',[])
    write('status.json',{'status':'connection_complete_awaiting_runtime_cap','long_training_started':False})
except Exception as e:
    write('driver_error.json',{'error':repr(e),'traceback':traceback.format_exc()})
    write('status.json',{'status':'error','error':repr(e),'long_training_started':False});raise
