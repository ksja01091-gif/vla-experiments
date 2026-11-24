"""Recreate the paired experiment in a NEW directory; --check never trains."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, hashlib, subprocess, shutil
R=Path(__file__).resolve().parent
FILES=['policy.py','rl_probe.py','check_math.py','worker.py','driver.py','report.py','upload_results.py']
PY='/home/jovyan/projects/venvs/rlrc-smoke312/bin/python'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--check',action='store_true',help='Verify source/config files only; no directory or training is created')
p.add_argument('--start',action='store_true',help='Create a fresh run and launch its three-hour bounded driver')
a=p.parse_args()
assert not (a.check and a.start)
cfg=json.loads((R/'experiment_config.json').read_text())
hashes=json.loads((R/'sources_at_start.json').read_text())
for path,h in hashes.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
assert Path(cfg['data_root']).is_dir()
assert (Path(cfg['starting_adapter'])/'adapter_model.safetensors').is_file()
if a.check:
    print('REPRODUCTION_SOURCE_CHECK_PASSED; no training launched')
    raise SystemExit(0)
if not a.start:
    p.error('Use --check to inspect, or --start to run a NEW experiment')
protected=json.loads((R/'protected_before.json').read_text())
assert all(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in protected.items())
interpreters=json.loads((R/'interpreters.json').read_text())
snapshots={}
for name,exe in interpreters.items():
    out=subprocess.run([exe,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    assert sorted(out.splitlines())==sorted((R/(name+'_before.txt')).read_text().splitlines())
    snapshots[name]=out
dest=R.parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_repeat')
dest.mkdir()
for name in FILES+[Path(__file__).name]:shutil.copy2(R/name,dest/name)
cfg['reproduced_from']=str(R)
(dest/'experiment_config.json').write_text(json.dumps(cfg,indent=2))
(dest/'protected_before.json').write_text(json.dumps(protected,indent=2))
(dest/'interpreters.json').write_text(json.dumps(interpreters,indent=2))
for name,out in snapshots.items():(dest/(name+'_before.txt')).write_text(out)
with (dest/'driver.log').open('w') as log:
    job=subprocess.Popen([PY,'-u',str(dest/'driver.py')],cwd=dest,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(dest/'driver.pid').write_text(str(job.pid))
print('NEW_EXPERIMENT_STARTED',dest,'PID',job.pid,flush=True)
