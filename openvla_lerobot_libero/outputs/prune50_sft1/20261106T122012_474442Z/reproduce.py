"""Use --check for files only; --start repeats in a new one-hour run directory."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,subprocess,shutil
R=Path(__file__).resolve().parent
FILES=['policy.py','check_math.py','worker.py','driver.py','report.py','upload_results.py','reproduce.py']
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--start',action='store_true');a=p.parse_args()
assert a.check!=a.start,'Choose exactly one of --check or --start'
cfg=json.loads((R/'experiment_config.json').read_text())
for name,h in json.loads((R/'sources_at_start.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
assert Path(cfg['data_root']).is_dir() and (Path(cfg['starting_adapter'])/'adapter_model.safetensors').is_file()
if a.check:
    print('P50_REPRODUCTION_CHECK_PASSED; no new training launched')
    raise SystemExit(0)
protected=json.loads((R/'protected_before.json').read_text())
assert all(Path(name).exists() and hashlib.sha256(Path(name).read_bytes()).hexdigest()==h for name,h in protected.items())
interpreters=json.loads((R/'interpreters.json').read_text());snaps={}
for name,exe in interpreters.items():
    snaps[name]=subprocess.run([exe,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    assert sorted(snaps[name].splitlines())==sorted((R/(name+'_before.txt')).read_text().splitlines())
dest=R.parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_repeat');dest.mkdir()
for name in FILES:shutil.copy2(R/name,dest/name)
cfg['reproduced_from']=str(R)
(dest/'experiment_config.json').write_text(json.dumps(cfg,indent=2))
(dest/'protected_before.json').write_text(json.dumps(protected,indent=2))
(dest/'interpreters.json').write_text(json.dumps(interpreters,indent=2))
for name,snap in snaps.items():(dest/(name+'_before.txt')).write_text(snap)
with (dest/'driver.log').open('w') as log:
    job=subprocess.Popen(['/home/jovyan/projects/venvs/rlrc-smoke312/bin/python','-u',str(dest/'driver.py')],cwd=dest,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(dest/'driver.pid').write_text(str(job.pid));print('NEW_P50_EXPERIMENT_STARTED',dest,'PID',job.pid,flush=True)
