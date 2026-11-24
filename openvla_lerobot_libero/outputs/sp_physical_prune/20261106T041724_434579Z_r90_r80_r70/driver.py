"""One unattended rank sweep: exports, isolated benchmarks, 15 trials, report, W&B."""
from pathlib import Path
import json,subprocess,time,hashlib,traceback
R=Path(__file__).resolve().parent
PY="/home/jovyan/anaconda3/envs/lerobot312/bin/python"
VIS="/home/jovyan/anaconda3/envs/openvla310/bin/python"
WB="/home/jovyan/projects/wandb-upload-env/bin/python"
def write(name,obj):
    p=R/name;t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(obj,indent=2));t.replace(p)
def start(script,args,label,exe=PY):
    with (R/(label+".log")).open("w") as log:
        return subprocess.Popen([exe,"-u",str(R/script),*args],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
def phase(name,script,args=(),exe=PY):
    write("status.json",{"status":"running","phase":name})
    p=start(script,args,name,exe);write("active_process.json",{"phase":name,"pid":p.pid})
    if p.wait():raise RuntimeError(name+" failed; inspect "+name+".log")
try:
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob("*.py")}
    write("source_hashes_at_start.json",sources)
    phase("unit_checks","check_compact.py")
    assert json.loads((R/"unit_checks.json").read_text())["passed"]
    phase("export","export_ranks.py")
    for tag in ["sp_full128","sp_pruned90","sp_pruned80","sp_pruned70"]:
        phase("benchmark_"+tag,"run_phase.py",["benchmark","--variant",tag])
    old=Path(json.loads((R/"experiment_config.json").read_text())["parent_pruning_run"])
    write("simulation_sp_full128.json",json.loads((old/"simulation_sp_full128.json").read_text()))
    write("status.json",{"status":"running","phase":"simulation_15_trials"})
    queue=["sp_pruned90","sp_pruned80","sp_pruned70"];active={}
    while queue or active:
        while queue and len(active)<2:
            tag=queue.pop(0);active[tag]=start("run_phase.py",["simulate","--variant",tag],"simulation_"+tag)
        write("simulation_schedule.json",{"active":{tag:p.pid for tag,p in active.items()},"pending":queue})
        for tag,p in list(active.items()):
            if p.poll() is None:continue
            if p.returncode:raise RuntimeError(tag+" simulation failed")
            del active[tag]
        time.sleep(3)
    phase("preservation_audit","run_phase.py",["audit"])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
    write("evaluation_complete.json",{"complete":True,"new_trials":15,"additional_training":False})
    phase("report","build_report.py",exe=VIS)
    try:
        phase("wandb_upload","upload_results.py",exe=WB)
    except Exception as e:
        write("wandb_error.json",{"error":repr(e),"local_results_complete":True})
        raise
    write("status.json",{"status":"complete","local_results_complete":True,"wandb_complete":True})
except Exception as e:
    write("driver_error.json",{"error":repr(e),"traceback":traceback.format_exc()})
    write("status.json",{"status":"error","error":repr(e),"local_results_complete":(R/"report_complete.json").exists()})
    raise
