"""Train-only static pruning -> isolated benchmarks -> paired simulation -> audit/report."""
from pathlib import Path
import json,subprocess,time,hashlib,traceback
R=Path(__file__).resolve().parent
PY="/home/jovyan/anaconda3/envs/lerobot312/bin/python"
VISPY="/home/jovyan/anaconda3/envs/openvla310/bin/python"
SOURCE=Path("/home/jovyan/projects/lerobot-openvla/outputs/lora_sp_compare/20261104T091818_826396Z")
def write(name,obj):
    p=R/name;t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(obj,indent=2));t.replace(p)
def start(script,args,label,exe=PY):
    with (R/(label+".log")).open("w") as log:
        return subprocess.Popen([exe,"-u",str(R/script),*args],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
def phase(name,script,args,label,exe=PY):
    write("status.json",{"status":"running","phase":name})
    p=start(script,args,label,exe)
    write("active_process.json",{"phase":name,"pid":p.pid})
    code=p.wait()
    if code:raise RuntimeError(label+" failed; preserved log "+label+".log")
try:
    assert json.loads((R/"unit_checks.json").read_text())["passed"]
    write("core_source_hashes.json",{str(R/f):hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ["pruned_policy.py","run_phase.py","check_compact.py","driver.py"]})
    phase("calibration","run_phase.py",["calibrate"],"calibration")
    for variant in ["sp_full128","sp_pruned96","sp_pruned64"]:
        phase("benchmark_"+variant,"run_phase.py",["benchmark","--variant",variant],"benchmark_"+variant)
    write("simulation_sp_full128.json",json.loads((SOURCE/"simulation_sp128_eta_090.json").read_text()))
    write("status.json",{"status":"running","phase":"paired_simulation"})
    processes={v:start("run_phase.py",["simulate","--variant",v],"simulation_"+v) for v in ["sp_pruned96","sp_pruned64"]}
    write("simulation_pids.json",{v:p.pid for v,p in processes.items()})
    pending=dict(processes)
    while pending:
        for v,p in list(pending.items()):
            code=p.poll()
            if code is None:continue
            if code:raise RuntimeError("simulation failed: "+v)
            del pending[v]
        time.sleep(3)
    phase("preservation_audit","run_phase.py",["audit"],"audit")
    source_hashes=json.loads((R/"core_source_hashes.json").read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items()),"Core implementation changed during experiment"
    write("evaluation_complete.json",{"complete":True,"new_simulation_trials":10,"reused_baseline_trials":5,"no_training":True})
    phase("report","build_report.py",[],"report",VISPY)
    write("status.json",{"status":"complete"})
except Exception as error:
    write("driver_error.json",{"error":repr(error),"traceback":traceback.format_exc()})
    write("status.json",{"status":"error","error":repr(error)})
    raise

