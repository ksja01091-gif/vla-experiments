from pathlib import Path
import importlib.util,inspect,traceback,json
r=Path(__file__).parent
spec=importlib.util.spec_from_file_location("parallel_epoch10",r/"ten_epoch_experiment.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=inspect.getsource(m.simulate_all)
assert "range(6,11)" in source
source=source.replace("range(6,11)","range(10,11)")
source=source.replace('ready=RUN/f"epoch_{epoch:02d}_ready.json"','ready=RUN/"epoch_10_parallel_ready.json"')
source=source.replace('"simulation_progress.json"','"parallel_simulation_progress.json"')
source=source.replace('"simulation_complete.json"','"parallel_epoch10_complete.json"')
source=source.replace('"epochs":[6,7,8,9,10],"new_trials":25','"epochs":[10],"new_trials":5')
exec(compile(source,str(r/"parallel_epoch10_worker.py"),"exec"),m.__dict__)
try:m.simulate_all()
except Exception:
 (r/"parallel_epoch10_error.txt").write_text(traceback.format_exc())
 raise
