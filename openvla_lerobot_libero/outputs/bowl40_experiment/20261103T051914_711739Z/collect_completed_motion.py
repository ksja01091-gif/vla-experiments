import importlib.util,json,time
from pathlib import Path
r=Path(__file__).parent
spec=importlib.util.spec_from_file_location('motion',r/'analyze_motion_no_camera.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
seen=set()
deadline=time.monotonic()+1800
while len(seen)<5:
 for p in sorted((r/'simulation/lora').glob('state_*/result.json')):
  if p.parent.name not in seen:
   result=m.analyze(p.parent)
   assert result and result['success_flags_match_original']
   seen.add(p.parent.name)
   print('MEASURED',p.parent.name,json.dumps(result),flush=True)
 if time.monotonic()>deadline:raise RuntimeError('Motion collection timed out')
 if len(seen)<5:time.sleep(10)
