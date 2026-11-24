import importlib.util,json
from pathlib import Path
r=Path(__file__).parent
spec=importlib.util.spec_from_file_location('motion',r/'analyze_motion_no_camera.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
reports=[]
for p in sorted((r/'diagnostics_train20/simulation/lora').glob('state_*/result.json')):
 reports.append(m.analyze(p.parent))
 print('MEASURED_PRIOR',p.parent.name,flush=True)
(r/'diagnostics_train20/failure_motion_analysis.json').write_text(json.dumps(reports,indent=2))
