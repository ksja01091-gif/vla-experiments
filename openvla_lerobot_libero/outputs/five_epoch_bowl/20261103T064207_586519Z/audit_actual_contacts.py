from pathlib import Path
import json,os,random
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
r=Path(__file__).parent;folder=r/"simulation/epoch5/state_00"
random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.set_num_threads(2)
suite=benchmark.get_benchmark_dict()["libero_spatial"]()
env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(0),camera_heights=256,camera_widths=256,use_camera_obs=False)
rows=[json.loads(x) for x in (folder/"steps.jsonl").read_text().splitlines()]
try:
 env.seed(0);env.reset();obs=env.set_init_state(np.load(folder/"selected_init_state.npy",allow_pickle=False))
 for _ in range(10):obs,_,_,_=env.step([0,0,0,0,0,0,-1])
 sim=env.env.sim;bowl_id=env.env.obj_body_id["akita_black_bowl_1"]
 subtree={bowl_id}
 changed=True
 while changed:
  changed=False
  for i in range(sim.model.nbody):
   if int(sim.model.body_parentid[i]) in subtree and i not in subtree:
    subtree.add(i);changed=True
 geoms={i for i in range(sim.model.ngeom) if int(sim.model.geom_bodyid[i]) in subtree}
 traces=[];z0=float(sim.data.body_xpos[bowl_id][2]);max_error=0.0
 def capture(step):
  pairs=[]
  for i in range(sim.data.ncon):
   c=sim.data.contact[i];g1=int(c.geom1);g2=int(c.geom2)
   if g1 in geoms or g2 in geoms:
    pairs.append([sim.model.geom_id2name(g1),sim.model.geom_id2name(g2)])
  return {"step":step,"bowl_rise_mm":1000*(float(sim.data.body_xpos[bowl_id][2])-z0),"contacts":pairs}
 traces.append(capture(0))
 for row in rows:
  obs,_,_,_=env.step(row["controller_action"])
  max_error=max(max_error,float(np.abs(np.array(obs["robot0_eef_pos"])-row["after"]["eef_position"]).max()))
  traces.append(capture(row["step"]))
 assert max_error<1e-7
 report={"goal_success":bool(env.check_success()),"maximum_eef_error":max_error,
   "bowl_body_id":bowl_id,"bowl_collision_geoms_by_body":[sim.model.geom_id2name(i) for i in sorted(geoms)],
   "gripper_groups":env.env.robots[0].gripper.important_geoms,
   "unique_bowl_contact_pairs":sorted({tuple(p) for t in traces for p in t["contacts"]}),
   "raised_frames":[t for t in traces if t["bowl_rise_mm"]>20],
   "all_trace":traces}
 (folder/"actual_contact_pair_audit.json").write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k not in ["all_trace","raised_frames"]}),flush=True)
finally:env.close()
