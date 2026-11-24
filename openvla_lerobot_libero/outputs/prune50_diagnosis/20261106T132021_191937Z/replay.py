"""Replay original commands, measure object contacts/poses, verify the old trajectory."""
from pathlib import Path
import os,json,time,hashlib,random,traceback
os.environ.update(MUJOCO_GL='egl',PYTHONPATH='/home/jovyan/projects/LIBERO',TF_CPP_MIN_LOG_LEVEL='2')
import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
R=Path(__file__).resolve().parent
CFG=json.loads((R/'diagnostic_config.json').read_text());SOURCE=Path(CFG['source_run'])
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def first(rows,key):
    hits=[r['step'] for r in rows if r[key]]
    return hits[0] if hits else None
def longest_run(values):
    longest=current=0
    for v in values:
        current=current+1 if v else 0;longest=max(longest,current)
    return longest

def one(tag,state):
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    folder=SOURCE/'simulation'/tag/f'state_{state:02d}'
    source=[json.loads(l) for l in (folder/'steps.jsonl').read_text().splitlines()]
    initial=json.loads((folder/'initialization.json').read_text())
    out=R/tag/f'state_{state:02d}';out.mkdir(parents=True,exist_ok=False)
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(0),camera_heights=256,camera_widths=256)
    rows=[];substats={'physics_samples':0,'max_lift':0.,'max_consecutive_grasp_samples':0,'current_grasp_run':0,
        'left_contact_samples':0,'right_contact_samples':0,'both_contact_samples':0,'near_lift_samples':0}
    errors={'initial_max_pixel_error':0,'periodic_max_pixel_error':0,'max_eef_error':0.,'max_joint_error':0.,'max_time_error':0.}
    tick=time.perf_counter()
    try:
        env.seed(0);env.reset();obs=env.set_init_state(suite.get_task_init_states(0)[state])
        for _ in range(10):obs,_,_,_=env.step([0,0,0,0,0,0,-1])
        robot=env.robots[0];target=env.env.objects_dict[CFG['target_object']];plate=env.env.objects_dict[CFG['goal_object']]
        target_id=env.sim.model.body_name2id(target.root_body);plate_id=env.sim.model.body_name2id(plate.root_body)
        geomset=lambda names:{env.sim.model.geom_name2id(n) for n in names}
        target_geoms=geomset(target.contact_geoms);plate_geoms=geomset(plate.contact_geoms)
        left=geomset(robot.gripper.important_geoms['left_fingerpad']);right=geomset(robot.gripper.important_geoms['right_fingerpad'])
        baseline_z=float(env.sim.data.body_xpos[target_id,2]);initial_target=env.sim.data.body_xpos[target_id].copy()
        initial_plate=env.sim.data.body_xpos[plate_id].copy()
        def contacts():
            l=r=p=False
            for i in range(env.sim.data.ncon):
                c=env.sim.data.contact[i];a,b=int(c.geom1),int(c.geom2)
                if a in target_geoms:
                    l=l or b in left;r=r or b in right;p=p or b in plate_geoms
                if b in target_geoms:
                    l=l or a in left;r=r or a in right;p=p or a in plate_geoms
            return bool(l),bool(r),bool(p)
        def physics_sample():
            l,r,_=contacts();both=l and r
            lift=float(env.sim.data.body_xpos[target_id,2]-baseline_z)
            substats['physics_samples']+=1;substats['left_contact_samples']+=int(l)
            substats['right_contact_samples']+=int(r);substats['both_contact_samples']+=int(both)
            substats['current_grasp_run']=substats['current_grasp_run']+1 if both else 0
            substats['max_consecutive_grasp_samples']=max(substats['max_consecutive_grasp_samples'],substats['current_grasp_run'])
            substats['max_lift']=max(substats['max_lift'],lift)
            substats['near_lift_samples']+=int(both and lift>=.05)
        native_step=env.sim.step
        def instrumented_step(*args,**kwargs):
            result=native_step(*args,**kwargs);physics_sample();return result
        env.sim.step=instrumented_step
        def compare(observation,reference,step):
            eef=float(np.max(np.abs(np.asarray(observation['robot0_eef_pos'])-reference['eef_position'])))
            joint=float(np.max(np.abs(np.asarray(observation['robot0_joint_pos'])-reference['joint_positions'])))
            ts=abs(float(env.sim.data.time)-reference['sim_time'])
            errors['max_eef_error']=max(errors['max_eef_error'],eef)
            errors['max_joint_error']=max(errors['max_joint_error'],joint);errors['max_time_error']=max(errors['max_time_error'],ts)
            assert eef<1e-8 and joint<1e-8 and ts<1e-8,('Replay does not reproduce recorded trajectory',tag,state,step,eef,joint,ts)
            assert bool(env.check_success())==bool(reference['task_success'])
            if step==0 or step%25==0 or step==len(source):
                pixels=np.ascontiguousarray(observation['agentview_image'][::-1,::-1])
                with Image.open(reference['frame']) as im:old=np.asarray(im.convert('RGB'))
                diff=int(np.abs(pixels.astype(np.int16)-old.astype(np.int16)).max())
                if step==0:errors['initial_max_pixel_error']=diff
                else:errors['periodic_max_pixel_error']=max(errors['periodic_max_pixel_error'],diff)
                assert diff==0,('Camera replay mismatch',tag,state,step,diff)
        def capture(observation,step,command=None):
            pos=env.sim.data.body_xpos[target_id].copy();platepos=env.sim.data.body_xpos[plate_id].copy()
            eef=np.asarray(observation['robot0_eef_pos']);l,r,p=contacts();grasp=l and r
            assert grasp==bool(env.env._check_grasp(robot.gripper,target.contact_geoms))
            target_height=float(pos[2]-baseline_z)
            return {'step':step,'target_pos':pos.tolist(),'plate_pos':platepos.tolist(),'eef_pos':eef.tolist(),
                'target_lift_m':target_height,'target_plate_xy_m':float(np.linalg.norm(pos[:2]-platepos[:2])),
                'eef_target_distance_m':float(np.linalg.norm(eef-pos)),
                'eef_target_xy_m':float(np.linalg.norm(eef[:2]-pos[:2])),
                'target_displacement_m':float(np.linalg.norm(pos-initial_target)),
                'left_finger_contact':l,'right_finger_contact':r,'grasp_both_pads':grasp,'target_plate_contact':p,
                'lifted_5cm':target_height>=.05,'grasp_and_lifted_5cm':bool(grasp and target_height>=.05),
                'official_success':bool(env.check_success()),'controller_gripper':None if command is None else float(command[-1]),
                'gripper_qpos':np.asarray(observation.get('robot0_gripper_qpos',[])).tolist(),
                'gripper_state':np.asarray(robot.gripper.current_action).tolist()}
        compare(obs,source[0]['before'],0);rows.append(capture(obs,0))
        write(out/'metadata.json',{'tag':tag,'state':state,'target':target.name,'plate':plate.name,
            'target_contact_geoms':target.contact_geoms,'left_fingerpads':robot.gripper.important_geoms['left_fingerpad'],
            'right_fingerpads':robot.gripper.important_geoms['right_fingerpad'],'baseline_target_z':baseline_z,
            'initial_target_pos':initial_target.tolist(),'initial_plate_pos':initial_plate.tolist(),
            'source_action_log_sha256':filehash(folder/'steps.jsonl'),
            'simulation_timestep':float(env.sim.model.opt.timestep),'sampled_every_physics_step':True})
        for row in source:
            obs,reward,done,_=env.step(row['controller_action'])
            compare(obs,row['after'],row['step']);rows.append(capture(obs,row['step'],row['controller_action']))
        assert len(rows)==len(source)+1
        with (out/'trace.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        grasp_lift_run=longest_run([r['grasp_and_lifted_5cm'] for r in rows])
        start_eef=np.asarray(rows[0]['eef_pos'])
        first_move=next((r['step'] for r in rows if np.linalg.norm(np.asarray(r['eef_pos'])-start_eef)>.005),None)
        report={'tag':tag,'state':state,'recorded_actions':len(source),'replay_exact':True,'errors':errors,
            'official_success':bool(rows[-1]['official_success']),
            'first_left_contact':first(rows,'left_finger_contact'),'first_right_contact':first(rows,'right_finger_contact'),
            'first_grasp_both_pads':first(rows,'grasp_both_pads'),'first_lift_5cm':first(rows,'lifted_5cm'),
            'first_grasp_and_lift_5cm':first(rows,'grasp_and_lifted_5cm'),
            'first_official_success':first(rows,'official_success'),
            'grasp_control_steps':sum(r['grasp_both_pads'] for r in rows),
            'grasp_lift_control_steps':sum(r['grasp_and_lifted_5cm'] for r in rows),
            'max_consecutive_grasp_lift_steps':grasp_lift_run,'stable_grasp_and_lift_observed':grasp_lift_run>=3,
            'first_motion_5mm':first_move,'max_target_lift_m':substats['max_lift'],
            'max_target_displacement_m':max(r['target_displacement_m'] for r in rows),
            'minimum_eef_target_distance_m':min(r['eef_target_distance_m'] for r in rows),
            'minimum_eef_target_xy_m':min(r['eef_target_xy_m'] for r in rows),
            'minimum_target_plate_xy_m':min(r['target_plate_xy_m'] for r in rows),
            'final_target_plate_xy_m':rows[-1]['target_plate_xy_m'],
            'gripper_closed_command_steps':sum(float(r['controller_action'][-1])>0 for r in source),
            'first_close_command':next((r['step'] for r in source if r['controller_action'][-1]>0),None),
            'first_open_after_grasp_command':next((r['step'] for r in source if first(rows,'grasp_both_pads') is not None and r['step']>first(rows,'grasp_both_pads') and r['controller_action'][-1]<0),None),
            'official_success_while_grasp_contact':bool(rows[-1]['official_success'] and rows[-1]['grasp_both_pads']),
            'physics_contact_measurements':substats,'seconds':time.perf_counter()-tick}
        write(out/'result.json',report)
        print('REPLAY_COMPLETE',tag,state,json.dumps(report),flush=True)
        return report
    finally:env.close()

try:
    write(R/'status.json',{'status':'running','phase':'exact_action_replay','no_training':True})
    reports=[]
    for tag in CFG['tags']:
        for state in CFG['states']:
            write(R/'progress.json',{'tag':tag,'state':state,'completed':len(reports),'total':10})
            reports.append(one(tag,state))
            write(R/'replay_progress.json',{'completed':len(reports),'total':10,'rows':reports})
    write(R/'replay_results.json',{'complete':True,'no_training':True,'no_model_inference':True,
        'scope':'Exact recorded-action replay with read-only physics instrumentation; not a new model evaluation cohort',
        'rows':reports,'all_replays_match_original_trajectories':True})
    write(R/'status.json',{'status':'replay_complete','no_training':True})
except Exception as e:
    write(R/'replay_error.json',{'error':repr(e),'traceback':traceback.format_exc()})
    write(R/'status.json',{'status':'error','error':repr(e),'no_training':True});raise
