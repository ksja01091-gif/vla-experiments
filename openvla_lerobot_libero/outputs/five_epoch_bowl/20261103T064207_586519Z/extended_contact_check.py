"""Replay completed commands to measure object motion without changing evaluation results."""
import argparse,contextlib,json,os,random,re,sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("OMP_NUM_THREADS","2")
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

def analyze(directory):
    result=json.loads((directory/"result.json").read_text())
    if not result.get("trial_executed"):return None
    destination=directory/"whole_finger_contact_diagnostics.json"
    if destination.exists():return json.loads(destination.read_text())
    random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.set_num_threads(2)
    suite=benchmark.get_benchmark_dict()["libero_spatial"]()
    bddl=Path(suite.get_task_bddl_file_path(0))
    text=bddl.read_text()
    goal=text[text.lower().rfind("(:goal"):]
    match=re.search(r"\(On\s+([^\s()]+)\s+([^\s()]+)\)",goal,re.I)
    if not match:raise RuntimeError("Could not identify the goal objects")
    bowl_name,plate_name=match.groups()
    environment=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,use_camera_obs=False)
    traces=[];maximum_eef_error=0.0;expected_success=[]
    commands=[json.loads(line) for line in (directory/"steps.jsonl").read_text().splitlines()]
    try:
        environment.seed(0);environment.reset()
        observation=environment.set_init_state(np.load(directory/"selected_init_state.npy",allow_pickle=False))
        for _ in range(10):observation,_,_,_=environment.step([0,0,0,0,0,0,-1])
        sim=environment.env.sim
        bowl_id=environment.env.obj_body_id[bowl_name]
        plate_id=environment.env.obj_body_id[plate_name]
        bowl_geoms=[sim.model.geom_id2name(i) for i in range(sim.model.ngeom)
                    if sim.model.geom_id2name(i) and bowl_name in sim.model.geom_id2name(i)]
        assert bowl_geoms

        def capture(step,obs):
            return {"step":step,"bilateral_finger_contact":bool(environment.env._check_grasp(environment.env.robots[0].gripper,bowl_geoms)),"whole_finger_contact":bool(environment.env._check_grasp(
                        [environment.env.robots[0].gripper.important_geoms["left_finger"],
                         environment.env.robots[0].gripper.important_geoms["right_finger"]],bowl_geoms)),"bowl":sim.data.body_xpos[bowl_id].copy().tolist(),
                    "plate":sim.data.body_xpos[plate_id].copy().tolist(),
                    "eef":np.asarray(obs["robot0_eef_pos"]).tolist(),
                    "success":bool(environment.check_success())}
        traces.append(capture(0,observation))
        initial=json.loads((directory/"initialization.json").read_text())
        maximum_eef_error=float(np.abs(np.asarray(traces[0]["eef"])-initial["eef_position"]).max())
        for row in commands:
            observation,_,_,_=environment.step(row["controller_action"])
            trace=capture(row["step"],observation);traces.append(trace)
            error=float(np.abs(np.asarray(trace["eef"])-np.asarray(row["after"]["eef_position"])).max())
            maximum_eef_error=max(maximum_eef_error,error)
            expected_success.append(trace["success"]==row["after"]["task_success"])
        assert maximum_eef_error<1e-7 and all(expected_success),(maximum_eef_error,all(expected_success))
        bowl=np.asarray([row["bowl"] for row in traces])
        plate=np.asarray([row["plate"] for row in traces])
        eef=np.asarray([row["eef"] for row in traces])
        rise=float((bowl[:,2]-bowl[0,2]).max())
        report={"model_tag":result["model_tag"],"init_state_index":result["init_state_index"],
                "goal_bowl":bowl_name,"goal_plate":plate_name,"success":result["success"],
                "replayed_policy_steps":len(commands),"original_actions_unchanged":True,
                "maximum_eef_replay_error_m":maximum_eef_error,"success_flags_match_original":all(expected_success),
                "maximum_bowl_height_increase_m":rise,
                "maximum_bowl_xy_displacement_m":float(np.linalg.norm(bowl[:,:2]-bowl[0,:2],axis=1).max()),
                "minimum_eef_to_bowl_center_distance_m":float(np.linalg.norm(eef-bowl,axis=1).min()),
                "initial_bowl_plate_xy_distance_m":float(np.linalg.norm(bowl[0,:2]-plate[0,:2])),
                "final_bowl_plate_xy_distance_m":float(np.linalg.norm(bowl[-1,:2]-plate[-1,:2])),
                "object_contact_geoms":bowl_geoms,
                "whole_finger_contact_steps":[t["step"] for t in traces if t["whole_finger_contact"]],
                "whole_finger_and_rise_steps":[t["step"] for t in traces if t["whole_finger_contact"] and t["bowl"][2]-traces[0]["bowl"][2]>0.02],

                "bilateral_contact_steps":[t["step"] for t in traces if t["bilateral_finger_contact"]],
                "contact_and_rise_over_2cm_steps":[t["step"] for t in traces if t["bilateral_finger_contact"] and t["bowl"][2]-traces[0]["bowl"][2]>0.02],
                "observed_failure_pattern":None if result["success"] else
                    ("target_bowl_did_not_rise_by_2cm" if rise<0.02 else "bowl_rose_by_2cm_but_goal_not_reached"),
                "diagnostic_camera_rendering_disabled":True,"interpretation_limit":"Height and distance are observations, not proof of grasp/contact or a causal explanation of model errors."}
        destination.write_text(json.dumps(report,indent=2))
        (directory/"whole_finger_contact_trace.jsonl").write_text(chr(10).join(json.dumps(row) for row in traces))
        return report
    finally:environment.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--models",nargs="+",default=["base","lora","reference"])
    args=parser.parse_args();reports=[]
    for model in args.models:
        for path in sorted((args.run/"simulation"/model).glob("state_*/result.json")):
            report=analyze(path.parent)
            if report:reports.append(report)
    (args.run/"failure_motion_analysis.json").write_text(json.dumps(reports,indent=2))
    print("MOTION_ANALYSIS_COMPLETE",json.dumps([{k:r[k] for k in ["model_tag","init_state_index","success","maximum_bowl_height_increase_m","final_bowl_plate_xy_distance_m","observed_failure_pattern"]} for r in reports]),flush=True)

if __name__=="__main__":
    with contextlib.redirect_stdout(sys.stderr):
        report=analyze(Path(sys.argv[1]))
    print(json.dumps(report),flush=True)
