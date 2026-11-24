"""Run the full user-authorized five-total-epoch experiment on remote Linux."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
RUN=Path(__file__).parent
MAIN=RUN/"five_epoch_experiment.py"
PY="/home/jovyan/anaconda3/envs/lerobot312/bin/python"
def write(p,v):Path(p).write_text(json.dumps(v,indent=2))
def command(phase,*extra):return [PY,"-u",str(MAIN),phase,*extra]
def run_phase(label,cmd):
 log=RUN/(label+".log")
 if log.exists():raise RuntimeError("Existing phase log preserved: "+str(log))
 with log.open("x") as handle:
  process=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,cwd="/home/jovyan/projects/lerobot-openvla",start_new_session=True)
  write(RUN/"driver_state.json",{"phase":label,"pid":process.pid,"log":str(log)})
  print("START",label,process.pid,flush=True)
  code=process.wait()
 if code:
  print(log.read_text()[-5000:],flush=True)
  raise RuntimeError(f"{label} exited {code}")
 print("DONE",label,flush=True)
def baseline_worker():
 run_phase("paired_epoch1",command("simulate","--tag","epoch1"))
 run_phase("motion_epoch1",command("motion","--tag","epoch1"))
 write(RUN/"epoch1_baseline_complete.json",{"complete":True,"attempts":5,"native_training_image_pipeline":True})

def final_checks():
 import imageio.v2 as imageio
 training=json.loads((RUN/"training_result.json").read_text())
 reload=json.loads((RUN/"checkpoint_reload_check.json").read_text())
 preservation=json.loads((RUN/"preservation_report.json").read_text())
 protected=json.loads((RUN/"protected_preservation.json").read_text())
 coverage=json.loads((RUN/"training_coverage.json").read_text())
 assert training["epochs_completed"]==5 and training["optimizer_steps"]==5035 and training["frames_seen"]==20125
 assert [x["epoch"] for x in coverage]==[2,3,4,5]
 assert all(x["frames"]==x["unique_frames"]==4025 and sorted(x["indices"])==list(range(4025)) for x in coverage)
 assert reload["fresh_python_process_reload"] and reload["all_actions_finite"] and reload["action_shape"]==[1,7]
 assert preservation["existing_sources_unchanged"] and all(v["packages_unchanged"] for v in preservation["environments"].values())
 assert not protected["changed_files"]
 cohorts={};videos=[];diagnostics={}
 for tag in ["epoch1","epoch5"]:
  cohort=json.loads((RUN/f"simulation_{tag}.json").read_text())
  assert cohort["attempts"]==5 and cohort["technical_errors"]==0 and cohort["states"]==[0,1,2,3,4]
  cohorts[tag]=cohort
  motion=json.loads((RUN/f"motion_{tag}.json").read_text())
  assert len(motion)==5 and all(x["success_flags_match_original"] for x in motion)
  diagnostics[tag]=motion
  for trial in cohort["trials"]:
   with imageio.get_reader(trial["video"]) as reader:
    count=reader.count_frames();assert count==trial["policy_steps"]+1
    reader.get_data(0);reader.get_data(count-1)
    videos.append({"tag":tag,"state":trial["init_state_index"],"frames":count,"fps":reader.get_meta_data()["fps"],"video":trial["video"]})
 for a,b in zip(cohorts["epoch1"]["trials"],cohorts["epoch5"]["trials"]):
  assert a["initial_image_sha256"]==b["initial_image_sha256"] and a["selected_init_state_sha256"]==b["selected_init_state_sha256"]
 write(RUN/"video_verification.json",videos)
 old_val=json.loads((Path("/home/jovyan/projects/lerobot-openvla/outputs/full_validation_actions/20261103T061517_441641Z")/"summary.json").read_text())
 old_train=json.loads((Path("/home/jovyan/projects/lerobot-openvla/outputs/error_cause_audit/20261103T062733_387868Z")/"summary.json").read_text())["training_sample"]
 summary={"training":training,"epoch1_free_validation":old_val,
          "epoch5_free_validation":json.loads((RUN/"validation_free_summary.json").read_text()),
          "epoch1_free_train_sample":old_train,"epoch5_free_train_sample":json.loads((RUN/"train_sample_free_summary.json").read_text()),
          "paired_simulation":cohorts,"motion_diagnostics":diagnostics,"reload":reload,"preservation":preservation,
          "limitations":["five fixed states, one task","training/evaluation initial-state disjointness not established",
                         "native training-matched inference for both cohorts; do not mix with prior cropped evaluations",
                         "final epoch5 is primary comparison even if another epoch has lower validation loss",
                         "simulator goal success and grasp/contact diagnostics are separate"]}
 write(RUN/"experiment_summary.json",summary)
 write(RUN/"core_complete.json",{"complete":True,"total_epochs":5,"paired_trials":10,"videos_verified":len(videos),
        "validation_frames":462,"train_sample_frames":480,"sources_and_environments_preserved":True})
 print("CORE_COMPLETE",RUN,flush=True)

def main():
 if "--baseline-only" in sys.argv:
  baseline_worker();return
 assert not (RUN/"driver_state.json").exists(),"Existing run is preserved; inspect before restarting"
 run_phase("prepare",command("prepare"))
 # Peak VRAM previously measured: training ~15 GiB plus inference ~14.5 GiB on 40 GiB A100.
 with (RUN/"baseline_driver.log").open("x") as log:
  baseline=subprocess.Popen([PY,"-u",str(Path(__file__)),"--baseline-only"],stdout=log,stderr=subprocess.STDOUT,
      cwd="/home/jovyan/projects/lerobot-openvla",start_new_session=True)
 write(RUN/"baseline_process.json",{"pid":baseline.pid,"overlaps_training":True,"expected_combined_peak_gib_under":34})
 try:
  run_phase("train",command("train"))
  run_phase("checkpoint_check",command("check"))
  run_phase("free_action_eval",command("free_eval"))
  write(RUN/"driver_state.json",{"phase":"waiting_paired_epoch1","pid":baseline.pid})
  code=baseline.wait()
  if code:raise RuntimeError("Paired epoch1 baseline failed; see baseline_driver.log")
  run_phase("paired_epoch5",command("simulate","--tag","epoch5"))
  run_phase("motion_epoch5",command("motion","--tag","epoch5"))
  run_phase("preservation",command("audit"))
  final_checks()
 except Exception as error:
  import traceback
  write(RUN/"pipeline_error.json",{"error":repr(error),"traceback":traceback.format_exc()})
  raise

if __name__=="__main__":main()
