"""Find minimum integer epoch on the same five fixed LIBERO states.
Saved checkpoints only. No training or package changes.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,types
from concurrent.futures import ThreadPoolExecutor
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
RUN=Path(__file__).parent
OLD=ROOT/"outputs/five_epoch_bowl/20261103T064207_586519Z"
INPUT=ROOT/"outputs/input_pipeline_check/20261103T055533_073692Z"
DATA=Path(json.loads((OLD/"training_config.json").read_text())["dataset_root"])
os.environ.update(OPENVLA_BOWL40_DATA_ROOT=str(DATA),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
sys.path.insert(0,str(ROOT/"examples"))
import torch
from PIL import Image
import numpy as np
import openvla_bowl40_experiment as exp
from run_openvla_bowl_full import capture_environment

def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def prepare():
 assert json.loads((OLD/"complete.json").read_text())["complete"]
 assert not (RUN/"plan.json").exists(),"Existing search preserved; use fresh directory for full reproduction"
 old_sources=json.loads((OLD/"sources_before.json").read_text())
 assert all(Path(p).exists() and digest(p)==h for p,h in old_sources.items()),"Old evaluation source changed"
 write(RUN/"sources_before.json",{str(p):digest(p) for p in [*ROOT.joinpath("src").rglob("*.py"),*ROOT.joinpath("examples").rglob("*.py")]})
 interpreters={"openvla310":"/home/jovyan/anaconda3/envs/openvla310/bin/python",
 "lerobot312":sys.executable,"eval310":"/home/jovyan/projects/venvs/openvla_eval310/bin/python"}
 with ThreadPoolExecutor(max_workers=3) as pool:
  environment=dict(pool.map(lambda item:capture_environment(RUN,item),interpreters.items()))
 write(RUN/"environment_audit.json",environment)
 for name in interpreters:
  assert sorted((RUN/(name+"_before.txt")).read_text().splitlines())==sorted((OLD/(name+"_after.txt")).read_text().splitlines()),f"Environment changed since prior evaluation: {name}"
 history=json.loads((OLD/"validation_history.json").read_text())
 checkpoints={str(x["epoch"]):x["checkpoint"] for x in history}
 assert sorted(map(int,checkpoints))==[1,2,3,4,5]
 protected={}
 for epoch,path in checkpoints.items():
  cp=Path(path)
  meta=json.loads((cp/"trainer_state.json").read_text())
  assert meta["epoch"]==int(epoch) and meta["optimizer_step"]==1007*int(epoch)
  for p in cp.iterdir():
   if p.is_file():protected[str(p)]=digest(p)
 for p in [DATA/"split_manifest.json",DATA/"action_statistics_train_only.json",
           INPUT/"audit_and_compare.py",INPUT/"matched_rollout_function.py",
           OLD/"simulation_epoch1.json",OLD/"simulation_epoch5.json"]:
  protected[str(p)]=digest(p)
 write(RUN/"protected_before.json",protected)
 write(RUN/"checkpoints.json",checkpoints)
 (RUN/"simulation").mkdir()
 (RUN/"simulation/base").symlink_to(OLD/"simulation/epoch1",target_is_directory=True)
 target=json.loads((OLD/"simulation_epoch5.json").read_text())["successes"]
 assert target==3
 write(RUN/"plan.json",{"primary_criterion":"at least the prior epoch5 performance: 3 successes out of same 5 states",
   "required_successes":target,"secondary_criterion":"at least one success out of same five states",
   "candidate_epochs_in_order":[2,3,4],"known_epoch1_successes":0,"known_epoch5_successes":3,
   "stop_rule":"complete all five trials per candidate; stop higher-epoch evaluation at first candidate with >=3 successes",
   "states":[0,1,2,3,4],"seed":7,"environment_seed":0,"warmup_steps":10,"horizon":220,
   "preprocessing":"same training-matched upright 256 RGB and native HF processor",
   "training_performed":False,"reference_results_reused_from":str(OLD),
   "scope":"minimum among integer epochs 1..5 in this saved training trajectory and fixed five states; not a universal minimum or unseen-state guarantee"})
 print("PREPARED_SEARCH",RUN,flush=True)

def evaluate_epoch(epoch):
 assert epoch in [2,3,4]
 checkpoints=json.loads((RUN/"checkpoints.json").read_text())
 checkpoint=Path(checkpoints[str(epoch)])
 _,val,stats=exp.get_data()
 policy=exp.make_policy(stats);policy.load_saved_adapter(checkpoint)
 before=policy.adapter_hash()
 assert before==(checkpoint/"adapter_sha256.txt").read_text().strip()
 for parameter in policy.model.parameters():parameter.requires_grad_(False)
 policy.eval()
 with torch.no_grad():loss,_=policy(exp.batch_from(policy,val,list(range(4))))
 original_first=json.loads((OLD/f"validation_step_{1007*epoch:06d}.jsonl").read_text().splitlines()[0])
 assert abs(float(loss)-original_first["loss"])<1e-4
 helper=load(INPUT/"audit_and_compare.py",f"epoch{epoch}_native_input")
 image=Image.fromarray(helper.pixels(val[0],policy.config.image_key))
 action=helper.predict_from_image(policy,image,val[0]["task"])
 assert action.shape==(1,7) and bool(torch.isfinite(action).all())
 write(RUN/f"reload_epoch{epoch}.json",{"checkpoint":str(checkpoint),"adapter_hash":before,
    "fresh_python_process":True,"same_validation_batch_loss":float(loss),
    "original_validation_batch_loss":original_first["loss"],"action_shape":[1,7],"all_finite":True})
 policy.select_action=types.MethodType(helper.matched_select_action,policy)
 ns=dict(vars(exp))
 source=(INPUT/"matched_rollout_function.py").read_text()
 exec(compile(source,str(INPUT/"matched_rollout_function.py"),"exec"),ns)
 trials=[]
 for state in range(5):
  folder=RUN/"simulation"/f"epoch{epoch}"/f"state_{state:02d}"
  if (folder/"result.json").exists():trial=json.loads((folder/"result.json").read_text())
  else:trial=ns["simulate_trial"](policy,f"epoch{epoch}",state,types.SimpleNamespace(run=RUN))
  assert trial["trial_executed"],"Technical error, not policy failure: "+repr(trial)
  reference=json.loads((OLD/"simulation/epoch1"/f"state_{state:02d}"/"result.json").read_text())
  assert reference["initial_image_sha256"]==trial["initial_image_sha256"]
  assert reference["selected_init_state_sha256"]==trial["selected_init_state_sha256"]
  trials.append(trial)
  write(RUN/f"epoch{epoch}_progress.json",{"epoch":epoch,"completed_trials":len(trials),
    "successes":sum(t["success"] for t in trials),"attempts_target":5})
 assert policy.adapter_hash()==before
 summary={"epoch":epoch,"checkpoint":str(checkpoint),"attempts":5,
    "successes":sum(t["success"] for t in trials),"success_rate":sum(t["success"] for t in trials)/5,
    "mean_policy_steps":float(np.mean([t["policy_steps"] for t in trials])),"technical_errors":0,
    "states":[0,1,2,3,4],"all_initial_states_match_prior":True,"checkpoint_unchanged":True,
    "trials":trials}
 write(RUN/f"simulation_epoch{epoch}.json",summary)
 print("EPOCH_EVALUATION_COMPLETE",json.dumps(summary),flush=True)

def search():
 prepare()
 known={}
 for epoch in [1,5]:
  d=json.loads((OLD/f"simulation_epoch{epoch}.json").read_text())
  assert d["attempts"]==5 and d["technical_errors"]==0
  known[epoch]={"epoch":epoch,**d,"reused_previous_result":True}
 evaluated=[]
 for epoch in [2,3,4]:
  log=RUN/f"epoch{epoch}.log"
  with log.open("x") as handle:
   process=subprocess.Popen([sys.executable,"-u",str(Path(__file__)),"--epoch",str(epoch)],
       cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
   write(RUN/"search_state.json",{"phase":"evaluate","epoch":epoch,"pid":process.pid,"log":str(log),
       "evaluated_epochs":evaluated,"required_successes":3})
   print("EPOCH_STARTED",epoch,process.pid,flush=True)
   code=process.wait()
  if code:raise RuntimeError(f"Epoch {epoch} evaluator failed with {code}; see {log}")
  known[epoch]=json.loads((RUN/f"simulation_epoch{epoch}.json").read_text());evaluated.append(epoch)
  write(RUN/"intermediate_summary.json",{str(e):v for e,v in sorted(known.items())})
  if known[epoch]["successes"]>=3:break
 first_any=min(e for e,r in known.items() if r["successes"]>=1)
 first_target=min(e for e,r in known.items() if r["successes"]>=3)
 assert all(e in known and known[e]["successes"]==0 for e in range(1,first_any))
 assert all(e in known and known[e]["successes"]<3 for e in range(1,first_target))
 write(RUN/"search_state.json",{"phase":"verify","evaluated_epochs":evaluated,"minimum_any":first_any,"minimum_target":first_target})
 import imageio.v2 as imageio
 videos=[]
 for epoch in evaluated:
  for trial in known[epoch]["trials"]:
   with imageio.get_reader(trial["video"]) as reader:
    frames=reader.count_frames();assert frames==trial["policy_steps"]+1
    reader.get_data(0);reader.get_data(frames-1)
    videos.append({"epoch":epoch,"state":trial["init_state_index"],"frames":frames,"fps":reader.get_meta_data()["fps"],"video":trial["video"]})
 write(RUN/"video_verification.json",videos)
 exp.audit_after(types.SimpleNamespace(run=RUN))
 protected=json.loads((RUN/"protected_before.json").read_text())
 changed=[p for p,h in protected.items() if digest(p)!=h]
 assert not changed
 write(RUN/"protected_preservation.json",{"files_checked":len(protected),"changed_files":changed})
 summary={"minimum_epoch_at_least_one_success":first_any,
   "minimum_epoch_matching_5epoch_performance":first_target,"target_successes":3,"attempts_per_epoch":5,
   "evaluated_new_epochs":evaluated,"not_evaluated_intermediate_epochs":[e for e in [2,3,4] if e not in evaluated],
   "cohorts":{str(e):r for e,r in sorted(known.items())},
   "scope":"fixed five states and this training trajectory; integer checkpoints only, not a universal minimum",
   "training_performed":False,"all_old_checkpoints_preserved":True,"all_initial_states_paired":True}
 write(RUN/"summary.json",summary)
 write(RUN/"core_complete.json",{"complete":True,"new_epoch_evaluations":evaluated,
   "minimum_any":first_any,"minimum_target":first_target,"videos_verified":len(videos),
   "original_sources_environments_checkpoints_preserved":True})
 print("MINIMUM_EPOCH_SEARCH_COMPLETE",json.dumps(summary),flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--epoch",type=int,choices=[2,3,4]);args=parser.parse_args()
 try:
  if args.epoch:evaluate_epoch(args.epoch)
  else:search()
 except Exception as error:
  import traceback
  name=f"epoch{args.epoch}_error.json" if args.epoch else "pipeline_error.json"
  write(RUN/name,{"error":repr(error),"traceback":traceback.format_exc()});raise
if __name__=="__main__":main()
