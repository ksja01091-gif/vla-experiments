"""Ten total epochs: exact resume from epoch5; evaluate every epoch6-10; preserve originals."""
from pathlib import Path
import argparse,contextlib,csv,hashlib,importlib.util,json,math,os,random,subprocess,sys,time,types
from concurrent.futures import ThreadPoolExecutor
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
RUN=Path(__file__).parent
ORIGINAL=ROOT/"outputs/bowl40_experiment/20261103T051914_711739Z"
PREVIOUS=ROOT/"outputs/five_epoch_bowl/20261103T064207_586519Z"
DATA=ORIGINAL/"data"
INPUT=ROOT/"outputs/input_pipeline_check/20261103T055533_073692Z"
VALIDATION=ROOT/"outputs/full_validation_actions/20261103T061517_441641Z"
CAUSE=ROOT/"outputs/error_cause_audit/20261103T062733_387868Z"
os.environ.update(OPENVLA_BOWL40_DATA_ROOT=str(DATA),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",WANDB_MODE="offline")
sys.path.insert(0,str(ROOT/"examples"))
import numpy as np
import torch
from PIL import Image
import openvla_bowl40_experiment as exp
from run_openvla_bowl_full import capture_environment

def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def old_result():return json.loads((PREVIOUS/"training_result.json").read_text())

def prepare():
 sources=[*ROOT.joinpath("src").rglob("*.py"),*ROOT.joinpath("examples").rglob("*.py")]
 write(RUN/"sources_before.json",{str(p):digest(p) for p in sources})
 interpreters={"openvla310":"/home/jovyan/anaconda3/envs/openvla310/bin/python","lerobot312":sys.executable,"eval310":"/home/jovyan/projects/venvs/openvla_eval310/bin/python"}
 with ThreadPoolExecutor(max_workers=3) as pool:env=dict(pool.map(lambda item:capture_environment(RUN,item),interpreters.items()))
 write(RUN/"environment_audit.json",env)
 checkpoint=Path(old_result()["final_checkpoint"])
 protected=[*checkpoint.iterdir(),DATA/"split_manifest.json",DATA/"action_statistics_train_only.json",
            PREVIOUS/"training_config.json",PREVIOUS/"training_result.json"]
 for folder in [DATA/"train",DATA/"validation"]:
  protected.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in [".json",".jsonl",".parquet"])
 write(RUN/"protected_before.json",{str(p):digest(p) for p in protected if p.is_file()})
 (RUN/"simulation").mkdir()
 (RUN/"simulation/base").symlink_to(PREVIOUS/"simulation/epoch5",target_is_directory=True)
 config=json.loads((PREVIOUS/"training_config.json").read_text())
 config.update({"epochs":10,"total_target_epochs":10,"resume_completed_epochs":5,"additional_epochs":5,
    "resume_checkpoint":str(checkpoint),"total_optimizer_steps":10070,"additional_optimizer_steps":5035,
    "total_frames_seen":40250,"additional_frames_seen":20125,
    "validation_every":"each completed additional epoch; no optimizer on validation",
    "evaluation_checkpoint_policy":"every epoch6-10 and final epoch10, no early stopping",
    "simulator_preprocessing":"epochs5-10: upright 256px RGB directly into native HF processor",
    "paired_simulator_states":[0,1,2,3,4],"previous_cropped_1epoch_5trials_not_reused_for_primary_comparison":True,
    "concurrent_work":"one rollout evaluator may overlap training; wall time not an isolated speed benchmark",
    "validation_statistics":"fixed original training20 calibration, same as epoch5 checkpoint",
    "learning_changes_only":"five more passes; same data, LR, optimizer state, LoRA, normalization and training image processing"})
 write(RUN/"training_config.json",config)
 write(RUN/"reproducibility_sources.json",{str(p):digest(p) for p in [Path(__file__),ROOT/"examples/openvla_bowl40_experiment.py",INPUT/"audit_and_compare.py",INPUT/"matched_rollout_function.py"]})
 print("PREPARED",json.dumps(config),flush=True)

def train():
 assert not (RUN/"training_metrics.jsonl").exists(),"Partial training preserved; use a new run or explicit resume"
 train,val,stats=exp.get_data();policy=exp.make_policy(stats)
 checkpoint=Path(old_result()["final_checkpoint"])
 policy.load_saved_adapter(checkpoint)
 initial_hash=policy.adapter_hash()
 state=torch.load(checkpoint/"trainer_state.pt",map_location="cpu",weights_only=False)
 assert state["epoch"]==5 and state["optimizer_step"]==5035 and state["seen_frames"]==20125
 names=[n for n,p in policy.model.named_parameters() if p.requires_grad]
 saved_names=json.loads((ORIGINAL/"calibration.json").read_text())["parameter_names"]
 assert names==saved_names
 optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=0.0)
 optimizer.load_state_dict(state["optimizer"])
 assert {float(v["step"]) for v in optimizer.state.values()}=={5035.0}
 policy.eval()
 with torch.no_grad():loss,metrics=policy(exp.batch_from(policy,val,list(range(4))))
 old_first=json.loads((PREVIOUS/"validation_step_005035.jsonl").read_text().splitlines()[0])
 assert abs(float(loss)-old_first["loss"])<1e-4
 write(RUN/"resume_check.json",{"epoch":5,"optimizer_step":5035,"seen_frames":20125,"adapter_hash":initial_hash,
    "optimizer_parameter_order_exact":True,"optimizer_moments_loaded":True,"same_validation_batch_loss":float(loss),
    "old_validation_batch_loss":old_first["loss"],"new_training_not_started_yet":True})
 frozen=exp.frozen_sample_hash(policy)
 wb=None
 try:
  import wandb
  wb=wandb.init(project="openvla-libero-bowl-baseline",name="ten-total-epochs-"+RUN.name,mode="offline",dir=str(RUN),config=json.loads((RUN/"training_config.json").read_text()))
  wb.define_metric("optimizer_step");wb.define_metric("train/*",step_metric="optimizer_step");wb.define_metric("validation/*",step_metric="optimizer_step")
  write(RUN/"wandb_status.json",{"mode":"offline","uploaded":False,"run_id":wb.id,"directory":wb.dir})
 except Exception as error:write(RUN/"wandb_status.json",{"mode":"local_logs_only","uploaded":False,"error":repr(error)})
 torch.set_rng_state(state["torch_rng_state"])
 torch.cuda.set_rng_state_all(state["cuda_rng_state"])
 np.random.set_state(state["numpy_rng_state"]);random.setstate(state["python_rng_state"])
 step=5035;seen=20125;history=[{"epoch":5,**old_result()["final_validation"]}]
 coverage=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 try:
  for epoch in range(6,11):
   order=torch.randperm(len(train),generator=torch.Generator().manual_seed(7+epoch-1)).tolist()
   epoch_seen=[];loss_sum=0.0;epoch_started=time.monotonic()
   for offset in range(0,len(order),4):
    indices=order[offset:offset+4]
    policy.train();policy.model.vision_backbone.eval();optimizer.zero_grad(set_to_none=True)
    tick=time.monotonic()
    loss,metrics=policy(exp.batch_from(policy,train,indices))
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(policy.get_optim_params(),1.0)
    assert torch.isfinite(norm)
    optimizer.step();step+=1;seen+=len(indices);epoch_seen.extend(indices);loss_sum+=metrics["loss"]*len(indices)
    row={"epoch":epoch,"optimizer_step":step,"frames_seen":seen,"batch_frames":len(indices),"indices":indices,
      **metrics,"gradient_norm":float(norm),"iteration_seconds":time.monotonic()-tick,
      "elapsed_additional_training_seconds":time.monotonic()-started,**exp.memory()}
    exp.append_json(RUN/"training_metrics.jsonl",row)
    if wb:wb.log({"optimizer_step":step,**{"train/"+k:v for k,v in row.items() if k!="indices"}})
    if step%25==0 or offset+len(indices)==len(order):
     write(RUN/"training_progress.json",{"epoch":epoch,"target_epoch":10,"optimizer_step":step,"target_step":10070,
       "frames_this_epoch":len(epoch_seen),"epoch_frames":4025,"additional_training_seconds":time.monotonic()-started,
       "last_loss":metrics["loss"],**exp.memory()})
     print("TRAIN_PROGRESS",epoch,step,len(epoch_seen),"/4025","loss",metrics["loss"],flush=True)
    del loss
   assert sorted(epoch_seen)==list(range(4025))
   optimizer.zero_grad(set_to_none=True)
   validation=exp.evaluate(policy,val,4,RUN/f"validation_step_{step:06d}.jsonl",step)
   cp=exp.save_checkpoint(policy,optimizer,RUN,step,epoch,seen)
   record={"epoch":epoch,"training_loss_frame_weighted":loss_sum/4025,"epoch_seconds":time.monotonic()-epoch_started,
           **validation,"checkpoint":str(cp)}
   history.append(record);coverage.append({"epoch":epoch,"frames":len(epoch_seen),"unique_frames":len(set(epoch_seen)),"indices":epoch_seen})
   write(RUN/"validation_history.json",history);write(RUN/"training_coverage.json",coverage)
   exp.append_json(RUN/"validation_history.jsonl",record)
   if wb:wb.log({"optimizer_step":step,**{"validation/"+k:v for k,v in record.items() if isinstance(v,(int,float))}})
   write(RUN/f"epoch_{epoch:02d}_ready.json",record)
   print("EPOCH_COMPLETE",json.dumps(record),flush=True)
  assert step==10070 and seen==40250 and policy.adapter_hash()!=initial_hash
  assert exp.frozen_sample_hash(policy)==frozen
  last=history[-1];best=min(history,key=lambda x:x["loss"])
  result={"training_completed":True,"epochs_completed":10,"additional_epochs_completed":5,
    "frames_seen":seen,"additional_frames_seen":20125,"unique_train_frames_per_epoch":4025,"optimizer_steps":step,
    "selected_checkpoint":last["checkpoint"],"final_checkpoint":last["checkpoint"],"selected_validation":last,
    "final_validation":last,"baseline_validation":history[0],"best_teacher_validation_checkpoint":best["checkpoint"],
    "best_teacher_validation_epoch":best["epoch"],"primary_evaluation_is_final_epoch10":True,
    "adapter_changed":True,"frozen_parameter_samples_unchanged":True,"validation_uses_optimizer":False,
    "full_validation_frames":462,"additional_training_wall_seconds":time.monotonic()-started,**exp.memory()}
  write(RUN/"training_result.json",result)
  print("TEN_EPOCH_TRAINING_COMPLETE",json.dumps(result),flush=True)
 finally:
  if wb:wb.finish()

def audit():
 exp.audit_after(types.SimpleNamespace(run=RUN))
 protected=json.loads((RUN/"protected_before.json").read_text())
 changed=[p for p,h in protected.items() if digest(p)!=h]
 assert not changed
 write(RUN/"protected_preservation.json",{"files_checked":len(protected),"changed_files":changed})


def check():
 exp.check_checkpoint(types.SimpleNamespace(run=RUN,rank=8,batch_size=4))
 import gc
 gc.collect();torch.cuda.empty_cache()
 benchmark_final()

def benchmark_final():
 helper=load(INPUT/"audit_and_compare.py","ten_epoch_latency")
 _,val,stats=exp.get_data();policy=exp.make_policy(stats)
 checkpoint=json.loads((RUN/"training_result.json").read_text())["final_checkpoint"]
 policy.load_saved_adapter(checkpoint)
 for p in policy.model.parameters():p.requires_grad_(False)
 policy.eval();policy.select_action=types.MethodType(helper.matched_select_action,policy)
 indices=list(np.linspace(0,len(val)-1,32,dtype=int));rows=[]
 def predict(index):
  sample=val[int(index)];pixels=np.rint(sample[policy.config.image_key].permute(1,2,0).numpy()*255).astype(np.uint8)
  tensor=torch.from_numpy(pixels.copy()).permute(2,0,1).float()/255
  return policy.select_action({policy.config.image_key:tensor.unsqueeze(0),"task":[sample["task"]]})
 for index in indices[:3]:predict(index)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 for index in indices:
  torch.cuda.synchronize();tick=time.perf_counter()
  action=predict(index);torch.cuda.synchronize()
  assert action.shape==(1,7) and bool(torch.isfinite(action).all())
  rows.append({"validation_index":int(index),"seconds":time.perf_counter()-tick,"shape":[1,7],
               "finite":True,"action":action.cpu().tolist()})
 write(RUN/"epoch10_inference_benchmark.json",{"epoch":10,"checkpoint":checkpoint,"samples":32,"warmup":3,
    "native_256px_preprocessing":True,"simulation_not_running_during_measurement":True,
    "mean_action_seconds":float(np.mean([r["seconds"] for r in rows])),
    "median_action_seconds":float(np.median([r["seconds"] for r in rows])),
    "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
    "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"rows":rows})
 print("EPOCH10_INFERENCE_BENCHMARK_COMPLETE",flush=True)

def simulate_all():
 helper=load(INPUT/"audit_and_compare.py","ten_epoch_input")
 ns=dict(vars(exp));source=(INPUT/"matched_rollout_function.py").read_text()
 exec(compile(source,str(INPUT/"matched_rollout_function.py"),"exec"),ns)
 _,_,stats=exp.get_data()
 policy=None
 for epoch in range(6,11):
  ready=RUN/f"epoch_{epoch:02d}_ready.json"
  while not ready.exists():
   if (RUN/"train_error.json").exists():raise RuntimeError("Training failed; saved results preserved.")
   time.sleep(5)
  record=json.loads(ready.read_text())
  if policy is None:policy=exp.make_policy(stats)
  # Re-enable adapter parameters only for loading; then freeze all for inference.
  policy.load_saved_adapter(record["checkpoint"])
  for parameter in policy.model.parameters():parameter.requires_grad_(False)
  policy.eval();policy.select_action=types.MethodType(helper.matched_select_action,policy)
  before=policy.adapter_hash();trials=[]
  for state in range(5):
   folder=RUN/"simulation"/f"epoch{epoch}"/f"state_{state:02d}"
   if (folder/"result.json").exists():trial=json.loads((folder/"result.json").read_text())
   else:trial=ns["simulate_trial"](policy,f"epoch{epoch}",state,types.SimpleNamespace(run=RUN))
   assert trial["trial_executed"],trial
   baseline=json.loads((PREVIOUS/"simulation/epoch5"/f"state_{state:02d}"/"result.json").read_text())
   for key in ["initial_image_sha256","selected_init_state_sha256","init_state_file_sha256","reference_sha256","simulator_versions","max_policy_steps","warmup_steps"]:
    assert trial[key]==baseline[key],(key,trial[key],baseline[key])
   trials.append(trial)
   write(RUN/"simulation_progress.json",{"epoch":epoch,"completed_trials":len(trials),"total_trials_this_epoch":5,"successes":sum(t["success"] for t in trials)})
  assert policy.adapter_hash()==before
  summary={"epoch":epoch,"checkpoint":record["checkpoint"],"attempts":5,"successes":sum(t["success"] for t in trials),
   "success_rate":sum(t["success"] for t in trials)/5,"mean_policy_steps":float(np.mean([t["policy_steps"] for t in trials])),
   "states":[0,1,2,3,4],"technical_errors":0,"trials":trials,"initial_conditions_match_epoch5":True}
  write(RUN/f"simulation_epoch{epoch}.json",summary)
  print("EPOCH_SIM_COMPLETE",epoch,summary["successes"],"/5",flush=True)
 write(RUN/"simulation_complete.json",{"complete":True,"epochs":[6,7,8,9,10],"new_trials":25})

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("phase",choices=["prepare","train","check","simulate_all","audit"])
 args=parser.parse_args()
 try:globals()[args.phase]()
 except Exception as error:
  import traceback
  write(RUN/f"{args.phase}_error.json",{"error":repr(error),"traceback":traceback.format_exc()});raise
if __name__=="__main__":main()
