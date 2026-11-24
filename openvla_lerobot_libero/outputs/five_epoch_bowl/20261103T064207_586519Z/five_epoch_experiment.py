"""Five total epochs: resume epoch1 for four passes, preserve originals, paired native-input evaluation."""
from pathlib import Path
import argparse,contextlib,csv,hashlib,importlib.util,json,math,os,random,subprocess,sys,time,types
from concurrent.futures import ThreadPoolExecutor
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
RUN=Path(__file__).parent
PREVIOUS=ROOT/"outputs/bowl40_experiment/20261103T051914_711739Z"
DATA=PREVIOUS/"data"
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

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
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
 (RUN/"simulation/base").symlink_to(PREVIOUS/"simulation/base",target_is_directory=True)
 config=json.loads((PREVIOUS/"training_config.json").read_text())
 config.update({"epochs":5,"total_target_epochs":5,"resume_completed_epochs":1,"additional_epochs":4,
    "resume_checkpoint":str(checkpoint),"total_optimizer_steps":5035,"additional_optimizer_steps":4028,
    "total_frames_seen":20125,"additional_frames_seen":16100,
    "validation_every":"each completed additional epoch; no optimizer on validation",
    "evaluation_checkpoint_policy":"epoch5 final checkpoint for primary comparison, not automatic best",
    "simulator_preprocessing":"both epoch1 and epoch5: upright 256px RGB directly into native HF processor",
    "paired_simulator_states":[0,1,2,3,4],"previous_cropped_1epoch_5trials_not_reused_for_primary_comparison":True,
    "concurrent_work":"epoch1 paired baseline rollouts overlap resumed training on the same GPU; affects wall time, not simulated control steps",
    "validation_statistics":"fixed original training20 calibration, same as epoch1 checkpoint",
    "learning_changes_only":"four more passes; same data, LR, optimizer state, LoRA, normalization and training image processing"})
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
 assert state["epoch"]==1 and state["optimizer_step"]==1007 and state["seen_frames"]==4025
 names=[n for n,p in policy.model.named_parameters() if p.requires_grad]
 saved_names=json.loads((PREVIOUS/"calibration.json").read_text())["parameter_names"]
 assert names==saved_names
 optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=0.0)
 optimizer.load_state_dict(state["optimizer"])
 assert {float(v["step"]) for v in optimizer.state.values()}=={1007.0}
 policy.eval()
 with torch.no_grad():loss,metrics=policy(exp.batch_from(policy,val,list(range(4))))
 old_first=json.loads((PREVIOUS/"validation_step_001007.jsonl").read_text().splitlines()[0])
 assert abs(float(loss)-old_first["loss"])<1e-4
 write(RUN/"resume_check.json",{"epoch":1,"optimizer_step":1007,"seen_frames":4025,"adapter_hash":initial_hash,
    "optimizer_parameter_order_exact":True,"optimizer_moments_loaded":True,"same_validation_batch_loss":float(loss),
    "old_validation_batch_loss":old_first["loss"],"new_training_not_started_yet":True})
 frozen=exp.frozen_sample_hash(policy)
 wb=None
 try:
  import wandb
  wb=wandb.init(project="openvla-libero-bowl-baseline",name="five-total-epochs-"+RUN.name,mode="offline",dir=str(RUN),config=json.loads((RUN/"training_config.json").read_text()))
  wb.define_metric("optimizer_step");wb.define_metric("train/*",step_metric="optimizer_step");wb.define_metric("validation/*",step_metric="optimizer_step")
  write(RUN/"wandb_status.json",{"mode":"offline","uploaded":False,"run_id":wb.id,"directory":wb.dir})
 except Exception as error:write(RUN/"wandb_status.json",{"mode":"local_logs_only","uploaded":False,"error":repr(error)})
 torch.set_rng_state(state["torch_rng_state"])
 torch.cuda.set_rng_state_all(state["cuda_rng_state"])
 np.random.set_state(state["numpy_rng_state"]);random.setstate(state["python_rng_state"])
 step=1007;seen=4025;history=[{"epoch":1,**old_result()["final_validation"]}]
 coverage=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 try:
  for epoch in range(2,6):
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
     write(RUN/"training_progress.json",{"epoch":epoch,"target_epoch":5,"optimizer_step":step,"target_step":5035,
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
   print("EPOCH_COMPLETE",json.dumps(record),flush=True)
  assert step==5035 and seen==20125 and policy.adapter_hash()!=initial_hash
  assert exp.frozen_sample_hash(policy)==frozen
  last=history[-1];best=min(history,key=lambda x:x["loss"])
  result={"training_completed":True,"epochs_completed":5,"additional_epochs_completed":4,
    "frames_seen":seen,"additional_frames_seen":16100,"unique_train_frames_per_epoch":4025,"optimizer_steps":step,
    "selected_checkpoint":last["checkpoint"],"final_checkpoint":last["checkpoint"],"selected_validation":last,
    "final_validation":last,"baseline_validation":history[0],"best_teacher_validation_checkpoint":best["checkpoint"],
    "best_teacher_validation_epoch":best["epoch"],"primary_evaluation_is_final_epoch5":True,
    "adapter_changed":True,"frozen_parameter_samples_unchanged":True,"validation_uses_optimizer":False,
    "full_validation_frames":462,"additional_training_wall_seconds":time.monotonic()-started,**exp.memory()}
  write(RUN/"training_result.json",result)
  print("FIVE_EPOCH_TRAINING_COMPLETE",json.dumps(result),flush=True)
 finally:
  if wb:wb.finish()

def check():
 exp.check_checkpoint(types.SimpleNamespace(run=RUN,rank=8,batch_size=4))

def free_eval():
 helper=load(INPUT/"audit_and_compare.py","five_epoch_input")
 old_audit=load(VALIDATION/"validate_free_actions.py","five_epoch_val")
 cause=load(CAUSE/"investigate_errors.py","five_epoch_cause")
 training,validation,stats=exp.get_data()
 result=json.loads((RUN/"training_result.json").read_text())
 policy=exp.make_policy(stats);policy.load_saved_adapter(result["final_checkpoint"])
 for p in policy.model.parameters():p.requires_grad_(False)
 policy.eval();before=policy.adapter_hash()
 owner=policy.model.predict_action.__self__;original=owner.generate;had="generate" in owner.__dict__;saved=owner.__dict__.get("generate");captured=[]
 def capture(*args,**kwargs):
  out=original(*args,**kwargs);captured.append(out[0,-7:].detach().cpu().numpy().copy());return out
 owner.generate=capture
 plan=json.loads((CAUSE/"sampling_plan.json").read_text())
 summaries={}
 try:
  for name,data,indices in [("validation",validation,list(range(462))),("train_sample",training,plan["training_indices"])]:
   rows=[];started=time.monotonic()
   with (RUN/f"{name}_free_predictions.jsonl").open("x") as f:
    for count,index in enumerate(indices,1):
     sample=data[index];image=Image.fromarray(helper.pixels(sample,policy.config.image_key))
     prepared=policy.prepare_training_batch(sample);inp=helper.matched_inputs(policy,image,sample["task"])
     assert torch.equal(prepared["pixel_values"],inp["pixel_values"])
     ids=inp["input_ids"]
     if int(ids[0,-1])!=29871:ids=torch.cat([ids,torch.tensor([[29871]],dtype=ids.dtype)],dim=1)
     assert torch.equal(prepared["input_ids"][:,:-8],ids)
     captured.clear();action=helper.predict_from_image(policy,image,sample["task"])[0].cpu().numpy()
     assert len(captured)==1 and action.shape==(7,) and np.isfinite(action).all()
     tokens=captured[0];truth=sample["action"].numpy()
     decoded=policy.action_tokenizer.decode_token_ids_to_actions(tokens);gold=prepared["labels"][0,-8:-1].numpy()
     errors=np.abs(decoded-policy.normalize_action(truth))
     row={"validation_index":index,"episode_index":int(sample["episode_index"]),"source_step_index":int(sample["source_step_index"].reshape(-1)[0]),
        "true_action":truth.tolist(),"predicted_action":action.tolist(),"true_action_token_ids":gold.tolist(),"predicted_action_token_ids":tokens.tolist(),
        "action_token_correct":int((tokens==gold).sum()),"token_correct_per_dimension":(tokens==gold).tolist(),
        "normalized_action_l1":float(errors.mean()),"normalized_action_l1_per_dimension":errors.tolist(),
        "true_normalized_action":policy.normalize_action(truth).tolist(),"predicted_normalized_action":decoded.tolist(),
        "valid_generated_action_tokens":int(((tokens>policy.action_tokenizer.action_token_begin_idx)&(tokens<policy.processor.tokenizer.vocab_size)).sum()),
        "true_gripper_class":old_audit.classify_gripper(float(truth[-1])),"predicted_gripper_class":old_audit.classify_gripper(float(action[-1])),
        "true_arm_max_abs":float(np.abs(truth[:6]).max()),"predicted_arm_max_abs":float(np.abs(action[:6]).max()),
        "image_and_prompt_match_training":True}
     rows.append(row);f.write(json.dumps(row)+"\n");f.flush()
     if count%25==0 or count==len(indices):
      write(RUN/"free_eval_progress.json",{"dataset":name,"completed":count,"total":len(indices),"seconds":time.monotonic()-started})
      print("FREE_EVAL",name,count,"/",len(indices),flush=True)
   summary=old_audit.summarize(rows,result["final_validation"]) if name=="validation" else cause.metrics(rows)
   summary["checkpoint"]=result["final_checkpoint"];summary["epoch"]=5
   write(RUN/f"{name}_free_summary.json",summary);summaries[name]=summary
 finally:
  if had:owner.generate=saved
  else:delattr(owner,"generate")
 assert policy.adapter_hash()==before
 write(RUN/"free_eval_complete.json",{"complete":True,"validation_frames":462,"train_sample_frames":480,"checkpoint_unchanged":True})
 print("FREE_EVAL_COMPLETE",json.dumps(summaries),flush=True)

def simulate(tag):
 assert tag in ["epoch1","epoch5"]
 helper=load(INPUT/"audit_and_compare.py","five_epoch_sim_input")
 _,_,stats=exp.get_data();policy=exp.make_policy(stats)
 checkpoint=old_result()["final_checkpoint"] if tag=="epoch1" else json.loads((RUN/"training_result.json").read_text())["final_checkpoint"]
 policy.load_saved_adapter(checkpoint)
 for p in policy.model.parameters():p.requires_grad_(False)
 policy.eval();before=policy.adapter_hash()
 policy.select_action=types.MethodType(helper.matched_select_action,policy)
 ns=dict(vars(exp))
 source=(INPUT/"matched_rollout_function.py").read_text()
 exec(compile(source,str(INPUT/"matched_rollout_function.py"),"exec"),ns)
 trials=[]
 for state in range(5):
  folder=RUN/"simulation"/tag/f"state_{state:02d}"
  if (folder/"result.json").exists():trial=json.loads((folder/"result.json").read_text())
  else:trial=ns["simulate_trial"](policy,tag,state,types.SimpleNamespace(run=RUN))
  assert trial["trial_executed"],trial
  if tag=="epoch5":
   baseline=json.loads((RUN/"simulation/epoch1"/f"state_{state:02d}"/"result.json").read_text())
   assert trial["initial_image_sha256"]==baseline["initial_image_sha256"]
   assert trial["selected_init_state_sha256"]==baseline["selected_init_state_sha256"]
  trials.append(trial)
  write(RUN/f"{tag}_simulation_progress.json",{"completed_trials":len(trials),"successes":sum(t["success"] for t in trials),"target_trials":5})
 assert policy.adapter_hash()==before
 summary={"model_tag":tag,"checkpoint":checkpoint,"attempts":5,"successes":sum(t["success"] for t in trials),
   "success_rate":sum(t["success"] for t in trials)/5,"mean_policy_steps":float(np.mean([t["policy_steps"] for t in trials])),
   "technical_errors":0,"states":[0,1,2,3,4],"preprocessing":"training matched native 256 RGB","trials":trials}
 write(RUN/f"simulation_{tag}.json",summary)
 print("SIMULATION_COMPLETE",json.dumps(summary),flush=True)

def audit():
 exp.audit_after(types.SimpleNamespace(run=RUN))
 protected=json.loads((RUN/"protected_before.json").read_text())
 changed=[p for p,h in protected.items() if digest(p)!=h]
 assert not changed
 write(RUN/"protected_preservation.json",{"files_checked":len(protected),"changed_files":changed})

def motion(tag):
 env=os.environ.copy();env["PYTHONPATH"]="/home/jovyan/projects/LIBERO";env["MUJOCO_GL"]="egl"
 subprocess.run(["/home/jovyan/anaconda3/envs/openvla310/bin/python","-u",str(RUN/"motion_worker.py"),tag],env=env,check=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("phase",choices=["prepare","train","check","free_eval","simulate","audit","motion"])
 parser.add_argument("--tag",choices=["epoch1","epoch5"],default="epoch5")
 args=parser.parse_args()
 try:
  if args.phase in ["simulate","motion"]:globals()[args.phase](args.tag)
  else:globals()[args.phase]()
 except Exception as error:
  import traceback
  write(RUN/f"{args.phase}_{args.tag}_error.json",{"error":repr(error),"traceback":traceback.format_exc()});raise

if __name__=="__main__":main()
