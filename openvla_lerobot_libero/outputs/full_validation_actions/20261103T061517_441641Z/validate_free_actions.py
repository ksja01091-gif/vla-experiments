"""Free-generation action audit on all 462 held-out LIBERO frames.
No training, simulator rollout, package changes, or upload.
"""
from pathlib import Path
import csv,hashlib,importlib.util,json,os,subprocess,sys,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
RUN=Path(__file__).parent
PREVIOUS=ROOT/"outputs/bowl40_experiment/20261103T051914_711739Z"
INPUT_AUDIT=ROOT/"outputs/input_pipeline_check/20261103T055533_073692Z"
os.environ.update(OPENVLA_BOWL40_DATA_ROOT=str(PREVIOUS/"data"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
sys.path.insert(0,str(ROOT/"examples"))
import numpy as np
import torch
from PIL import Image
import openvla_bowl40_experiment as experiment
from run_openvla_bowl_full import capture_environment

def write(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def classify_gripper(value):
 return "close" if value<0.5 else ("open" if value>0.5 else "neutral")

def summarize(rows,teacher):
 correct=sum(r["action_token_correct"] for r in rows)
 frame_l1=np.array([r["normalized_action_l1_per_dimension"] for r in rows])
 token_by_dim=np.array([r["token_correct_per_dimension"] for r in rows],dtype=np.float64)
 confusion={truth:{pred:0 for pred in ["close","open","neutral"]} for truth in ["close","open"]}
 for r in rows:confusion[r["true_gripper_class"]][r["predicted_gripper_class"]]+=1
 arms=np.array([r["predicted_arm_max_abs"] for r in rows])
 gold=np.array([r["true_arm_max_abs"] for r in rows])
 thresholds=[]
 for active in [0.05,0.1,0.2]:
  for near in [0.005,0.01,0.05]:
   eligible=gold>=active
   stopped=(arms<=near)&eligible
   thresholds.append({"expert_active_threshold":active,"predicted_near_zero_threshold":near,
      "expert_active_frames":int(eligible.sum()),"active_but_predicted_near_zero_frames":int(stopped.sum()),
      "fraction":float(stopped.sum()/eligible.sum()) if eligible.any() else None})
 main=next(x for x in thresholds if x["expert_active_threshold"]==0.1 and x["predicted_near_zero_threshold"]==0.01)
 action_counts=Counter(tuple(r["predicted_action_token_ids"]) for r in rows)
 mode,frequency=action_counts.most_common(1)[0]
 per_episode=[]
 for episode in sorted({r["episode_index"] for r in rows}):
  part=[r for r in rows if r["episode_index"]==episode]
  per_episode.append({"episode_index":episode,"frames":len(part),
     "action_token_accuracy":sum(r["action_token_correct"] for r in part)/(7*len(part)),
     "normalized_action_l1":float(np.mean([r["normalized_action_l1"] for r in part])),
     "gripper_correct":sum(r["true_gripper_class"]==r["predicted_gripper_class"] for r in part)})
 return {"frames":len(rows),"episodes":len(per_episode),"all_frames_exactly_once":[r["validation_index"] for r in rows]==list(range(462)),
   "training_performed":False,"simulator_rollouts_performed":False,
   "free_generation_action_token_accuracy":correct/(7*len(rows)),
   "free_generation_normalized_action_l1":float(frame_l1.mean()),
   "free_generation_l1_per_dimension":frame_l1.mean(axis=0).tolist(),
   "free_generation_token_accuracy_per_dimension":token_by_dim.mean(axis=0).tolist(),
   "valid_generated_action_token_fraction":sum(r["valid_generated_action_tokens"] for r in rows)/(7*len(rows)),
   "gripper_confusion":confusion,"gripper_correct_frames":sum(r["true_gripper_class"]==r["predicted_gripper_class"] for r in rows),
   "gripper_accuracy":sum(r["true_gripper_class"]==r["predicted_gripper_class"] for r in rows)/len(rows),
   "arm_near_zero_frames":int((arms<=0.01).sum()),
   "expert_active_but_predicted_near_zero":main,"motion_threshold_sensitivity":thresholds,
   "dominant_predicted_action_token_ids":list(mode),"dominant_predicted_action_frames":frequency,
   "unique_predicted_action_token_sequences":len(action_counts),
   "reused_teacher_forced_validation":teacher,"teacher_metrics_source":str(PREVIOUS/"training_result.json"),
   "per_episode":per_episode,
   "limitations":["held-out demonstrations but their images are correlated within five episodes",
      "free-generation imitation metrics, not a simulator success rate",
      "near-zero controller values are a diagnostic threshold, not measured physical immobility",
      "teacher forcing reveals previous correct action tokens; free generation uses its own previous token predictions"]}

def main():
 started=time.monotonic()
 write(RUN/"sources_before.json",{str(p):digest(p) for p in [*ROOT.joinpath("src").rglob("*.py"),*ROOT.joinpath("examples").rglob("*.py")]})
 interpreters={"openvla310":"/home/jovyan/anaconda3/envs/openvla310/bin/python",
               "lerobot312":sys.executable,"eval310":"/home/jovyan/projects/venvs/openvla_eval310/bin/python"}
 with ThreadPoolExecutor(max_workers=3) as pool:
  env=dict(pool.map(lambda item:capture_environment(RUN,item),interpreters.items()))
 write(RUN/"environment_audit.json",env)
 spec=importlib.util.spec_from_file_location("verified_input_audit",INPUT_AUDIT/"audit_and_compare.py")
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 _,validation,stats=experiment.get_data()
 assert len(validation)==462 and validation.num_episodes==5
 previous=json.loads((PREVIOUS/"training_result.json").read_text())
 checkpoint=Path(previous["selected_checkpoint"])
 checkpoint_hashes={str(p):digest(p) for p in checkpoint.iterdir() if p.is_file() and p.suffix in [".json",".safetensors",".txt"]}
 write(RUN/"checkpoint_before.json",checkpoint_hashes)
 policy=experiment.make_policy(stats)
 policy.load_saved_adapter(checkpoint)
 for parameter in policy.model.parameters():parameter.requires_grad_(False)
 policy.eval()
 adapter_before=policy.adapter_hash()
 write(RUN/"config.json",{"checkpoint":str(checkpoint),"adapter_hash":adapter_before,
    "base_model":"openvla/openvla-7b","train_episodes_for_checkpoint":40,"validation_episodes":5,"validation_frames":462,
    "preprocessing":"source RGB 256 directly to native HF processor; exact match to training",
    "generation":"original cached predict_action; do_sample=False; all seven action tokens freely generated",
    "normalization":"fixed original train20 statistics, identical to checkpoint training",
    "diagnostic_motion_threshold":{"expert_active_max_abs_arm_command_at_least":0.1,
                                    "predicted_near_zero_max_abs_arm_command_at_most":0.01,
                                    "arm_dimensions":[0,1,2,3,4,5],
                                    "gripper_excluded":True},
    "generation_hook":"Records generated token IDs and returns the original generated tensor unchanged",
    "training_performed":False,"simulator_rollouts_performed":False})
 # Observe the original predictor's .generate output without changing generated results.
 owner=policy.model.predict_action.__self__
 original_generate=owner.generate
 had_instance_generate="generate" in owner.__dict__
 old_instance_generate=owner.__dict__.get("generate")
 captured=[]
 def capture_generate(*args,**kwargs):
  output=original_generate(*args,**kwargs)
  assert isinstance(output,torch.Tensor)
  captured.append(output[0,-7:].detach().cpu().numpy().copy())
  return output
 owner.generate=capture_generate
 rows=[];timings=[]
 try:
  with (RUN/"frame_predictions.jsonl").open("x") as log:
   for index in range(len(validation)):
    sample=validation[index]
    image=Image.fromarray(helper.pixels(sample,policy.config.image_key))
    prepared=policy.prepare_training_batch(sample)
    infer=helper.matched_inputs(policy,image,sample["task"])
    assert torch.equal(prepared["pixel_values"],infer["pixel_values"])
    ids=infer["input_ids"]
    if int(ids[0,-1])!=29871:ids=torch.cat([ids,torch.tensor([[29871]],dtype=ids.dtype)],dim=1)
    assert torch.equal(prepared["input_ids"][:,:-8],ids)
    true_tokens=prepared["labels"][0,-8:-1].numpy()
    truth=sample["action"].numpy()
    normalized_truth=policy.normalize_action(truth)
    captured.clear();torch.cuda.synchronize();tick=time.monotonic()
    prediction=helper.predict_from_image(policy,image,sample["task"])
    torch.cuda.synchronize();seconds=time.monotonic()-tick
    timings.append(seconds)
    assert len(captured)==1
    predicted_tokens=captured[0]
    decoded=policy.action_tokenizer.decode_token_ids_to_actions(predicted_tokens)
    action=prediction[0].cpu().numpy()
    mask=np.asarray(policy.action_statistics["mask"],dtype=bool)
    low=np.asarray(policy.action_statistics["q01"]);high=np.asarray(policy.action_statistics["q99"])
    restored=np.where(mask,0.5*(decoded+1)*(high-low)+low,decoded)
    assert np.allclose(restored,action,atol=1e-6,rtol=1e-6)
    errors=np.abs(decoded-normalized_truth)
    token_correct=(predicted_tokens==true_tokens)
    assert np.isfinite(action).all() and action.shape==(7,)
    assert truth[-1] in [0,1]
    row={"validation_index":index,"episode_index":int(sample["episode_index"]),
       "source_step_index":int(sample["source_step_index"].reshape(-1)[0]),
       "true_action":truth.tolist(),"predicted_action":action.tolist(),
       "true_normalized_action":normalized_truth.tolist(),"predicted_normalized_action":decoded.tolist(),
       "true_action_token_ids":true_tokens.tolist(),"predicted_action_token_ids":predicted_tokens.tolist(),
       "action_token_correct":int(token_correct.sum()),"token_correct_per_dimension":token_correct.tolist(),
       "valid_generated_action_tokens":int(((predicted_tokens>policy.action_tokenizer.action_token_begin_idx)&(predicted_tokens<policy.processor.tokenizer.vocab_size)).sum()),
       "normalized_action_l1":float(errors.mean()),"normalized_action_l1_per_dimension":errors.tolist(),
       "true_gripper_class":classify_gripper(float(truth[-1])),
       "predicted_gripper_class":classify_gripper(float(action[-1])),
       "true_arm_max_abs":float(np.abs(truth[:6]).max()),"predicted_arm_max_abs":float(np.abs(action[:6]).max()),
       "true_translation_max_abs":float(np.abs(truth[:3]).max()),"predicted_translation_max_abs":float(np.abs(action[:3]).max()),
       "image_and_prompt_match_training":True,"prediction_seconds":seconds}
    rows.append(row);log.write(json.dumps(row,allow_nan=False)+"\n");log.flush()
    if index==9:
     estimate={"completed":10,"mean_seconds_per_frame_after_warmup":float(np.mean(timings[2:])),
               "estimated_remaining_minutes":float(np.mean(timings[2:])*(462-10)/60),
               "gpu_peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3}
     write(RUN/"timing_estimate.json",estimate)
     print("MEASURED_ESTIMATE",json.dumps(estimate),flush=True)
    if (index+1)%25==0 or index==461:
     write(RUN/"progress.json",{"completed":index+1,"total":462,"elapsed_seconds":time.monotonic()-started,
         "gripper_errors_so_far":sum(r["true_gripper_class"]!=r["predicted_gripper_class"] for r in rows)})
     print("PROGRESS",index+1,"/462","seconds",round(time.monotonic()-started,1),flush=True)
 finally:
  if had_instance_generate:owner.generate=old_instance_generate
  else:delattr(owner,"generate")
 assert policy.adapter_hash()==adapter_before
 assert all(digest(p)==h for p,h in checkpoint_hashes.items())
 summary=summarize(rows,previous["selected_validation"])
 summary.update({"checkpoint_unchanged":True,"all_462_image_and_prompt_checks_passed":True,
     "wall_seconds":time.monotonic()-started,"prediction_seconds":sum(timings),
     "gpu_peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3})
 write(RUN/"summary.json",summary)
 keys=["validation_index","episode_index","source_step_index","normalized_action_l1","action_token_correct",
       "true_gripper_class","predicted_gripper_class","true_arm_max_abs","predicted_arm_max_abs",
       "true_translation_max_abs","predicted_translation_max_abs"]
 with (RUN/"frame_predictions.csv").open("w",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in rows)
 # Save first examples by predefined categories, with exact source actions for explanation.
 categories={"move_but_near_zero":lambda r:r["true_arm_max_abs"]>=0.1 and r["predicted_arm_max_abs"]<=0.01,
             "should_close_but_open":lambda r:r["true_gripper_class"]=="close" and r["predicted_gripper_class"]=="open",
             "correct_close":lambda r:r["true_gripper_class"]=="close" and r["predicted_gripper_class"]=="close"}
 examples={}
 for name,condition in categories.items():
  match=next((r for r in rows if condition(r)),None)
  if match:
   Image.fromarray(helper.pixels(validation[match["validation_index"]],policy.config.image_key)).save(RUN/(name+".png"))
   examples[name]=match
 write(RUN/"examples.json",examples)
 experiment.audit_after(type("Args",(),{"run":RUN})())
 write(RUN/"complete.json",{"complete":True,"frames":462,"episodes":5,"training_performed":False,
    "checkpoint_unchanged":True,"exact_input_checks":462,"preservation_audit_passed":True})
 print("FULL_FREE_GENERATION_COMPLETE",json.dumps(summary),flush=True)

if __name__=="__main__":
 try:main()
 except Exception as error:
  import traceback
  write(RUN/"error.json",{"error":repr(error),"traceback":traceback.format_exc()})
  raise
