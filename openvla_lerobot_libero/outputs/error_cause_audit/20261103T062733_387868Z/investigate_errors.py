"""Evidence audit: seen-vs-held-out images, modal baseline, and shuffled image/action pairs.
No optimizer or training. All model inference uses the verified native image pipeline.
"""
from pathlib import Path
import csv,hashlib,importlib.util,json,os,sys,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
RUN=Path(__file__).parent
OLD=ROOT/"outputs/full_validation_actions/20261103T061517_441641Z"
TRAIN_RUN=ROOT/"outputs/bowl40_experiment/20261103T051914_711739Z"
INPUT_AUDIT=ROOT/"outputs/input_pipeline_check/20261103T055533_073692Z"
os.environ.update(OPENVLA_BOWL40_DATA_ROOT=str(TRAIN_RUN/"data"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
sys.path.insert(0,str(ROOT/"examples"))
import numpy as np
import torch
from PIL import Image
import openvla_bowl40_experiment as experiment
from run_openvla_bowl_full import capture_environment

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module

def pick(records):
 cursor=0;indices=[]
 for record in records:
  local=np.linspace(0,record["frames"]-1,12,dtype=int).tolist()
  assert len(set(local))==12
  indices.extend(cursor+x for x in local);cursor+=record["frames"]
 return indices

def metrics(rows):
 n=len(rows);a=np.array([r["normalized_action_l1_per_dimension"] for r in rows])
 active=[r for r in rows if r["true_arm_max_abs"]>=.1]
 zero=[r for r in active if r["predicted_arm_max_abs"]<=.01]
 confused={t:{p:0 for p in ["open","close","neutral"]} for t in ["open","close"]}
 for r in rows:confused[r["true_gripper_class"]][r["predicted_gripper_class"]]+=1
 return {"frames":n,"episodes":len({r["episode_index"] for r in rows}),
  "token_accuracy":sum(r["action_token_correct"] for r in rows)/(7*n),
  "normalized_action_l1":float(a.mean()),"l1_per_dimension":a.mean(axis=0).tolist(),
  "gripper_accuracy":sum(r["true_gripper_class"]==r["predicted_gripper_class"] for r in rows)/n,
  "expert_active_frames":len(active),"active_predicted_near_zero":len(zero),
  "active_predicted_near_zero_fraction":len(zero)/len(active),
  "gripper_confusion":confused}

def main():
 started=time.monotonic()
 write(RUN/"sources_before.json",{str(p):digest(p) for p in [*ROOT.joinpath("src").rglob("*.py"),*ROOT.joinpath("examples").rglob("*.py")]})
 interpreters={"openvla310":"/home/jovyan/anaconda3/envs/openvla310/bin/python",
 "lerobot312":sys.executable,"eval310":"/home/jovyan/projects/venvs/openvla_eval310/bin/python"}
 with ThreadPoolExecutor(max_workers=3) as pool:
  env=dict(pool.map(lambda item:capture_environment(RUN,item),interpreters.items()))
 write(RUN/"environment_audit.json",env)
 helper=load(INPUT_AUDIT/"audit_and_compare.py","known_input_helper")
 training,validation,stats=experiment.get_data()
 split=json.loads((TRAIN_RUN/"data/split_manifest.json").read_text())
 indices=pick(split["train"]);validation_indices=pick(split["validation"])
 assert len(indices)==480 and len(validation_indices)==60
 val_rows=[json.loads(line) for line in (OLD/"frame_predictions.jsonl").read_text().splitlines()]
 assert [x["validation_index"] for x in val_rows]==list(range(462))
 val_by_index={r["validation_index"]:r for r in val_rows}
 matched_val=[val_by_index[i] for i in validation_indices]
 old_config=json.loads((OLD/"config.json").read_text())
 write(RUN/"sampling_plan.json",{"training_indices":indices,"validation_indices_for_matched_phase_comparison":validation_indices,
  "selection":"12 evenly spaced source frames per whole demonstration, including first and last; no selection using errors",
  "train_photos":480,"train_demonstrations":40,"validation_photos":60,"validation_demonstrations":5,
  "all_validation_context_photos":462,"training_performed":False,
  "scope":"Sampled seen images versus held-out images; rates are descriptive, not a unique causal attribution."})
 policy=experiment.make_policy(stats);policy.load_saved_adapter(old_config["checkpoint"])
 assert policy.adapter_hash()==old_config["adapter_hash"]
 for parameter in policy.model.parameters():parameter.requires_grad_(False)
 policy.eval();before=policy.adapter_hash()
 # Inspect exact target labels and how much a constant per-dimension mode can explain.
 normalized=[]
 for record in split["train"]:
  with np.load(record["archive"],allow_pickle=False) as z:
   normalized.extend(policy.normalize_action(action) for action in z["actions"])
 normalized=np.stack(normalized)
 target_tokens=policy.processor.tokenizer.vocab_size-np.digitize(normalized,policy.action_tokenizer.bins)
 modes=np.array([Counter(target_tokens[:,i].tolist()).most_common(1)[0][0] for i in range(7)])
 mode_values=policy.action_tokenizer.decode_token_ids_to_actions(modes)
 true_val=np.array([r["true_normalized_action"] for r in val_rows])
 predicted_val=np.array([r["predicted_normalized_action"] for r in val_rows])
 val_targets=np.array([r["true_action_token_ids"] for r in val_rows])
 # Same saved predictions, intentionally paired with shuffled labels: no new model calls.
 rng=np.random.default_rng(7)
 shuffled=np.array([np.abs(predicted_val-true_val[rng.permutation(462)]).mean() for _ in range(500)])
 baseline={"training_frames":len(normalized),"per_dimension_mode_token_ids":modes.tolist(),
   "per_dimension_mode_normalized_actions":mode_values.tolist(),
   "fixed_mode_validation_token_accuracy":float((val_targets==modes).mean()),
   "fixed_mode_validation_normalized_l1":float(np.abs(true_val-mode_values).mean()),
   "actual_model_validation_normalized_l1":float(np.abs(predicted_val-true_val).mean()),
   "actual_model_validation_token_accuracy":sum(r["action_token_correct"] for r in val_rows)/(462*7),
   "shuffle_pairing":{"permutations":500,"seed":7,"shuffled_l1_mean":float(shuffled.mean()),
      "shuffled_l1_min":float(shuffled.min()),"shuffled_l1_max":float(shuffled.max()),
      "note":"Descriptive association check only; correlated frames and shared task prevent an independent-sample causal claim."},
   "training_mode_frequency_per_dimension":[float((target_tokens[:,i]==modes[i]).mean()) for i in range(7)],
   "model_validation_mode_frequency_per_dimension":[float((np.array([r["predicted_action_token_ids"] for r in val_rows])[:,i]==modes[i]).mean()) for i in range(7)],
   "encoding_control":"Normalization and token mapping are the exact training functions; no semantic relabeling."}
 write(RUN/"mode_and_image_pairing_diagnostics.json",baseline)
 print("BASELINE_DIAGNOSTICS",json.dumps(baseline),flush=True)
 owner=policy.model.predict_action.__self__;original=owner.generate
 had="generate" in owner.__dict__;saved=owner.__dict__.get("generate");captured=[]
 def capture(*args,**kwargs):
  output=original(*args,**kwargs);captured.append(output[0,-7:].detach().cpu().numpy().copy());return output
 owner.generate=capture
 rows=[]
 try:
  with (RUN/"train_sample_predictions.jsonl").open("x") as f:
   for count,index in enumerate(indices,1):
    sample=training[index];image=Image.fromarray(helper.pixels(sample,policy.config.image_key))
    target=policy.prepare_training_batch(sample)
    inp=helper.matched_inputs(policy,image,sample["task"])
    assert torch.equal(target["pixel_values"],inp["pixel_values"])
    ids=inp["input_ids"]
    if int(ids[0,-1])!=29871:ids=torch.cat([ids,torch.tensor([[29871]],dtype=ids.dtype)],dim=1)
    assert torch.equal(ids,target["input_ids"][:,:-8])
    captured.clear()
    output=helper.predict_from_image(policy,image,sample["task"])
    assert len(captured)==1
    pred=output[0].cpu().numpy()
    tokens=captured[0];gold=target["labels"][0,-8:-1].numpy()
    decoded=policy.action_tokenizer.decode_token_ids_to_actions(tokens)
    truth=sample["action"].numpy();normalized_truth=policy.normalize_action(truth)
    errors=np.abs(decoded-normalized_truth)
    classify=lambda v:"close" if v<.5 else ("open" if v>.5 else "neutral")
    row={"training_index":index,"episode_index":int(sample["episode_index"]),
      "source_step_index":int(sample["source_step_index"].reshape(-1)[0]),
      "true_action":truth.tolist(),"predicted_action":pred.tolist(),
      "true_action_token_ids":gold.tolist(),"predicted_action_token_ids":tokens.tolist(),
      "action_token_correct":int((tokens==gold).sum()),"normalized_action_l1_per_dimension":errors.tolist(),
      "normalized_action_l1":float(errors.mean()),
      "true_gripper_class":classify(float(truth[-1])),"predicted_gripper_class":classify(float(pred[-1])),
      "true_arm_max_abs":float(np.abs(truth[:6]).max()),"predicted_arm_max_abs":float(np.abs(pred[:6]).max()),
      "image_and_prompt_match":True}
    assert np.isfinite(pred).all()
    rows.append(row);f.write(json.dumps(row)+"\n");f.flush()
    if count%25==0 or count==480:
     progress={"completed":count,"total":480,"elapsed_seconds":time.monotonic()-started}
     write(RUN/"progress.json",progress);print("PROGRESS",json.dumps(progress),flush=True)
 finally:
  if had:owner.generate=saved
  else:delattr(owner,"generate")
 assert policy.adapter_hash()==before
 assert len(rows)==480 and len(set(r["training_index"] for r in rows))==480
 summary={"training_performed":False,"checkpoint_unchanged":True,"checkpoint":old_config["checkpoint"],
  "training_sample":metrics(rows),"validation_matched_phase_sample":metrics(matched_val),
  "validation_all":metrics(val_rows),"baseline_diagnostics":baseline,
  "comparison_limitations":["480/4025 training frames sampled, not full training evaluation",
    "12 equally spaced photos per episode for the main train-vs-validation comparison; 40 versus 5 episodes",
    "a train/validation gap alone is not a unique cause or proof of underfitting/generalization",
    "no simulator rollout, extra training, or controlled capacity/training-duration intervention in this audit"],
  "wall_seconds":time.monotonic()-started,"gpu_peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3}
 write(RUN/"summary.json",summary)
 categories={"seen_but_near_zero":lambda x:x["true_arm_max_abs"]>=.1 and x["predicted_arm_max_abs"]<=.01,
             "seen_but_wrong_gripper":lambda x:x["true_gripper_class"]!=x["predicted_gripper_class"]}
 examples={}
 for name,test in categories.items():
  row=next((r for r in rows if test(r)),None)
  if row:
   examples[name]=row;Image.fromarray(helper.pixels(training[row["training_index"]],policy.config.image_key)).save(RUN/(name+".png"))
 write(RUN/"seen_error_examples.json",examples)
 experiment.audit_after(type("Args",(),{"run":RUN})())
 write(RUN/"complete.json",{"complete":True,"training_frames_evaluated":480,"training_demonstrations":40,
   "validation_predictions_reused":462,"training_performed":False,"checkpoint_unchanged":True,"preservation_audit_passed":True})
 print("CAUSE_AUDIT_COMPLETE",json.dumps(summary),flush=True)

if __name__=="__main__":
 try:main()
 except Exception as error:
  import traceback
  write(RUN/"error.json",{"error":repr(error),"traceback":traceback.format_exc()})
  raise
