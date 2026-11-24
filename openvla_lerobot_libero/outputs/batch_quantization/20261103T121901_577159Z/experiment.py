"""Isolated batch throughput and INT8 inference experiment. Never overwrites source checkpoints."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, math, os, random, subprocess, sys, time, types, traceback
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
RUN=Path(__file__).parent
OLD=ROOT/"outputs/five_epoch_bowl/20261103T064207_586519Z"
CP=OLD/"checkpoints/step_005035"
DATA=ROOT/"outputs/bowl40_experiment/20261103T051914_711739Z/data"
INPUT=ROOT/"outputs/input_pipeline_check/20261103T055533_073692Z"
os.environ.update(OPENVLA_BOWL40_DATA_ROOT=str(DATA),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",WANDB_MODE="offline")
sys.path.insert(0,str(ROOT/"examples"))
# Isolated optional package; absent from ordinary conda imports.
sys.path.insert(0,str(RUN/"deps"))
import numpy as np
import torch
from PIL import Image
import openvla_bowl40_experiment as exp
from transformers import AutoProcessor,AutoModelForVision2Seq,BitsAndBytesConfig
from peft import PeftModel

def write(name,value):
 path=RUN/name
 tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
 tmp.replace(path)
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
helper=load(INPUT/"audit_and_compare.py","batch_quant_input")
def sync():torch.cuda.synchronize()
def gpu_process():
 out=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True,capture_output=True).stdout
 return [s.strip() for s in out.splitlines() if s.strip().split(",")[0]==str(os.getpid())]
def batch_benchmark(size):
 train,_,stats=exp.get_data();assert len(train)==4025
 policy=exp.make_policy(stats);policy.load_saved_adapter(CP)
 original_hash=policy.adapter_hash()
 indices=torch.randperm(len(train),generator=torch.Generator().manual_seed(713)).tolist()[:144]
 # warm kernels on disposable weights, then reset adapter and optimizer.
 def new_optimizer():return torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=0.0)
 opt=new_optimizer()
 def step(ids):
  policy.train();policy.model.vision_backbone.eval();opt.zero_grad(set_to_none=True)
  loss,metrics=policy(exp.batch_from(policy,train,ids));loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(policy.get_optim_params(),1.0)
  assert torch.isfinite(norm);opt.step()
  return metrics
 for start in range(0,16,size):step(indices[128:144][start:start+size])
 opt.zero_grad(set_to_none=True);del opt
 policy.load_saved_adapter(CP);assert policy.adapter_hash()==original_hash
 opt=new_optimizer();sync();torch.cuda.reset_peak_memory_stats()
 rows=[];start_time=time.perf_counter()
 for offset in range(0,128,size):
  sync();tick=time.perf_counter();metrics=step(indices[offset:offset+size]);sync()
  row={"batch_size":size,"frames":size,"seconds":time.perf_counter()-tick,**metrics}
  rows.append(row)
  if offset%(size*8)==0:print("BATCH_PROGRESS",size,offset+size,"/128",flush=True)
 sync();elapsed=time.perf_counter()-start_time
 result={"batch_size":size,"measured_frames":128,"warmup_frames":16,"starting_checkpoint":str(CP),
 "starting_adapter_hash":original_hash,"same_measured_frame_indices":indices[:128],"optimizer_updates":len(rows),
 "wall_seconds":elapsed,"frames_per_second":128/elapsed,"mean_step_seconds":float(np.mean([r["seconds"] for r in rows])),
 "estimated_five_epoch_train_minutes_excluding_validation":elapsed/128*4025*5/60,
 "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
 "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"gpu_process_mib":gpu_process(),
 "disposable_updates_only":True,"adapter_changed_in_memory":policy.adapter_hash()!=original_hash,
 "checkpoint_saved":False,"not_a_model_quality_comparison":True,"steps":rows}
 write(f"batch_{size}.json",result);print("BATCH_COMPLETE",size,elapsed,result["frames_per_second"],flush=True)

def make_inference(mode):
 train,val,stats=exp.get_data()
 if mode=="bf16":
  policy=exp.make_policy(stats);policy.load_saved_adapter(CP)
 else:
  from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
  from lerobot.policies.pretrained import PreTrainedPolicy
  from lerobot.policies.openvla.full_experiment_openvla import FullExperimentOpenVLAPolicy
  cfg=json.loads((OLD/"training_config.json").read_text())
  config=OpenVLAConfig(model_id="openvla/openvla-7b",unnorm_key="libero_bowl_train20",device="cuda:0")
  policy=FullExperimentOpenVLAPolicy.__new__(FullExperimentOpenVLAPolicy)
  PreTrainedPolicy.__init__(policy,config)
  policy.processor=AutoProcessor.from_pretrained(cfg["base_snapshot"],trust_remote_code=True,local_files_only=True)
  quant=BitsAndBytesConfig(load_in_8bit=True,llm_int8_skip_modules=["vision_backbone","projector","lm_head"])
  # Transformers 4.40 sets is_loaded_in_8bit only after dispatch;
  # Accelerate 1.15 otherwise calls forbidden model.to(). Use its public
  # force_hooks option only for this load; restore original dispatcher afterwards.
  import transformers.modeling_utils as modeling_utils
  from functools import wraps
  from unittest.mock import patch
  dispatch=modeling_utils.dispatch_model
  @wraps(dispatch)
  def dispatch_with_hooks(*args,**kwargs):
   kwargs["force_hooks"]=True
   return dispatch(*args,**kwargs)
  with patch.object(modeling_utils,"dispatch_model",dispatch_with_hooks):
   base=AutoModelForVision2Seq.from_pretrained(cfg["base_snapshot"],trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,quantization_config=quant,device_map={"":0})
  base.norm_stats.update(json.loads((CP/"action_normalization.json").read_text()))
  # PEFT 0.11 reads this removed, unused constructor metadata.
  # Its Linear8bitLt wrapper accepts it via **kwargs but does not use it.
  # Restore only the legacy flag in memory; do not change quantization math.
  import bitsandbytes as bnb
  legacy_flags=0
  for module in base.modules():
   if isinstance(module,bnb.nn.Linear8bitLt) and not hasattr(module.state,"memory_efficient_backward"):
    module.state.memory_efficient_backward=False
    legacy_flags+=1
  write("peft_int8_compatibility.json",{"legacy_metadata_flags_added":legacy_flags,
       "peft_version":"0.11.1","bitsandbytes_version":bnb.__version__,
       "unused_metadata_only":True,"package_files_modified":False})
  policy.model=PeftModel.from_pretrained(base,str(CP),is_trainable=False)
  for n,p in policy.model.named_parameters():
   if "lora_" in n:p.data=p.data.float()
  policy.model.language_model.config.use_cache=False
  # Read original adapter exactly; no merging into or overwriting the quantized base.
  from safetensors.torch import load_file
  from peft import get_peft_model_state_dict
  saved=load_file(str(CP/"adapter_model.safetensors"))
  actual=get_peft_model_state_dict(policy.model)
  assert set(saved)==set(actual)
  assert all(torch.equal(actual[n].detach().cpu(),saved[n]) for n in saved),"adapter reload mismatch"
 for p in policy.model.parameters():p.requires_grad_(False)
 policy.eval()
 policy.select_action=types.MethodType(helper.matched_select_action,policy)
 return policy,val

def infer(mode):
 exp.seed()
 started=time.perf_counter()
 policy,val=make_inference(mode);sync()
 import bitsandbytes as bnb
 quantized=[n for n,m in policy.model.named_modules() if isinstance(m,bnb.nn.Linear8bitLt)]
 if mode=="int8":
  assert quantized
  assert all("language_model" in n for n in quantized)
 else:assert not quantized
 index=list(np.linspace(0,len(val)-1,32,dtype=int))
 def predict(i):
  sample=val[int(i)]
  pixels=np.rint(sample[policy.config.image_key].permute(1,2,0).numpy()*255).astype(np.uint8)
  tensor=torch.from_numpy(pixels.copy()).permute(2,0,1).float()/255
  return policy.select_action({policy.config.image_key:tensor.unsqueeze(0),"task":[sample["task"]]})
 for i in index[:3]:predict(i)
 sync();torch.cuda.reset_peak_memory_stats();rows=[]
 for i in index:
  sync();tick=time.perf_counter();action=predict(i);sync()
  assert action.shape==(1,7) and bool(torch.isfinite(action).all())
  rows.append({"validation_index":int(i),"seconds_including_image_preprocessing":time.perf_counter()-tick,
               "action":action.cpu().tolist(),"shape":[1,7],"finite":True})
 result={"mode":mode,"checkpoint":str(CP),"sample_count":len(rows),"all_actions_finite":True,
 "mean_action_seconds":float(np.mean([r["seconds_including_image_preprocessing"] for r in rows])),
 "median_action_seconds":float(np.median([r["seconds_including_image_preprocessing"] for r in rows])),
 "allocated_gib":torch.cuda.memory_allocated()/1024**3,"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
 "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"gpu_process_mib":gpu_process(),
 "model_tensor_footprint_bytes":policy.model.get_memory_footprint(),"quantized_linear_count":len(quantized),
 "quantized_modules":quantized,"scope":"language model linear layers except lm_head; vision/projector remain BF16; adapter FP32",
 "rows":rows,"loading_and_test_seconds":time.perf_counter()-started}
 write(f"inference_{mode}.json",result)
 print("INFERENCE_CHECK",mode,result["peak_allocated_gib"],result["mean_action_seconds"],flush=True)
 if mode=="int8":
  ns=dict(vars(exp));source=(INPUT/"matched_rollout_function.py").read_text()
  exec(compile(source,str(INPUT/"matched_rollout_function.py"),"exec"),ns)
  baseline=RUN/"simulation/base"
  baseline.parent.mkdir(exist_ok=True)
  if not baseline.exists():baseline.symlink_to(OLD/"simulation/epoch5",target_is_directory=True)
  before=policy.adapter_hash();trials=[]
  for state in range(5):
   folder=RUN/"simulation/int8"/f"state_{state:02d}"
   if (folder/"result.json").exists():trial=json.loads((folder/"result.json").read_text())
   else:trial=ns["simulate_trial"](policy,"int8",state,types.SimpleNamespace(run=RUN))
   assert trial["trial_executed"],trial
   old=json.loads((OLD/"simulation/epoch5"/f"state_{state:02d}"/"result.json").read_text())
   for k in ["initial_image_sha256","selected_init_state_sha256","init_state_file_sha256","reference_sha256","simulator_versions","max_policy_steps","warmup_steps"]:
    assert old[k]==trial[k],(k,old[k],trial[k])
   trials.append(trial)
   write("simulation_progress.json",{"completed":len(trials),"total":5,"successes":sum(t["success"] for t in trials)})
  assert policy.adapter_hash()==before
  write("simulation_int8.json",{"attempts":5,"successes":sum(t["success"] for t in trials),
       "success_rate":sum(t["success"] for t in trials)/5,"mean_policy_steps":float(np.mean([t["policy_steps"] for t in trials])),
       "trials":trials,"baseline_reused":str(OLD/"simulation_epoch5.json"),
       "identical_initial_states_images_reference_and_simulator_versions":True})
  print("QUANT_SIM_COMPLETE",sum(t["success"] for t in trials),"/5",flush=True)

def audit():
 before=json.loads((RUN/"protected_before.json").read_text())
 changed=[p for p,h in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
 env={}
 for n,p in json.loads((RUN/"interpreters.json").read_text()).items():
  now=subprocess.run([p,"-m","pip","freeze"],capture_output=True,text=True,check=True).stdout
  (RUN/(n+"_after.txt")).write_text(now)
  env[n]=sorted(now.splitlines())==sorted((RUN/(n+"_before.txt")).read_text().splitlines())
 report={"protected_files_checked":len(before),"changed_files":changed,"environments_unchanged":env}
 write("preservation_report.json",report)
 assert not changed and all(env.values())
 print("PRESERVATION_OK",flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("phase",choices=["batch","infer","audit"])
 p.add_argument("--batch-size",type=int,choices=[4,8],default=4);p.add_argument("--mode",choices=["bf16","int8"],default="bf16")
 a=p.parse_args()
 try:
  if a.phase=="batch":batch_benchmark(a.batch_size)
  elif a.phase=="infer":infer(a.mode)
  else:audit()
 except Exception as e:
  write(f"error_{a.phase}_{a.batch_size}_{a.mode}.json",{"error":repr(e),"traceback":traceback.format_exc()})
  raise
if __name__=="__main__":main()
