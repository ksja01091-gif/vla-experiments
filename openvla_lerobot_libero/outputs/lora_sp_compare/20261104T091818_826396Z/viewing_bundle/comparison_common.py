"""Shared data/model/checkpoint utilities for the controlled comparison."""
from pathlib import Path
import os,sys,json,hashlib,time,types,importlib.util,random
RUN=Path(__file__).parent
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
DATA=ROOT/"outputs/bowl40_experiment/20261103T051914_711739Z/data"
OLD=ROOT/"outputs/five_epoch_bowl/20261103T064207_586519Z"
INPUT=ROOT/"outputs/input_pipeline_check/20261103T055533_073692Z"
os.environ.update(OPENVLA_BOWL40_DATA_ROOT=str(DATA),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",WANDB_MODE="offline")
sys.path.insert(0,str(ROOT/"examples"))
import numpy as np
import torch
from PIL import Image
import openvla_bowl40_experiment as exp
from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from sp_policy import SPOPenvlaPolicy
METHODS=["lora8","lora128","sp128"]
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
helper=load_module(INPUT/"audit_and_compare.py","sp_comparison_images")
def make_policy(method,stats):
 exp.seed()
 if method=="sp128":
  policy=SPOPenvlaPolicy(OpenVLAConfig(model_id="openvla/openvla-7b",unnorm_key="bridge_orig",device="cuda:0"),stats,rank=128,eta=.9)
 else:policy=exp.make_policy(stats,8 if method=="lora8" else 128)
 return policy
def checkpoint_path(method):
 if method=="lora8":return OLD/"checkpoints/step_005035"
 return Path(json.loads((RUN/method/"training_result.json").read_text())["final_checkpoint"])
def clear_aux(policy):
 if hasattr(policy,"clear_aux"):policy.clear_aux()
def prepare_inference(policy):
 for p in policy.model.parameters():p.requires_grad_(False)
 policy.eval();clear_aux(policy)
 policy.select_action=types.MethodType(helper.matched_select_action,policy)
 return policy
def sample_action(policy,data,index):
 sample=data[int(index)]
 pixels=np.rint(sample[policy.config.image_key].permute(1,2,0).numpy()*255).astype(np.uint8)
 tensor=torch.from_numpy(pixels.copy()).permute(2,0,1).float()/255
 return policy.select_action({policy.config.image_key:tensor.unsqueeze(0),"task":[sample["task"]]})
def step(policy,optimizer,data,indices):
 policy.train();policy.model.vision_backbone.eval();optimizer.zero_grad(set_to_none=True)
 loss,metrics=policy(exp.batch_from(policy,data,indices))
 loss.backward()
 grad=torch.nn.utils.clip_grad_norm_(policy.get_optim_params(),1.)
 assert torch.isfinite(grad)
 optimizer.step();metrics["gradient_norm"]=float(grad)
 del loss
 clear_aux(policy)
 return metrics
def configure():
 cfg={"base_model":"openvla/openvla-7b","data_root":str(DATA),"train_demonstrations":40,"train_frames":4025,
 "validation_demonstrations":5,"validation_frames":462,"epochs":5,"batch_size":4,"lr":1e-4,
 "optimizer":"AdamW","weight_decay":0.,"gradient_clip":1.,"seed":7,"target_modules":["language_model q_proj","language_model v_proj"],
 "base_dtype":"BF16","adapter_dtype":"FP32","lora_scale_alpha_over_rank":1.,"image_preprocessing":"unchanged 256 RGB, native processor; no augmentation",
 "normalization":"fixed original train20 q01/q99, identical to prior comparisons; no validation calibration",
 "checkpoint_selection":"final epoch5, fixed in advance",
 "sp":{"initial_bank_rank":128,"eta":.9,"router_hidden":32,"router":"two Linear layers; SiLU then Softplus for nonnegative scores",
 "A_init":"Gaussian std=1/128","B_init":"zeros","scale":1.,"spec_weight":.01,"router_weight":.001,
 "router_regularizer":"Switch top1 load balance + mean(logsumexp(raw logits)^2)",
 "selection":"minimum sorted squared-score prefix reaching eta; include crossing component; selector FP64 detached; differentiable energy FP32",
 "dense_full_bank":True,"orthogonal_basis_constraint":False,"score_energy_not_a_guaranteed_true_matrix_SVD_error_bound":True,
 "adaptation_not_exact_paper_replication":"OpenVLA LIBERO single-task q/v only; paper pi0/SmolVLA PiPER multi-task all-module study; hidden/activation/init explicitly chosen"},
 "threshold_ablation":{"trained_eta":.9,"inference_only":[.9,.8,.7],"same_SP_checkpoint":True,"retraining":False},
 "simulation":{"states":[0,1,2,3,4],"horizon":220,"new_state_generalization_test":False},
 "lora8_primary_reuse":"existing 5 epoch checkpoint and same-state simulation; fresh isolated resource benchmark",
 "package_installation":False,"paper":"arXiv:2603.07404v1"}
 write(RUN/"experiment_config.json",cfg)
 return cfg
