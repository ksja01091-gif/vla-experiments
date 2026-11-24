"""Audit photo/prompt/action parity, then one paired diagnostic rollout.
No training or package changes. Existing reference evaluation remains intact.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,subprocess,sys,time,types
from concurrent.futures import ThreadPoolExecutor
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
RUN=Path(__file__).parent
PREVIOUS=ROOT/"outputs/bowl40_experiment/20261103T051914_711739Z"
DATA=PREVIOUS/"data"
os.environ.update(OPENVLA_BOWL40_DATA_ROOT=str(DATA),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
sys.path.insert(0,str(ROOT/"examples"))
import numpy as np
import torch
from PIL import Image,ImageDraw
import openvla_bowl40_experiment as experiment
from run_openvla_bowl_full import capture_environment

def write(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixels(sample,key):
 return np.rint(sample[key].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
def tensor_hash(tensor):
 return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

def matched_inputs(policy,image,task):
 prompt=f"In: What action should the robot take to {task.strip().lower()}?\nOut:"
 return policy.processor(prompt,image,return_tensors="pt")

@torch.inference_mode()
def predict_from_image(policy,image,task):
 inputs=matched_inputs(policy,image,task).to(policy.config.device,dtype=torch.bfloat16)
 result=policy.model.predict_action(**inputs,unnorm_key=policy.config.unnorm_key,do_sample=False)
 action=torch.tensor(result,dtype=torch.float32,device=policy.config.device).unsqueeze(0)
 assert action.shape==(1,7) and bool(torch.isfinite(action).all())
 return action

@torch.inference_mode()
def matched_select_action(self,batch,**kwargs):
 images=batch[self.config.image_key]
 assert images.shape==(1,3,256,256)
 assert bool(torch.isfinite(images).all()) and bool(((images>=0)&(images<=1)).all())
 image=Image.fromarray(np.rint(images[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
 return predict_from_image(self,image,batch["task"][0])

def prepare_variant_images(records):
 # Invoke the exact existing TF-CPU preprocessing in its original environment.
 helper=RUN/"prepare_photo_variants.py"
 helper.write_text("""from pathlib import Path
import json,sys
sys.path.insert(0,'/home/jovyan/projects/lerobot-openvla/examples')
import numpy as np
from PIL import Image
from openvla_eval_preprocessing import prepare_image,REFERENCE_NAMESPACE
r=Path(__file__).parent
for item in json.loads((r/'photo_records.json').read_text()):
 p=r/'photos'/str(item['validation_index'])
 image=np.array(Image.open(p/'source.png').convert('RGB'))
 raw=np.ascontiguousarray(image[::-1,::-1])
 Image.fromarray(prepare_image(raw)).save(p/'original_eval.png')
 Image.fromarray(REFERENCE_NAMESPACE['get_libero_image']({'agentview_image':raw},224)).save(p/'no_crop_eval.png')
print('PHOTO_VARIANTS_COMPLETE')
""")
 env=os.environ.copy()
 env.update(TF_CPP_MIN_LOG_LEVEL="2",TF_NUM_INTRAOP_THREADS="2",TF_NUM_INTEROP_THREADS="1")
 with (RUN/"prepare_variants.log").open("w") as log:
  subprocess.run(["/home/jovyan/projects/venvs/openvla_eval310/bin/python",str(helper)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
 sources=[*ROOT.joinpath("src").rglob("*.py"),*ROOT.joinpath("examples").rglob("*.py")]
 write(RUN/"sources_before.json",{str(p):digest(p) for p in sources})
 interpreters={"openvla310":"/home/jovyan/anaconda3/envs/openvla310/bin/python",
               "lerobot312":sys.executable,"eval310":"/home/jovyan/projects/venvs/openvla_eval310/bin/python"}
 with ThreadPoolExecutor(max_workers=3) as pool:
  environments=dict(pool.map(lambda item:capture_environment(RUN,item),interpreters.items()))
 write(RUN/"environment_audit.json",environments)
 _,validation,stats=experiment.get_data()
 indices=np.linspace(0,len(validation)-1,8,dtype=int).tolist()
 records=[]
 for index in indices:
  sample=validation[index]
  folder=RUN/"photos"/str(index);folder.mkdir(parents=True)
  array=pixels(sample,"observation.images.image")
  Image.fromarray(array).save(folder/"source.png")
  records.append({"validation_index":index,"episode_index":int(sample["episode_index"]),
                  "source_step":int(sample["source_step_index"].reshape(-1)[0])})
 write(RUN/"photo_records.json",records)
 prepare_variant_images(records)
 policy=experiment.make_policy(stats)
 selected=json.loads((PREVIOUS/"training_result.json").read_text())["selected_checkpoint"]
 policy.load_saved_adapter(selected)
 for p in policy.model.parameters():p.requires_grad_(False)
 policy.eval()
 adapter_before=policy.adapter_hash()
 write(RUN/"plan.json",{"checkpoint":selected,"adapter_sha256":adapter_before,"training_performed":False,
   "photo_samples":indices,"simulator_state":2,"state_selection":"Previously observed bowl rise, chosen for targeted diagnosis; not an unbiased success-rate sample.",
   "changed":"entire pre-model image pipeline: no extra JPEG, external resize or center crop; unchanged native HF processor",
   "held_fixed":["checkpoint","action normalization","task","initial state","seed","controller conversion","horizon 220","warmup 10"],
   "not_a_crop_only_ablation":True,"not_an_official_benchmark_result":True})
 audits=[]
 for index in indices:
  sample=validation[index];folder=RUN/"photos"/str(index)
  source=Image.open(folder/"source.png").convert("RGB")
  original=Image.open(folder/"original_eval.png").convert("RGB")
  no_crop=Image.open(folder/"no_crop_eval.png").convert("RGB")
  train_batch=policy.prepare_training_batch(sample)
  matched=matched_inputs(policy,source,sample["task"])
  input_ids=matched["input_ids"]
  if int(input_ids[0,-1])!=29871:
   input_ids=torch.cat([input_ids,torch.tensor([[29871]],dtype=input_ids.dtype)],dim=1)
  prefix_equal=torch.equal(train_batch["input_ids"][:,:-8],input_ids)
  exact_pixels=torch.equal(train_batch["pixel_values"],matched["pixel_values"])
  assert prefix_equal and exact_pixels,("Prompt or image parity failed",index)
  original_processed=matched_inputs(policy,original,sample["task"])["pixel_values"]
  crop_free_processed=matched_inputs(policy,no_crop,sample["task"])["pixel_values"]
  original_delta=(original_processed-train_batch["pixel_values"]).abs()
  crop_free_delta=(crop_free_processed-train_batch["pixel_values"]).abs()
  # Reproduce the old policy entry point, including its image tensor conversion.
  old_tensor=torch.from_numpy(np.asarray(original).copy()).permute(2,0,1).float().div(255).unsqueeze(0)
  with torch.inference_mode():
   old_action=policy.select_action({policy.config.image_key:old_tensor,"task":[sample["task"]]})
   new_action=predict_from_image(policy,source,sample["task"])
  truth=policy.normalize_action(sample["action"].numpy())
  row={"validation_index":index,"train_input_shape":list(train_batch["pixel_values"].shape),
       "matched_pixel_values_exact":exact_pixels,"inference_prefix_matches_training":prefix_equal,
       "original_eval_pixel_values_exact":bool(torch.equal(train_batch["pixel_values"],original_processed)),
       "original_eval_model_input_mean_abs_difference":float(original_delta.mean()),
       "no_crop_only_model_input_mean_abs_difference":float(crop_free_delta.mean()),
       "train_input_sha256":tensor_hash(train_batch["pixel_values"]),
       "matched_input_sha256":tensor_hash(matched["pixel_values"]),
       "old_action":old_action.cpu().tolist(),"training_matched_action":new_action.cpu().tolist(),
       "old_free_generation_normalized_l1":float(np.abs(policy.normalize_action(old_action[0].cpu().numpy())-truth).mean()),
       "matched_free_generation_normalized_l1":float(np.abs(policy.normalize_action(new_action[0].cpu().numpy())-truth).mean()),
       "teacher_forced_metric":False,"small_diagnostic_sample_only":True}
  audits.append(row)
  print("PHOTO_AUDITED",index,"exact_input",exact_pixels,"prefix",prefix_equal,
        "old_input_difference",row["original_eval_model_input_mean_abs_difference"],flush=True)
 write(RUN/"photo_audit.json",audits)
 # Show the visible transformations, keeping the exact processed tensors audited above.
 index=indices[0];folder=RUN/"photos"/str(index)
 panel=Image.new("RGB",(768,288),"white");draw=ImageDraw.Draw(panel)
 for x,name,label in [(0,"source.png","Training source: 256px"),(256,"original_eval.png","Previous runtime: JPEG + crop"),(512,"no_crop_eval.png","No crop only: still JPEG + resize")]:
  im=Image.open(folder/name).convert("RGB").resize((256,256))
  panel.paste(im,(x,32));draw.text((x+5,10),label,fill="black")
 panel.save(RUN/"image_pipeline_comparison.png")
 # Reuse only the previous reference images/state records for pairing checks.
 (RUN/"simulation").mkdir()
 (RUN/"simulation/base").symlink_to(PREVIOUS/"simulation/base",target_is_directory=True)
 source=(ROOT/"examples/openvla_bowl40_experiment.py").read_text()
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="simulate_trial")
 function=ast.get_source_segment(source,node)
 assert 'with Image.open(current["policy_frame"])' in function
 function=function.replace('with Image.open(current["policy_frame"])','with Image.open(current["frame"])')
 function=function.replace('"reference_preprocessing_pixel_parity":True',
    '"actual_policy_preprocessing":"raw upright 256px RGB to native HF processor, exactly matching training","worker_original_reference_check_passed":True')
 ns=dict(vars(experiment))
 exec(compile(function,str(RUN/"matched_rollout_function.py"),"exec"),ns)
 (RUN/"matched_rollout_function.py").write_text(function)
 policy.select_action=types.MethodType(matched_select_action,policy)
 args=types.SimpleNamespace(run=RUN)
 started=time.monotonic()
 result=ns["simulate_trial"](policy,"training_matched",2,args)
 assert result["trial_executed"],result
 original=json.loads((PREVIOUS/"simulation/lora/state_02/result.json").read_text())
 assert original["selected_init_state_sha256"]==result["selected_init_state_sha256"]
 assert original["initial_image_sha256"]==result["initial_image_sha256"]
 assert policy.adapter_hash()==adapter_before
 write(RUN/"rollout_comparison.json",{"previous":original,"training_matched":result,
   "paired_initial_state_exact":True,"checkpoint_unchanged":True,"training_performed":False,
   "diagnostic_only":True,"wall_seconds":time.monotonic()-started})
 env=os.environ.copy();env["PYTHONPATH"]="/home/jovyan/projects/LIBERO";env["MUJOCO_GL"]="egl"
 with (RUN/"motion_diagnostics.log").open("w") as log:
  subprocess.run(["/home/jovyan/anaconda3/envs/openvla310/bin/python",str(PREVIOUS/"analyze_motion_no_camera.py"),
                  "--run",str(RUN),"--models","training_matched"],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (RUN/"grasp_contact.log").open("w") as log:
  subprocess.run(["/home/jovyan/anaconda3/envs/openvla310/bin/python",str(PREVIOUS/"check_grasp_contacts.py"),
                  str(RUN/"simulation/training_matched/state_02")],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 experiment.audit_after(args)
 contacts=json.loads((RUN/"simulation/training_matched/state_02/grasp_contact_diagnostics.json").read_text())
 old_contacts=json.loads((PREVIOUS/"simulation/lora/state_02/grasp_contact_diagnostics.json").read_text())
 write(RUN/"diagnostic_summary.json",{
   "training_performed":False,"checkpoint_unchanged":True,"paired_initial_state_exact":True,
   "photo_samples":len(audits),"all_matched_model_inputs_exact":all(a["matched_pixel_values_exact"] for a in audits),
   "all_prompt_prefixes_match":all(a["inference_prefix_matches_training"] for a in audits),
   "original_model_input_exact_count":sum(a["original_eval_pixel_values_exact"] for a in audits),
   "previous_success":original["success"],"training_matched_success":result["success"],
   "previous_steps":original["policy_steps"],"training_matched_steps":result["policy_steps"],
   "previous_bowl_rise_mm":old_contacts["maximum_bowl_height_increase_m"]*1000,
   "training_matched_bowl_rise_mm":contacts["maximum_bowl_height_increase_m"]*1000,
   "previous_bilateral_contact_steps":old_contacts["bilateral_contact_steps"],
   "training_matched_bilateral_contact_steps":contacts["bilateral_contact_steps"],
   "training_matched_contact_and_rise_steps":contacts["contact_and_rise_over_2cm_steps"],
   "previous_free_generation_sample_l1":float(np.mean([a["old_free_generation_normalized_l1"] for a in audits])),
   "matched_free_generation_sample_l1":float(np.mean([a["matched_free_generation_normalized_l1"] for a in audits])),
   "limitations":["one state selected after prior observation, not an unbiased success-rate estimate",
                 "whole image pipeline changed, not crop alone",
                 "eight demonstration images are a diagnostic sample, not full validation",
                 "bilateral contact checked at control-step boundaries",
                 "matched pixels do not establish sufficient training or successful closed-loop control"]})
 # Video verification and a step-aligned comparison for viewing.
 import imageio.v2 as imageio
 old_video=imageio.get_reader(original["video"])
 new_video=imageio.get_reader(result["video"])
 assert old_video.count_frames()==original["policy_steps"]+1
 assert new_video.count_frames()==result["policy_steps"]+1
 with imageio.get_writer(str(RUN/"previous_vs_training_matched.mp4"),fps=10,codec="libx264",quality=8) as writer:
  for step in range(max(original["policy_steps"],result["policy_steps"])+1):
   canvas=Image.new("RGB",(512,288),"white");draw=ImageDraw.Draw(canvas)
   oi=min(step,original["policy_steps"]);ni=min(step,result["policy_steps"])
   canvas.paste(Image.fromarray(old_video.get_data(oi)),(0,32))
   canvas.paste(Image.fromarray(new_video.get_data(ni)),(256,32))
   draw.text((5,9),f"Previous pipeline | step {oi}",fill="black")
   draw.text((261,9),f"Training-matched | step {ni}",fill="black")
   writer.append_data(np.asarray(canvas))
 old_video.close();new_video.close()
 write(RUN/"complete.json",{"complete":True,"evidence":["8 exact image and prefix checks","same checkpoint",
       "one paired diagnostic rollout","physical replay parity","video decoded","sources and 3 environments unchanged"]})
 print("DIAGNOSTIC_COMPLETE",(RUN/"diagnostic_summary.json").read_text(),flush=True)

if __name__=="__main__":
 try:main()
 except Exception as error:
  import traceback
  write(RUN/"error.json",{"error":repr(error),"traceback":traceback.format_exc()})
  raise
