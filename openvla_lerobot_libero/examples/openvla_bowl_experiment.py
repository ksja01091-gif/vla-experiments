"""Full-data OpenVLA/LeRobot bowl baseline: calibrate, train, reload, simulate, summarize.
All experiments run on the remote Linux server. Existing source files are not edited.
"""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from PIL import Image, ImageDraw
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from lerobot.policies.openvla.modeling_openvla import OpenVLAPolicy
from lerobot.policies.openvla.full_experiment_openvla import FullExperimentOpenVLAPolicy
from openvla_libero_single_task import reference_settings, set_reference_seed, save_media
from openvla_libero_motion import controller_action, send, read_observation

ROOT = Path("/home/jovyan/projects/lerobot-openvla")
DATA = Path("/home/jovyan/projects/datasets/libero_bowl_25/20261101T120613_454595Z")
TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
METRICS = ["loss", "action_token_accuracy", "normalized_action_l1", "predicted_action_token_fraction"]

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))

def append_json(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + chr(10))

def memory():
    return {"gpu_allocated_gib":torch.cuda.memory_allocated()/1024**3,
            "gpu_reserved_gib":torch.cuda.memory_reserved()/1024**3,
            "gpu_peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3}

def seed():
    random.seed(7); np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    torch.set_num_threads(4)

def get_data():
    split = json.loads((DATA/"split_manifest.json").read_text())
    assert len(split["train"])==20 and len(split["validation"])==5
    assert not {r["content_id"] for r in split["train"]} & {r["content_id"] for r in split["validation"]}
    train=LeRobotDataset("local/openvla-libero-bowl-train",root=DATA/"train")
    validation=LeRobotDataset("local/openvla-libero-bowl-validation",root=DATA/"validation")
    assert len(train)==1928 and len(validation)==462
    stats=json.loads((DATA/"action_statistics_train_only.json").read_text())
    assert stats["episodes"]==20 and stats["frames"]==1928
    return train,validation,stats

def make_policy(stats, rank=8):
    seed()
    return FullExperimentOpenVLAPolicy(OpenVLAConfig(model_id="openvla/openvla-7b",unnorm_key="bridge_orig",device="cuda:0"),stats,rank)

def batch_from(policy, data, indices):
    samples=[data[int(i)] for i in indices]
    assert all(s["task"].strip().lower().rstrip(".")==TASK for s in samples)
    return policy.collate_samples(samples)

@torch.no_grad()
def evaluate(policy, data, batch_size, destination, optimizer_step):
    policy.eval()
    sums={key:0.0 for key in METRICS};count=0
    started=time.perf_counter();seen=[]
    with destination.open("x") as handle:
        for start in range(0,len(data),batch_size):
            indices=list(range(start,min(start+batch_size,len(data))))
            _,metrics=policy(batch_from(policy,data,indices))
            for key in METRICS:sums[key]+=metrics[key]*len(indices)
            count+=len(indices);seen.extend(indices)
            handle.write(json.dumps({"indices":indices,"count":len(indices),**metrics})+chr(10))
            if count%100<batch_size or count==len(data):
                print("VALIDATION_PROGRESS",optimizer_step,count,"/",len(data),flush=True)
    assert seen==list(range(len(data)))
    summary={key:sums[key]/count for key in METRICS}
    summary.update({"frames":count,"optimizer_step":optimizer_step,"seconds":time.perf_counter()-started,**memory()})
    print("VALIDATION_COMPLETE",json.dumps(summary),flush=True)
    return summary

def frozen_sample_hash(policy):
    chunks=[]
    for _,param in policy.model.named_parameters():
        if not param.requires_grad:
            flat=param.detach().flatten();n=flat.numel()
            for first in [0,max(0,n//2-8),max(0,n-16)]:chunks.append(flat[first:first+16].float())
    return hashlib.sha256(torch.cat(chunks).cpu().numpy().tobytes()).hexdigest()

def calibrate(args):
    seed();train,validation,stats=get_data();policy=make_policy(stats,args.rank)
    batch=batch_from(policy,train,list(range(args.batch_size)))
    trainables=policy.get_optim_params()
    optimizer=torch.optim.AdamW(trainables,lr=args.lr,weight_decay=0.0)
    before=policy.adapter_hash();timings=[]
    policy.train();policy.model.vision_backbone.eval();torch.cuda.reset_peak_memory_stats()
    for trial in range(4):
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.perf_counter()
        loss,_=policy(batch);loss.backward();torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        if trial:timings.append(elapsed)
    optimizer.zero_grad(set_to_none=True)
    assert policy.adapter_hash()==before
    evaluation=[];policy.eval()
    with torch.no_grad():
        for _ in range(3):
            torch.cuda.synchronize();started=time.perf_counter();policy(batch);torch.cuda.synchronize()
            evaluation.append(time.perf_counter()-started)
    batches=math.ceil(len(train)/args.batch_size)
    validation_batches=math.ceil(len(validation)/args.batch_size)
    train_seconds=float(np.mean(timings))*batches*args.epochs
    eval_seconds=float(np.mean(evaluation))*validation_batches*(args.epochs+2)
    report={"batch_size":args.batch_size,"train_frames":len(train),"validation_frames":len(validation),
            "epochs":args.epochs,"optimizer_updates":batches*args.epochs,"train_batch_seconds":float(np.mean(timings)),
            "validation_batch_seconds":float(np.mean(evaluation)),"estimated_train_seconds":train_seconds,
            "estimated_validation_seconds":eval_seconds,"estimated_total_minutes_with_30pct_margin":(train_seconds+eval_seconds)*1.3/60,
            "no_optimizer_updates_performed":True,"trainable_parameters":sum(p.numel() for p in trainables),
            "trainable_dtype":str(trainables[0].dtype),"parameter_names":[n for n,p in policy.model.named_parameters() if p.requires_grad],**memory()}
    write_json(args.run/"calibration.json",report)
    print("CALIBRATION_COMPLETE",json.dumps({k:v for k,v in report.items() if k!="parameter_names"}),flush=True)

def save_checkpoint(policy,optimizer,run,step,epoch,seen):
    checkpoint=run/"checkpoints"/f"step_{step:06d}"
    policy.save_adapter(checkpoint)
    torch.save({"optimizer":optimizer.state_dict(),"optimizer_step":step,"epoch":epoch,
                "seen_frames":seen,"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all(),
                "numpy_rng_state":np.random.get_state(),"python_rng_state":random.getstate()},checkpoint/"trainer_state.pt")
    write_json(checkpoint/"trainer_state.json",{"optimizer_step":step,"epoch":epoch,"seen_frames":seen,"adapter_hash":policy.adapter_hash()})
    return checkpoint

def train_full(args):
    seed();train,validation,stats=get_data();policy=make_policy(stats,args.rank)
    config={"base_model":"openvla/openvla-7b","base_snapshot":policy.checkpoint_path,"dataset_root":str(DATA),"task":TASK,
            "public_data_description":"public LIBERO demonstrations selected and preprocessed into an experimental split",
            "train_episodes":20,"train_frames":1928,"validation_episodes":5,"validation_frames":462,
            "epochs":args.epochs,"batch_size":args.batch_size,"gradient_accumulation_steps":1,"learning_rate":args.lr,
            "optimizer":"AdamW","weight_decay":0.0,"gradient_clip_norm":1.0,"scheduler":"constant",
            "lora_rank":args.rank,"lora_alpha":args.rank,"lora_dropout":0.0,"lora_targets":["language_model q_proj","language_model v_proj"],
            "base_dtype":"bfloat16","lora_dtype":"float32","seed":7,"validation_every_optimizer_steps":args.validate_every,
            "image_preprocessing":"existing native HF image processor on unchanged 256px source; no augmentation; same as connection test",
            "action_normalization":"training split q01/q99 on first six dimensions; gripper unchanged; constant dimensions zero",
            "loss":"teacher-forced cross entropy on seven action tokens plus EOS; prompt ignored",
            "normalized_action_l1":"decoded argmax action versus continuous normalized demonstration action; teacher-forced",
            "checkpoint_selection":"lowest full validation loss among trained checkpoints; baseline excluded",
            "evaluation_initial_state_disjointness":"not established; data split is at demonstration level",
            "reference_functions":policy.reference_hashes}
    write_json(args.run/"training_config.json",config)
    with (args.run/"used_sources.json").open("w") as handle:
        json.dump({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"src/lerobot/policies/openvla/full_experiment_openvla.py"]},handle,indent=2)
    wandb_run=None
    try:
        import wandb
        wandb_run=wandb.init(project="openvla-libero-bowl-baseline",name=args.run.name,mode="offline",dir=str(args.run),config=config)
        wandb_run.define_metric("optimizer_step")
        wandb_run.define_metric("train/*",step_metric="optimizer_step")
        wandb_run.define_metric("validation/*",step_metric="optimizer_step")
        write_json(args.run/"wandb_status.json",{"mode":"offline","run_id":wandb_run.id,"directory":wandb_run.dir,"uploaded":False,
            "logged":["training loss","validation loss","action token accuracy","normalized action L1","GPU memory","elapsed time","checkpoint step"],
            "local_only":["base model","LoRA checkpoint","optimizer state","source snapshots","raw data","simulator frames and videos"]})
    except Exception as error:
        write_json(args.run/"wandb_status.json",{"mode":"local_logs_only","error":str(error),"uploaded":False})
    baseline=evaluate(policy,validation,args.batch_size,args.run/"validation_step_000000.jsonl",0)
    write_json(args.run/"baseline_validation.json",baseline)
    if wandb_run:wandb_run.log({"optimizer_step":0,**{"validation/"+k:v for k,v in baseline.items()}})
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=args.lr,weight_decay=0.0)
    frozen_before=frozen_sample_hash(policy);adapter_before=policy.adapter_hash()
    validation_history=[baseline];checkpoints=[];coverage=[];step=0;seen=0;started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(args.epochs):
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(7+epoch)).tolist()
            assert len(set(order))==len(train)
            epoch_seen=[]
            for offset in range(0,len(order),args.batch_size):
                indices=order[offset:offset+args.batch_size]
                policy.train();policy.model.vision_backbone.eval();optimizer.zero_grad(set_to_none=True)
                iteration=time.perf_counter()
                loss,metrics=policy(batch_from(policy,train,indices))
                loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(policy.get_optim_params(),1.0)
                if not torch.isfinite(grad_norm):raise RuntimeError("Non-finite gradient")
                optimizer.step();step+=1;seen+=len(indices);epoch_seen.extend(indices)
                row={"optimizer_step":step,"epoch":epoch+1,"frames_seen":seen,"indices":indices,"batch_frames":len(indices),
                     **metrics,"gradient_norm":float(grad_norm),"iteration_seconds":time.perf_counter()-iteration,
                     "elapsed_training_seconds":time.perf_counter()-started,**memory()}
                append_json(args.run/"training_metrics.jsonl",row)
                if step==1 or step%25==0:print("TRAIN_UPDATE",json.dumps(row),flush=True)
                if wandb_run:wandb_run.log({"optimizer_step":step,**{"train/"+k:v for k,v in row.items() if k!="indices"}})
                del loss
                final_batch=(epoch==args.epochs-1 and offset+len(indices)==len(order))
                if step%args.validate_every==0 or final_batch:
                    optimizer.zero_grad(set_to_none=True)
                    current=evaluate(policy,validation,args.batch_size,args.run/f"validation_step_{step:06d}.jsonl",step)
                    checkpoint=save_checkpoint(policy,optimizer,args.run,step,epoch+1,seen)
                    current["checkpoint"]=str(checkpoint)
                    validation_history.append(current);checkpoints.append(current)
                    append_json(args.run/"validation_history.jsonl",current)
                    if wandb_run:wandb_run.log({"optimizer_step":step,**{"validation/"+k:v for k,v in current.items() if k!="checkpoint"}})
                    print("CHECKPOINT_SAVED",checkpoint,flush=True)
            assert sorted(epoch_seen)==list(range(len(train)))
            coverage.append({"epoch":epoch+1,"frames":len(epoch_seen),"unique_frames":len(set(epoch_seen)),"indices":epoch_seen})
        assert seen==len(train)*args.epochs and policy.adapter_hash()!=adapter_before
        assert frozen_sample_hash(policy)==frozen_before
        selected=min(checkpoints,key=lambda r:r["loss"])
        report={"training_completed":True,"epochs_completed":args.epochs,"frames_seen":seen,"unique_train_frames_per_epoch":1928,
                "optimizer_steps":step,"baseline_validation":baseline,"final_validation":validation_history[-1],"selected_validation":selected,
                "selected_checkpoint":selected["checkpoint"],"final_checkpoint":checkpoints[-1]["checkpoint"],
                "training_loop_wall_seconds":time.perf_counter()-started,"adapter_changed":True,"frozen_parameter_samples_unchanged":True,
                "validation_uses_optimizer":False,"full_validation_frames":462,"simulator_evaluation_completed":False,**memory()}
        write_json(args.run/"training_coverage.json",coverage);write_json(args.run/"training_result.json",report)
        print("FULL_TRAINING_COMPLETE",json.dumps(report),flush=True)
    finally:
        if wandb_run:wandb_run.finish()

def check_checkpoint(args):
    _,validation,stats=get_data();result=json.loads((args.run/"training_result.json").read_text())
    checkpoint=Path(result["selected_checkpoint"])
    policy=make_policy(stats,args.rank);policy.load_saved_adapter(checkpoint)
    batch=batch_from(policy,validation,list(range(min(args.batch_size,len(validation)))))
    with torch.no_grad():loss,metrics=policy(batch)
    # select_action expects one 224px RGB image; this check is separate from simulator evaluation.
    sample=validation[0]
    array=np.rint(sample[policy.config.image_key].permute(1,2,0).numpy()*255).astype(np.uint8)
    image=Image.fromarray(array).resize((224,224),Image.Resampling.BICUBIC)
    tensor=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float().div(255).unsqueeze(0)
    action=policy.select_action({policy.config.image_key:tensor,"task":[TASK]})
    assert action.shape==(1,7) and bool(torch.isfinite(action).all())
    validation_file=args.run/f"validation_step_{result['selected_validation']['optimizer_step']:06d}.jsonl"
    first_record=json.loads(validation_file.read_text().splitlines()[0])
    assert first_record["indices"]==list(range(args.batch_size))
    assert abs(float(loss)-first_record["loss"])<1e-4,(float(loss),first_record["loss"])
    report={"fresh_python_process_reload":True,"checkpoint":str(checkpoint),"adapter_hash_exact":True,
            "same_validation_batch_loss_matches":True,"loss":float(loss),"action_shape":list(action.shape),
            "all_actions_finite":True,"action":action.cpu().tolist(),"metrics":metrics,"simulator_rollout":False}
    write_json(args.run/"checkpoint_reload_check.json",report)
    print("CHECKPOINT_RELOAD_COMPLETE",json.dumps(report),flush=True)

def simulate_trial(policy,tag,state_index,args):
    horizon,warmup=reference_settings();set_reference_seed()
    output=args.run/"simulation"/tag/f"state_{state_index:02d}"
    output.mkdir(parents=True,exist_ok=False)
    environment=os.environ.copy()
    environment.update({"MUJOCO_GL":"egl","TF_CPP_MIN_LOG_LEVEL":"2","TF_NUM_INTRAOP_THREADS":"2","TF_NUM_INTEROP_THREADS":"1"})
    environment["PYTHONPATH"]="/home/jovyan/projects/LIBERO"+(os.pathsep+environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    worker=ROOT/"examples/libero_eval_worker.py"
    process=None;records=[];initial=None;current=None;started=time.perf_counter()
    with (output/"simulator.log").open("w") as log:
        try:
            process=subprocess.Popen(["/home/jovyan/projects/venvs/openvla_eval310/bin/python","-u",str(worker),
                "--output",str(output),"--task-id","0","--init-state-index",str(state_index)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=environment)
            initial=current=read_observation(process,timeout=240)
            assert initial["type"]=="ready" and initial["task"]==TASK
            assert initial["actual_initial_image_reference_check"]["max_pixel_error"]==0
            write_json(output/"initialization.json",initial)
            initial_hash=hashlib.sha256(Path(initial["frame"]).read_bytes()).hexdigest()
            baseline_file=args.run/"simulation/base"/f"state_{state_index:02d}"/"result.json"
            if tag!="base" and baseline_file.exists():
                baseline=json.loads(baseline_file.read_text())
                if baseline.get("trial_executed"):
                    assert baseline["selected_init_state_sha256"]==initial["selected_init_state_sha256"]
                    assert baseline["initial_image_sha256"]==initial_hash
            low,high=np.asarray(initial["action_low"]),np.asarray(initial["action_high"])
            frames=[initial["frame"]];policy.reset()
            for step in range(1,horizon+1):
                with Image.open(current["policy_frame"]) as source:image=source.convert("RGB")
                pixels=np.asarray(image).copy()
                tensor=torch.from_numpy(pixels).permute(2,0,1).float().div(255)
                assert np.array_equal(np.rint(tensor.permute(1,2,0).numpy()*255).astype(np.uint8),pixels)
                action=policy.select_action({policy.config.image_key:tensor.unsqueeze(0),"task":[TASK]})
                assert action.shape==(1,7) and bool(torch.isfinite(action).all())
                raw=action[0].cpu().numpy()
                converted,command=controller_action(raw,low,high)
                assert np.array_equal(converted,command),"Action would require clipping"
                before=current;send(process,{"type":"step","action":command.tolist()});current=read_observation(process)
                assert current["type"]=="observation" and current["step"]==step
                row={"step":step,"raw_action":raw.tolist(),"controller_action":command.tolist(),"before":before,"after":current}
                records.append(row);append_json(output/"steps.jsonl",row);frames.append(current["frame"])
                if step==1 or step%25==0 or current["task_success"]:
                    print("SIM_STEP",tag,state_index,step,"success",current["task_success"],flush=True)
                if current["task_success"]:break
            success=bool(current["task_success"])
            save_media(frames,output,success)
            # The old media helper labels state 0; correct that label in this new output only.
            comparison=Image.open(output/"before_after.png").convert("RGB")
            draw=ImageDraw.Draw(comparison);draw.rectangle((0,0,255,23),fill="white")
            draw.text((6,6),f"Initial state {state_index}",fill="black");comparison.save(output/"before_after.png")
            report={"trial_executed":True,"model_tag":tag,"task":TASK,"task_id":0,"init_state_index":state_index,
                "success":success,"termination_reason":"task_success" if success else "horizon_exhausted",
                "failure_detail":None if success else "LIBERO success predicate not reached within the fixed 220-action horizon; see video for motion-level diagnosis",
                "policy_steps":len(records),"max_policy_steps":horizon,"warmup_steps":warmup,
                "environment_seed":0,"global_seed":7,"action_shape":[1,7],"all_actions_finite":True,"action_clipping_used":False,
                "reference_preprocessing_pixel_parity":True,"initial_image_sha256":initial_hash,
                "selected_init_state_sha256":initial["selected_init_state_sha256"],"init_state_file_sha256":initial["init_state_file_sha256"],
                "reference_sha256":initial["reference_sha256"],"simulator_versions":initial["simulator_versions"],
                "initial_eef_position":initial["eef_position"],"final_eef_position":current["eef_position"],
                "wall_seconds":time.perf_counter()-started,"video":str(output/"episode.mp4"),"overview":str(output/"episode_overview.png")}
        except Exception as error:
            report={"trial_executed":False,"model_tag":tag,"init_state_index":state_index,"success":False,
                    "termination_reason":"technical_error","error":repr(error),"policy_steps":len(records),"wall_seconds":time.perf_counter()-started}
            print("SIM_TECHNICAL_ERROR",tag,state_index,repr(error),flush=True)
        finally:
            if process is not None and process.poll() is None:
                try:send(process,{"type":"close"});process.wait(timeout=15)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
    write_json(output/"result.json",report)
    print("SIM_TRIAL_COMPLETE",json.dumps(report),flush=True)
    return report

def simulate(args):
    assert args.model in ["base","lora","reference"]
    stats=json.loads((DATA/"action_statistics_train_only.json").read_text())
    if args.model=="reference":
        seed();policy=OpenVLAPolicy(OpenVLAConfig(device="cuda:0"));checkpoint=None
    else:
        policy=make_policy(stats,args.rank);checkpoint=None
        if args.model=="lora":
            checkpoint=json.loads((args.run/"training_result.json").read_text())["selected_checkpoint"]
            policy.load_saved_adapter(checkpoint)
        for parameter in policy.model.parameters():parameter.requires_grad_(False)
        policy.eval()
    context=policy.model.disable_adapter() if args.model=="base" else contextlib.nullcontext()
    with context:
        trials=[simulate_trial(policy,args.model,index,args) for index in args.states]
    successes=sum(bool(row["success"]) for row in trials)
    executed=[row for row in trials if row["trial_executed"]]
    summary={"model_tag":args.model,"model_id":policy.config.model_id,"checkpoint":checkpoint,"successes":successes,
             "attempts":len(trials),"success_rate":successes/len(trials),"technical_errors":len(trials)-len(executed),
             "executed_trials":len(executed),"valid_trial_success_rate":successes/len(executed) if executed else None,
             "mean_policy_steps":float(np.mean([row["policy_steps"] for row in executed])) if executed else None,
             "states":args.states,"horizon":220,"warmup_steps":10,"environment_seed":0,"global_seed":7,
             "action_normalization":"published libero_spatial checkpoint statistics" if args.model=="reference" else "shared training-only bowl-20 statistics; identical for base and LoRA",
             "scope":"one task, five fixed initial states; not a LIBERO-wide benchmark or evidence of unseen-state generalization",
             "trials":trials}
    write_json(args.run/f"simulation_{args.model}.json",summary)
    print("SIMULATION_COHORT_COMPLETE",json.dumps(summary),flush=True)

def audit_after(args):
    before=json.loads((args.run/"sources_before.json").read_text())
    changed=[p for p,digest in before.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest]
    environment=json.loads((args.run/"environment_audit.json").read_text())
    environments={}
    for name,entry in environment.items():
        result=subprocess.run([entry["interpreter"],"-m","pip","freeze"],capture_output=True,text=True,check=True,timeout=120)
        (args.run/(name+"_after.txt")).write_text(result.stdout)
        environments[name]={"packages_unchanged":sorted(result.stdout.splitlines())==sorted((args.run/(name+"_before.txt")).read_text().splitlines())}
    report={"original_source_files":len(before),"changed_original_sources":changed,"existing_sources_unchanged":not changed,"environments":environments}
    write_json(args.run/"preservation_report.json",report)
    assert not changed and all(e["packages_unchanged"] for e in environments.values())
    print("PRESERVATION_COMPLETE",json.dumps(report),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["calibrate","train","check","simulate","audit"])
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--rank",type=int,default=8)
    parser.add_argument("--validate-every",type=int,default=241)
    parser.add_argument("--model",choices=["base","lora","reference"],default="base")
    parser.add_argument("--states",type=int,nargs="+",default=[0,1,2,3,4])
    args=parser.parse_args();args.run=args.run.absolute()
    assert args.run.is_dir() and args.batch_size>=1 and args.epochs>=1
    try:
        {"calibrate":calibrate,"train":train_full,"check":check_checkpoint,"simulate":simulate,"audit":audit_after}[args.phase](args)
    except Exception as error:
        import traceback
        write_json(args.run/(args.phase+"_error.json"),{"phase":args.phase,"error":repr(error),"traceback":traceback.format_exc()})
        raise

if __name__=="__main__":main()
