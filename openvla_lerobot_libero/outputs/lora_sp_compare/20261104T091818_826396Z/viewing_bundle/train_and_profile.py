"""Actual loop: read data -> prepare batch -> loss -> backward/update -> validate/save."""
from pathlib import Path
import argparse,json,time,math,random,gc
from comparison_common import *
def profile(method):
 out=RUN/method;out.mkdir(exist_ok=True)
 train,val,stats=exp.get_data();policy=make_policy(method,stats)
 params=policy.get_optim_params();count=sum(p.numel() for p in params)
 batch=exp.batch_from(policy,train,[0,1,2,3])
 policy.eval()
 with torch.no_grad():initial_loss,initial_metrics=policy(exp.batch_from(policy,val,[0,1,2,3]))
 initial_hash=policy.adapter_hash()
 policy.save_adapter(out/"profile_initial")
 ids=torch.randperm(len(train),generator=torch.Generator().manual_seed(713)).tolist()[:144]
 optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
 for offset in range(128,144,4):step(policy,optimizer,train,ids[offset:offset+4])
 optimizer.zero_grad(set_to_none=True);del optimizer
 policy.load_saved_adapter(out/"profile_initial")
 assert policy.adapter_hash()==initial_hash
 optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 start=time.perf_counter();rows=[]
 for offset in range(0,128,4):
  tick=time.perf_counter();metrics=step(policy,optimizer,train,ids[offset:offset+4]);torch.cuda.synchronize()
  rows.append({"offset":offset,"seconds":time.perf_counter()-tick,**metrics})
 elapsed=time.perf_counter()-start
 optimizer.zero_grad(set_to_none=True);clear_aux(policy)
 policy.eval()
 with torch.no_grad():smoke_loss,smoke_metrics=policy(exp.batch_from(policy,val,[0,1,2,3]))
 policy.save_adapter(out/"profile_smoke")
 shape_batch={k:list(v.shape) for k,v in batch.items()}
 policy=prepare_inference(policy)
 action=sample_action(policy,val,0)
 assert action.shape==(1,7) and bool(torch.isfinite(action).all())
 report={"method":method,"trainable_parameters":count,"initial_validation_loss_four_frames":float(initial_loss),
 "profile_smoke_validation_loss_four_frames":float(smoke_loss),"profile_smoke_checkpoint":str(out/"profile_smoke"),
 "measured_training_frames":128,"warmup_frames":16,"same_training_indices":ids[:128],
 "seconds":elapsed,"frames_per_second":128/elapsed,"peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
 "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,
 "estimated_five_epoch_training_minutes_excluding_validation_and_saving":elapsed/128*4025*5/60,
 "action_shape":list(action.shape),"all_actions_finite":True,"batch_shapes":shape_batch,
 "training_names":[n for n,p in policy.model.named_parameters() if any(key in n for key in ["lora_A","lora_B","sp_A","sp_B","router1","router2"])],
 "profile_weights_not_used_for_full_training":True,"steps":rows}
 write(out/"profile.json",report)
 print("PROFILE_COMPLETE",method,count,elapsed,report["peak_allocated_gib"],flush=True)

def reload_smoke(method):
 _,val,stats=exp.get_data();policy=make_policy(method,stats)
 profile=json.loads((RUN/method/"profile.json").read_text())
 policy.load_saved_adapter(profile["profile_smoke_checkpoint"])
 with torch.no_grad():loss,_=policy(exp.batch_from(policy,val,[0,1,2,3]))
 assert abs(float(loss)-profile["profile_smoke_validation_loss_four_frames"])<1e-4
 policy=prepare_inference(policy);action=sample_action(policy,val,0)
 assert action.shape==(1,7) and bool(torch.isfinite(action).all())
 write(RUN/method/"profile_reload.json",{"fresh_process":True,"adapter_hash_exact":True,"four_frame_loss_matches":True,
 "shape":list(action.shape),"finite":True,"action":action.cpu().tolist()})
 print("FRESH_RELOAD_OK",method,flush=True)

def train_full(method):
 assert method in ["lora128","sp128"]
 out=RUN/method
 assert not (out/"training_metrics.jsonl").exists(),"Existing partial run preserved; explicit resume required."
 train,val,stats=exp.get_data()
 policy=make_policy(method,stats)
 optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=0.)
 frozen=exp.frozen_sample_hash(policy);initial_hash=policy.adapter_hash()
 if method=="sp128":policy.record_rank_stats(True)
 baseline=exp.evaluate(policy,val,4,out/"validation_step_000000.jsonl",0)
 if method=="sp128":
  baseline["mean_active_rank"]=policy.rank_stats()["mean_active_rank"];policy.record_rank_stats(False)
 write(out/"baseline_validation.json",baseline)
 wb=None
 try:
  import wandb
  wb=wandb.init(project="openvla-libero-bowl-baseline",name=method+"-"+RUN.name,mode="offline",dir=str(out),
       config={"comparison":str(RUN),"method":method,"epochs":5,"rank":128,"eta":.9 if method=="sp128" else None,
       "batch_size":4,"lr":1e-4,"main_loss":"action+EOS cross entropy","validation_optimizer":False})
  write(out/"wandb_status.json",{"mode":"offline","uploaded":False,"run_id":wb.id,"directory":wb.dir})
 except Exception as e:write(out/"wandb_status.json",{"mode":"local_only","error":repr(e),"uploaded":False})
 step_number=0;seen=0;history=[];coverage=[]
 start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 try:
  for epoch in range(1,6):
   order=torch.randperm(len(train),generator=torch.Generator().manual_seed(7+epoch-1)).tolist()
   losses=0.;total_losses=0.;epoch_seen=[];tick=time.monotonic()
   for offset in range(0,len(order),4):
    indices=order[offset:offset+4]
    metrics=step(policy,optimizer,train,indices)
    step_number+=1;seen+=len(indices);epoch_seen.extend(indices)
    losses+=metrics["loss"]*len(indices);total_losses+=metrics.get("total_loss",metrics["loss"])*len(indices)
    row={"epoch":epoch,"optimizer_step":step_number,"frames_seen":seen,"batch_frames":len(indices),
         **metrics,"elapsed_seconds":time.monotonic()-start,**exp.memory()}
    exp.append_json(out/"training_metrics.jsonl",row)
    if wb and step_number%10==0:wb.log(row,step=step_number)
    if step_number%25==0 or offset+len(indices)==len(order):
     write(out/"training_progress.json",{"epoch":epoch,"frames_this_epoch":len(epoch_seen),"epoch_frames":4025,
       "optimizer_step":step_number,"target_steps":5035,"elapsed_seconds":time.monotonic()-start,"last_task_loss":metrics["loss"],
       "method":method,**exp.memory()})
     print("TRAIN_PROGRESS",method,epoch,len(epoch_seen),"/4025",step_number,flush=True)
   assert sorted(epoch_seen)==list(range(4025))
   optimizer.zero_grad(set_to_none=True);clear_aux(policy)
   if method=="sp128":policy.record_rank_stats(True)
   validation=exp.evaluate(policy,val,4,out/f"validation_step_{step_number:06d}.jsonl",step_number)
   if method=="sp128":
    ranks=policy.rank_stats();validation["mean_active_rank"]=ranks["mean_active_rank"]
    write(out/f"active_rank_epoch_{epoch:02d}.json",ranks);policy.record_rank_stats(False)
   checkpoint=exp.save_checkpoint(policy,optimizer,out,step_number,epoch,seen)
   record={"epoch":epoch,"training_loss_frame_weighted":losses/4025,"training_total_loss":total_losses/4025,
     "epoch_seconds":time.monotonic()-tick,**validation,"checkpoint":str(checkpoint)}
   history.append(record);coverage.append({"epoch":epoch,"frames":len(epoch_seen),"unique_frames":len(set(epoch_seen))})
   write(out/"validation_history.json",history);write(out/"training_coverage.json",coverage)
   write(out/f"epoch_{epoch:02d}_ready.json",record)
   if wb:wb.log({"validation/"+k:v for k,v in validation.items() if isinstance(v,(float,int))},step=step_number)
   print("EPOCH_COMPLETE",method,epoch,validation["loss"],flush=True)
  assert step_number==5035 and seen==20125
  assert policy.adapter_hash()!=initial_hash
  assert exp.frozen_sample_hash(policy)==frozen
  result={"method":method,"training_completed":True,"epochs_completed":5,"frames_seen":seen,"optimizer_steps":step_number,
    "baseline_validation":baseline,"final_validation":history[-1],"final_checkpoint":history[-1]["checkpoint"],
    "selected_checkpoint":history[-1]["checkpoint"],"selected_validation":history[-1],
    "adapter_changed":True,"frozen_parameter_samples_unchanged":True,"validation_uses_optimizer":False,
    "training_wall_seconds":time.monotonic()-start,**exp.memory()}
  write(out/"training_result.json",result)
  print("TRAINING_COMPLETE",method,json.dumps(result),flush=True)
 finally:
  if wb:wb.finish()
def main():
 p=argparse.ArgumentParser();p.add_argument("phase",choices=["profile","reload","train"])
 p.add_argument("--method",choices=METHODS,required=True);a=p.parse_args()
 if not (RUN/"experiment_config.json").exists():configure()
 try:{"profile":profile,"reload":reload_smoke,"train":train_full}[a.phase](a.method)
 except Exception as e:
  import traceback
  write(RUN/a.method/(a.phase+"_error.json"),{"error":repr(e),"traceback":traceback.format_exc()});raise
if __name__=="__main__":main()
