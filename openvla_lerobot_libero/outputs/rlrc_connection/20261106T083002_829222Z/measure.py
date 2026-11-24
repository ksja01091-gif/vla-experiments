"""Fresh-process SFT/inference/PPO probes for one branch, using disposable updates."""
import argparse,time,traceback
from policy import *
from rl_probe import rollout,update_probe
p=argparse.ArgumentParser();p.add_argument('--branch',choices=['dense','pruned'],required=True);a=p.parse_args()
out=R/a.branch;out.mkdir(exist_ok=False)
try:
    train,val,stats=exp.get_data();exp.seed();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    policy=load_checkpoint(R/('checkpoints/'+a.branch+'_initial'),stats)
    load_seconds=time.perf_counter()-tick;load_memory=memory()
    frozen_before=exp.frozen_sample_hash(policy)
    actor_before=policy.actor_hash();policy.configure_trainables(False)
    assert all('.recovery_' in n for n,p in policy.model.named_parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=0.)
    times=[];rows=[];torch.cuda.reset_peak_memory_stats()
    for i in range(4):
        policy.train();policy.model.vision_backbone.eval();optimizer.zero_grad(set_to_none=True)
        batch=exp.batch_from(policy,train,list(range(i*4,i*4+4)))
        torch.cuda.synchronize();tick=time.perf_counter()
        loss,metrics=policy(batch);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(policy.get_optim_params(),1.)
        assert torch.isfinite(grad);optimizer.step();torch.cuda.synchronize();seconds=time.perf_counter()-tick
        times.append(seconds);rows.append({'optimizer_step':i+1,'train_indices':list(range(i*4,i*4+4)),
            'seconds':seconds,'gradient_norm':float(grad),**metrics})
        write(out/'sft_progress.json',{'completed':i+1,'total':4})
        print('SFT_PROBE',a.branch,i+1,seconds,metrics['loss'],flush=True)
        del loss,batch
    train_memory=memory();assert policy.actor_hash()!=actor_before
    assert exp.frozen_sample_hash(policy)==frozen_before
    optimizer.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()
    policy.eval()
    with torch.no_grad():
        tick=time.perf_counter();_,vm=policy(exp.batch_from(policy,val,[0,100,300,461]));torch.cuda.synchronize()
        validation_seconds=time.perf_counter()-tick
    del vm
    infer_indices=list(map(int,np.linspace(0,len(val)-1,8,dtype=int)))
    common.sample_action(policy,val,infer_indices[0]);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    inference=[]
    for i in infer_indices:
        torch.cuda.synchronize();tick=time.perf_counter();action=common.sample_action(policy,val,i)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        assert action.shape==(1,7) and bool(torch.isfinite(action).all())
        inference.append({'index':i,'seconds':seconds,'shape':[1,7],'finite':True,'action':action.cpu().tolist()})
    inference_memory=memory()
    sft={'updates':4,'training_frames':16,'batch_size':4,'lr':1e-4,'rank':8,'not_full_training':True,
         'mean_update_seconds_excluding_first':float(np.mean(times[1:])),
         'all_update_seconds':times,'validation_four_frame_seconds':validation_seconds,
         'memory':train_memory,'frozen_samples_unchanged':True,'rows':rows}
    write(out/'sft_probe.json',sft)
    write(out/'inference_probe.json',{'samples':8,'mean_seconds':float(np.mean([x['seconds'] for x in inference])),
         'memory':inference_memory,'rows':inference,'not_a_repeated_final_speed_benchmark':True})
    before_rl=policy.parameter_report();rlrows,rlstats=rollout(policy,out/'rl_rollout',max_steps=8)
    write(out/'rollout_probe.json',rlstats)
    ppo=update_probe(policy,rlrows,train);write(out/'ppo_probe.json',ppo)
    expected_action=common.sample_action(policy,val,0).cpu().tolist()
    checkpoint=out/'disposable_after_ppo';filebytes=save_checkpoint(policy,checkpoint)
    write(out/'reload_expected.json',{'checkpoint':str(checkpoint),'actor_hash':policy.actor_hash(),
        'validation_index':0,'action':expected_action,'shape':[1,7],'finite':True})
    write(out/'complete.json',{'complete':True,'branch':a.branch,'load_seconds':load_seconds,'load_memory':load_memory,
        'parameters':before_rl,'checkpoint':str(checkpoint),'saved_file_bytes':filebytes,
        'sft':sft,'inference_seconds':float(np.mean([x['seconds'] for x in inference])),
        'inference_peak_gib':inference_memory['peak_gib'],'rollout':rlstats,'ppo':ppo,
        'not_a_full_episode_or_quality_test':True,'original_start_adapter_unchanged':True})
    print('BRANCH_SMOKE_COMPLETE',a.branch,flush=True)
except Exception as e:
    write(R/(a.branch+'_error.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
