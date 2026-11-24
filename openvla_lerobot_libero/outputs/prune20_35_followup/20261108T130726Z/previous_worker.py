"""Bounded matched SFT/PPO recovery and fixed-state evaluation on remote Linux."""
import argparse, os, subprocess, time, traceback, signal
from policy import *
from rl_probe import inputs_from_image, teacher, sample, bootstrap_value
from PIL import Image

CFG=json.loads((R/'experiment_config.json').read_text())

class BudgetExceeded(RuntimeError): pass
def check_budget(reserve=90):
    deadline=json.loads((R/'budget.json').read_text())['deadline_unix']
    if time.time()+reserve>=deadline: raise BudgetExceeded('Twelve-hour safety budget nearly exhausted')

def terminate(signum,frame): raise BudgetExceeded('Run supervisor requested a checkpoint stop')
signal.signal(signal.SIGTERM,terminate)

def append(path,row): exp.append_json(Path(path),row)
def progress(phase,branch,**kw):
    write(R/'progress.json',{'phase':phase,'branch':branch,'updated_unix':time.time(),**kw})

def load(branch,stage):
    train,val,stats=exp.get_data();exp.seed();check_budget()
    if stage=='after_sft' and CFG.get('stage2_start_checkpoint')=='epoch3': stage='epoch3'
    cp=R/'checkpoints'/branch/stage
    policy=load_checkpoint(cp,stats)
    return policy,train,val,cp

def save(policy,branch,stage,optimizer=None,**state):
    cp=R/'checkpoints'/branch/stage
    n=save_checkpoint(policy,cp)
    trainer={'optimizer':optimizer.state_dict() if optimizer else None,
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
        'numpy_rng':np.random.get_state(),**state}
    torch.save(trainer,cp/'trainer_state.pt')
    return cp,n

@torch.no_grad()
def validation(policy,data,branch,stage):
    check_budget();out=R/branch;out.mkdir(exist_ok=True)
    before=policy.actor_hash();policy.eval()
    result=exp.evaluate(policy,data,4,out/f'validation_{stage}.jsonl',0)
    assert result['frames']==462 and before==policy.actor_hash()
    write(out/f'validation_{stage}.json',result)
    return result

def prepare():
    check_budget();train,val,stats=exp.get_data();exp.seed()
    policy=RecoveryPolicy(stats)
    before=policy.parameter_report()
    cp,n=save(policy,'dense','initial')
    old=json.loads(Path(CFG['calibration_source']).read_text())
    indices=[int(i) for e in old['episodes'] for i in e['indices']]
    assert len(indices)==len(set(indices))==160 and all(0<=i<len(train) for i in indices)
    write(R/'calibration_selection.json',{'split':'train','indices':indices,'frames':160,'validation_used':False})
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    plan=score_and_plan(policy,train,indices,R/'importance_progress.json')
    importance_mem=memory();write(R/'pruning_plan.json',plan)
    apply_plan(policy,plan);after=policy.parameter_report()
    cp,bytes_=save(policy,'pruned','initial')
    assert before['model_parameters']>after['model_parameters']
    write(R/'prepare_complete.json',{'complete':True,'before':before,'after':after,
        'importance_frames':160,'importance_seconds':time.perf_counter()-tick,
        'importance_memory':importance_mem,'pruned_weights_bytes':bytes_,
        'target_internal_pruning':.9,'actual_target_heads_removed':29/32,
        'actual_target_mlp_removed':1-1088/11008,
        'whole_parameter_reduction':1-after['model_parameters']/before['model_parameters']})

def sft(branch):
    policy,train,val,_=load(branch,'initial');optimizer=None
    out=R/branch;out.mkdir(exist_ok=True)
    try:
        base=validation(policy,val,branch,'initial')
        policy.configure_trainables(False);frozen=exp.frozen_sample_hash(policy)
        actor_before=policy.actor_hash();order=torch.randperm(len(train),generator=torch.Generator().manual_seed(7)).tolist()
        optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=0.)
        torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();coverage=[];sums={k:0. for k in exp.METRICS}
        for offset in range(0,len(order),4):
            check_budget();indices=order[offset:offset+4]
            t=time.perf_counter();m=common.step(policy,optimizer,train,indices)
            coverage.extend(indices)
            for k in sums:sums[k]+=m[k]*len(indices)
            row={'step':offset//4+1,'indices':indices,'frames_seen':len(coverage),
                 'seconds':time.perf_counter()-t,**m,**memory()}
            append(out/'sft_metrics.jsonl',row)
            if offset==0 or row['step']%25==0:
                progress('sft',branch,completed_frames=len(coverage),total_frames=len(train),loss=m['loss'])
                print('SFT',branch,row['step'],len(coverage),'/',len(train),'loss',m['loss'],flush=True)
        elapsed=time.perf_counter()-tick;mem=memory();optimizer.zero_grad(set_to_none=True)
        assert sorted(coverage)==list(range(4025)) and policy.actor_hash()!=actor_before
        assert exp.frozen_sample_hash(policy)==frozen
        after=validation(policy,val,branch,'after_sft')
        cp,n=save(policy,branch,'after_sft',optimizer,training_phase='sft',updates=math.ceil(4025/4),coverage=coverage)
        write(out/'sft_result.json',{'complete':True,'epochs':1,'frames_seen':len(coverage),
            'unique_frames':len(set(coverage)),'updates':math.ceil(4025/4),
            'coverage_sha256':hashlib.sha256(json.dumps(coverage).encode()).hexdigest(),
            'training_metrics':{k:v/len(coverage) for k,v in sums.items()},
            'before_validation':base,'after_validation':after,'checkpoint':str(cp),
            'checkpoint_weight_bytes':n,'seconds':elapsed,'memory':mem,'frozen_samples_unchanged':True})
    except BudgetExceeded:
        save(policy,branch,'interrupted_sft',optimizer,training_phase='sft',covered=locals().get('coverage',[]))
        raise

def collect_episode(policy,branch,state,episode_index=None):
    check_budget();output=R/'rl_collection'/branch/f'episode_{episode_index:03d}_state_{state:02d}'
    output.mkdir(parents=True,exist_ok=False)
    environment=os.environ.copy();environment.update(MUJOCO_GL='egl',TF_CPP_MIN_LOG_LEVEL='2',
        TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
    environment['PYTHONPATH']='/home/jovyan/projects/LIBERO'+(os.pathsep+environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    process=None;rows=[];times=[];frames=[];success=False;bootstrap=0.;started=time.perf_counter()
    policy._rollout_generator=torch.Generator(device=policy.config.device).manual_seed(71000+state+1000*episode_index)
    with (output/'simulator.log').open('w') as log:
        try:
            process=subprocess.Popen(['/home/jovyan/projects/venvs/openvla_eval310/bin/python','-u',
                '/home/jovyan/projects/lerobot-openvla/examples/libero_eval_worker.py','--output',str(output),
                '--task-id','0','--init-state-index',str(state)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=log,text=True,bufsize=1,env=environment)
            observation=exp.read_observation(process,timeout=240)
            assert observation['type']=='ready' and observation['task']==exp.TASK
            assert observation['actual_initial_image_reference_check']['max_pixel_error']==0
            assert state not in CFG['evaluation_states']
            write(output/'initialization.json',observation);frames=[observation['frame']]
            low,high=np.asarray(observation['action_low']),np.asarray(observation['action_high'])
            for step in range(220):
                check_budget();policy.eval()
                with Image.open(observation['frame']) as image:inputs=inputs_from_image(policy,image.convert('RGB'))
                t=time.perf_counter();tokens,raw,logp,value,diff=sample(policy,inputs)
                converted,command=exp.controller_action(raw,low,high)
                assert np.array_equal(converted,command)
                exp.send(process,{'type':'step','action':command.tolist()})
                observation=exp.read_observation(process);success=bool(observation['task_success'])
                reward=float(success)-.001
                rows.append({'inputs':{k:v.detach().cpu() for k,v in inputs.items()},'tokens':tokens,
                    'old_logp':logp,'value':value,'reward':reward,'terminal':success,
                    'state':state,'step':step+1,'generation_logp_max_difference':diff})
                frames.append(observation['frame']);times.append(time.perf_counter()-t)
                append(output/'steps.jsonl',{'step':step+1,'action':raw.tolist(),'tokens':tokens.tolist(),
                    'old_joint_logp':logp,'value':value,'reward':reward,'success':success,'logp_difference':diff})
                if (step+1)%25==0 or success:
                    progress('rl_collect',branch,state=state,step=step+1,max_steps=220,task_success=success)
                    print('RL_COLLECT',branch,state,step+1,success,flush=True)
                if success: break
            if not success:
                with Image.open(observation['frame']) as image:bootstrap=bootstrap_value(policy,inputs_from_image(policy,image.convert('RGB')))
        finally:
            if process is not None and process.poll() is None:
                try:exp.send(process,{'type':'close'});process.wait(timeout=15)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):process.terminate();process.wait(timeout=15)
    gae=0.;next_value=bootstrap
    for row in reversed(rows):
        continuation=0. if row['terminal'] else 1.
        delta=row['reward']+.99*next_value*continuation-row['value']
        gae=delta+.99*.95*continuation*gae
        row['return']=gae+row['value'];row['advantage_raw']=gae;next_value=row['value']
    exp.save_media(frames,output,success)
    report={'state':state,'episode_index':episode_index,'steps':len(rows),'success':success,'bootstrap':bootstrap,
        'truncated':not success,'reward_sum':sum(r['reward'] for r in rows),
        'seconds':time.perf_counter()-started,'mean_transition_seconds':float(np.mean(times)),
        'max_sampling_logp_difference':max(r['generation_logp_max_difference'] for r in rows),
        'video':str(output/'episode.mp4'),'training_rollout_not_final_evaluation':True}
    write(output/'result.json',report)
    return rows,report

def collect(branch):
    policy,train,val,_=load(branch,'after_sft');before=policy.actor_hash()
    rows=[];trials=[]
    for episode_index,state in enumerate(CFG['rl_training_states']):
        r,t=collect_episode(policy,branch,state,episode_index);rows.extend(r);trials.append(t)
        torch.save(r,R/'rl_collection'/branch/f'episode_{episode_index:03d}_state_{state:02d}'/'transitions.pt')
        write(R/branch/'collection_progress.json',{'episodes':len(trials),'transitions':len(rows),'trials':trials})
    torch.save(rows,R/branch/'on_policy_transitions.pt')
    assert policy.actor_hash()==before
    write(R/branch/'collection_result.json',{'complete':True,'episodes':len(trials),'transitions':len(rows),
        'successes':sum(t['success'] for t in trials),'trials':trials,'actor_unchanged':True})

def ppo(branch):
    policy,train,val,_=load(branch,'after_sft');out=R/branch;optimizer=None
    try:
        match=json.loads((R/'matched_ppo_budget.json').read_text());n=match['transitions_each']
        source=torch.load(out/'on_policy_transitions.pt',weights_only=False,map_location='cpu')
        chosen=np.linspace(0,len(source)-1,n,dtype=int).tolist();assert len(set(chosen))==n
        rows=[source[i] for i in chosen]
        advantages=np.asarray([r['advantage_raw'] for r in rows],dtype=np.float32)
        norm=(advantages-advantages.mean())/(advantages.std()+1e-8)
        for row,adv in zip(rows,norm):row['advantage']=float(adv)
        policy.configure_trainables(False);frozen=exp.frozen_sample_hash(policy);actor=policy.actor_hash()
        policy.model.requires_grad_(False);policy.value_head.requires_grad_(True);policy.eval()
        critic=torch.optim.AdamW(policy.value_head.parameters(),lr=1e-4,weight_decay=0.)
        for row in rows[:2]:
            check_budget();critic.zero_grad(set_to_none=True)
            inputs={k:v.to(policy.config.device) for k,v in row['inputs'].items()}
            _,value,_,_=teacher(policy,inputs,row['tokens'])
            loss=.5*(value-row['return']).square().mean();loss.backward();critic.step()
        assert policy.actor_hash()==actor
        critic.zero_grad(set_to_none=True);del critic
        policy.configure_trainables(True)
        optimizer=torch.optim.AdamW([*policy.get_optim_params(),*policy.value_head.parameters()],lr=1e-5,weight_decay=0.)
        torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();updates=0;used=[];first_ratios=[]
        bc_order=torch.randperm(len(train),generator=torch.Generator().manual_seed(17007)).tolist()
        for epoch in range(2):
            order=torch.randperm(n,generator=torch.Generator().manual_seed(9000+epoch)).tolist();used.extend(order)
            for offset in range(0,n,4):
                check_budget();policy.train();policy.model.vision_backbone.eval();optimizer.zero_grad(set_to_none=True)
                metrics=[];t=time.perf_counter()
                for idx in order[offset:offset+4]:
                    row=rows[idx];inputs={k:v.to(policy.config.device) for k,v in row['inputs'].items()}
                    logp,value,entropy,_=teacher(policy,inputs,row['tokens'])
                    ratio=(logp-row['old_logp']).exp()
                    assert torch.isfinite(ratio).all()
                    if updates==0:
                        first_ratios.append(float(ratio.detach()));assert abs(float(ratio.detach())-1.)<1e-4
                    actor_loss=-torch.minimum(ratio*row['advantage'],ratio.clamp(.8,1.2)*row['advantage']).mean()
                    value_loss=.5*(value-row['return']).square().mean()
                    loss=(actor_loss+.5*value_loss)/4
                    loss.backward()
                    metrics.append({'actor_loss':float(actor_loss.detach()),'value_loss':float(value_loss.detach()),
                        'ratio':float(ratio.detach()),'entropy':float(entropy.detach())})
                bc_index=bc_order[updates%len(bc_order)]
                bc_loss,bc_m=policy(exp.batch_from(policy,train,[bc_index]));(.1*bc_loss).backward()
                grad=torch.nn.utils.clip_grad_norm_([*policy.get_optim_params(),*policy.value_head.parameters()],1.)
                assert torch.isfinite(grad);optimizer.step();updates+=1
                row={'update':updates,'ppo_pass':epoch+1,'transitions':[int(i) for i in order[offset:offset+4]],
                    'bc_train_index':bc_index,'bc_loss':bc_m['loss'],'gradient_norm':float(grad),
                    'seconds':time.perf_counter()-t,**{k:float(np.mean([x[k] for x in metrics])) for k in metrics[0]},**memory()}
                append(out/'ppo_metrics.jsonl',row)
                if updates==1 or updates%10==0:
                    progress('ppo',branch,completed_updates=updates,total_updates=match['optimizer_updates_each'],actor_loss=row['actor_loss'])
                    print('PPO',branch,updates,'/',match['optimizer_updates_each'],row['actor_loss'],flush=True)
        elapsed=time.perf_counter()-tick;mem=memory();optimizer.zero_grad(set_to_none=True)
        assert updates==match['optimizer_updates_each'] and len(used)==2*n
        assert policy.actor_hash()!=actor
        policy.configure_trainables(False);assert exp.frozen_sample_hash(policy)==frozen
        result=validation(policy,val,branch,'after_ppo')
        action=common.sample_action(policy,val,0).cpu().tolist()
        cp,bytes_=save(policy,branch,'after_ppo',optimizer,training_phase='ppo',updates=updates,selected_source_indices=chosen)
        write(out/'reload_expected.json',{'actor_hash':policy.actor_hash(),'action':action,'validation_index':0})
        write(out/'ppo_result.json',{'complete':True,'transitions_each':n,'source_transitions':len(source),
            'unused_source_transitions':len(source)-n,'passes':2,'updates':updates,'first_update_ratios':first_ratios,
            'critic_warmup_updates':2,'warmup_actor_unchanged':True,'frozen_samples_unchanged':True,
            'final_validation':result,'checkpoint':str(cp),'checkpoint_weight_bytes':bytes_,
            'seconds':elapsed,'memory':mem,'normalized_advantage_std':float(norm.std()),
            'return_min':min(r['return'] for r in rows),'return_max':max(r['return'] for r in rows)})
    except BudgetExceeded:
        save(policy,branch,'interrupted_ppo',optimizer,training_phase='ppo',updates=locals().get('updates',0))
        raise

def evaluate(branch):
    policy,train,val,cp=load(branch,'after_ppo');out=R/branch
    expected=json.loads((out/'reload_expected.json').read_text());before=policy.actor_hash()
    assert before==expected['actor_hash']
    action=common.sample_action(policy,val,0).cpu()
    assert action.shape==(1,7) and bool(torch.isfinite(action).all())
    assert np.array_equal(action.numpy(),np.asarray(expected['action'],dtype=np.float32))
    with torch.no_grad():loss,m=policy(exp.batch_from(policy,val,[0,1,2,3]))
    prior=json.loads((out/'validation_after_ppo.jsonl').read_text().splitlines()[0])
    assert abs(m['loss']-prior['loss'])<1e-4
    write(out/'fresh_reload.json',{'passed':True,'action_exact':True,'actor_hash_exact':True,
        'first_validation_batch_loss_exact':True,'shape':[1,7],'finite':True})
    indices=list(map(int,np.linspace(0,461,32,dtype=int)))
    for i in indices[:3]:common.sample_action(policy,val,i)
    torch.cuda.reset_peak_memory_stats();times=[]
    for i in indices:
        check_budget();torch.cuda.synchronize();t=time.perf_counter();a=common.sample_action(policy,val,i);torch.cuda.synchronize()
        assert a.shape==(1,7) and bool(torch.isfinite(a).all());times.append(time.perf_counter()-t)
    write(out/'inference.json',{'samples':32,'warmup':3,'mean_seconds':float(np.mean(times)),
        'median_seconds':float(np.median(times)),'all_finite_1x7':True,'memory':memory(),
        'parameters':policy.parameter_report(),'checkpoint':str(cp)})
    namespace=dict(vars(exp))
    source=Path('/home/jovyan/projects/lerobot-openvla/outputs/input_pipeline_check/20261103T055533_073692Z/matched_rollout_function.py')
    exec(compile(source.read_text(),str(source),'exec'),namespace)
    base=R/'simulation/base'
    base.parent.mkdir(exist_ok=True)
    if not base.exists():base.symlink_to('/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z/simulation/epoch5',target_is_directory=True)
    trials=[]
    for state in CFG['evaluation_states']:
        check_budget(reserve=240);progress('evaluate',branch,state=state,completed_trials=len(trials),total_trials=5)
        trial=namespace['simulate_trial'](policy,branch+'_rl',state,types.SimpleNamespace(run=R))
        assert trial['trial_executed'],trial
        baseline=json.loads((base/f'state_{state:02d}/result.json').read_text())
        for k in ['initial_image_sha256','selected_init_state_sha256','init_state_file_sha256','reference_sha256',
                  'simulator_versions','max_policy_steps','warmup_steps']:
            assert trial[k]==baseline[k],k
        trials.append(trial)
        write(out/'simulation_after_ppo_progress.json',{'completed':len(trials),'successes':sum(t['success'] for t in trials),'trials':trials})
    assert before==policy.actor_hash()
    write(out/'simulation_after_ppo_result.json',{'complete':True,'attempts':5,'successes':sum(t['success'] for t in trials),
        'success_rate':sum(t['success'] for t in trials)/5,'mean_episode_steps':float(np.mean([t['policy_steps'] for t in trials])),
        'states':CFG['evaluation_states'],'max_steps':220,'warmup_steps':10,
        'technical_errors':0,'actor_unchanged':True,'trials':trials,
        'scope':'one bowl task, five previously reused fixed states; not unseen-state generalization'})

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['collect','ppo','evaluate']);p.add_argument('--branch',choices=['p20','p35'],required=True);a=p.parse_args()
    try:
        progress(a.phase,a.branch)
        if a.phase=='prepare':prepare()
        else:globals()[a.phase](a.branch)
        print('PHASE_COMPLETE',a.phase,a.branch,flush=True)
    except Exception as e:
        write(R/(a.phase+'_'+a.branch+'_error.json'),{'error':repr(e),'traceback':traceback.format_exc(),'budget_exceeded':isinstance(e,BudgetExceeded)})
        raise
if __name__=='__main__':main()
