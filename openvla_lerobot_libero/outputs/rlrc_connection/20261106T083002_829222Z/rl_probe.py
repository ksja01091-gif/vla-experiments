"""Tiny on-policy action-token/PPO connection probe, not a success-rate experiment."""
import os,subprocess,time,json
from PIL import Image
from transformers import LogitsProcessor,LogitsProcessorList
from policy import *

class ActionOnly(LogitsProcessor):
    def __init__(self,low,high):self.low=low;self.high=high
    def __call__(self,input_ids,scores):
        out=torch.full_like(scores,float('-inf'));out[:,self.low:self.high]=scores[:,self.low:self.high]
        return out

def inputs_from_image(policy,image):
    inputs=common.helper.matched_inputs(policy,image,exp.TASK)
    if int(inputs['input_ids'][0,-1])!=29871:
        inputs['input_ids']=torch.cat([inputs['input_ids'],torch.tensor([[29871]],dtype=inputs['input_ids'].dtype)],dim=1)
        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
    return {k:v.to(policy.config.device,dtype=torch.bfloat16) if k=='pixel_values' else v.to(policy.config.device) for k,v in inputs.items()}

def teacher(policy,inputs,tokens,return_distribution=False):
    ids=torch.cat([inputs['input_ids'],tokens.reshape(1,7).to(policy.config.device)],dim=1)
    mask=torch.ones_like(ids)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=policy.model(input_ids=ids,attention_mask=mask,pixel_values=inputs['pixel_values'],
                         use_cache=False,output_hidden_states=True)
    low=policy.action_tokenizer.action_token_begin_idx+1;high=policy.processor.tokenizer.vocab_size
    logits=out.logits[:,-8:-1,low:high].float()
    logp=F.log_softmax(logits,dim=-1)
    chosen=tokens.reshape(1,7).to(logp.device)-low
    selected=logp.gather(-1,chosen.unsqueeze(-1)).squeeze(-1)
    entropy=-(logp.exp()*logp).sum(-1).mean()
    patches=out.logits.shape[1]-ids.shape[1]
    hidden=out.hidden_states[-1][:,patches+inputs['input_ids'].shape[1]-1,:]
    value=policy.value_head(hidden.float()).squeeze(-1)
    if return_distribution:return logp,value
    return selected.sum(-1),value,entropy,selected

@torch.no_grad()
def bootstrap_value(policy,inputs):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=policy.model(**inputs,use_cache=False,output_hidden_states=True)
    return float(policy.value_head(out.hidden_states[-1][:,-1].float()).squeeze())

@torch.no_grad()
def sample(policy,inputs):
    policy.eval();low=policy.action_tokenizer.action_token_begin_idx+1;high=policy.processor.tokenizer.vocab_size
    # Keep the full prompt+7 shape identical during sampling and PPO scoring.
    # Causal masking prevents not-yet-sampled future placeholders influencing logits.
    # This avoids the observed cached-generation vs teacher-scoring discrepancy.
    tokens=torch.full((1,7),low,device=policy.config.device,dtype=torch.long)
    generated_logp=torch.empty((1,7),device=policy.config.device,dtype=torch.float32)
    for i in range(7):
        logp,_=teacher(policy,inputs,tokens,return_distribution=True)
        chosen=torch.multinomial(logp[:,i].exp(),1,generator=policy._rollout_generator)
        tokens[:,i]=chosen[:,0]+low
        generated_logp[:,i]=logp[:,i].gather(-1,chosen)[:,0]
    assert bool(((tokens>=low)&(tokens<high)).all())
    old_logp,value,_,token_logp=teacher(policy,inputs,tokens)
    max_diff=float((generated_logp-token_logp).abs().max())
    assert max_diff<1e-4,('Identical-path sampling/teacher action log-probability mismatch',max_diff)
    norm=policy.action_tokenizer.decode_token_ids_to_actions(tokens[0].cpu().numpy())
    stats=policy.action_statistics;lo=np.asarray(stats['q01']);hi=np.asarray(stats['q99']);mask=np.asarray(stats['mask'])
    raw=np.where(mask,.5*(norm+1)*(hi-lo)+lo,norm).astype(np.float32)
    assert raw.shape==(7,) and np.isfinite(raw).all()
    return tokens.detach().cpu(),raw,float(old_logp),float(value),max_diff

def rollout(policy,output,max_steps=8):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(MUJOCO_GL='egl',TF_CPP_MIN_LOG_LEVEL='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
    env['PYTHONPATH']='/home/jovyan/projects/LIBERO'+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    worker=Path('/home/jovyan/projects/lerobot-openvla/examples/libero_eval_worker.py')
    policy._rollout_generator=torch.Generator(device=policy.config.device).manual_seed(71000)
    rows=[];times=[];wall_times=[];process=None;success=False;bootstrap=0.;wall_started=time.perf_counter();ready_at=None
    with (output/'simulator.log').open('w') as log:
        try:
            process=subprocess.Popen(['/home/jovyan/projects/venvs/openvla_eval310/bin/python','-u',str(worker),
                '--output',str(output),'--task-id','0','--init-state-index','0'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
            observation=exp.read_observation(process,timeout=240)
            assert observation['type']=='ready' and observation['actual_initial_image_reference_check']['max_pixel_error']==0
            ready_at=time.perf_counter()
            write(output/'initialization.json',observation)
            low,high=np.asarray(observation['action_low']),np.asarray(observation['action_high'])
            for step in range(max_steps):
                step_tick=time.perf_counter()
                image=Image.open(observation['frame']).convert('RGB');inputs=inputs_from_image(policy,image)
                torch.cuda.synchronize();t=time.perf_counter()
                tokens,raw,logp,value,diff=sample(policy,inputs)
                torch.cuda.synchronize();times.append(time.perf_counter()-t)
                converted,command=exp.controller_action(raw,low,high)
                assert np.array_equal(converted,command),'Controller clipping required'
                exp.send(process,{'type':'step','action':command.tolist()});observation=exp.read_observation(process)
                success=bool(observation['task_success'])
                wall_times.append(time.perf_counter()-step_tick)
                rows.append({'inputs':{k:v.detach().cpu() for k,v in inputs.items()},'tokens':tokens,
                    'old_logp':logp,'value':value,'reward':float(success)-.001,'terminal':success,
                    'generation_logp_max_difference':diff})
                exp.append_json(output/'steps.jsonl',{'step':step+1,'raw_action':raw.tolist(),'tokens':tokens.tolist(),
                    'reward':float(success)-.001,'task_success':success,'old_joint_logp':logp,'value':value,'logp_difference':diff})
                if success:break
            if not success:
                inputs=inputs_from_image(policy,Image.open(observation['frame']).convert('RGB'))
                bootstrap=bootstrap_value(policy,inputs)
        finally:
            if process is not None and process.poll() is None:
                try:exp.send(process,{'type':'close'});process.wait(timeout=15)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):process.terminate();process.wait(timeout=15)
    gae=0.;next_value=bootstrap;advantages=[]
    for row in reversed(rows):
        continuation=0. if row['terminal'] else 1.
        delta=row['reward']+.99*next_value*continuation-row['value']
        gae=delta+.99*.95*continuation*gae
        advantages.append(gae);next_value=row['value']
    advantages=list(reversed(advantages));a=np.asarray(advantages,dtype=np.float32)
    normalized=(a-a.mean())/(a.std()+1e-8)
    for row,adv,norm in zip(rows,advantages,normalized):row['return']=adv+row['value'];row['advantage']=float(norm)
    return rows,{'steps':len(rows),'max_steps':max_steps,'success_rate_not_evaluated':True,
        'truncated':not success,'bootstrap':bootstrap,'mean_sample_and_score_seconds':float(np.mean(times)),
        'mean_transition_wall_seconds':float(np.mean(wall_times)),
        'worker_initialization_seconds':ready_at-wall_started,'total_rollout_wall_seconds':time.perf_counter()-wall_started,
        'max_generation_teacher_logp_difference':max(r['generation_logp_max_difference'] for r in rows),
        'reward':'task success +1, step penalty -0.001; smoke adaptation, not a verified paper coefficient',
        'sampler':'identical-shape uncached autoregressive sampling over256 action tokens; temperature1; per-rollout generator seed71000',
        'gamma':.99,'gae_lambda':.95}

def update_probe(policy,rows,train):
    before=policy.actor_hash();frozen_before=exp.frozen_sample_hash(policy)
    policy.configure_trainables(False)
    policy.model.requires_grad_(False);policy.value_head.requires_grad_(True)
    critic=torch.optim.AdamW(policy.value_head.parameters(),lr=1e-4,weight_decay=0.)
    tick=time.perf_counter()
    for row in rows[:2]:
        critic.zero_grad(set_to_none=True)
        inputs={k:v.to(policy.config.device) for k,v in row['inputs'].items()}
        _,value,_,_=teacher(policy,inputs,row['tokens'])
        loss=.5*(value-row['return']).square().mean();loss.backward();critic.step()
    assert policy.actor_hash()==before,'Critic warm-up modified the actor'
    warmup=time.perf_counter()-tick
    policy.configure_trainables(True)
    optimizer=torch.optim.AdamW([*policy.get_optim_params(),*policy.value_head.parameters()],lr=1e-5,weight_decay=0.)
    optimizer.zero_grad(set_to_none=True);policy.train();policy.model.vision_backbone.eval()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
    metrics=[]
    for row in rows[:4]:
        inputs={k:v.to(policy.config.device) for k,v in row['inputs'].items()}
        logp,value,entropy,_=teacher(policy,inputs,row['tokens'])
        ratio=(logp-row['old_logp']).exp()
        actor=-torch.minimum(ratio*row['advantage'],ratio.clamp(.8,1.2)*row['advantage']).mean()
        value_loss=.5*(value-row['return']).square().mean()
        loss=(actor+.5*value_loss)/min(4,len(rows));loss.backward()
        metrics.append({'ratio':float(ratio.detach()),'actor_loss':float(actor.detach()),'value_loss':float(value_loss.detach()),'entropy':float(entropy.detach())})
    bc_loss,bc_metrics=policy(exp.batch_from(policy,train,[0]))
    (.1*bc_loss).backward()
    grad=torch.nn.utils.clip_grad_norm_([*policy.get_optim_params(),*policy.value_head.parameters()],1.)
    assert torch.isfinite(grad);optimizer.step();torch.cuda.synchronize()
    elapsed=time.perf_counter()-tick
    assert policy.actor_hash()!=before,'PPO+BC did not update the recovery actor'
    policy.configure_trainables(False)
    assert exp.frozen_sample_hash(policy)==frozen_before,'Frozen backbone/start adapter changed'
    report={'critic_warmup_steps':2,'critic_warmup_seconds':warmup,'actor_unchanged_during_critic_warmup':True,
            'ppo_optimizer_steps':1,'ppo_transitions':min(4,len(rows)),'ppo_update_seconds':elapsed,
            'ppo_gradient_norm':float(grad),'bc_training_only_loss':bc_metrics['loss'],
            'actor_changed':True,'frozen_samples_unchanged':True,'rows':metrics,**memory()}
    policy.eval();return report
