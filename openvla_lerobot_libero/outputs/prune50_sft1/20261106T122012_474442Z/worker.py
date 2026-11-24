"""50% target structure removal; same SFT epoch; no new RL or quantization."""
import argparse,os,time,signal,traceback
from policy import *
CFG=json.loads((R/'experiment_config.json').read_text())
OLD=Path(CFG['previous_run'])

class BudgetExceeded(RuntimeError):pass
def check_budget(reserve=90):
    if time.time()+reserve>=json.loads((R/'budget.json').read_text())['deadline_unix']:
        raise BudgetExceeded('One-hour experiment cap is nearly exhausted')
def signal_stop(signum,frame):raise BudgetExceeded('Supervisor requested checkpoint stop')
signal.signal(signal.SIGTERM,signal_stop)
def progress(phase,**kw):write(R/'progress.json',{'phase':phase,'updated_unix':time.time(),**kw})
def checkpoint_for(tag):
    if tag=='p50':return R/'checkpoints/p50/after_sft'
    return OLD/'checkpoints'/('dense' if tag=='dense_sft' else 'pruned')/'after_sft'

@torch.no_grad()
def validation(policy,data,stage,step):
    check_budget();policy.eval();before=policy.actor_hash()
    report=exp.evaluate(policy,data,4,R/'p50'/f'validation_{stage}.jsonl',step)
    assert report['frames']==462 and before==policy.actor_hash()
    write(R/'p50'/f'validation_{stage}.json',report)
    return report

def prepare():
    check_budget();train,val,stats=exp.get_data();exp.seed();policy=RecoveryPolicy(stats)
    before=policy.parameter_report();frozen=exp.frozen_sample_hash(policy);actor=policy.actor_hash()
    old=json.loads(Path(CFG['calibration_source']).read_text())
    indices=[int(i) for e in old['episodes'] for i in e['indices']]
    prior=json.loads((OLD/'calibration_selection.json').read_text())['indices']
    assert indices==prior and len(indices)==len(set(indices))==160
    write(R/'calibration_selection.json',{'split':'train','indices':indices,'frames':160,'validation_used':False,'same_as_p90':True})
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    plan=score_and_plan(policy,train,indices,R/'importance_progress.json')
    assert actor==policy.actor_hash() and exp.frozen_sample_hash(policy)==frozen
    importance_mem=memory();write(R/'pruning_plan.json',plan)
    assert all(len(l['heads'])==16 and len(l['neurons'])==5504 for l in plan['layers'])
    original_outer={str(i):digest_tensor(layer) for i,layer in enumerate(policy.model.language_model.model.layers) if i in [0,31]}
    apply_plan(policy,plan)
    assert original_outer=={str(i):digest_tensor(layer) for i,layer in enumerate(policy.model.language_model.model.layers) if i in [0,31]}
    after=policy.parameter_report();cp=R/'checkpoints/p50/initial';filebytes=save_checkpoint(policy,cp)
    before_plan=json.loads((OLD/'pruning_plan.json').read_text())
    nesting=[]
    for p50,p90 in zip(plan['layers'],before_plan['layers']):
        nesting.append({'layer':p50['layer'],'p90_heads_subset_of_p50':set(p90['heads'])<=set(p50['heads']),
            'p90_neurons_subset_of_p50':set(p90['neurons'])<=set(p50['neurons'])})
    write(R/'prepare_complete.json',{'complete':True,'before':before,'after':after,
        'importance_frames':160,'importance_seconds':time.perf_counter()-tick,'importance_memory':importance_mem,
        'target_internal_removal':.5,'actual_head_removal':.5,'actual_mlp_removal':.5,
        'protected_outer_layers_exact':True,'pruned_checkpoint':str(cp),'pruned_weights_bytes':filebytes,
        'whole_parameter_reduction':1-after['model_parameters']/before['model_parameters'],'prior90_selection_nesting':nesting})
    print('P50_PREPARED',before['model_parameters'],after['model_parameters'],flush=True)

def digest_tensor(module):
    h=hashlib.sha256()
    for name,t in module.state_dict().items():
        h.update(name.encode());h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def sft():
    check_budget();train,val,stats=exp.get_data();exp.seed()
    policy=load_checkpoint(R/'checkpoints/p50/initial',stats);optimizer=None
    (R/'p50').mkdir(exist_ok=True)
    try:
        initial=validation(policy,val,'initial',0)
        policy.configure_trainables(False);frozen=exp.frozen_sample_hash(policy);actor=policy.actor_hash()
        optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4,weight_decay=0.)
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(7)).tolist()
        coverage=[];totals={k:0. for k in exp.METRICS};started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for offset in range(0,len(order),4):
            check_budget();indices=order[offset:offset+4];tick=time.perf_counter()
            metrics=common.step(policy,optimizer,train,indices);coverage.extend(indices)
            for k in totals:totals[k]+=metrics[k]*len(indices)
            row={'optimizer_step':offset//4+1,'epoch':1,'frames_seen':len(coverage),'indices':indices,
                'seconds':time.perf_counter()-tick,**metrics,**memory()}
            exp.append_json(R/'p50/sft_metrics.jsonl',row)
            if row['optimizer_step']==1 or row['optimizer_step']%25==0:
                progress('sft',completed_frames=len(coverage),total_frames=4025,loss=metrics['loss'])
                print('P50_SFT',row['optimizer_step'],len(coverage),metrics['loss'],flush=True)
        elapsed=time.perf_counter()-started;mem=memory();optimizer.zero_grad(set_to_none=True)
        assert sorted(coverage)==list(range(4025)) and policy.actor_hash()!=actor
        assert frozen==exp.frozen_sample_hash(policy)
        coverage_hash=hashlib.sha256(json.dumps(coverage).encode()).hexdigest()
        for b in ['dense','pruned']:
            assert coverage_hash==json.loads((OLD/b/'sft_result.json').read_text())['coverage_sha256']
        final=validation(policy,val,'after_sft',1007)
        action=common.sample_action(policy,val,0).cpu().tolist()
        cp=R/'checkpoints/p50/after_sft';n=save_checkpoint(policy,cp)
        torch.save({'optimizer':optimizer.state_dict(),'updates':1007,'epoch':1,'coverage':coverage,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},cp/'trainer_state.pt')
        write(R/'p50/reload_expected.json',{'actor_hash':policy.actor_hash(),'validation_index':0,'action':action})
        write(R/'p50/sft_result.json',{'complete':True,'epochs':1,'frames_seen':len(coverage),'unique_frames':len(set(coverage)),
            'updates':1007,'coverage_sha256':coverage_hash,'coverage_same_as_dense_and_p90':True,
            'training_metrics':{k:v/4025 for k,v in totals.items()},'seconds':elapsed,'memory':mem,
            'before_validation':initial,'after_validation':final,'checkpoint':str(cp),'checkpoint_weight_bytes':n,
            'frozen_samples_unchanged':True,'no_reinforcement_learning':True})
    except BudgetExceeded:
        cp=R/'checkpoints/p50/interrupted_sft';save_checkpoint(policy,cp)
        torch.save({'optimizer':optimizer.state_dict() if optimizer else None,'coverage':locals().get('coverage',[])},cp/'trainer_state.pt')
        raise

def simulate(tag,states,folder):
    check_budget();train,val,stats=exp.get_data();exp.seed();cp=checkpoint_for(tag)
    policy=load_checkpoint(cp,stats);actor=policy.actor_hash();policy.eval()
    if tag=='p50':
        expected=json.loads((R/'p50/reload_expected.json').read_text());assert actor==expected['actor_hash']
        action=common.sample_action(policy,val,0).cpu()
        assert action.shape==(1,7) and bool(torch.isfinite(action).all())
        assert np.array_equal(action.numpy(),np.asarray(expected['action'],dtype=np.float32))
        with torch.no_grad():_,m=policy(exp.batch_from(policy,val,[0,1,2,3]))
        first=json.loads((R/'p50/validation_after_sft.jsonl').read_text().splitlines()[0])
        assert abs(m['loss']-first['loss'])<1e-4
        write(R/'p50/fresh_reload.json',{'passed':True,'actor_hash_exact':True,'action_exact':True,
            'same_validation_batch_loss':True,'shape':[1,7],'finite':True})
    if folder=='evaluation':
        indices=list(map(int,np.linspace(0,461,32,dtype=int)))
        for i in indices[:3]:common.sample_action(policy,val,i)
        torch.cuda.reset_peak_memory_stats();times=[]
        for i in indices:
            check_budget();torch.cuda.synchronize();tick=time.perf_counter();action=common.sample_action(policy,val,i);torch.cuda.synchronize()
            assert action.shape==(1,7) and bool(torch.isfinite(action).all());times.append(time.perf_counter()-tick)
        write(R/(tag+'_inference.json'),{'samples':32,'warmup':3,'mean_seconds':float(np.mean(times)),
            'median_seconds':float(np.median(times)),'all_finite_1x7':True,'memory':memory(),
            'parameters':policy.parameter_report(),'checkpoint':str(cp),'fresh_process':True})
    namespace=dict(vars(exp))
    source=Path('/home/jovyan/projects/lerobot-openvla/outputs/input_pipeline_check/20261103T055533_073692Z/matched_rollout_function.py')
    exec(compile(source.read_text(),str(source),'exec'),namespace)
    base=R/'simulation/base';base.parent.mkdir(exist_ok=True)
    if not base.exists():base.symlink_to('/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z/simulation/epoch5',target_is_directory=True)
    trials=[];label=tag if folder=='evaluation' else tag+'_diagnostic'
    for state in states:
        check_budget(reserve=240);progress(folder,model=tag,state=state,completed_trials=len(trials),total_trials=len(states))
        trial=namespace['simulate_trial'](policy,label,state,types.SimpleNamespace(run=R))
        assert trial['trial_executed'],trial
        init=json.loads((R/'simulation'/label/f'state_{state:02d}/initialization.json').read_text())
        if folder=='evaluation':
            reference=json.loads((base/f'state_{state:02d}/result.json').read_text())
        else:
            reference=json.loads((OLD/'rl_collection/dense'/f'state_{state:02d}/initialization.json').read_text())
        for k in ['selected_init_state_sha256','init_state_file_sha256','reference_sha256','simulator_versions']:
            assert trial[k]==reference[k],k
        if folder=='evaluation':assert trial['initial_image_sha256']==reference['initial_image_sha256']
        else:
            reference_frame=Path(reference['frame'])
            assert trial['initial_image_sha256']==hashlib.sha256(reference_frame.read_bytes()).hexdigest()
        trials.append(trial)
        write(R/(label+'_progress.json'),{'completed':len(trials),'attempts_planned':len(states),
            'successes':sum(t['success'] for t in trials),'trials':trials})
    assert actor==policy.actor_hash()
    write(R/(label+'_result.json'),{'complete':True,'tag':tag,'checkpoint':str(cp),'stage':'after_sft_before_any_PPO',
        'attempts':len(trials),'successes':sum(t['success'] for t in trials),'success_rate':sum(t['success'] for t in trials)/len(trials),
        'mean_episode_steps':float(np.mean([t['policy_steps'] for t in trials])),'states':states,'trials':trials,
        'technical_errors':0,'actor_unchanged':True,'diagnostic_only':folder!='evaluation',
        'scope':'one bowl task, fixed previously used initial states; no independent generalization claim'})

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','sft','simulate']);p.add_argument('--tag',choices=['p50','p90_sft','dense_sft'],default='p50')
    p.add_argument('--states',type=int,nargs='+',default=[0,1,2,3,4]);p.add_argument('--folder',choices=['diagnostic','evaluation'],default='evaluation');a=p.parse_args()
    try:
        progress(a.phase,model=a.tag)
        if a.phase=='simulate':simulate(a.tag,a.states,a.folder)
        else:globals()[a.phase]()
        print('PHASE_COMPLETE',a.phase,a.tag,flush=True)
    except Exception as e:
        write(R/(a.phase+'_'+a.tag+'_error.json'),{'error':repr(e),'traceback':traceback.format_exc(),'budget_exceeded':isinstance(e,BudgetExceeded)})
        raise
if __name__=='__main__':main()
