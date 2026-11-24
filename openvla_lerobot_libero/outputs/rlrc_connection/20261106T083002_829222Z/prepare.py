"""Short train-only Taylor calibration and real 90%-target backbone export."""
import time,traceback
from policy import *
try:
    train,val,stats=exp.get_data()
    torch.cuda.reset_peak_memory_stats()
    reference=common.make_policy('lora8',stats);reference.load_saved_adapter(START);common.prepare_inference(reference)
    indices=[0,100,300,461]
    expected=[common.sample_action(reference,val,i).cpu().tolist() for i in indices]
    with torch.no_grad():baseline_loss=float(reference(exp.batch_from(reference,val,indices))[0])
    original_count=sum(p.numel() for p in reference.model.parameters())
    del reference;gc.collect();torch.cuda.empty_cache()
    policy=RecoveryPolicy(stats)
    actual=[common.sample_action(policy,val,i).cpu().tolist() for i in indices]
    assert np.array_equal(np.asarray(actual),np.asarray(expected)),('Starting policy changed',actual,expected)
    with torch.no_grad():wrapper_loss=float(policy(exp.batch_from(policy,val,indices))[0])
    assert abs(wrapper_loss-baseline_loss)<1e-6,(wrapper_loss,baseline_loss)
    write(R/'starting_parity.json',{'exact_actions':True,'original_loss':baseline_loss,'wrapper_loss':wrapper_loss,
        'indices':indices,'actions':expected,'old_model_parameters':original_count,
        'starting_adapter_unmerged':True,'new_recovery_zero_initialized':True})
    before=policy.parameter_report();dense_bytes=save_checkpoint(policy,R/'checkpoints/dense_initial')
    calibration_source=Path('/home/jovyan/projects/lerobot-openvla/outputs/sp_physical_prune/20261105T085625_138557Z/calibration_indices.json')
    old=json.loads(calibration_source.read_text());allindices=[i for e in old['episodes'] for i in e['indices']]
    assert len(allindices)==160
    calibration=[allindices[i] for i in np.linspace(0,159,8,dtype=int)]
    write(R/'calibration_selection.json',{'smoke_indices':calibration,'full_candidate_indices':allindices,
        'split':'train','validation_used':False,'smoke_only_frames':8,'final_long_run_would_recalibrate_on_160':True})
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
    plan=score_and_plan(policy,train,calibration,R/'importance_progress.json')
    torch.cuda.synchronize();elapsed=time.perf_counter()-tick;importance_memory=memory()
    write(R/'pruning_plan.json',plan)
    apply_plan(policy,plan);after=policy.parameter_report()
    actual=[common.sample_action(policy,val,i).cpu().tolist() for i in indices]
    assert all(np.asarray(a).shape==(1,7) and np.isfinite(a).all() for a in actual)
    pruned_bytes=save_checkpoint(policy,R/'checkpoints/pruned_initial')
    write(R/'prepare_complete.json',{'complete':True,'target_internal_pruning':.9,
        'before':before,'after':after,'whole_model_parameter_reduction':1-after['model_parameters']/before['model_parameters'],
        'dense_checkpoint_bytes':dense_bytes,'dense_checkpoint_is_adapter_only':True,'pruned_full_model_file_bytes':pruned_bytes,
        'actual_target_heads_removed':29/32,'actual_target_mlp_removed':1-1088/11008,
        'importance_frames':8,'importance_seconds':elapsed,'importance_memory':importance_memory,
        'pruned_initial_finite_actions':True,'pruned_initial_actions':actual,
        'not_a_quality_or_success_rate_evaluation':True})
    print('PREPARE_COMPLETE',json.dumps(json.loads((R/'prepare_complete.json').read_text())),flush=True)
except Exception as e:
    write(R/'prepare_error.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
