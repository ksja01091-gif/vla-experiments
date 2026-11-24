"""Frozen policies on identical saved observations; never updates model parameters."""
from pathlib import Path
import sys,json,time,gc,hashlib
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent
CFG=json.loads((R/'diagnostic_config.json').read_text());SOURCE=Path(CFG['source_run'])
sys.path.insert(0,str(SOURCE))
from policy import load_checkpoint,common,exp,torch
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False))
images=[]
for tag,frames in [('dense_sft',[0,30,60,68,69,76,90,103]),('p50',[0,30,60,120,180,219])]:
    folder=SOURCE/'simulation'/tag/'state_00'
    old=[json.loads(l) for l in (folder/'steps.jsonl').read_text().splitlines()]
    for frame in frames:
        images.append({'source_tag':tag,'state':0,'frame':frame,'path':str(folder/'frames'/f'{frame:04d}.png'),
            'recorded_next_action':old[frame]['raw_action'],'image_sha256':hashlib.sha256((folder/'frames'/f'{frame:04d}.png').read_bytes()).hexdigest()})
old_run=Path('/home/jovyan/projects/lerobot-openvla/outputs/rlrc_matched_3h/20261106T090120_404808Z')
_,_,stats=exp.get_data();predictions={};start=time.perf_counter()
for model,cp in [('dense_sft',old_run/'checkpoints/dense/after_sft'),('p50',SOURCE/'checkpoints/p50/after_sft')]:
    exp.seed();policy=load_checkpoint(cp,stats);common.prepare_inference(policy)
    assert not any(p.requires_grad for p in policy.model.parameters())
    before=policy.actor_hash();rows=[]
    for source in images:
        with Image.open(source['path']) as im:pixels=np.asarray(im.convert('RGB')).copy()
        tensor=torch.from_numpy(pixels).permute(2,0,1).float().div(255)
        with torch.inference_mode():action=policy.select_action({policy.config.image_key:tensor.unsqueeze(0),'task':[exp.TASK]})
        assert action.shape==(1,7) and bool(torch.isfinite(action).all())
        raw=action[0].cpu().numpy();controller=exp.controller_action(raw,np.full(7,-1.),np.ones(7))[1]
        own=source['source_tag']==model
        error=float(np.max(np.abs(raw-np.asarray(source['recorded_next_action'],dtype=np.float32))))
        if own:assert error<1e-7,('Frozen replay prediction changed',model,source['frame'],error)
        rows.append({**source,'model':model,'raw_action':raw.tolist(),'controller_action':controller.tolist(),
            'same_branch_recorded_action_match':bool(own and error<1e-7),'own_branch_match_checked':own})
    assert before==policy.actor_hash()
    predictions[model]=rows
    write(R/('same_image_'+model+'.json'),{'frozen':True,'actor_unchanged':True,'rows':rows})
    print('SAME_IMAGE_MODEL_COMPLETE',model,len(rows),flush=True)
    del policy;gc.collect();torch.cuda.empty_cache()
comparisons=[]
for a,b in zip(predictions['dense_sft'],predictions['p50']):
    assert a['image_sha256']==b['image_sha256']
    av=np.asarray(a['controller_action']);bv=np.asarray(b['controller_action'])
    comparisons.append({'source_tag':a['source_tag'],'frame':a['frame'],'image_sha256':a['image_sha256'],
        'dense_controller_action':av.tolist(),'p50_controller_action':bv.tolist(),
        'translation_command_L2_difference':float(np.linalg.norm(av[:3]-bv[:3])),
        'rotation_command_L2_difference':float(np.linalg.norm(av[3:6]-bv[3:6])),
        'gripper_command_disagrees':bool(av[-1]!=bv[-1]),
        'dense_translation_magnitude':float(np.linalg.norm(av[:3])),
        'p50_translation_magnitude':float(np.linalg.norm(bv[:3]))})
write(R/'same_image_comparison.json',{'complete':True,'no_training':True,'all_parameters_frozen':True,
    'same_image_same_instruction':True,'own_recorded_actions_reproduced':True,
    'images':len(images),'seconds':time.perf_counter()-start,'rows':comparisons,
    'scope':'Counterfactual action comparison on the same observations; does not establish which weight or head caused an error.'})
print('SAME_IMAGE_CHECK_COMPLETE',len(images),flush=True)
