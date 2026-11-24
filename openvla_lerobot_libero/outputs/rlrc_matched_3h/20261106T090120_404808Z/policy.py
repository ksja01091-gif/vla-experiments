"""Paper-informed structural pruning and frozen-start/recovery LoRA policy.
No edits to existing packages; no merging/rounding of the starting adapter.
The LIBERO single-task adaptation is not an exact reproduction of paper training.
"""
from pathlib import Path
import sys,json,hashlib,types,gc,math
R=Path(__file__).resolve().parent
SOURCE=Path('/home/jovyan/projects/lerobot-openvla/outputs/lora_sp_compare/20261104T091818_826396Z')
sys.path.insert(0,str(SOURCE))
import comparison_common as common
from comparison_common import exp,torch,np,OpenVLAConfig
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from lerobot.policies.openvla.full_experiment_openvla import FullExperimentOpenVLAPolicy
START=Path('/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z/checkpoints/step_005035')

def write(path,data):common.write(Path(path),data)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def memory():return {'allocated_gib':torch.cuda.memory_allocated()/1024**3,'reserved_gib':torch.cuda.memory_reserved()/1024**3,'peak_gib':torch.cuda.max_memory_allocated()/1024**3}

class RecoveryLinear(nn.Module):
    def __init__(self,source,index,rank=8):
        super().__init__()
        assert set(source.lora_A)=={'default'} and float(source.scaling['default'])==1.
        self.base_layer=source.base_layer
        self.frozen_A=source.lora_A['default'];self.frozen_B=source.lora_B['default']
        device=self.base_layer.weight.device
        self.recovery_A=nn.Linear(self.in_features,rank,bias=False,device=device,dtype=torch.float32)
        self.recovery_B=nn.Linear(rank,self.out_features,bias=False,device=device,dtype=torch.float32)
        gen=torch.Generator(device=device).manual_seed(7000+index)
        with torch.no_grad():
            self.recovery_A.weight.normal_(std=1/rank,generator=gen);self.recovery_B.weight.zero_()
        for p in self.parameters():p.requires_grad_(False)
        for m in [self.recovery_A,self.recovery_B]:m.requires_grad_(True)
    @property
    def weight(self):return self.base_layer.weight
    @property
    def in_features(self):return self.base_layer.in_features
    @property
    def out_features(self):return self.base_layer.out_features
    def forward(self,x):
        result=self.base_layer(x);dtype=result.dtype
        result=result+self.frozen_B(self.frozen_A(x.to(self.frozen_A.weight.dtype)))
        result=result+self.recovery_B(self.recovery_A(x.to(self.recovery_A.weight.dtype)))
        return result.to(dtype)

class RecoveryPolicy(FullExperimentOpenVLAPolicy):
    def __init__(self,stats):
        super().__init__(OpenVLAConfig(model_id='openvla/openvla-7b',unnorm_key='bridge_orig',device='cuda:0'),stats,rank=8)
        self.load_saved_adapter(START)
        native=self.model.get_base_model()
        todo=[(n,m) for n,m in native.named_modules() if n.startswith('language_model.') and hasattr(m,'lora_A')]
        assert len(todo)==64
        for i,(name,module) in enumerate(todo):
            parent,leaf=name.rsplit('.',1)
            setattr(native.get_submodule(parent),leaf,RecoveryLinear(module,i))
        self.model=native
        self.plan=None
        self.value_head=nn.Linear(self.model.language_model.config.hidden_size,1,device='cuda:0',dtype=torch.float32)
        nn.init.zeros_(self.value_head.weight);nn.init.zeros_(self.value_head.bias)
        self.configure_trainables(False)
        self.select_action=types.MethodType(common.helper.matched_select_action,self)
        self.eval()
    def recovery_parameters(self):
        return [p for n,p in self.model.named_parameters() if '.recovery_A.' in n or '.recovery_B.' in n]
    def configure_trainables(self,critic=False):
        self.model.requires_grad_(False)
        for p in self.recovery_parameters():p.requires_grad_(True)
        self.value_head.requires_grad_(critic)
    def actor_hash(self):
        h=hashlib.sha256()
        for n,p in self.model.named_parameters():
            if '.recovery_' in n:h.update(n.encode());h.update(p.detach().cpu().float().numpy().tobytes())
        return h.hexdigest()
    def adapter_hash(self):
        if not hasattr(self,'value_head'):return FullExperimentOpenVLAPolicy.adapter_hash(self)
        return self.actor_hash()
    def get_optim_params(self):return [p for p in self.model.parameters() if p.requires_grad]
    def parameter_report(self):
        return {'model_parameters':sum(p.numel() for p in self.model.parameters()),
                'recovery_parameters':sum(p.numel() for p in self.recovery_parameters()),
                'model_tensor_bytes':self.model.get_memory_footprint(),
                'critic_parameters':sum(p.numel() for p in self.value_head.parameters())}

def cut_linear(module,keep,axis):
    """Physically index matching rows or columns, preserving a frozen start LoRA."""
    if isinstance(module,RecoveryLinear):
        module.base_layer=cut_linear(module.base_layer,keep,axis)
        if axis==0:
            module.frozen_B=cut_linear(module.frozen_B,keep,0)
            module.recovery_B=cut_linear(module.recovery_B,keep,0)
        else:
            module.frozen_A=cut_linear(module.frozen_A,keep,1)
            module.recovery_A=cut_linear(module.recovery_A,keep,1)
        return module
    assert isinstance(module,nn.Linear)
    keep=torch.as_tensor(keep,device=module.weight.device,dtype=torch.long)
    value=module.weight.detach().index_select(axis,keep).contiguous()
    out=nn.Linear(value.shape[1],value.shape[0],bias=module.bias is not None,
                  device=value.device,dtype=value.dtype)
    out.weight=nn.Parameter(value,requires_grad=module.weight.requires_grad)
    if module.bias is not None:
        b=module.bias.detach().index_select(0,keep) if axis==0 else module.bias.detach().clone()
        out.bias=nn.Parameter(b,requires_grad=module.bias.requires_grad)
    return out

def apply_plan(policy,plan):
    layers=policy.model.language_model.model.layers
    assert len(layers)==32 and plan['protected_layers']==[0,31]
    for row in plan['layers']:
        i=row['layer'];assert 0<i<31
        att=layers[i].self_attn;mlp=layers[i].mlp
        assert att.head_dim==128 and att.num_key_value_groups==1
        idx=[h*att.head_dim+j for h in row['heads'] for j in range(att.head_dim)]
        for n in ['q_proj','k_proj','v_proj']:setattr(att,n,cut_linear(getattr(att,n),idx,0))
        att.o_proj=cut_linear(att.o_proj,idx,1)
        att.num_heads=len(row['heads']);att.num_key_value_heads=att.num_heads;att.num_key_value_groups=1
        att.hidden_size=att.num_heads*att.head_dim
        for n in ['gate_proj','up_proj']:setattr(mlp,n,cut_linear(getattr(mlp,n),row['neurons'],0))
        mlp.down_proj=cut_linear(mlp.down_proj,row['neurons'],1)
        mlp.intermediate_size=len(row['neurons'])
        assert att.q_proj.weight.shape[0]==att.o_proj.weight.shape[1]==att.hidden_size
        assert mlp.gate_proj.weight.shape[0]==mlp.down_proj.weight.shape[1]==mlp.intermediate_size
    policy.plan=plan;policy.configure_trainables(False)
    gc.collect();torch.cuda.empty_cache()

def score_and_plan(policy,train,indices,progress):
    """First-order |effective W * grad| grouped into complete heads/MLP neurons."""
    layers=policy.model.language_model.model.layers
    policy.model.requires_grad_(False)
    targets=[];scores={i:{'heads':torch.zeros(32,dtype=torch.float64),
                         'neurons':torch.zeros(11008,dtype=torch.float64)} for i in range(1,31)}
    for i in range(1,31):
        layer=layers[i]
        for family,names in [('attn',['q_proj','k_proj','v_proj','o_proj']),('mlp',['gate_proj','up_proj','down_proj'])]:
            parent=layer.self_attn if family=='attn' else layer.mlp
            for n in names:
                module=getattr(parent,n);base=module.base_layer if isinstance(module,RecoveryLinear) else module
                base.weight.requires_grad_(True);targets.append((i,family,n,module,base))
    for j,index in enumerate(indices,1):
        policy.train();policy.model.vision_backbone.eval();policy.model.zero_grad(set_to_none=True)
        loss,_=policy(exp.batch_from(policy,train,[int(index)]));loss.backward()
        with torch.no_grad():
            for i,family,n,module,base in targets:
                assert base.weight.grad is not None and torch.isfinite(base.weight.grad).all(),(i,n)
                w=base.weight.detach().float()
                if isinstance(module,RecoveryLinear):w=w+module.frozen_B.weight.float()@module.frozen_A.weight.float()
                importance=(w*base.weight.grad.float()).abs()
                if family=='attn':
                    channels=importance.sum(0 if n=='o_proj' else 1).reshape(32,128).sum(1)
                    scores[i]['heads']+=channels.double().cpu()
                else:
                    channels=importance.sum(0 if n=='down_proj' else 1)
                    scores[i]['neurons']+=channels.double().cpu()
                del w,importance
        policy.model.zero_grad(set_to_none=True);del loss
        write(progress,{'completed':j,'total':len(indices),**memory()})
        print('IMPORTANCE',j,'/',len(indices),flush=True)
    policy.configure_trainables(False)
    plan={'method':'RLRC-inspired grouped Taylor structural pruning','target_pruning':.9,
          'protected_layers':[0,31],'head_dim':128,'calibration_split':'train',
          'calibration_indices':list(map(int,indices)),'calibration_is_smoke_only':False,
          'head_keep':3,'mlp_keep':1088,'actual_target_head_removal':29/32,
          'actual_target_mlp_removal':1-1088/11008,'layers':[]}
    for i,row in scores.items():
        assert all(torch.isfinite(t).all() and t.sum()>0 for t in row.values())
        h=torch.argsort(row['heads'],descending=True,stable=True)[:3].sort().values.tolist()
        m=torch.argsort(row['neurons'],descending=True,stable=True)[:1088].sort().values.tolist()
        plan['layers'].append({'layer':i,'heads':h,'neurons':m})
    return plan

def save_checkpoint(policy,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    if policy.plan is None:
        weights={n:t.detach().cpu().contiguous() for n,t in policy.model.state_dict().items() if '.recovery_' in n}
    else:
        weights={n:t.detach().cpu().contiguous() for n,t in policy.model.state_dict().items()}
    save_file(weights,str(dest/'model.safetensors'))
    save_file({n:t.detach().cpu().contiguous() for n,t in policy.value_head.state_dict().items()},str(dest/'critic.safetensors'))
    write(dest/'plan.json',policy.plan)
    write(dest/'metadata.json',{'source_adapter':str(START),'source_adapter_sha256':digest(START/'adapter_model.safetensors'),
        'physical_full_model_saved':policy.plan is not None,'actor_hash':policy.actor_hash(),
        'weights_sha256':digest(dest/'model.safetensors'),'parameters':policy.parameter_report()})
    return (dest/'model.safetensors').stat().st_size

def load_checkpoint(dest,stats):
    dest=Path(dest);meta=json.loads((dest/'metadata.json').read_text())
    assert digest(START/'adapter_model.safetensors')==meta['source_adapter_sha256']
    assert digest(dest/'model.safetensors')==meta['weights_sha256']
    policy=RecoveryPolicy(stats);plan=json.loads((dest/'plan.json').read_text())
    if plan is not None:apply_plan(policy,plan)
    weights=load_file(str(dest/'model.safetensors'),device='cpu')
    result=policy.model.load_state_dict(weights,strict=plan is not None)
    if plan is None:assert not result.unexpected_keys and all('.recovery_' not in n for n in result.missing_keys)
    policy.value_head.load_state_dict(load_file(str(dest/'critic.safetensors'),device='cpu'))
    assert policy.actor_hash()==meta['actor_hash']
    policy.eval();return policy
