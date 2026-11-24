"""Physical A/B bank pruning of the preserved trained SP adapter.
The original 128-score router and eta rule are kept, but removed A rows/B
columns no longer exist in memory or in the adapter checkpoint.
This is a static, potentially lossy extension, not an exact SP refactor.
"""
from pathlib import Path
import sys, json, hashlib, gc, types
SOURCE=Path("/home/jovyan/projects/lerobot-openvla/outputs/lora_sp_compare/20261104T091818_826396Z")
sys.path.insert(0,str(SOURCE))
import comparison_common as common
from comparison_common import exp, torch, np, OpenVLAConfig
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from sp_policy import SPOPenvlaPolicy, SPLinear, energy_mask
from lerobot.policies.openvla.full_experiment_openvla import FullExperimentOpenVLAPolicy
from lerobot.policies.openvla.training_openvla import NORM_KEY
RUN=Path(__file__).resolve().parent
CHECKPOINT=SOURCE/"sp128/checkpoints/step_005035"
VARIANTS={"sp_full128":128,"sp_pruned96":96,"sp_pruned64":64}

def write(path,obj):
    common.write(Path(path),obj)

class CompactSPLinear(nn.Module):
    def __init__(self,base,keep,original_rank=128,hidden=32,eta=.9):
        super().__init__()
        self.base=base
        for p in self.base.parameters():p.requires_grad_(False)
        self.original_rank=original_rank
        self.rank=len(keep)
        self.eta=eta
        self.register_buffer("keep_indices",torch.tensor(keep,device=base.weight.device,dtype=torch.long))
        device=base.weight.device
        self.sp_A=nn.Linear(base.in_features,self.rank,bias=False,device=device,dtype=torch.float32)
        self.sp_B=nn.Linear(self.rank,base.out_features,bias=False,device=device,dtype=torch.float32)
        self.router1=nn.Linear(base.in_features,hidden,device=device,dtype=torch.float32)
        self.router2=nn.Linear(hidden,original_rank,device=device,dtype=torch.float32)
        self.record_ranks=False
        self.reset_rank_stats()
    def clear_aux(self):pass
    def reset_rank_stats(self):
        self.rank_sum=None;self.rank_count=0;self.rank_min=None;self.rank_max=None;self.energy_min=None
    def forward(self,x):
        base=self.base(x)
        with torch.autocast(device_type=x.device.type,enabled=False):
            logits=self.router2(F.silu(self.router1(x.float())))
            scores=F.softplus(logits).clamp_min(1e-6)
            mask,_,_=energy_mask(scores,self.eta)
            # Keep the SAME original routing decisions. Static pruning then removes
            # some contributions permanently, which may change policy behavior.
            selected=(scores*mask).index_select(-1,self.keep_indices)
            if self.record_ranks:
                selected_mask=mask.index_select(-1,self.keep_indices)
                counts=selected_mask.sum(-1)
                energy=selected.square().sum(-1)/scores.square().sum(-1).clamp_min(1e-12)
                self.rank_sum=counts.sum() if self.rank_sum is None else self.rank_sum+counts.sum()
                self.rank_count+=counts.numel()
                lo=counts.min();hi=counts.max();en=energy.detach().min()
                self.rank_min=lo if self.rank_min is None else torch.minimum(self.rank_min,lo)
                self.rank_max=hi if self.rank_max is None else torch.maximum(self.rank_max,hi)
                self.energy_min=en if self.energy_min is None else torch.minimum(self.energy_min,en)
        # These matrix products now have actual width 96 or 64 instead of 128.
        projected=self.sp_A(x.to(self.sp_A.weight.dtype))
        update=self.sp_B(projected*selected)
        return base+update.to(base.dtype)

class PrunedSPPolicy(SPOPenvlaPolicy):
    def __init__(self,config,action_statistics,plan):
        # Build COMPACT modules directly; never place a full SP bank on GPU.
        FullExperimentOpenVLAPolicy.__init__(self,config,action_statistics,rank=8)
        self.model=self.model.unload()
        for p in self.model.parameters():p.requires_grad_(False)
        self.sp_names=[]
        replacements=[]
        for name,module in self.model.named_modules():
            if name.startswith("language_model.") and name.rsplit(".",1)[-1] in ["q_proj","v_proj"]:
                assert isinstance(module,nn.Linear),name
                replacements.append((name,module))
        assert len(replacements)==64 and {n for n,_ in replacements}==set(plan["modules"])
        self.plan=plan
        for name,module in replacements:
            parent,leaf=name.rsplit(".",1)
            keep=plan["modules"][name]["keep_indices"]
            setattr(self.model.get_submodule(parent),leaf,CompactSPLinear(module,keep,eta=.9))
            self.sp_names.append(name)
        self.sp_rank=128
        self.eta=.9
        for p in self.model.parameters():p.requires_grad_(False)
        self.eval()
    def adapter_hash(self):
        h=hashlib.sha256()
        h.update(json.dumps({n:m.keep_indices.cpu().tolist() for n,m in self.sp_modules()},sort_keys=True).encode())
        for name,t in sorted(self.adapter_state().items()):
            h.update(name.encode());h.update(t.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()
    def load_saved_adapter(self,out):
        out=Path(out);cfg=json.loads((out/"adapter_config.json").read_text())
        assert cfg["method"]=="static-physical-SP-bank-pruning"
        assert cfg["modules"]==self.plan["modules"]
        saved=load_file(str(out/"adapter_model.safetensors"),device="cpu")
        current=self.adapter_state()
        assert set(current)==set(saved)
        with torch.no_grad():
            for name,t in current.items():
                assert t.shape==saved[name].shape,(name,t.shape,saved[name].shape)
                t.copy_(saved[name].to(t.device))
        assert self.adapter_hash()==(out/"adapter_sha256.txt").read_text().strip()
        self.eval();self.clear_aux()

def load_policy(variant):
    assert variant in VARIANTS
    exp.seed()
    _,val,stats=exp.get_data()
    if variant=="sp_full128":
        policy=common.make_policy("sp128",stats)
        policy.load_saved_adapter(CHECKPOINT)
        cp=CHECKPOINT
    else:
        cp=RUN/"checkpoints"/variant
        plan=json.loads((cp/"adapter_config.json").read_text())
        policy=PrunedSPPolicy(OpenVLAConfig(model_id="openvla/openvla-7b",unnorm_key="bridge_orig",device="cuda:0"),stats,plan)
        policy.load_saved_adapter(cp)
    policy=common.prepare_inference(policy)
    assert not any(p.requires_grad for p in policy.model.parameters())
    return policy,val,cp

def physical_shapes(policy):
    return [{"module":name,"A":list(m.sp_A.weight.shape),"B":list(m.sp_B.weight.shape),
             "router_out":m.router2.out_features} for name,m in policy.sp_modules()]

def source_hash():
    return hashlib.sha256((CHECKPOINT/"adapter_model.safetensors").read_bytes()).hexdigest()

def export_compact(policy,rank,importance):
    dest=RUN/"checkpoints"/("sp_pruned"+str(rank))
    assert not dest.exists(),"Existing checkpoint preserved"
    dest.mkdir(parents=True)
    weights={};modules={}
    for name,module in policy.sp_modules():
        score=importance[name]
        keep=torch.argsort(score,descending=True,stable=True)[:rank].sort().values.cpu()
        fraction=float(score[keep].sum()/score.sum().clamp_min(1e-30))
        modules[name]={"keep_indices":keep.tolist(),"retained_calibration_importance":fraction,
                       "original_rank":128,"physical_rank":rank}
        for suffix,p in module.named_parameters():
            if suffix.startswith("base."):continue
            value=p.detach().cpu()
            if suffix=="sp_A.weight":value=value.index_select(0,keep)
            elif suffix=="sp_B.weight":value=value.index_select(1,keep)
            weights[name+"."+suffix]=value.contiguous()
    config={"method":"static-physical-SP-bank-pruning","original_rank":128,"physical_rank":rank,"eta":.9,
            "router_full_128_retained":True,"selection":"original 128-score eta mask, then retained component indices",
            "static_pruning_changes_policy":True,"guaranteed_90pct_energy_after_removal":False,
            "importance_method":"train-only squared component output norm heuristic","modules":modules,
            "source_checkpoint":str(CHECKPOINT),"source_adapter_file_sha256":source_hash(),
            "additional_training":False,"trained_epochs":5}
    write(dest/"adapter_config.json",config)
    save_file(weights,str(dest/"adapter_model.safetensors"))
    for name in ["action_normalization.json","base_model.json"]:
        (dest/name).write_bytes((CHECKPOINT/name).read_bytes())
    # Hash compact weights and indices in exactly the same format as the loader.
    h=hashlib.sha256()
    h.update(json.dumps({n:modules[n]["keep_indices"] for n in modules},sort_keys=True).encode())
    for name,t in sorted(weights.items()):h.update(name.encode());h.update(t.float().numpy().tobytes())
    (dest/"adapter_sha256.txt").write_text(h.hexdigest())
    return {"variant":"sp_pruned"+str(rank),"checkpoint":str(dest),
            "adapter_file_bytes":(dest/"adapter_model.safetensors").stat().st_size,
            "adapter_parameters":sum(t.numel() for t in weights.values()),
            "mean_retained_calibration_importance":float(np.mean([x["retained_calibration_importance"] for x in modules.values()]))}

