"""OpenVLA adaptation of LoRA-SP equations 9-14, not a reproduction of PiPER experiments.
Explicit choices absent from paper: router hidden32, SiLU then Softplus, scale=1,
Gaussian A(std=1/R), zero B; FP32 adapters/router, BF16 frozen backbone.
Use a full dense bank. Active rank is a proxy, NOT a storage/FLOP reduction claim.
"""
import hashlib,json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from lerobot.policies.openvla.full_experiment_openvla import FullExperimentOpenVLAPolicy
from lerobot.policies.openvla.training_openvla import NORM_KEY

def energy_mask(scores,eta):
    """Keep the minimum prefix whose squared-score energy reaches eta."""
    if not 0<eta<=1:raise ValueError("eta must be in (0,1]")
    values,order=torch.sort(scores.detach().double().square(),dim=-1,descending=True,stable=True)
    cumulative=values.cumsum(-1)
    total=cumulative[...,-1:]
    # Previous prefix still below target => this component must be included.
    previous=torch.cat([torch.zeros_like(cumulative[...,:1]),cumulative[...,:-1]],dim=-1)
    keep=previous<eta*total
    if eta==1:keep=torch.ones_like(keep)
    keep[...,0]=True
    mask=torch.zeros_like(keep).scatter(-1,order,keep)
    energy=((scores.square()*mask).sum(-1)/scores.square().sum(-1).clamp_min(1e-12)).clamp(0,1)
    return mask,mask.sum(-1),energy

class SPLinear(nn.Module):
    def __init__(self,base,rank=128,hidden=32,eta=.9):
        super().__init__()
        self.base=base
        for p in base.parameters():p.requires_grad_(False)
        self.rank=rank;self.eta=eta
        device=base.weight.device
        self.sp_A=nn.Linear(base.in_features,rank,bias=False,device=device,dtype=torch.float32)
        self.sp_B=nn.Linear(rank,base.out_features,bias=False,device=device,dtype=torch.float32)
        self.router1=nn.Linear(base.in_features,hidden,device=device,dtype=torch.float32)
        self.router2=nn.Linear(hidden,rank,device=device,dtype=torch.float32)
        nn.init.normal_(self.sp_A.weight,std=1/rank);nn.init.zeros_(self.sp_B.weight)
        nn.init.kaiming_uniform_(self.router1.weight,a=math.sqrt(5));nn.init.zeros_(self.router1.bias)
        nn.init.normal_(self.router2.weight,std=.01);nn.init.zeros_(self.router2.bias)
        self.aux_spec=None;self.aux_router=None
        self.record_ranks=False;self.rank_sum=None;self.rank_count=0;self.rank_min=None;self.rank_max=None;self.energy_min=None
    def clear_aux(self):self.aux_spec=None;self.aux_router=None
    def reset_rank_stats(self):
        self.rank_sum=None;self.rank_count=0;self.rank_min=None;self.rank_max=None;self.energy_min=None
    def forward(self,x):
        base=self.base(x)
        with torch.autocast(device_type=x.device.type,enabled=False):
            logits=self.router2(F.silu(self.router1(x.float())))
            scores=F.softplus(logits).clamp_min(1e-6)
            mask,counts,energy=energy_mask(scores,self.eta)
            selected=scores*mask
            if self.training and torch.is_grad_enabled():
                probs=torch.softmax(logits,dim=-1)
                hard=F.one_hot(probs.argmax(-1),self.rank).float()
                axes=tuple(range(probs.ndim-1))
                balance=self.rank*(hard.mean(axes)*probs.mean(axes)).sum()
                z=torch.logsumexp(logits,dim=-1).square().mean()
                self.aux_spec=(1-energy).mean()
                self.aux_router=balance+z
            if self.record_ranks:
                c=counts.detach()
                self.rank_sum=c.sum() if self.rank_sum is None else self.rank_sum+c.sum()
                self.rank_count+=c.numel()
                lo=c.min();hi=c.max();en=energy.detach().min()
                self.rank_min=lo if self.rank_min is None else torch.minimum(self.rank_min,lo)
                self.rank_max=hi if self.rank_max is None else torch.maximum(self.rank_max,hi)
                self.energy_min=en if self.energy_min is None else torch.minimum(self.energy_min,en)
        # Match existing LoRA: adapter params FP32; outer training autocast controls GEMMs.
        projected=self.sp_A(x.to(self.sp_A.weight.dtype))
        update=self.sp_B(projected*selected)
        return base+update.to(base.dtype)

class SPOPenvlaPolicy(FullExperimentOpenVLAPolicy):
    def __init__(self,config,action_statistics,rank=128,eta=.9):
        # Reuse existing processor/normalization/action-token training interface.
        super().__init__(config,action_statistics,rank=8)
        self.model=self.model.unload()  # remove disposable LoRA WITHOUT merging
        for p in self.model.parameters():p.requires_grad_(False)
        self.sp_names=[]
        replacements=[]
        for name,mod in self.model.named_modules():
            if name.startswith("language_model.") and name.rsplit(".",1)[-1] in ["q_proj","v_proj"]:
                assert isinstance(mod,nn.Linear),name
                replacements.append((name,mod))
        assert len(replacements)==64,len(replacements)
        for name,mod in replacements:
            parent,leaf=name.rsplit(".",1)
            setattr(self.model.get_submodule(parent),leaf,SPLinear(mod,rank=rank,eta=eta))
            self.sp_names.append(name)
        self.sp_rank=rank;self.eta=eta
        trainable=[n for n,p in self.model.named_parameters() if p.requires_grad]
        assert trainable and all(any("."+key+"." in n for key in ["sp_A","sp_B","router1","router2"]) for n in trainable)
    def sp_modules(self):
        return [(name,self.model.get_submodule(name)) for name in self.sp_names]
    def set_eta(self,eta):
        for _,module in self.sp_modules():module.eta=eta
        self.eta=eta
    def clear_aux(self):
        for _,module in self.sp_modules():module.clear_aux()
    def record_rank_stats(self,enabled):
        for _,module in self.sp_modules():
            module.record_ranks=enabled;module.reset_rank_stats()
    def rank_stats(self):
        rows=[]
        for name,module in self.sp_modules():
            if module.rank_count:
                rows.append({"module":name,"mean":float(module.rank_sum)/module.rank_count,
                "min":int(module.rank_min),"max":int(module.rank_max),"minimum_retained_score_energy":float(module.energy_min),
                "tokens":module.rank_count})
        tokens=sum(r["tokens"] for r in rows)
        return {"mean_active_rank":sum(r["mean"]*r["tokens"] for r in rows)/tokens if tokens else None,
                "rows":rows,"initial_bank_rank":self.sp_rank,"eta":self.eta}
    def forward(self,batch):
        self.clear_aux()
        ce,metrics=super().forward(batch)
        if self.training and torch.is_grad_enabled():
            modules=[m for _,m in self.sp_modules()]
            assert all(m.aux_spec is not None for m in modules)
            spec=torch.stack([m.aux_spec for m in modules]).mean()
            router=torch.stack([m.aux_router for m in modules]).mean()
            total=ce+.01*spec+.001*router
            assert torch.isfinite(total)
            metrics.update({"total_loss":float(total.detach()),"spectral_loss":float(spec.detach()),"router_loss":float(router.detach())})
            return total,metrics
        metrics.update({"total_loss":metrics["loss"],"spectral_loss":0.,"router_loss":0.})
        return ce,metrics
    def adapter_state(self):
        result={}
        for name,module in self.sp_modules():
            for suffix,p in module.named_parameters():
                if not suffix.startswith("base."):result[name+"."+suffix]=p
        return result
    def adapter_hash(self):
        h=hashlib.sha256()
        for name,tensor in sorted(self.adapter_state().items()):
            h.update(name.encode());h.update(tensor.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()
    def save_adapter(self,out):
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        save_file({n:p.detach().cpu().contiguous() for n,p in self.adapter_state().items()},str(out/"adapter_model.safetensors"))
        (out/"adapter_config.json").write_text(json.dumps({"method":"OpenVLA application of LoRA-SP equations","rank":self.sp_rank,"eta":self.eta,
          "router_hidden":32,"router_activation":"SiLU + Softplus","scale":1.,"spec_weight":.01,"router_weight":.001,
          "dense_full_bank":True,"orthogonality_constraint":False,"score_energy_not_guaranteed_true_matrix_SVD_energy":True},indent=2))
        (out/"action_normalization.json").write_text(json.dumps({NORM_KEY:{"action":self.action_statistics}},indent=2))
        (out/"base_model.json").write_text(json.dumps({"checkpoint":self.checkpoint_path,"model_id":self.config.model_id,"reference_hashes":self.reference_hashes},indent=2))
        (out/"adapter_sha256.txt").write_text(self.adapter_hash())
    def load_saved_adapter(self,out):
        out=Path(out)
        cfg=json.loads((out/"adapter_config.json").read_text())
        assert cfg["rank"]==self.sp_rank
        saved=load_file(str(out/"adapter_model.safetensors"),device="cpu")
        actual=self.adapter_state()
        assert set(saved)==set(actual)
        with torch.no_grad():
            for name,tensor in actual.items():tensor.copy_(saved[name].to(tensor.device))
        assert all(torch.equal(actual[n].detach().cpu(),saved[n]) for n in saved)
        assert self.adapter_hash()==(out/"adapter_sha256.txt").read_text().strip()
        self.set_eta(cfg["eta"]);self.eval();self.clear_aux()
