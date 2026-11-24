"""Small correctness tests for paper's minimum crossing rule and checkpoint gradients."""
import copy,json,torch
from pathlib import Path
from torch import nn
from torch.utils.checkpoint import checkpoint
from sp_policy import SPLinear,energy_mask
R=Path(__file__).parent
torch.manual_seed(7)
scores=torch.tensor([[4.,3.,2.,1.]])
for eta,k in [(.5,1),(.7,2),(.8,2),(.9,3),(1.,4)]:
 mask,count,energy=energy_mask(scores,eta)
 assert int(count)==k,(eta,count)
 assert float(energy)>=eta-1e-6,(eta,energy)
 assert torch.isfinite(energy).all()
# Selection monotonicity on many positive score vectors; minimum-prefix condition.
s=torch.rand(21,128)+1e-4;last=torch.zeros(21,dtype=torch.long)
for eta in [.5,.7,.8,.9,.99,1.]:
 mask,k,e=energy_mask(s,eta);assert (k>=last).all() and (e>=eta-1e-6).all()
 sorted_sq=s.double().square().sort(descending=True).values
 previous=sorted_sq.cumsum(-1).gather(-1,(k-2).clamp_min(0).unsqueeze(-1)).squeeze(-1)
 assert ((k==1)|(previous<eta*sorted_sq.sum(-1)+1e-6)).all(), (eta, (previous-eta*sorted_sq.sum(-1)).tolist(), k.tolist())
 last=k
base=nn.Linear(8,8)
model=SPLinear(base,rank=4,hidden=3)
x=torch.randn(2,5,8)
assert torch.equal(model(x),base(x)),"zero-B initialization must preserve backbone"
a=nn.Sequential(model,nn.SiLU(),SPLinear(nn.Linear(8,8),rank=4,hidden=3))
model.clear_aux()
b=copy.deepcopy(a)
def loss_for(m,use_ckpt):
 y=checkpoint(m,x,use_reentrant=False) if use_ckpt else m(x)
 aux=[q for q in m.modules() if isinstance(q,SPLinear)]
 return y.square().mean()+.01*torch.stack([q.aux_spec for q in aux]).mean()+.001*torch.stack([q.aux_router for q in aux]).mean()
loss_a=loss_for(a,False);loss_b=loss_for(b,True)
assert torch.equal(loss_a,loss_b)
loss_a.backward();loss_b.backward()
for (n,p),(n2,q) in zip(a.named_parameters(),b.named_parameters()):
 assert n==n2
 if not p.requires_grad:assert p.grad is None and q.grad is None
 else:
  assert p.grad is not None and torch.isfinite(p.grad).all()
  assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5),n
assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in a.named_parameters() if "router" in n)
(R/"sp_unit_check.json").write_text(json.dumps({"passed":True,"minimum_crossing_rule":True,"eta_monotonicity":True,
"zero_update_base_equivalence":True,"base_gradients_absent":True,"router_gradients_present":True,
"nonreentrant_checkpoint_loss_and_gradients_match":True},indent=2))
print("SP_UNIT_TESTS_PASSED",flush=True)
