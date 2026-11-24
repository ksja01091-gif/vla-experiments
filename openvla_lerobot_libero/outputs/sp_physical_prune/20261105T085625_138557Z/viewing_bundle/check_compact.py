"""Meaningful compact-vs-dense reference checks before touching the 7B model."""
from pathlib import Path
import copy,tempfile,json
from pruned_policy import *
def run():
    torch.manual_seed(7)
    base=nn.Linear(24,20,bias=False)
    original=SPLinear(base,rank=128,hidden=32,eta=.9).eval()
    with torch.no_grad():original.sp_B.weight.normal_(std=.03)
    x=torch.randn(2,13,24)
    checks=[]
    for k in [128,96,64]:
        keep=torch.randperm(128)[:k].sort().values
        compact=CompactSPLinear(copy.deepcopy(base),keep.tolist(),eta=.9).eval()
        with torch.no_grad():
            compact.sp_A.weight.copy_(original.sp_A.weight[keep])
            compact.sp_B.weight.copy_(original.sp_B.weight[:,keep])
            compact.router1.load_state_dict(original.router1.state_dict())
            compact.router2.load_state_dict(original.router2.state_dict())
            scores=F.softplus(original.router2(F.silu(original.router1(x)))).clamp_min(1e-6)
            mask,_,_=energy_mask(scores,.9)
            projected=original.sp_A(x)
            gates=(scores*mask)
            dense_keep=torch.zeros(128);dense_keep[keep]=1
            reference=base(x)+original.sp_B(projected*gates*dense_keep)
            result=compact(x)
            error=float((reference-result).abs().max())
            assert torch.allclose(reference,result,rtol=2e-5,atol=2e-6),(k,error)
            if k==128:assert torch.allclose(result,original(x),rtol=2e-5,atol=2e-6)
            assert compact.sp_A.weight.shape==(k,24)
            assert compact.sp_B.weight.shape==(20,k)
            assert compact.router2.out_features==128
            assert compact.sp_A.weight.numel()+compact.sp_B.weight.numel()==k*44
            assert compact.keep_indices.tolist()==keep.tolist()
            checks.append({"rank":k,"max_abs_error_against_dense_removed_components":error,
                           "A_shape":list(compact.sp_A.weight.shape),"B_shape":list(compact.sp_B.weight.shape),
                           "A_B_parameter_count":k*44})
    write(RUN/"unit_checks.json",{"passed":True,"checks":checks,
        "meaning":"Compact multiplication equals the full-bank reference with the same permanently removed components. It is not guaranteed equivalent to the original unpruned policy."})
    print("COMPACT_MATH_CHECKS_PASSED",json.dumps(checks),flush=True)
if __name__=="__main__":run()

