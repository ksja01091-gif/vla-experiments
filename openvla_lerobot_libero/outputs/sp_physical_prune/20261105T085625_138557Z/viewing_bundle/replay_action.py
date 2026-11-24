"""Reload one saved variant and predict one real validation image; no training."""
import argparse,json
from pruned_policy import *
p=argparse.ArgumentParser()
p.add_argument("--variant",choices=list(VARIANTS),default="sp_pruned64")
p.add_argument("--index",type=int,default=0)
a=p.parse_args()
policy,val,cp=load_policy(a.variant)
assert 0<=a.index<len(val)
action=common.sample_action(policy,val,a.index)
assert action.shape==(1,7) and bool(torch.isfinite(action).all())
print(json.dumps({"variant":a.variant,"checkpoint":str(cp),"index":a.index,"shape":list(action.shape),"finite":True,"action":action.cpu().tolist()}))

