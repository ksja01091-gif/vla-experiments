"""Check a saved physically pruned policy and PPO actor in a fresh process."""
import argparse,json
from policy import *
p=argparse.ArgumentParser();p.add_argument('--branch',choices=['dense','pruned'],required=True);a=p.parse_args()
_,val,stats=exp.get_data();expected=json.loads((R/a.branch/'reload_expected.json').read_text())
policy=load_checkpoint(expected['checkpoint'],stats)
action=common.sample_action(policy,val,expected['validation_index'])
assert action.shape==(1,7) and bool(torch.isfinite(action).all())
assert np.array_equal(np.asarray(action.cpu().tolist()),np.asarray(expected['action']))
write(R/a.branch/'fresh_reload.json',{'passed':True,'actor_hash_exact':True,'action_exact':True,'shape':[1,7],'finite':True})
print('FRESH_RELOAD_PASSED',a.branch,flush=True)
