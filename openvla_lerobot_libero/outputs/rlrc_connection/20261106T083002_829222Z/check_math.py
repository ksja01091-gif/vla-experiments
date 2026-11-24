"""CPU checks against dense references; not a substitute for GPU integration."""
import copy,types,json
from policy import *
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

def lora_source(base):
    a=nn.Linear(base.in_features,8,bias=False);b=nn.Linear(8,base.out_features,bias=False)
    return types.SimpleNamespace(base_layer=base,lora_A={'default':a},lora_B={'default':b},scaling={'default':1.})

torch.manual_seed(7)
base=nn.Linear(32,32,bias=False);src=lora_source(base)
wrapped=RecoveryLinear(src,0)
x=torch.randn(2,5,32)
expected=base(x)+src.lora_B['default'](src.lora_A['default'](x))
assert torch.equal(expected,wrapped(x))
cfg=LlamaConfig(hidden_size=32,intermediate_size=64,num_attention_heads=4,num_key_value_heads=4,attention_dropout=0.)
full=LlamaAttention(cfg,layer_idx=0).eval()
full.q_proj=RecoveryLinear(lora_source(full.q_proj),1)
full.v_proj=RecoveryLinear(lora_source(full.v_proj),2)
compact=copy.deepcopy(full);keep_heads=[0,2];keep=[h*8+j for h in keep_heads for j in range(8)]
removed=[j for j in range(32) if j not in keep]
with torch.no_grad():full.o_proj.weight[:,removed]=0
for n in ['q_proj','k_proj','v_proj']:setattr(compact,n,cut_linear(getattr(compact,n),keep,0))
compact.o_proj=cut_linear(compact.o_proj,keep,1)
compact.num_heads=compact.num_key_value_heads=2;compact.num_key_value_groups=1;compact.hidden_size=16
pos=torch.arange(5).unsqueeze(0).expand(2,-1)
ref=full(x,position_ids=pos)[0];got=compact(x,position_ids=pos)[0]
assert torch.allclose(ref,got,atol=2e-6,rtol=2e-5)
gate=nn.Linear(32,64,bias=False);up=nn.Linear(32,64,bias=False);down=nn.Linear(64,32,bias=False)
keep_mlp=[1,5,13,24,40,61]
projected=F.silu(gate(x))*up(x);mask=torch.zeros(64);mask[keep_mlp]=1
ref_mlp=down(projected*mask)
got_mlp=cut_linear(down,keep_mlp,1)(F.silu(cut_linear(gate,keep_mlp,0)(x))*cut_linear(up,keep_mlp,0)(x))
assert torch.allclose(ref_mlp,got_mlp,atol=2e-6,rtol=2e-5)
report={'passed':True,'zero_recovery_matches_unmerged_start_exactly':True,
        'attention_max_error':float((ref-got).abs().max()),'mlp_max_error':float((ref_mlp-got_mlp).abs().max()),
        'attention_shape':list(compact.q_proj.weight.shape),'attention_output_shape':list(got.shape),
        'meaning':'Compact attention and MLP match dense references with the same removed contributions. Not equivalent to the original unpruned model.'}
write(R/'math_checks.json',report);print('MATH_CHECKS_PASSED',json.dumps(report),flush=True)
