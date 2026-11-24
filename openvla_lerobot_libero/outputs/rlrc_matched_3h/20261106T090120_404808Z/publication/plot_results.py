"""Plot recorded results only; does not load any robot model."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
d=json.loads((R/'final_results.json').read_text());table=d['comparison']
fig,axes=plt.subplots(1,3,figsize=(12,4.1))
names=['Dense 7.55B','Pruned 2.06B'];colors=['#3877bb','#df873d']
rates=[r['success_rate']*100 for r in table]
bars=axes[0].bar(names,rates,color=colors,width=.55)
axes[0].set_ylim(0,105);axes[0].set_ylabel('Success (%)');axes[0].set_title('Task success: five fixed states')
for bar,row in zip(bars,table):axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,f"{row['successes']}/5",ha='center',fontsize=11)
bars=axes[1].bar(names,[r['sft_peak_GiB'] for r in table],color=colors,width=.55)
axes[1].set_ylim(0,18);axes[1].set_ylabel('GiB');axes[1].set_title('SFT peak CUDA memory')
for bar,row in zip(bars,table):axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.4,f"{row['sft_peak_GiB']:.2f}",ha='center',fontsize=11)
for name,b,color in zip(names,['dense','pruned'],colors):
    history=d['branches'][b]['validation_history']
    vals=[history[s]['loss'] for s in ['initial','after_sft','after_ppo']]
    axes[2].plot(['Initial','After SFT','After PPO'],vals,marker='o',label=name,color=color)
axes[2].set_ylabel('Cross-entropy loss');axes[2].set_title('Validation: all 462 frames');axes[2].legend(fontsize=9)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('RLRC-inspired pilot | Same extra SFT1 + PPO298 | One LIBERO bowl task',fontsize=12)
fig.text(.5,.02,'CUDA peaks exclude model loading. Initial pruned loss is measured after pruning. Not a full paper reproduction.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
(R/'plot_recovery.json').write_text(json.dumps({'created':True,'environment':'existing openvla310; plotting only','no_package_changes':True,'no_model_execution':True},indent=2))
print('PLOT_CREATED',R/'comparison.png',flush=True)
