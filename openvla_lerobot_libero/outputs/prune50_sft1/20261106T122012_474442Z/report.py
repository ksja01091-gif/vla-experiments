"""Separate matched SFT controls, two-state diagnostics, and prior PPO references."""
from pathlib import Path
import json,csv,hashlib,subprocess
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2))
cfg=read(R/'experiment_config.json');old=Path(cfg['previous_run'])
protected=read(R/'protected_before.json')
changed=[p for p,h in protected.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
env={}
for name,exe in read(R/'interpreters.json').items():
    snap=subprocess.run([exe,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (R/(name+'_after.txt')).write_text(snap)
    env[name]=sorted(snap.splitlines())==sorted((R/(name+'_before.txt')).read_text().splitlines())
assert not changed and all(env.values())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in read(R/'sources_at_start.json').items())
audit={'protected_files':len(protected),'changed_files':changed,'environments_unchanged':env,'run_sources_unchanged':True}
write(R/'preservation_report.json',audit)
assert read(R/'p50/fresh_reload.json')['passed']
primary=[]
for tag in ['dense_sft','p50']:
    inf=read(R/(tag+'_inference.json'));sim=read(R/(tag+'_result.json'))
    train=read(old/'dense/sft_result.json') if tag=='dense_sft' else read(R/'p50/sft_result.json')
    assert sim['attempts']==5 and sim['technical_errors']==0 and sim['states']==[0,1,2,3,4]
    assert train['frames_seen']==4025 and train['updates']==1007
    primary.append({'method':tag,'stage':'after_sft_before_PPO','parameters_B':inf['parameters']['model_parameters']/1e9,
        'successes':sim['successes'],'attempts':5,'success_rate':sim['success_rate'],'mean_episode_steps':sim['mean_episode_steps'],
        'validation_loss':train['after_validation']['loss'],'action_token_accuracy':train['after_validation']['action_token_accuracy'],
        'normalized_action_l1':train['after_validation']['normalized_action_l1'],
        'prediction_ms':1000*inf['mean_seconds'],'inference_peak_GiB':inf['memory']['peak_gib'],
        'sft_peak_GiB':train['memory']['peak_gib'],'sft_seconds':train['seconds'],
        'sft_resource_measurement':'prior isolated run' if tag=='dense_sft' else 'this run',
        'model_tensor_bytes':inf['parameters']['model_tensor_bytes'],'checkpoint':inf['checkpoint']})
    if tag=='p50':assert train['coverage_sha256']==read(old/'dense/sft_result.json')['coverage_sha256']
for a,b in zip(read(R/'dense_sft_result.json')['trials'],read(R/'p50_result.json')['trials']):
    for k in ['initial_image_sha256','selected_init_state_sha256','init_state_file_sha256','reference_sha256','simulator_versions','max_policy_steps','warmup_steps']:
        assert a[k]==b[k],k
diagnostics={tag:read(R/(tag+'_diagnostic_result.json')) for tag in ['p90_sft','p50']}
for a,b in zip(diagnostics['p90_sft']['trials'],diagnostics['p50']['trials']):
    for k in ['initial_image_sha256','selected_init_state_sha256','reference_sha256','simulator_versions']:
        assert a[k]==b[k],k
reference=read(old/'final_results.json')['comparison']
for row in reference:row['reference_only_different_PPO_stage']=True
with (R/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(primary[0]));writer.writeheader();writer.writerows(primary)
result={'complete':True,'primary_comparison':primary,'diagnostics_two_states':diagnostics,
    'prior_PPO_results_reference_only':reference,'config':cfg,'pruning':read(R/'prepare_complete.json'),
    'new_sft':read(R/'p50/sft_result.json'),'preservation':audit,
    'limitations':['Single bowl task, five previously reused states; not independent generalization.',
        'No new reinforcement learning. Primary controls are both after the same extra SFT1 epoch.',
        'P90 diagnostic is two greedy episodes after SFT, not a new five-state success rate.',
        'The earlier 2/5 versus0/5 results included PPO and are separate reference results.',
        'Recovery rank8 is fixed but trainable parameter count changes with physical structure.',
        'Current pruned model loader constructs original7B first; inference CUDA peak excludes model loading.',
        'Frozen sample hashes and protected original checkpoint file hashes do not compare every frozen tensor byte after every update.']}
write(R/'final_results.json',result)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,4.1));names=['Dense + SFT1','50% prune + SFT1'];colors=['#3877bb','#43a68e']
for ax,key,title in zip(axes,['success_rate','inference_peak_GiB','prediction_ms'],['Task success: five states (%)','Inference CUDA peak (GiB)','Action prediction (ms)']):
    vals=[r[key]*(100 if key=='success_rate' else 1) for r in primary]
    bars=ax.bar(names,vals,color=colors,width=.55)
    ax.set_title(title,fontsize=10);ax.set_ylim(0,105 if key=='success_rate' else max(vals)*1.2)
    for bar,row,val in zip(bars,primary,vals):
        label=f"{row['successes']}/5" if key=='success_rate' else f'{val:.2f}'
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+ax.get_ylim()[1]*.025,label,ha='center',fontsize=10)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',labelsize=8)
fig.suptitle('50% internal-structure removal | Same extra SFT1 | No new PPO',fontsize=12)
fig.text(.5,.02,'One bowl task, five previously used states. CUDA phase peaks exclude original7B construction during loading.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
lines=['50% 제거 목표 실험 결과','',
    '이번 결과는 원본과50%모델 모두 기존5epoch LoRA8모델 + 추가시범학습1epoch 후의 비교다.',
    '새 강화학습은 하지 않았다. 이전 원본2/5·90%모델0/5는PPO까지 한 별도 결과다.',
    '전체 시범40개(사진40장이 아님), 학습4025장면. 검증5개시범462장면은 가중치 수정에 사용하지 않았다.',
    '공개 LIBERO 데이터에서 우리가 선택·전처리한 실험용 분할이다.','',
    '모델 | 파라미터 B | 성공/5 | 검증loss | 예측ms | 추론GPU GiB']
for r in primary:lines.append(f"{r['method']} | {r['parameters_B']:.3f} | {r['successes']}/5 | {r['validation_loss']:.4f} | {r['prediction_ms']:.1f} | {r['inference_peak_GiB']:.2f}")
lines+=['','초기상태5,8의2회진단(최종5회평가와별도):']
for tag,d in diagnostics.items():lines.append(f"{tag}: {d['successes']}/2, 모두시범학습직후·greedy행동예측, 파라미터업데이트없음.")
lines+=['','50%는층1~30의head32개중16개,MLP11008개중5504개제거다. 층0,31과시각모델은보존했다.',
    '전체 모델이정확히절반이라는뜻은아니다. prepare_complete.json에서실제크기와감소율확인.',
    '실패의확인된사실:220행동내LIBERO성공판정미도달. 집기/놓기세부원인은영상으로따로진단해야한다.',
    '기존코드·모델·환경보존검사통과. 저장·새프로세스재로딩·유효한(1,7)행동검사통과.',
    '핵심코드:policy.py의score_and_plan/apply_plan,worker.py의prepare/sft/simulate,driver.py전체실행순서.',
    '같은초기상태0의영상:simulation/dense_sft/state_00/episode.mp4와simulation/p50/state_00/episode.mp4.',
    '손실감소와실제작업성공은다르다. 기존부터쓴5상태이므로일반화성능으로확대해석하지않는다.',
    '현재로더는시작할때원본7B를구성한다. 추론GPU메모리수치가최대시작메모리를뜻하지않는다.']
(R/'RESULTS_KO.txt').write_text('\n'.join(lines),encoding='utf-8')
print('P50_REPORT_COMPLETE',json.dumps(primary),flush=True)
