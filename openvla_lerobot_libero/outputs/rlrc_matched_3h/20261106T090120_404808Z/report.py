"""Actual paired results and preservation audit; no missing result is guessed."""
from pathlib import Path
import json, csv, hashlib, subprocess
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2))
protected=read(R/'protected_before.json')
changed=[p for p,h in protected.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
environments={}
for name,exe in read(R/'interpreters.json').items():
    completed=subprocess.run([exe,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (R/(name+'_after.txt')).write_text(completed.stdout)
    environments[name]=sorted(completed.stdout.splitlines())==sorted((R/(name+'_before.txt')).read_text().splitlines())
sources=read(R/'sources_at_start.json')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
assert not changed and all(environments.values())
audit={'protected_files':len(protected),'changed_files':changed,'environments_unchanged':environments,'run_sources_unchanged':True}
write(R/'preservation_report.json',audit)
results={};table=[]
for b in ['dense','pruned']:
    d={n:read(R/b/(n+'.json')) for n in ['sft_result','collection_result','ppo_result','fresh_reload','inference','simulation_result']}
    assert d['fresh_reload']['passed'] and d['simulation_result']['attempts']==5
    d['validation_history']={s:read(R/b/('validation_'+s+'.json')) for s in ['initial','after_sft','after_ppo']}
    results[b]=d
    row={'method':b,'parameters_B':d['inference']['parameters']['model_parameters']/1e9,
        'initial_validation_loss':d['validation_history']['initial']['loss'],
        'after_sft_validation_loss':d['validation_history']['after_sft']['loss'],
        'after_ppo_validation_loss':d['validation_history']['after_ppo']['loss'],
        'action_token_accuracy':d['validation_history']['after_ppo']['action_token_accuracy'],
        'normalized_action_l1':d['validation_history']['after_ppo']['normalized_action_l1'],
        'successes':d['simulation_result']['successes'],'attempts':5,'success_rate':d['simulation_result']['success_rate'],
        'mean_episode_steps':d['simulation_result']['mean_episode_steps'],
        'sft_peak_GiB':d['sft_result']['memory']['peak_gib'],'ppo_peak_GiB':d['ppo_result']['memory']['peak_gib'],
        'inference_peak_GiB':d['inference']['memory']['peak_gib'],
        'prediction_ms':1000*d['inference']['mean_seconds'],
        'sft_seconds':d['sft_result']['seconds'],'ppo_seconds':d['ppo_result']['seconds'],
        'checkpoint':d['ppo_result']['checkpoint']}
    table.append(row)
assert results['dense']['sft_result']['coverage_sha256']==results['pruned']['sft_result']['coverage_sha256']
assert results['dense']['ppo_result']['updates']==results['pruned']['ppo_result']['updates']
for x,y in zip(results['dense']['simulation_result']['trials'],results['pruned']['simulation_result']['trials']):
    for k in ['initial_image_sha256','selected_init_state_sha256','init_state_file_sha256','reference_sha256','simulator_versions']:
        assert x[k]==y[k],k
with (R/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
write(R/'final_results.json',{'complete':True,'not_exact_paper_reproduction':True,
    'description':'RLRC-inspired structured backbone pruning and new rank8 q/v LoRA SFT/PPO recovery on one LIBERO task',
    'public_data_description':'public LIBERO demonstrations selected and preprocessed into an experimental split',
    'config':read(R/'experiment_config.json'),'pruning':read(R/'prepare_complete.json'),
    'matched_ppo_budget':read(R/'matched_ppo_budget.json'),'comparison':table,'branches':results,'preservation':audit,
    'limitations':['Only five previously reused fixed evaluation states; not independent LIBERO-wide performance.',
        '90% internal structure pruning and short recovery may fail; failure does not settle the full paper method.',
        'Teacher-forced validation loss and robot task success are different measurements.',
        'Shared rank8 recovery targets q/v, but physical pruning changes trainable parameter count.',
        'Dense saved checkpoint is adapter-only; compare full tensor bytes, not adapter file against full pruned file.']})
lines=['RLRC 본체 압축 비교 결과','',
    '원본과 압축 모델은 같은 기존 5epoch LoRA8 모델에서 출발했다.',
    '각각 학습 장면4025개를1epoch 추가 학습하고, 상태5~8에서4번 시도한 뒤 같은 수의 PPO 업데이트를 했다.',
    '검증 시범5개/462장면은 가중치 업데이트에 쓰지 않았다. 평가 상태0~4도 강화학습에서 제외했다.',
    '공개 LIBERO 데이터에서 우리가 선택·전처리한 실험용 분할이다.',
    '논문 전체 재현이 아니라 한 작업에 적용한 작은 비교 실험이다.','',
    '모델 | 크기 B | 성공/5 | 예측ms | 시범학습 GPU GiB | PPO GPU GiB']
for t in table:
    lines.append(f"{t['method']} | {t['parameters_B']:.3f} | {t['successes']}/5 | {t['prediction_ms']:.1f} | {t['sft_peak_GiB']:.2f} | {t['ppo_peak_GiB']:.2f}")
lines+=['','검증loss: 시작 → 시범학습 후 → 강화학습 후']
for t in table:lines.append(f"{t['method']}: {t['initial_validation_loss']:.4f} → {t['after_sft_validation_loss']:.4f} → {t['after_ppo_validation_loss']:.4f}")
lines+=['','성공 판정: LIBERO check_success()가220행동 이내에 참이 되었는지.',
    '실패 원인은 먼저 제한 시간 내 성공하지 못했다는 사실만 기록. 집기/놓기 진단은 영상을 본 뒤 판단한다.',
    '이전부터 반복 사용한5상태이므로 새로운 환경 일반화 성능으로 해석하지 않는다.',
    '크기 감소, GPU 메모리 감소, 속도 개선, 작업 성공은 각각 구분해서 본다.','',
    '코드 읽는 순서: experiment_config.json → driver.py → worker.py → policy.py.',
    'worker.py: sft(시범학습), collect_episode(직접시도), ppo(보상학습), evaluate(최종시험).',
    'policy.py: score_and_plan(제거할 구조 선택), apply_plan(실제 제거), save_checkpoint/load_checkpoint(저장/재로딩).',
    'simulation/dense 와 simulation/pruned: 상태별 영상 episode.mp4, 결과 result.json.',
    'checkpoints/*/after_ppo: 최종 모델. dense는 원본 모델과 기존 adapter에 의존하는 추가 adapter만 저장한다.',
    '비교표 comparison.csv / 상세 final_results.json / 보존 검사 preservation_report.json.',
    '기존 환경3개와 보호 대상 파일은 변경하지 않았다.']
(R/'RESULTS_KO.txt').write_text('\n'.join(lines),encoding='utf-8')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    names=['Dense','Pruned'];colors=['#4273b9','#e88d3a']
    axes[0].bar(names,[t['success_rate']*100 for t in table],color=colors);axes[0].set_ylim(0,100);axes[0].set_title('Task success (%), n=5')
    axes[1].bar(names,[t['sft_peak_GiB'] for t in table],color=colors);axes[1].set_title('SFT peak GPU memory (GiB)')
    for name,d,c in zip(names,results.values(),colors):
        axes[2].plot(['Initial','SFT','PPO'],[d['validation_history'][s]['loss'] for s in ['initial','after_sft','after_ppo']],marker='o',label=name,color=c)
    axes[2].set_title('Validation loss (462 frames)');axes[2].legend();fig.tight_layout();fig.savefig(R/'comparison.png',dpi=170);plt.close(fig)
except ImportError as e:write(R/'plot_status.json',{'created':False,'reason':str(e),'csv_preserved':True})
print('PAIRED_RESULTS_COMPLETE',json.dumps(table),flush=True)
