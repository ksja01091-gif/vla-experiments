"""Measured feasibility and transparent runtime projections; no invented success rate."""
from pathlib import Path
import json,subprocess,hashlib,math
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2))
reports={branch:read(R/branch/'complete.json') for branch in ['dense','pruned']}
for branch in reports:
    assert read(R/branch/'fresh_reload.json')['passed']
    assert reports[branch]['sft']['frozen_samples_unchanged'] and reports[branch]['ppo']['frozen_samples_unchanged']
initial=[read(R/b/'rl_rollout/initialization.json') for b in ['dense','pruned']]
for key in ['selected_init_state_sha256','init_state_file_sha256','simulator_versions','reference_sha256']:
    assert initial[0][key]==initial[1][key],key
before=read(R/'protected_before.json')
changed=[p for p,h in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
env={}
for name,exe in read(R/'interpreters.json').items():
    result=subprocess.run([exe,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    env[name]=sorted(result.stdout.splitlines())==sorted((R/(name+'_before.txt')).read_text().splitlines())
assert not changed and all(env.values())
prep_hashes=read(R/'preparation_source_hashes.json')
used_prep={p:h for p,h in prep_hashes.items() if Path(p).name in ['policy.py','prepare.py','check_math.py']}
run_hashes=read(R/'connection_source_hashes.json')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in {**used_prep,**run_hashes}.items())
preservation={'protected_files':len(before),'changed_files':changed,'environments_unchanged':env,'used_sources_unchanged':True}
write(R/'preservation_report.json',preservation)
budgets=[]
for sft_epochs,rl_iterations in [(1,1),(3,3),(5,5)]:
    detail={}
    for branch,d in reports.items():
        sft=math.ceil(4025/4)*sft_epochs*d['sft']['mean_update_seconds_excluding_first']
        validation=math.ceil(462/4)*d['sft']['validation_four_frame_seconds']*(sft_epochs+2)
        # Explicit small single-task adaptation: 4 whole episodes per PPO collection,
        # 2 passes, microbatch1 accumulated over4 transitions, plus a train-only BC frame.
        rollout=4*220*d['rollout']['mean_transition_wall_seconds']
        updates=math.ceil(4*220/4)*2*d['ppo']['ppo_update_seconds']
        rl=(rollout+updates)*rl_iterations
        detail[branch]={'sft_seconds':sft,'validation_seconds':validation,'rl_seconds':rl,
                        'four_episode_rollout_seconds_per_iteration':rollout,'ppo_updates_seconds_per_iteration':updates}
    seconds=sum(sum(v[k] for k in ['sft_seconds','validation_seconds','rl_seconds']) for v in detail.values())
    budgets.append({'sft_epochs_each':sft_epochs,'rl_iterations_each':rl_iterations,'episodes_per_iteration_each':4,
        'max_steps':220,'ppo_passes':2,'matched_models':2,'estimate_minutes_30pct_margin':seconds*1.3/60,
        'not_a_guarantee_or_permission_to_start':True,'components':detail})
summary={'connection_passed':True,'not_full_RLRC_reproduction':True,'no_full_training_started':True,
    'no_success_rate_evaluated':True,'accepted_choices':read(R/'accepted_choices.json'),
    'pruning':read(R/'prepare_complete.json'),'branches':reports,'runtime_options':budgets,
    'projection_limits':['Only 4 SFT updates, 8 inference images and 8 RL transitions per branch were measured.',
        'Full220-step episodes projected assuming same per-step cost; actual task lengths and throughput may differ.',
        'This single-task 4-episode PPO collection is much smaller than paper training.',
        'Final calibration will use160 train frames instead of the8 smoke frames.'],
    'preservation':preservation}
write(R/'smoke_summary.json',summary)
text=['RLRC 90% 목표 연결 시험','',
      '사용자 선택: 동일 추가 학습 대조군 / 시범 학습과 강화학습 포함 / LLM 구조90% 제거 목표.',
      '이번 실행은 긴 학습 전 연결·메모리·시간 측정이다. 성능이나 성공률을 평가한 결과가 아니다.',
      '가지치기 중요도는 학습 사진8장면으로 임시 측정. 본 학습 전에160장면으로 다시 정한다.',
      '기존 LoRA를 합치지 않고 고정해서 보존. 새 회복 LoRA만 시범 학습/PPO로 소량 업데이트.',
      '처음4개 검증 장면에서 기존 정책과 행동·loss가 정확히 일치한 뒤 분기.',
      '', '모델 | 파라미터 B | SFT peak GiB | PPO peak GiB | 예측 ms']
for b,d in reports.items():text.append(f"{b} | {d['parameters']['model_parameters']/1e9:.3f} | {d['sft']['memory']['peak_gib']:.3f} | {d['ppo']['peak_gib']:.3f} | {1000*d['inference_seconds']:.1f}")
text+=['','저장·재로딩 및 유효한(1,7)행동 확인, 원본 본체·출발 LoRA 고정 확인.',
       'RL은 실제 시뮬8스텝에서 수집한 행동으로 critic2업데이트와 PPO+BC 1업데이트만 연결 시험.',
       '8스텝 중 미완료를 그릇 옮기기 실패율로 해석하지 않는다.',
       '', '학습 예산 후보(양쪽 모델 같은 횟수):']
for b in budgets:text.append(f"SFT각{b['sft_epochs_each']}epoch + RL각{b['rl_iterations_each']}회(회당4episode, 최대220step): 합계 약{b['estimate_minutes_30pct_margin']:.1f}분. 추정치일 뿐이다.")
text+=['','핵심코드: policy.py(실제 본체 제거), rl_probe.py(시뮬에서 행동 수집·PPO+시범유지 손실), measure.py(짧은 측정).',
       '전체 논문 그대로의 조건이 아닌 LIBERO 한 작업에 적용한 구현. 새 환경에서 추가 패키지 설치 없음.',
       '학습률·회복 LoRA rank·reward·PPO batch/iteration 등은 본 학습 전에 확정해야 한다.',
       '기존3환경과 원래 코드·checkpoint 보존 확인.']
(R/'RESULTS_KO.txt').write_text('\n'.join(text),encoding='utf-8')
print('CONNECTION_ALL_PASSED',json.dumps({'runtime_options':budgets,'preservation':preservation}),flush=True)
