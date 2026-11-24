"""Beginner guide and verifiable viewing bundle from completed Linux results."""
from pathlib import Path
import json,hashlib,zipfile
R=Path(__file__).resolve().parents[1]
def read(name):return json.loads((R/name).read_text())
d=read('final_results.json');status=read('status.json');assert status['paired_experiment_complete']
rows=d['comparison'];dense,pruned=rows
wb=read('wandb_upload_result.json');assert wb['verified']
text=f'''이번 실험에서 먼저 볼 것

결론: 모델 본체와 GPU 메모리는 줄였다. 이번5상태에서는 작업 성능을 유지하지 못했다.
원본 추가학습 모델: {dense['successes']}/5 성공. 압축 추가학습 모델: {pruned['successes']}/5 성공.
총 실행: {status['elapsed_seconds']/60:.1f}분. 승인 상한180분 이내.

1. 같은 초기 상태의 영상부터 보기
01_dense7B_state00.mp4: 원본7B, 시범학습+강화학습 후. 상태0에서 성공.
02_pruned2B_state00.mp4: 압축2B, 동일 추가학습 후. 상태0에서220행동 내 미완료.
둘의 최초 사진과 로봇 상태는 정확히 같았다.
실패의 확인된 사실은220행동 내 LIBERO 성공 판정에 도달하지 못했다는 것이다.
어느 집기/놓기 동작이 원인인지는 별도의 자세한 영상·행동 분석이 필요하다.

2. 무엇을 한 실험인가
출발점은 둘 다 기존 일반 LoRA rank8·5epoch 모델이다. 순수 사전학습 모델과의 비교가 아니다.
기존 본체와 기존 LoRA는 고정하고, 새 회복용 rank8 LoRA(q/v)를 학습했다.
한쪽은 본체를 유지하고, 다른 쪽은 LLM 내부 대상 구조90% 제거를 목표로 실제로 잘랐다.
전체 파라미터는7.55B→2.06B(약72.7% 감소)다. 전체 모델90% 제거라는 뜻은 아니다.
공개 LIBERO 데이터에서 우리가 선택·전처리한 실험용 분할을 사용했다.
학습: 전체 시범40개,4025장면. 검증: 전체 시범5개,462장면.
시범1개는 사진1장이 아니라 처음부터 끝까지의 로봇 시연1개다.
시범학습: 각 모델4025장면을 한 번씩, batch4,1007업데이트, 학습률0.0001.
직접 연습: 초기 상태5~8에서 각4회. 원본2/4 성공, 압축0/4 성공. 최종 시험과 다른 기록이다.
원본599장면/압축880장면 중 각596장면을 뽑아, 각2회 통과·298PPO업데이트로 학습량을 맞췄다.
PPO 학습률0.00001. 업데이트마다 학습용 시범1장면의 오차도0.1가중치로 함께 반영했다.
검증462장면은 가중치 수정에 쓰지 않았다.
최종 시험: 상태0~4, 각5회, 각최대220행동,10warmup행동. 이번 강화학습에서는 제외한 상태다.
이전부터 반복 사용한5상태이므로, 새로운 환경 일반화나 LIBERO 전체 성능의 증거는 아니다.
논문의 아이디어를 한 작업에 적용한 작은 실험이며 논문 전체 재현은 아니다.

3. 핵심 숫자
항목                    원본 추가학습             압축 추가학습
파라미터                {dense['parameters_B']:.3f}B                    {pruned['parameters_B']:.3f}B
성공                    {dense['successes']}/5                       {pruned['successes']}/5
행동예측1회             {dense['prediction_ms']:.1f}ms                   {pruned['prediction_ms']:.1f}ms
시범학습 peak CUDA      {dense['sft_peak_GiB']:.2f}GiB                 {pruned['sft_peak_GiB']:.2f}GiB
PPO peak CUDA           {dense['ppo_peak_GiB']:.2f}GiB                 {pruned['ppo_peak_GiB']:.2f}GiB
평균 episode행동        {dense['mean_episode_steps']:.1f}                      {pruned['mean_episode_steps']:.1f}
최종 검증loss           {dense['after_ppo_validation_loss']:.4f}                    {pruned['after_ppo_validation_loss']:.4f}
최종 행동토큰정확도     {dense['action_token_accuracy']:.4f}                    {pruned['action_token_accuracy']:.4f}
최종 정규화행동L1       {dense['normalized_action_l1']:.4f}                    {pruned['normalized_action_l1']:.4f}
상세 수치는 comparison.csv / final_results.json을 보면 된다.
낮은loss나 높은토큰정확도가 실제 작업 성공을 보장하지 않는다.
메모리는 모델을 불러온 뒤의 단계별 CUDA할당 최대값이다.
일회성 구조선택에는 약25.95GiB가 필요했다.
현재 로더는 원본7B를 먼저 구성하므로, 시작할 때 필요한 메모리까지4GiB라는 뜻은 아니다.
압축 모델 파일은 본체 전체 약4.14GB, 원본 checkpoint는 추가LoRA만 저장해 파일크기를 직접 비교하면 안 된다.

4. 코드는 이 순서로 읽기
experiment_config.json: 실험 설정표.
driver.py: 전체 순서와3시간 제한.
worker.py의 sft(): 정답 시범 보고 학습.
worker.py의 collect_episode(): 시뮬레이터에서 직접 움직여 경험 수집.
worker.py의 ppo(): 받은 보상과 시범 오차로 새LoRA 수정.
worker.py의 evaluate(): 저장 모델 재로딩, 행동 검사, 실제 로봇 시험.
policy.py의 score_and_plan()/apply_plan(): 제거할 구조 선택/실제 제거.
policy.py의 save_checkpoint()/load_checkpoint(): 모델 저장/복구.
local source/ 폴더는 실제 실행 코드의 사본이다. 모델 실행은 원격Linux에서만 했다.
원격 실험 폴더:
{R}
최종checkpoint:
{R}/checkpoints/dense/after_ppo
{R}/checkpoints/pruned/after_ppo

5. 다시 실행하기(원격VS Code 터미널)
아래 명령은 기존 결과를 덮어쓰지 않고 새 폴더에 최대3시간짜리 실험을 시작한다.
/home/jovyan/projects/venvs/rlrc-smoke312/bin/python {R}/reproduce.py --start
파일 검사만 하고 싶으면 끝의 --start를 --check로 바꾼다.
--check는 실제로 통과했다. --start로 전체 실험을 두 번 반복 검증한 것은 아니다.

6. 오류와 보존
연결 시험에서 행동을 뽑을 때와 학습 때의 확률 계산 차이를 발견했다.
같은 입력 길이·계산 경로로 통일했고, 본 실험은 샘플링/재계산 일치 검사와 첫PPO확률비1 검사를 통과했다.
본 학습·시뮬레이터·재로딩에 실행 오류는 없었다. 작업 실패와 코드 실행 오류는 다르다.
기본 그래프 환경에 matplotlib이 없어, 기존 openvla310 환경에서 그래프만 그렸다. 패키지는 바꾸지 않았다.
기존3환경과 보호 파일322개 보존 확인. GPU메모리는 실행 후0MiB로 반환됐다.

7. W&B
아래 두 개가 이번 실험 기록이다. Finished는 기록 종료이지 로봇 성공이라는 뜻이 아니다.
RLRC90-dense-control-SFT1-PPO4
RLRC90-pruned-2B-SFT1-PPO4
검증 그래프 stage_index: 0=추가학습전(압축은 가지치기직후),1=시범학습후,2=PPO후.
새 실험 URL:
'''
text+='\n'.join(x['url'] for x in wb['runs'])
text+='\n\n이번에 이해할 개념2개\n압축: 본체 구조와 저장해야 할 숫자를 줄이는 것.\n회복학습: 줄인 뒤 성능을 되찾으려고 정답 시범과 보상으로 남은 학습값을 수정하는 것.\n'
(R/'START_HERE_KO.txt').write_text(text,encoding='utf-8')
entries={}
for name in ['START_HERE_KO.txt','RESULTS_KO.txt','comparison.csv','comparison.png','final_results.json','status.json',
             'experiment_config.json','pruning_plan.json','preservation_report.json','execution_notes.json',
             'training_condition_audit.json','reproduction_check.json','wandb_upload_result.json']:
    if (R/name).exists():entries[name]=R/name
for name in ['policy.py','worker.py','driver.py','rl_probe.py','report.py','upload_results.py','check_math.py','reproduce.py']:
    entries['source/'+name]=R/name
for b in ['dense','pruned']:
    for name in ['fresh_reload.json','sft_result.json','ppo_result.json','simulation_result.json','inference.json']:
        entries['checks/'+b+'_'+name]=R/b/name
entries['01_dense7B_state00.mp4']=R/'simulation/dense/state_00/episode.mp4'
entries['02_pruned2B_state00.mp4']=R/'simulation/pruned/state_00/episode.mp4'
entries['pruned_state00_overview.png']=R/'simulation/pruned/state_00/episode_overview.png'
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in entries.items()}
(R/'viewing_bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(R/'viewing_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name,path in entries.items():archive.write(path,name)
    archive.write(R/'viewing_bundle_manifest.json','viewing_bundle_manifest.json')
print('VIEWING_BUNDLE_READY',len(entries),(R/'viewing_bundle.zip').stat().st_size,flush=True)
