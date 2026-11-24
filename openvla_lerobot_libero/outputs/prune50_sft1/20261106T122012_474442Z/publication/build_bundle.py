"""Package recorded remote results and a short Korean reading guide; no model runs."""
from pathlib import Path
import json,hashlib,zipfile,subprocess
R=Path(__file__).resolve().parents[1]
def read(name):return json.loads((R/name).read_text())
status=read('status.json');assert status['experiment_complete']
d=read('final_results.json');dense,p50=d['primary_comparison'];prep=d['pruning']
diagnostic=d['diagnostics_two_states'];wb=read('wandb_upload_result.json')
readme=f'''이번에는 이 파일부터 읽기

무엇을 비교했나?
같은 기존5epoch LoRA rank8모델에서 출발했다.
원본은 본체를 유지하고,50%모델은 LLM 내부 대상 구조의 절반을 실제로 제거했다.
두 모델 모두 추가 시범 학습1epoch 후의 상태다. 이번에 새 강화학습은 하지 않았다.
원본의 추가 시범 학습checkpoint는 이미 정확히 같은 데이터순서로 학습돼 있어 재사용하고, 로봇시험은 새로 실행했다.
앞서 알려준 원본2/5와90%모델0/5는PPO까지한 별도실험이다. 이번 표와학습단계를섞지않는다.

실제 결과
항목                    원본+추가시범학습       50%제거+추가시범학습
모델 파라미터           {dense['parameters_B']:.3f}B                   {p50['parameters_B']:.3f}B
최종 시험 성공          {dense['successes']}/5                       {p50['successes']}/5
행동 예측1회            {dense['prediction_ms']:.1f}ms                  {p50['prediction_ms']:.1f}ms
추론 CUDA최대           {dense['inference_peak_GiB']:.2f}GiB                {p50['inference_peak_GiB']:.2f}GiB
시범학습 CUDA최대       {dense['sft_peak_GiB']:.2f}GiB                {p50['sft_peak_GiB']:.2f}GiB
검증 loss               {dense['validation_loss']:.4f}                   {p50['validation_loss']:.4f}
검증 토큰 정확도        {dense['action_token_accuracy']:.4f}                   {p50['action_token_accuracy']:.4f}
검증 정규화 행동L1      {dense['normalized_action_l1']:.4f}                   {p50['normalized_action_l1']:.4f}
평균 episode행동        {dense['mean_episode_steps']:.1f}                     {p50['mean_episode_steps']:.1f}
전체 실행 시간          {status['elapsed_seconds']/60:.1f}분, 상한60분
원본 시범학습의시간/메모리는이전독립실행측정치, 원본추론/로봇평가는이번새실행측정치다.

50% 제거는 무슨 뜻?
중간 층1~30의Attention head32개중16개,MLP11008개중5504개를 남겼다.
시각모델과 맨앞/맨뒤decoder층0,31은보존했다.
전체 파라미터 감소는 {prep['whole_parameter_reduction']*100:.1f}%다. 전체7B의정확히절반제거라는뜻은아니다.
구조 선택에는 학습용160장면을 썼다. 검증데이터는 쓰지 않았다.
기존 본체와기존LoRA는고정, 새 회복용LoRA rank8의q/v부분만 학습했다.

데이터를 어떻게 썼나?
공개 LIBERO 데이터에서 우리가 선택·전처리한 실험용 분할이다.
학습용은전체시범40개,사진·행동쌍4025개다. 시범40개는사진40장이아니다.
추가1epoch=4025장면을한번씩사용, batch4로1007번업데이트, 학습률0.0001.
검증용은전체시범5개,462장면이다. 가중치수정에는사용하지않았다.
50%모델의검증loss: 압축직후 {d['new_sft']['before_validation']['loss']:.4f} → 추가시범학습후 {p50['validation_loss']:.4f}.
동일사진처리,기존train20분할의고정q01/q99행동정규화를유지했다.

진단과최종시험은별도
진단상태5,8:90%모델은시범학습직후 {diagnostic['p90_sft']['successes']}/2 성공,
50%모델은시범학습직후 {diagnostic['p50']['successes']}/2 성공.
이는강화학습전의greedy행동예측으로시험한결과다. 이전확률샘플링연습결과와혼동하지않는다.
진단을보고checkpoint나조건을고르지않았고,성패와관계없이예정된최종시험을실행했다.
최종상태0~4: 각5회,각최대220행동,시작전10warmup,공식LIBERO성공판정사용.
이전부터반복사용한상태라서완전히새로운상황일반화나LIBERO전체성능으로볼수없다.
작업실패의확인된사실은220행동내성공판정미도달이다. 정확한집기/놓기원인은영상으로별도분석한다.

어디서보면되나?
comparison.png:성공률·추론메모리·속도한눈에비교.
comparison.csv:엑셀로여는실제수치표.
01_dense_SFT1_state00.mp4 / 02_prune50_SFT1_state00.mp4:같은상태0의원본/50%모델영상.
50_state00_overview.png:50%모델대표장면.
final_results.json:전체상세결과. diagnostics와priorPPO참고결과는따로분리.
preservation_report.json:기존파일/환경보존검사.

코드읽는순서
source/experiment_config.json:설정표.
source/driver.py:최대1시간안에어떤순서로실행하는지.
source/policy.py의score_and_plan():남길head와MLP통로고르기.
source/policy.py의apply_plan():선택한구조만실제로남기기.
source/worker.py의sft():시범데이터로loss계산→backward→optimizer.step→검증→저장.
source/worker.py의simulate():저장모델재로딩→사진/지시→(1,7)행동→시뮬레이터반복.
source/report.py:단계가같은원본과50%모델의표·그림·보존검사.
실제원격폴더: {R}
새최종checkpoint: {p50['checkpoint']}

나중에다시실행할명령(원격Linux터미널)
/home/jovyan/projects/venvs/rlrc-smoke312/bin/python {R}/reproduce.py --start
이명령은기존결과를덮어쓰지않고새폴더에1시간상한실험을시작한다.
파일검사만하려면 --start 대신 --check. --check 통과여부는reproduction_check.json참조.
전체실험을두번반복검증한것은아니다.

주의해서읽을숫자
메모리수치는모델을불러온뒤의단계별CUDA할당최대다.
현재로더는원본7B를먼저구성하므로이값으로시작단계메모리까지줄었다고말할수없다.
loss감소와그릇옮기기성공은다르다. 속도감소도별도로실측해야한다.
원본checkpoint는추가adapter만,50%checkpoint는압축본체전체를저장한다. 파일크기를그대로비교하면안된다.
논문전체재현이아닌한작업의소규모조건비교다.

W&B실제기록
'''
if wb.get('verified'):readme+='\n'.join(entry['url'] for entry in wb['runs'])
else:readme+='업로드검증안됨. 로컬표/로그/영상은저장돼있음.'
readme+='\n\n이번에기억할개념2개\n제거비율: 대상구조중얼마나잘라냈는지.\n회복학습: 자른뒤남은학습값을조정해성능을되찾는과정.\n'
(R/'DETAILS_KO.txt').write_text(readme,encoding='utf-8')
readme=f'''이번 실험은 여기부터 읽어줘

결론: 대상 구조를 50% 제거해도, 이번 짧은 회복 학습으로는 작업 성공을 유지하지 못했어.
총 실행 시간은 {status['elapsed_seconds']/60:.1f}분. 승인한 60분 상한 안에 끝났어.

같은 조건끼리 비교한 결과
항목                 원본 모델             50% 제거 모델
파라미터 수          {dense['parameters_B']:.2f}B                {p50['parameters_B']:.2f}B
작업 성공            {dense['successes']}/5회                {p50['successes']}/5회
추론 GPU 메모리      {dense['inference_peak_GiB']:.2f}GiB             {p50['inference_peak_GiB']:.2f}GiB
학습 GPU 메모리      {dense['sft_peak_GiB']:.2f}GiB             {p50['sft_peak_GiB']:.2f}GiB
행동 예측 시간       약{dense['prediction_ms']:.0f}ms              약{p50['prediction_ms']:.0f}ms

무엇을 했나?
1. 기존 5epoch LoRA8 모델에서 출발했어.
2. 중요도를 계산할 때는 학습용 160장면만 사용했어.
3. 한쪽은 원본을 유지하고, 한쪽은 중간 LLM 층의 head와 MLP 통로 절반을 제거했어.
4. 양쪽 모두 추가 시범 학습 1epoch 직후의 모델로 비교했어.
   원본은 이미 같은 학습을 마친 checkpoint를 재사용했고, 로봇 시험은 새로 실행했어.
5. 초기 상태 0~4에서 각 5번, 최대 220행동씩 시험했어. 이번에 새 강화학습은 하지 않았어.

데이터는 공개 LIBERO 데이터에서 우리가 선택·전처리한 실험용 분할이야.
학습은 전체 시범 40개 / 4,025장면, 검증은 전체 시범 5개 / 462장면이야.
시범 40개는 사진 40장이 아니라, 처음부터 끝까지의 로봇 시연 40개야.
검증 장면은 가중치를 수정하는 데 쓰지 않았어.

어떻게 해석하나?
50%는 대상 LLM 구조의 제거 비율이야. 시각 모델과 양 끝 층은 남겼어.
그래서 전체 파라미터 감소는 {prep['whole_parameter_reduction']*100:.1f}%야.
50% 모델의 검증 loss는 {d['new_sft']['before_validation']['loss']:.2f} → {p50['validation_loss']:.2f}로 줄었어.
하지만 로봇 성공은 {p50['successes']}/5였어. 오차 감소와 작업 성공은 다른 지표라는 예야.
속도는 약{dense['prediction_ms']:.0f}ms와 약{p50['prediction_ms']:.0f}ms로, 큰 개선은 확인하지 못했어.
이전 원본 2/5·90% 모델 0/5는 강화학습까지 한 별도 결과라 이번 표와 섞지 않았어.
진단 상태 5,8에서는 시범 학습 직후의 90% 모델과 50% 모델 모두 0/2였어.
이전부터 반복 사용한 다섯 초기 상태라, 새로운 환경 일반화 성능으로 확대해석하면 안 돼.

무엇부터 열어보면 되나?
comparison.png: 한눈에 보는 비교 그림.
comparison.csv: 실제 수치표.
01_dense_SFT1_state00.mp4: 상태 0에서 성공한 원본 모델.
02_prune50_SFT1_state00.mp4: 같은 상태에서 실패한 50% 모델.
DETAILS_KO.txt: checkpoint, 측정 범위, W&B 주소 등 자세한 기록.

코드는 이 두 군데부터 보면 돼
source/policy.py의 score_and_plan(), apply_plan(): 남길 구조 선택 → 실제 제거.
source/worker.py의 sft(), simulate(): 시범으로 학습 → 저장 모델로 로봇 시험.
실제로 실행한 코드 한 줄:
  metrics=common.step(policy,optimizer,train,indices)
common.step은 이 프로젝트의 공통 함수야. policy는 모델, train은 학습 데이터,
indices는 이번에 볼 장면 번호야. 이 함수 안에서 오차 계산 → backward → optimizer.step을 실행해.
함수에 재료를 전달하는 것은 Python 문법, backward와 optimizer.step은 PyTorch 기능이야.

다시 실행할 명령 (원격 Linux 터미널)
/home/jovyan/projects/venvs/rlrc-smoke312/bin/python {R}/reproduce.py --start
기존 결과를 덮어쓰지 않고 새 폴더에 최대 1시간 실험을 시작해.
--start 대신 --check를 넣으면 검사만 해. --check 통과를 확인했고, 전체 실험을 두 번 반복한 것은 아니야.

완료 확인과 남은 일
저장·새 프로세스 재로딩·유효한 (1,7) 행동 출력·기존 환경과 파일 보존 검사를 통과했어.
모델 학습·시뮬레이터 실행 오류는 없었어. 실패 이유로 확인된 건 220행동 내 성공 판정 미도달이야.
정확한 집기/놓기 실패 지점과, 어떻게 회복시킬지는 다음 진단으로 남아 있어.
W&B 업로드 검증: {wb.get('verified',False)}.
GPU 메모리는 모델 로딩 뒤의 단계별 최대값이야. 현재 로더는 처음에 원본 7B를 구성한다는 제한이 있어.

기억할 개념 두 개
제거 비율 = 모델의 대상 구조를 얼마나 잘랐는지.
회복 학습 = 자른 뒤 남은 학습값을 조정해 실력을 되찾는 과정.
'''
(R/'START_HERE_KO.txt').write_text(readme,encoding='utf-8')
entries={}
for name in ['START_HERE_KO.txt','DETAILS_KO.txt','RESULTS_KO.txt','comparison.csv','comparison.png','final_results.json',
    'status.json','prepare_complete.json','calibration_selection.json','pruning_plan.json','importance_scores.json',
    'preservation_report.json','wandb_upload_result.json','reproduction_check.json',
    'p50_result.json','dense_sft_result.json','p50_diagnostic_result.json','p90_sft_diagnostic_result.json']:
    if (R/name).exists():entries[name]=R/name
for name in ['policy.py','worker.py','driver.py','check_math.py','report.py','upload_results.py','reproduce.py','experiment_config.json']:
    entries['source/'+name]=R/name
for name in ['sft_result.json','fresh_reload.json','validation_initial.json','validation_after_sft.json']:
    entries['checks/p50_'+name]=R/'p50'/name
entries['01_dense_SFT1_state00.mp4']=R/'simulation/dense_sft/state_00/episode.mp4'
entries['02_prune50_SFT1_state00.mp4']=R/'simulation/p50/state_00/episode.mp4'
entries['50_state00_overview.png']=R/'simulation/p50/state_00/episode_overview.png'
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in entries.items()}
(R/'viewing_bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(R/'viewing_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name,path in entries.items():archive.write(path,name)
    archive.write(R/'viewing_bundle_manifest.json','viewing_bundle_manifest.json')
print('P50_VIEWING_BUNDLE_READY',len(entries),(R/'viewing_bundle.zip').stat().st_size,flush=True)
