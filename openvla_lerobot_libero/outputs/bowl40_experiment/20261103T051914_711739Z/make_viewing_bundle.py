"""Create a readable evidence-based result and a viewing bundle after completion."""
from pathlib import Path
import csv,hashlib,json,shutil
r=Path(__file__).parent
assert json.loads((r/"pipeline_complete.json").read_text())["complete"]
s=json.loads((r/"comparison_summary.json").read_text())
old_motion=json.loads((r/"diagnostics_train20/failure_motion_analysis.json").read_text())
contact=json.loads((r/"simulation/lora/state_02/grasp_contact_diagnostics.json").read_text())
motion=s["train40_bowl_motion"]
old_above=sum(x["maximum_bowl_height_increase_m"]>0.02 for x in old_motion)
new_above=sum(x["maximum_bowl_height_increase_m"]>0.02 for x in motion)
rows=[]
for label,items in [("train20",old_motion),("train40",motion)]:
 for x in items:
  rows.append({"model":label,"state_index":x["init_state_index"],"maximum_bowl_rise_mm":1000*x["maximum_bowl_height_increase_m"],
               "rise_over_2cm":x["maximum_bowl_height_increase_m"]>0.02,"task_success":x["success"],
               "replay_eef_max_error_m":x["maximum_eef_replay_error_m"]})
with (r/"bowl_height_comparison.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plan=s["plan"]
config=json.loads((r/"training_config.json").read_text())
wandb=json.loads((r/"wandb_status.json").read_text())
text=f"""40개 시범 추가 학습 실험 결과 — 2026-11-03

무엇을 바꿨나?
- 공개 LIBERO 데이터에서 선택·전처리한 학습용 전체 시범을 20개에서 40개로 늘렸다.
- 검증용 전체 시범 5개(462장면)는 그대로 두고 학습에 사용하지 않았다.
- 학습용은 4,025장면이며, 전체를 한 번씩 사용했다. 시범 40개는 사진 40장이 아니다.
- 이전 학습 모델을 이어 학습하지 않고 같은 기본 openvla/openvla-7b에서 새로 시작했다.
- 기존 파일·환경·실험 결과는 보존했다.

실제 결과
- 검증 loss: 이전 20개 모델 {s['prior_train20_validation']['loss']:.4f} -> 새 40개 모델 {s['train40_validation']['loss']:.4f}
- 행동 토큰 정확도: {100*s['prior_train20_validation']['action_token_accuracy']:.2f}% -> {100*s['train40_validation']['action_token_accuracy']:.2f}%
- 정규화 행동 L1 오차: {s['prior_train20_validation']['normalized_action_l1']:.6f} -> {s['train40_validation']['normalized_action_l1']:.6f}
- 같은 초기 상태 5개에서 최종 그릇 옮기기: 기본 모델 0/5, 이전 20개 모델 0/5, 새 40개 모델 {s['simulation']['train40']['successes']}/5
- 평균 실행 단계: 새 모델 {s['simulation']['train40']['mean_steps']:.1f}단계.
- 목표 그릇의 높이 증가가 2cm를 넘은 시도: 이전 {old_above}/5, 새 모델 {new_above}/5.
  이 값은 '그릇의 위치가 올라간 관찰'이며 '양손가락으로 안정적으로 잡음'의 성공률이 아니다.
- 세 번째 시도(내부 상태 번호 2): 최대 높이 증가 {contact['maximum_bowl_height_increase_m']*100:.2f}cm.
  영상에서는 그릇이 올라갔다 다시 내려왔다. 기록된 제어 단계의 양손가락 패드 동시 접촉 판정은 {len(contact['bilateral_contact_steps'])}회.
  따라서 안정적으로 잡고 들어 올렸다고 확정하지 않는다. 놓기까지의 최종 작업에도 실패했다.
- 시뮬레이션 실행 오류 0개. 원래 행동 재현 시 손 위치와 성공 판정의 일치를 확인했다.
- 별도 공개 LIBERO-Spatial 추가 학습 모델의 기존 참고 결과 4/5는 위 기본 모델 결과와 구분한다.

영상 보는 순서
1. 01_같은 시작상태_20개 학습 vs 40개 학습_느린 비교.mp4
   왼쪽은 이전 20개 모델, 오른쪽은 이번 40개 모델. 같은 초기 상태, 같은 단계끼리 비교.
   저장된 제어 장면을 초당 10장으로 재생하며 실제 계산 시간 영상이 아니다.
2. 02_40개 학습_그릇이 올라갔다 내려오는 순간_느리게.mp4
   세 번째 시도의 25~60단계, 초당 5장으로 재생.
3. 전체 시도 영상 1~5: 새 모델의 다섯 초기 상태 결과. 초당 20장.

학습 조건
- 1 epoch / batch 4 / learning rate 0.0001 / AdamW / seed 7
- optimizer update 1,007회 (이전은 482회).
- LoRA rank 8 / alpha 8 / dropout 0 / 언어 모델 q_proj, v_proj만 변경.
- 기본 가중치는 BF16, LoRA 가중치는 FP32.
- 행동 정규화는 이전 학습용 20개에서 구한 값을 고정해 비교 기준을 유지했다.
- 학습 루프 시간 {s['training_seconds']:.1f}초, 최대 GPU 메모리 {s['gpu_peak_gib']:.2f}GiB.
- 검증 loss가 가장 낮은 별도 체크포인트를 선택했다.
- 새 Python 프로세스에서 정확한 LoRA 재로딩, 행동 (1,7) 형태, NaN/무한대 없음 확인.

무엇까지 말할 수 있나?
- 데이터를 늘렸을 때 예측 loss는 조금 낮아졌고, 한 시도에서 목표 그릇이 크게 움직이는 변화가 있었다.
- 최종 작업 성공률은 개선되지 않았으며 안정적 파지 성공도 확인하지 못했다.
- 1 epoch를 유지했기 때문에 데이터와 함께 총 가중치 수정 횟수도 늘었다.
  동일 계산량에서 데이터 개수만 바꾼 실험은 아니다.
- 초기 상태 5개만 평가했으므로 전체 LIBERO 성능으로 일반화할 수 없다.
- 학습 시범과 평가 초기 상태의 완전한 분리는 확인되지 않았다.
- 양손가락 접촉은 제어 단계 끝에서 검사했다. 모든 물리 내부 순간을 연속 관찰한 판정은 아니다.
- loss는 시범 장면의 다음 행동 예측 오차이며 로봇 작업 성공률과 다르다.

원격 Linux 경로
실험 폴더:
{r}
데이터 분할:
{r/'data/split_manifest.json'}
학습 설정:
{r/'training_config.json'}
선택 체크포인트:
{s['selected_checkpoint']}
실행 코드:
 /home/jovyan/projects/lerobot-openvla/examples/run_openvla_bowl40.py
학습·평가 코드:
 /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl40_experiment.py
LeRobot 정책 클래스:
 /home/jovyan/projects/lerobot-openvla/src/lerobot/policies/openvla/full_experiment_openvla.py
그릇 위치·접촉 진단 코드:
{r/'analyze_motion_no_camera.py'}
{r/'check_grasp_contacts.py'}

저장 모델 재확인 명령 (원격 터미널)
conda activate lerobot312
export OPENVLA_BOWL40_DATA_ROOT={r/'data'}
python /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl40_experiment.py check --run {r}

새 실험 재실행 방법 (기존 결과를 덮어쓰지 않음)
python /home/jovyan/projects/lerobot-openvla/examples/run_openvla_bowl40.py prepare
위 명령이 출력한 새 PREPARED_RUN 경로를 RUN으로 사용:
python /home/jovyan/projects/lerobot-openvla/examples/run_openvla_bowl40.py calibrate --run RUN
python /home/jovyan/projects/lerobot-openvla/examples/run_openvla_bowl40.py finish --run RUN

W&B
{wandb['mode']}로 기록했으며 웹 업로드는 하지 않았다. CSV·PNG·체크포인트·영상은 원격에 보관했다.
"""
(r/"RESULTS_KO.txt").write_text(text,encoding="utf-8")
bundle=r/"viewing_bundle"
bundle.mkdir(exist_ok=False)
mapping={
 "train20_vs_train40_state2_slow.mp4":"01_같은 시작상태_20개 학습 vs 40개 학습_느린 비교.mp4",
 "train40_state2_lift_attempt_slow.mp4":"02_40개 학습_그릇이 올라갔다 내려오는 순간_느리게.mp4",
 "comparison.png":"03_검증 loss와 성공률 비교.png",
 "RESULTS_KO.txt":"00_먼저 읽기_실험 결과와 영상 설명.txt",
 "simulation_comparison.csv":"실험별_작업성공_비교.csv",
 "bowl_height_comparison.csv":"시도별_그릇높이_비교.csv",
 "validation_metrics.csv":"40개학습_검증결과.csv",
}
for src,dst in mapping.items():shutil.copyfile(r/src,bundle/dst)
for state in range(5):
 shutil.copyfile(r/f"simulation/lora/state_{state:02d}/episode.mp4",bundle/f"전체시도_{state+1:02d}_40개학습_시작상태{state}.mp4")
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False))
write(r/"viewing_bundle_hashes.json",manifest)
write(r/"lift_comparison_summary.json",{"old20_rise_over_2cm":old_above,"new40_rise_over_2cm":new_above,
      "new40_bilateral_contact_state2":contact["bilateral_contact_steps"],
      "not_a_stable_grasp_success_metric":True})
print("VIEWING_BUNDLE_READY",bundle)
print(json.dumps({"training_seconds":s["training_seconds"],"old_val_loss":s["prior_train20_validation"]["loss"],
                 "new_val_loss":s["train40_validation"]["loss"],"old20_rise_over_2cm":old_above,
                 "new40_rise_over_2cm":new_above,"new40_bowl_rise_mm":[round(x["maximum_bowl_height_increase_m"]*1000,3) for x in motion]}))
