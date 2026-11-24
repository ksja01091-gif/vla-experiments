"""Render existing full-validation evidence using the existing matplotlib environment."""
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).parent
assert json.loads((r/"complete.json").read_text())["complete"]
s=json.loads((r/"summary.json").read_text())
examples=json.loads((r/"examples.json").read_text())
rows=[json.loads(x) for x in (r/"frame_predictions.jsonl").read_text().splitlines()]
c=s["gripper_confusion"];m=s["expert_active_but_predicted_near_zero"]
bundle=r/"viewing_bundle";bundle.mkdir(exist_ok=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
x=np.arange(2)
correct=[c["close"]["close"],c["open"]["open"]]
wrong=[c["close"]["open"],c["open"]["close"]]
axes[0].bar(x,correct,color="#4589bd",label="Correct")
axes[0].bar(x,wrong,bottom=correct,color="#d96850",label="Opposite command")
axes[0].set_xticks(x,["Expert: close","Expert: open"])
axes[0].set(title="Gripper decisions on 462 frames",ylabel="Frames")
for i,(a,b) in enumerate(zip(correct,wrong)):
 axes[0].text(i,a/2,str(a),ha="center",va="center",color="white",fontsize=12)
 axes[0].text(i,a+b/2,str(b),ha="center",va="center",color="white",fontsize=11)
axes[0].legend(loc="lower left")
n=m["expert_active_frames"];bad=m["active_but_predicted_near_zero_frames"]
axes[1].bar(["Expert arm-active frames"],[n-bad],color="#b8bfc4",label="Other predictions (not all correct)")
axes[1].bar(["Expert arm-active frames"],[bad],bottom=[n-bad],color="#d96850",label="Near-zero arm prediction")
axes[1].text(0,(n-bad)/2,str(n-bad),ha="center",va="center",fontsize=12)
axes[1].text(0,n-bad+bad/2,f"{bad} / {n}",ha="center",va="center",color="white",fontsize=12)
axes[1].set(title="Near-zero diagnostic, not task success",ylabel="Frames")
axes[1].legend(loc="lower left",fontsize=8)
fig.tight_layout();fig.savefig(bundle/"01_검증462장면_전체 결과.png",dpi=160);plt.close(fig)
categories=[("move_but_near_zero","Expert moves; model near zero"),
            ("should_close_but_open","Expert closes; model opens"),
            ("correct_close","Expert closes; model closes")]
available=[(key,title) for key,title in categories if key in examples]
fig,axes=plt.subplots(len(available),2,figsize=(11,3.7*len(available)),squeeze=False)
for i,(key,title) in enumerate(available):
 row=examples[key]
 axes[i,0].imshow(np.asarray(Image.open(r/(key+".png"))))
 axes[i,0].set_title(f"{title}\nValidation frame {row['validation_index']}, episode {row['episode_index']}")
 axes[i,0].axis("off")
 xx=np.arange(7)
 axes[i,1].bar(xx-.18,row["true_action"],width=.36,label="Demonstration",color="#4589bd")
 axes[i,1].bar(xx+.18,row["predicted_action"],width=.36,label="Model prediction",color="#df8e46")
 axes[i,1].set_xticks(xx,["dx","dy","dz","rx","ry","rz","grip"])
 axes[i,1].set(ylim=(-1.05,1.15),ylabel="Action values (not metres)",
               title=f"Gripper: expert {row['true_gripper_class']} / model {row['predicted_gripper_class']}")
 axes[i,1].axhline(0,color="gray",linewidth=.7)
 axes[i,1].legend(loc="lower left",fontsize=8)
fig.tight_layout();fig.savefig(bundle/"02_실제사진_정답행동과 모델예측 비교.png",dpi=160);plt.close(fig)
flat=[]
for row in rows:
 a={k:row[k] for k in ["validation_index","episode_index","source_step_index","true_gripper_class","predicted_gripper_class","normalized_action_l1"]}
 a["expert_arm_active"]=row["true_arm_max_abs"]>=.1
 a["predicted_arm_near_zero"]=row["predicted_arm_max_abs"]<=.01
 for i,name in enumerate(["dx","dy","dz","rx","ry","rz","gripper"]):
  a["expert_"+name]=row["true_action"][i];a["predicted_"+name]=row["predicted_action"][i]
 flat.append(a)
with (bundle/"03_462장면_정답과예측_행동7개.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for key,title in available:
 names={"move_but_near_zero":"04_이동해야하는데거의정지_시범사진.png",
        "should_close_but_open":"05_닫아야하는데열기_시범사진.png",
        "correct_close":"06_닫기를올바르게예측_시범사진.png"}
 shutil.copyfile(r/(key+".png"),bundle/names[key])
shutil.copyfile(r/"summary.json",bundle/"전체검증_원본집계.json")
ex=examples.get("move_but_near_zero")
grip=examples.get("should_close_but_open")
def action_text(values):return "["+", ".join(f"{x:.4f}" for x in values)+"]"
text=f"""검증용 462장면 전체 행동 예측 검사 — 2026-11-03

무엇을 한 건가?
공개 LIBERO 시범 40개로 학습한 같은 모델에, 학습에 쓰지 않은 시범 5개의 사진 462장을 한 장씩 보여줬다.
각 사진에서 행동 7개를 스스로 생성하게 하고 해당 사진의 시범 정답 행동과 비교했다.
추가 학습이나 새 시뮬레이터 실험은 하지 않았다. 시범 사진에서 답을 얼마나 맞히는지 보는 검사다.

1. 집게 열기·닫기
닫기가 정답인 228장면:
- 올바르게 닫기 {c['close']['close']}장면
- 잘못 열기 {c['close']['open']}장면
열기가 정답인 234장면:
- 올바르게 열기 {c['open']['open']}장면
- 잘못 닫기 {c['open']['close']}장면
전체 열기·닫기 정답: {s['gripper_correct_frames']}/462 = {s['gripper_accuracy']*100:.1f}%.
따라서 모델이 집게 닫기를 전혀 모르는 것은 아니다. 하지만 필요한 순간에 반대로 예측하는 경우가 있다.

2. 움직여야 하는데 거의 정지하는 예측
- 진단 기준상 팔 동작이 있는 시범 장면 {n}개 중, 모델의 팔 행동값이 거의 0인 장면 {bad}개 ({bad/n*100:.1f}%).
- 462장면 전체에서 모델의 팔 행동값이 거의 0인 장면 {s['arm_near_zero_frames']}개.
- 팔 6개 행동값 중 시범의 최대 절댓값 >=0.1을 '활동 명령', 예측의 6개 값 절댓값 모두 <=0.01을 '거의 0'으로 정했다.
- 이 값은 컨트롤러 명령값이다. 미터 단위의 이동 거리나 실제 시뮬레이터 정지를 측정한 결과는 아니다.
- 회전까지 포함한 팔 행동 6개이며 집게 값은 제외한다.
- 기준을 바꾼 결과도 원본 JSON의 motion_threshold_sensitivity에 저장했다.
- '거의 0으로 분류되지 않은 나머지 장면'이 전부 정답이라는 뜻은 아니다.

3. 같은 사진에서 실제 예시
이동 명령이 있는데 거의 정지로 예측한 첫 사례:
- 검증 사진 번호 {ex['validation_index'] if ex else '없음'}
- 시범 정답 행동: {action_text(ex['true_action']) if ex else '-'}
- 모델 예측 행동: {action_text(ex['predicted_action']) if ex else '-'}
숫자 순서는 손의 이동 3개, 회전 3개, 집게 1개다. 위치 좌표나 미터 단위 거리가 아니다.
시범 사진과 숫자를 02 PNG에서 함께 볼 수 있다.

닫아야 하는데 연 첫 사례:
- 검증 사진 번호 {grip['validation_index'] if grip else '없음'}
- 시범의 마지막 집게 값: {grip['true_action'][-1] if grip else '-'}
- 예측한 마지막 집게 값: {grip['predicted_action'][-1] if grip else '-'}
- 여기서 모델 출력의 0은 닫기, 1에 가까운 값은 열기를 뜻한다.
  실제 시뮬레이터로 보낼 때는 별도로 +1=닫기, -1=열기로 변환한다.

4. 학습 중 검증 숫자와 이번 숫자는 어떻게 다른가?
학습 중 기록(같은 체크포인트, 같은 462장면):
- 앞부분 정답 행동 토큰을 참고하며 다음 토큰 예측: 토큰 정확도 {s['reused_teacher_forced_validation']['action_token_accuracy']*100:.2f}%, 정규화 L1 {s['reused_teacher_forced_validation']['normalized_action_l1']:.6f}.
이번 검사:
- 행동 7개를 스스로 순서대로 예측: 토큰 정확도 {s['free_generation_action_token_accuracy']*100:.2f}%, 정규화 L1 {s['free_generation_normalized_action_l1']:.6f}.
- L1은 시범 정답과 예측의 평균 숫자 차이로 낮을수록 좋다.
- 두 수치는 모두 사진 단위 예측 지표이며 로봇 작업 성공률이 아니다.
- 집게 열기·닫기 분류 정확도 87.9%와 전체 행동 토큰 정확도 34.14%도 서로 다른 지표다.
- 토큰이 조금 달라도 비슷한 행동일 수 있으므로 토큰 정확도만으로 작업 성공을 판정하지 않는다.

무엇을 알게 됐나?
- 사진 입력을 학습과 맞추고 같은 모델을 써도, 시범 사진에서 이미 중요한 예측 오류가 나온다.
- 따라서 문제를 실제 시뮬레이터 실행에서만 생기는 것으로 볼 수 없다.
- 동시에 집게 동작을 전혀 배우지 못했다고 말할 수도 없다.
- 왜 실수하는지에 대한 원인은 이번 검사만으로 확정하지 않는다. 학습량·설정·새 상황에 대한 대응 등을 구분해 추가 실험해야 한다.
- 시범 사진에서의 오류가 실제 실패의 어느 정도를 설명하는지도 추가 확인이 필요하다.
- 다음 학습 전에는 학습용 시범에서도 같은 실수가 있는지 확인하면 학습 부족과 일반화 문제를 나누는 데 도움이 된다.

검사 신뢰성
- 462장면을 빠짐없이 한 번씩 검사.
- 462장면 모두 학습과 실제 이미지 입력·작업 지시 토큰 일치 확인.
- 실제 predict_action이 생성한 토큰을 그대로 기록. 기록용 관찰 함수가 생성 결과를 바꾸지 않았음.
- 예측 토큰을 행동으로 복원한 값과 실제 반환 행동이 일치하는지 확인.
- 모델/체크포인트 변경 없음. 기존 프로젝트 코드와 openvla310·lerobot312·eval310 패키지 변경 없음.
- 생성된 행동 토큰은 모두 유효 범위, 출력 숫자 NaN·무한대 없음.
- 처리 시간 약 {s['wall_seconds']:.1f}초(주요 검증), 최대 GPU 메모리 {s['gpu_peak_allocated_gib']:.2f}GiB.
- 이번 검사 실행 오류 없음.

이해하면 좋은 개념 두 가지
1) 사진별 예측 검사: 정해진 시범 사진에서 다음 행동을 맞히는지 확인. 실제 로봇을 움직이는 평가는 아님.
2) 자유 생성: 7개 행동을 모델이 혼자 순서대로 답하는 것. 앞부분의 정답을 참고하는 학습 중 계산과 다름.

원격 파일 위치
검사 코드: {r/'validate_free_actions.py'}
전체 사진별 원본 기록: {r/'frame_predictions.jsonl'}
집계: {r/'summary.json'}
분류 기준·체크포인트: {r/'config.json'}
오답 사례: {r/'examples.json'}
기존 환경·코드 보존 검사: {r/'preservation_report.json'}

결과 다시 확인
cat {r/'summary.json'}

같은 검사를 새 폴더에서 실행
conda activate lerobot312
NEW_RUN="/home/jovyan/projects/lerobot-openvla/outputs/full_validation_actions/$(date -u +%Y%m%dT%H%M%S)_manual"
mkdir "$NEW_RUN"
cp "{r/'validate_free_actions.py'}" "$NEW_RUN/"
python "$NEW_RUN/validate_free_actions.py"

주요 코드 읽는 순서
- main(): 사진을 한 장씩 읽고 실제 모델 예측을 받는다.
- capture_generate(): 생성된 행동 토큰을 관찰·기록한다.
- summarize(): 집게 오류와 거의 0인 팔 예측을 집계한다.
"""
(bundle/"00_먼저 읽기_462장면 검증 결과.txt").write_text(text,encoding="utf-8")
(r/"viewing_hashes.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()},ensure_ascii=False,indent=2))
print("REPORT_READY",bundle)
print("FIRST_MOVE_ERROR",ex["validation_index"] if ex else None,ex["true_action"] if ex else None,ex["predicted_action"] if ex else None)
print("FIRST_GRIPPER_ERROR",grip["validation_index"] if grip else None)
