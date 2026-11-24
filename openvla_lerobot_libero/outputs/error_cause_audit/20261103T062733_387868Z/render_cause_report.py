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
seen=s["training_sample"];held=s["validation_matched_phase_sample"];allval=s["validation_all"]
frequency=json.loads((r/"same_split_frequency_check.json").read_text())
examples=json.loads((r/"seen_error_examples.json").read_text())
base=s["baseline_diagnostics"]
bundle=r/"viewing_bundle";bundle.mkdir(exist_ok=False)
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
labels=["Seen / 480","Heldout / 60","Heldout / 462"]
sets=[seen,held,allval]
colors=["#4589bd","#df8e46","#84989b"]
for ax,key,title in zip(axes,["normalized_action_l1","gripper_accuracy","active_predicted_near_zero_fraction"],
 ["Action error (lower is better)","Gripper accuracy (not grasp success)","Near-zero arm errors / expert active"]):
 values=[m[key] for m in sets]
 ax.bar(labels,values,color=colors);ax.set_title(title,fontsize=10);ax.tick_params(axis="x",rotation=15)
 for i,value in enumerate(values):ax.text(i,value+.005,f"{value:.3f}",ha="center",fontsize=10)
 ax.set_ylim(0,max(values)*1.2)
fig.tight_layout();fig.savefig(bundle/"01_학습사진과 검증사진_오류 비교.png",dpi=150);plt.close(fig)
row=examples.get("seen_but_near_zero")
if row:
 fig,axes=plt.subplots(1,2,figsize=(10,3.7))
 axes[0].imshow(np.asarray(Image.open(r/"seen_but_near_zero.png")))
 axes[0].set_title("This exact photo was used in training");axes[0].axis("off")
 x=np.arange(3);a=np.asarray(row["true_action"][:3]);b=np.asarray(row["predicted_action"][:3])
 axes[1].bar(x-.18,a,.36,label="Demonstration",color="#4589bd")
 axes[1].bar(x+.18,b,.36,label="Model",color="#df8e46")
 for xx,y in zip(x-.18,a):axes[1].text(xx,y+.012,f"{y:.3f}",ha="center")
 for xx,y in zip(x+.18,b):axes[1].text(xx,y-.012,f"{y:.3f}",ha="center",va="top")
 axes[1].set_xticks(x,["Move X","Move Y","Move Z"])
 axes[1].set(ylim=(-.06,.55),ylabel="Action command (not metres)",title="Seen photo, still near-zero movement")
 axes[1].axhline(0,color="gray",linewidth=.7);axes[1].legend()
 fig.tight_layout();fig.savefig(bundle/"02_학습했던사진에서도_이동을못맞힌_실제사례.png",dpi=140);plt.close(fig)
indices=[0,1,3,4,5]
fig,ax=plt.subplots(figsize=(9,4))
x=np.arange(len(indices));a=np.array(frequency["demonstration_frequency"])[indices]*100;b=np.array(frequency["model_frequency"])[indices]*100
ax.bar(x-.18,a,.36,label="Demonstration frequency",color="#4589bd")
ax.bar(x+.18,b,.36,label="Model prediction frequency",color="#df8e46")
ax.set_xticks(x,["Move X","Move Y","Rotate X","Rotate Y","Rotate Z"])
ax.set(ylabel="Percent of the same 462 frames",ylim=(0,110),title="Overusing zero-like action bins (per direction)")
for xx,y in zip(x-.18,a):ax.text(xx,y+2,f"{y:.1f}",ha="center",fontsize=9)
for xx,y in zip(x+.18,b):ax.text(xx,y+2,f"{y:.1f}",ha="center",fontsize=9)
ax.legend(loc="upper left",fontsize=8);fig.tight_layout();fig.savefig(bundle/"03_특정값을 지나치게 반복하는 패턴.png",dpi=140);plt.close(fig)
flat=[]
for line in (r/"train_sample_predictions.jsonl").read_text().splitlines():
 x=json.loads(line)
 item={k:x[k] for k in ["training_index","episode_index","source_step_index","normalized_action_l1","true_gripper_class","predicted_gripper_class"]}
 item["near_zero_error"]=x["true_arm_max_abs"]>=.1 and x["predicted_arm_max_abs"]<=.01
 for i,name in enumerate(["dx","dy","dz","rx","ry","rz","gripper"]):
  item["expert_"+name]=x["true_action"][i];item["predicted_"+name]=x["predicted_action"][i]
 flat.append(item)
with (bundle/"04_학습사진480장_정답과예측.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
shutil.copyfile(r/"summary.json",bundle/"원인점검_원본집계.json")
if row:shutil.copyfile(r/"seen_but_near_zero.png",bundle/"05_학습때사용한_실제오답사진.png")
text=f"""모델이 왜 오답을 내는지 원인 점검 — 2026-11-03

질문
검증용 사진에서만 못하는가, 이미 학습했던 사진에서도 답을 잘 못하는가?

이번에 한 것
- 학습용 전체 시범 40개에서 각 시범의 처음~끝을 고르게 12장씩 선택: 480장.
- 학습했던 4,025장면 전체를 다시 평가한 것은 아니다. 선택 기준은 오류를 보기 전에 고정했다.
- 검증용 시범 5개에서도 같은 간격으로 12장씩 60장을 골라, 기존 462장 전체 결과 중 대응하는 기록을 사용.
- 같은 체크포인트, 같은 사진·지시 처리, 같은 행동 복원 방식. 재학습·추가 로봇 실행 없음.
- 실제 predict_action이 스스로 생성한 행동 토큰을 관찰했으며 생성 결과는 변경하지 않았다.

1. 이미 본 사진에서도 오류가 남아 있었다
[학습사진 표본 480장 / 40개 시범]
- 정규화 행동 평균 오차: {seen['normalized_action_l1']:.4f} (낮을수록 좋음).
- 집게 열기·닫기 정확도: {seen['gripper_accuracy']*100:.2f}%.
- 시범에 팔 활동 명령이 있는 {seen['expert_active_frames']}장면 중 거의 0인 팔 행동 예측 {seen['active_predicted_near_zero']}장면 ({100*seen['active_predicted_near_zero_fraction']:.1f}%).
[검증사진 동일 간격 표본 60장 / 5개 시범]
- 정규화 행동 평균 오차: {held['normalized_action_l1']:.4f}.
- 집게 정확도: {held['gripper_accuracy']*100:.2f}%.
- 팔 활동 장면 {held['expert_active_frames']}개 중 거의 0 예측 {held['active_predicted_near_zero']}개 ({100*held['active_predicted_near_zero_fraction']:.1f}%).
[검증사진 전체 462장 / 5개 시범]
- 정규화 행동 평균 오차: {allval['normalized_action_l1']:.4f}.
- 팔 활동 장면 {allval['expert_active_frames']}개 중 거의 0 예측 {allval['active_predicted_near_zero']}개 ({100*allval['active_predicted_near_zero_fraction']:.1f}%).
- 집게 정확도: {allval['gripper_accuracy']*100:.2f}%.

해석:
처음 보는 사진에만 약한 것이 아니다. 이미 배운 사진에서도 상황별 팔 행동을 충분히 맞히지 못한다.
동시에 학습 표본의 오차가 더 낮고 집게는 더 잘 맞히므로, 새 사진에 대한 일반화 차이도 전혀 없다고 할 수 없다.
480장과 60장은 다른 사진이며 시범 수도 다르다. 이 차이만으로 원인 하나를 확정할 수는 없다.
'거의 0'은 팔 6개 행동값 절댓값이 모두 0.01 이하인 경우이며, 시범 활동 기준은 최대 절댓값 0.1 이상이다.
이는 실제 이동 거리나 시뮬레이터 성공률이 아닌 명령값 진단 기준이다.

2. 사진을 완전히 무시하는 것은 아니다
- 올바른 사진-정답 짝에서 모델 평균 오차: {base['actual_model_validation_normalized_l1']:.4f}.
- 같은 모델 예측에 정답을 무작위로 섞어 붙였을 때: 평균 {base['shuffle_pairing']['shuffled_l1_mean']:.4f} (500회, seed 7).
- 학습 데이터에서 각 방향의 가장 흔한 값만 항상 답하는 단순 기준: {base['fixed_mode_validation_normalized_l1']:.4f}.
모델이 이 두 기준보다 낫다. 사진과 정답의 대응에서 유용한 정보를 어느 정도 배웠다는 근거다.
이는 독립 표본의 통계 검정이나 인과 증명이 아니라, 사진-행동 대응을 이용하는지 보는 진단이다.

3. 하지만 몇몇 행동값에 지나치게 쏠린다
같은 검증 사진 462장 안에서, 학습 데이터의 최빈 행동 bin과 같은 값을 사용하는 비율:
- Y 이동: 시범 {frequency['demonstration_frequency'][1]*100:.1f}% / 모델 {frequency['model_frequency'][1]*100:.1f}%.
- X 회전: 시범 {frequency['demonstration_frequency'][3]*100:.1f}% / 모델 {frequency['model_frequency'][3]*100:.1f}%.
- Y 회전: 시범 {frequency['demonstration_frequency'][4]*100:.1f}% / 모델 {frequency['model_frequency'][4]*100:.1f}%.
선택한 방향의 bin은 거의 0인 동작에 해당한다. bin 하나는 일정 폭의 행동값 구간을 뜻한다.
이는 상황에 맞게 다양한 값을 내기보다 몇 가지 답을 반복하는 패턴이다.
원본 시범이 모두 정지했다는 뜻은 아니다. 여러 방향 중 특정 방향만 0일 수도 있다.

4. 현재 가장 유력한 설명과 아직 추정인 부분
확인된 사실:
- 입력/프롬프트/토큰-행동 복원에서 앞선 검사에 불일치가 없었고 같은 체크포인트가 유지됐다.
- 이미 배운 사진에서도 팔 명령 오답이 남는다.
- 사진은 어느 정도 활용하지만, 행동값의 다양성이 부족하고 특정 값에 과도하게 치우친다.
유력한 설명:
- 현재 최소 추가 학습으로는 상황별 팔 동작을 충분히 구분하도록 적응하지 못한 상태일 가능성.
확인한 설정:
- 전체 시범을 1회 학습, LoRA rank 8, language_model의 q_proj/v_proj만 업데이트.
- 로컬 공식 vla-scripts/finetune.py 기본 예시는 rank 32, target_modules=all-linear로 더 넓은 부분을 업데이트한다.
주의:
공식 코드의 기본값이 이 그릇 작업의 유일한 정답 설정이라는 뜻은 아니다.
공식 기본 max_steps를 이번 작은 데이터셋의 필수 학습량으로 해석해서도 안 된다.
학습 횟수 부족인지, 변경 가능한 모델 범위 부족인지, 사진만으로 구분하기 어려운 상황인지 아직 분리되지 않았다.

다음 원인 확인에 적합한 대조 실험 (이번에는 실행하지 않음)
- 데이터·사진 처리·LoRA 범위를 고정하고 학습 횟수만 1회에서 2회로 늘려 같은 지표를 비교.
- 학습/검증 오류가 함께 줄면 현재 학습량이 부족했다는 설명을 지지.
- 학습 오류만 줄고 검증이 나빠지면 일반화/과적합 문제도 확인.
- 둘 다 개선되지 않으면 LoRA 범위·학습 설정·부분 관측 등 다른 후보를 추가 점검.
- 새 체크포인트와 기존 결과는 분리 보관해야 한다. 작업 성공률은 이후 로봇 실행으로 별도 확인해야 한다.

이해하면 좋은 두 개념
1) 학습 적합 부족: 이미 본 예제에서도 원하는 행동을 충분히 재현하지 못함.
2) 일반화: 배울 때 보지 않은 예제에도 배운 내용을 적용함.
이번 검사에서는 첫 번째 문제의 징후가 확인됐고, 두 번째도 완전히 배제하지 않는다.

실제 코드와 결과 위치
검사 코드: {r/'investigate_errors.py'}
학습 사진 선택 목록: {r/'sampling_plan.json'}
학습 사진별 예측: {r/'train_sample_predictions.jsonl'}
학습·검증 비교: {r/'summary.json'}
빈도·사진-정답 섞기 진단: {r/'mode_and_image_pairing_diagnostics.json'}
같은 검증 분할 내 빈도 비교: {r/'same_split_frequency_check.json'}
학습 사진의 실제 오답: {r/'seen_error_examples.json'}
보존 검사: {r/'preservation_report.json'}

결과 다시 확인
cat {r/'summary.json'}

새 폴더에서 동일 진단 재실행
conda activate lerobot312
NEW_RUN="/home/jovyan/projects/lerobot-openvla/outputs/error_cause_audit/$(date -u +%Y%m%dT%H%M%S)_manual"
mkdir "$NEW_RUN"
cp "{r/'investigate_errors.py'}" "$NEW_RUN/"
python "$NEW_RUN/investigate_errors.py"

이번 검사 실행 오류 없음. 기존 소스 292개와 세 환경 패키지 변경 없음. 원본 가중치·기존 데이터 변경 없음.
"""
(bundle/"00_먼저 읽기_오답 원인 점검.txt").write_text(text,encoding="utf-8")
(r/"viewing_hashes.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()},ensure_ascii=False,indent=2))
print("REPORT_READY",bundle)
