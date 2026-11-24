"""Build reviewable evidence and paired videos only after core checks completed."""
from pathlib import Path
import csv,hashlib,json,shutil
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RUN=Path(__file__).parent
assert json.loads((RUN/"core_complete.json").read_text())["complete"]
s=json.loads((RUN/"experiment_summary.json").read_text())
cfg=json.loads((RUN/"training_config.json").read_text())
history=json.loads((RUN/"validation_history.json").read_text())
bundle=RUN/"viewing_bundle";bundle.mkdir(exist_ok=False)
train1=s["epoch1_free_train_sample"];train5=s["epoch5_free_train_sample"]
v1=s["epoch1_free_validation"];v5=s["epoch5_free_validation"]
a=s["paired_simulation"]["epoch1"];b=s["paired_simulation"]["epoch5"]
fig,axes=plt.subplots(2,2,figsize=(11,8))
axes[0,0].plot([r["epoch"] for r in history],[r["loss"] for r in history],"o-")
axes[0,0].set(title="Same 462-frame teacher-forced validation",xlabel="Total epochs",ylabel="Loss")
x=np.arange(2)
axes[0,1].bar(x-.18,[train1["normalized_action_l1"],v1["free_generation_normalized_action_l1"]],.36,label="1 epoch")
axes[0,1].bar(x+.18,[train5["normalized_action_l1"],v5["free_generation_normalized_action_l1"]],.36,label="5 epochs")
axes[0,1].set_xticks(x,["Seen sample / 480","Heldout / 462"])
axes[0,1].set(title="Freely generated action error",ylabel="Normalized L1");axes[0,1].legend()
near1=v1["expert_active_but_predicted_near_zero"];near5=v5["expert_active_but_predicted_near_zero"]
axes[1,0].bar(["1 epoch","5 epochs"],[near1["fraction"],near5["fraction"]])
axes[1,0].set(title="Near-zero arm errors / expert-active frames",ylabel="Diagnostic fraction",ylim=(0,1))
axes[1,1].bar(["1 epoch","5 epochs"],[a["success_rate"],b["success_rate"]])
axes[1,1].set(title="Same 5 initial states, native input",ylabel="Simulator goal success rate",ylim=(0,1))
for ax in [axes[1,0],axes[1,1]]:
 for rect in ax.patches:ax.text(rect.get_x()+rect.get_width()/2,rect.get_height()+.02,f"{rect.get_height()*100:.1f}%",ha="center")
fig.tight_layout();fig.savefig(bundle/"01_1회와5회_학습검증과성공률.png",dpi=150);plt.close(fig)
raw=bundle/"원본영상";raw.mkdir()
video_metadata=[];trial_rows=[]
for state in range(5):
 ta=a["trials"][state];tb=b["trials"][state]
 assert ta["init_state_index"]==tb["init_state_index"]==state
 ra=imageio.get_reader(ta["video"]);rb=imageio.get_reader(tb["video"])
 counts=[ra.count_frames(),rb.count_frames()]
 assert counts==[ta["policy_steps"]+1,tb["policy_steps"]+1]
 filename=f"{state+2:02d}_1회vs5회_같은시작상태{state}.mp4"
 with imageio.get_writer(str(bundle/filename),fps=10,codec="libx264",quality=8) as writer:
  for step in range(max(counts)):
   ia=min(step,counts[0]-1);ib=min(step,counts[1]-1)
   panel=Image.new("RGB",(512,288),"white");draw=ImageDraw.Draw(panel)
   panel.paste(Image.fromarray(ra.get_data(ia)),(0,32));panel.paste(Image.fromarray(rb.get_data(ib)),(256,32))
   la=f"1 epoch | step {ia}"+(" | SUCCESS" if ta["success"] and step>=counts[0]-1 else "")
   lb=f"5 epochs | step {ib}"+(" | SUCCESS" if tb["success"] and step>=counts[1]-1 else "")
   draw.text((5,9),la,fill="black");draw.text((261,9),lb,fill="black")
   writer.append_data(np.asarray(panel))
 ra.close();rb.close()
 with imageio.get_reader(str(bundle/filename)) as reader:
  assert reader.count_frames()==max(counts);reader.get_data(0);reader.get_data(max(counts)-1)
 video_metadata.append({"file":filename,"state":state,"frames":max(counts),"fps":10})
 for tag,trial in [("1epoch",ta),("5epochs",tb)]:
  shutil.copyfile(trial["video"],raw/f"{tag}_시작상태{state}.mp4")
  trial_rows.append({"model":tag,"initial_state":state,"success":trial["success"],"policy_steps":trial["policy_steps"],
                    "termination_reason":trial["termination_reason"],"source_video":trial["video"]})
with (bundle/"07_동일조건_시도별성공결과.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(trial_rows[0]));w.writeheader();w.writerows(trial_rows)
with (bundle/"08_epoch별_검증기록.csv").open("w",newline="") as f:
 keys=["epoch","optimizer_step","loss","action_token_accuracy","normalized_action_l1","checkpoint"]
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in history)
for name in ["validation","train_sample"]:
 rows=[json.loads(line) for line in (RUN/f"{name}_free_predictions.jsonl").read_text().splitlines()]
 with (bundle/(name+"_5epochs_사진별예측.csv")).open("w",newline="") as f:
  keys=["validation_index","episode_index","source_step_index","normalized_action_l1","true_gripper_class","predicted_gripper_class","true_arm_max_abs","predicted_arm_max_abs"]
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
motion={}
for tag,reports in s["motion_diagnostics"].items():
 motion[tag]=[{"state":r["init_state_index"],"rise_mm":r["maximum_bowl_height_increase_m"]*1000,
     "bilateral_contact_frames":len(r["bilateral_contact_steps"]),
     "simultaneous_contact_and_rise_frames":len(r["contact_and_rise_over_2cm_steps"]),
     "success":r["success"]} for r in reports]
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False))
write(RUN/"motion_summary.json",motion)
write(RUN/"comparison_video_verification.json",video_metadata)
shutil.copyfile(RUN/"experiment_summary.json",bundle/"전체실험_원본집계.json")
selection=next((i for i,t in enumerate(b["trials"]) if t["success"]),None)
if selection is None:selection=max(range(5),key=lambda i:s["motion_diagnostics"]["epoch5"][i]["maximum_bowl_height_increase_m"])
representative=f"{selection+2:02d}_1회vs5회_같은시작상태{selection}.mp4"
contact_audit=RUN/"simulation/epoch5/state_00/actual_contact_pair_audit.json"
contact_note=""
if contact_audit.exists():
 audit=json.loads(contact_audit.read_text())
 support=[]
 for frame in audit["raised_frames"]:
  names={name for pair in frame["contacts"] for name in pair if name}
  if any("gripper0_hand_collision" in name for name in names) and any("gripper0_finger2" in name for name in names):
   support.append(frame["step"])
 contact_note=("첫 성공(초기 상태 0)의 추가 접촉 검사: 목표 그릇이 올라간 동안 한쪽 손가락(finger2)과 손바닥 형상이 함께 물체를 지지한 단계가 "
               +str(len(support))+"개 기록됐다. 양손가락 동시 접촉이 없어도 이 방식으로 그릇을 운반해 목표를 달성했다. "
               +"작업 성공과 양손가락 집기는 같은 지표가 아니다. 실제 접촉 쌍은 "+str(contact_audit)+"에 저장했다.")
text=f"""총 5 epochs 학습과 동일 조건 비교 — 2026-11-03

먼저 볼 결론
같은 초기 상태 5개에서 1회 모델은 {a['successes']}/5, 5회 모델은 {b['successes']}/5 성공했다.
멈춤에 가까운 팔 예측은 {near1['active_but_predicted_near_zero_frames']}/{near1['expert_active_frames']}에서 {near5['active_but_predicted_near_zero_frames']}/{near5['expert_active_frames']}로 줄었다.
반복 학습 후 실제 목표 달성은 개선됐지만, 모든 초기 상태에서 성공하거나 항상 같은 방식으로 안정적으로 잡는다는 뜻은 아니다.
검증 loss는 {history[0]['loss']:.3f}에서 {history[-1]['loss']:.3f}로 높아졌다. 행동 오차·토큰 확률 손실·실제 작업 성공은 서로 다른 지표이므로 하나로 대체하지 않는다.

무엇을 했나?
- 공개 LIBERO 데이터에서 선택·전처리한 같은 시범 40개, 4,025장면을 사용.
- 기존 1회 학습 가중치·AdamW 모멘텀·난수 상태를 복원해 4회를 추가. 총 5회.
- 총 학습 장면 사용 횟수 20,125, 총 optimizer update 5,035. 새로 본 사진이 20,125장 생긴 것은 아님.
- 검증용 전체 시범 5개에서 나온 462장면은 가중치 수정에 사용하지 않음.
- 로봇 평가의 초기 상태 5개는 검증 시범 5개와 다른 개념. 시범을 그대로 재생한 평가가 아님.

동일하게 유지한 것
batch 4 / learning rate 0.0001 / AdamW / weight decay 0 / gradient clip 1 /
LoRA rank 8, alpha 8, dropout 0, 언어모델 q_proj·v_proj /
기본 가중치 BF16, LoRA FP32 / 원본 256 RGB를 native HF 처리기에 입력 /
기존 학습용 20개에서 계산해 고정한 행동 정규화 / task / seed / initial states 0~4 /
warmup 10 / 최대 정책 단계 220 / 집게 변환 -sign(2*g-1).

핵심 결과
[1 epoch -> 5 epochs]
같은 462장면 teacher-forced validation loss:
{history[0]['loss']:.6f} -> {history[-1]['loss']:.6f}
행동 7개를 스스로 생성한 검증 오차(normalized L1):
{v1['free_generation_normalized_action_l1']:.6f} -> {v5['free_generation_normalized_action_l1']:.6f}
자유 생성 행동 토큰 정확도:
{100*v1['free_generation_action_token_accuracy']:.2f}% -> {100*v5['free_generation_action_token_accuracy']:.2f}%
검증 사진 집게 열기·닫기 정확도:
{100*v1['gripper_accuracy']:.2f}% -> {100*v5['gripper_accuracy']:.2f}%
팔 활동 명령이 있는데 거의 0으로 예측한 장면:
{near1['active_but_predicted_near_zero_frames']}/{near1['expert_active_frames']} -> {near5['active_but_predicted_near_zero_frames']}/{near5['expert_active_frames']}
학습 사진 동일 표본 480장의 자유 생성 오차:
{train1['normalized_action_l1']:.6f} -> {train5['normalized_action_l1']:.6f}
학습 사진 480장의 집게 정확도:
{100*train1['gripper_accuracy']:.2f}% -> {100*train5['gripper_accuracy']:.2f}%

로봇 실제 실행 (두 모델 모두 native 학습 일치 사진 처리로 새로 평가)
1회 모델: {a['successes']}/5, 성공률 {100*a['success_rate']:.1f}%, 평균 {a['mean_policy_steps']:.1f}단계
5회 모델: {b['successes']}/5, 성공률 {100*b['success_rate']:.1f}%, 평균 {b['mean_policy_steps']:.1f}단계
기술 오류: 두 모델 모두 0.
각 상태의 처음 이미지·초기 상태 해시가 완전히 같은지 확인.
예전 중앙 자르기 처리 영상의 성공률과 이번 native 입력 성공률은 같은 비교에 섞지 않음.
공개 LIBERO Spatial 추가 학습 모델 4/5 결과도 이번 두 모델 결과와 섞지 않음.

물체 움직임·접촉 진단
{json.dumps(motion,ensure_ascii=False,indent=2)}
rise_mm는 목표 그릇의 높이 증가. 그릇이 튀거나 기울어도 증가할 수 있으므로 이 값만으로 잡기 성공이라 하지 않음.
bilateral_contact_frames는 양쪽 손가락 패드의 동시 접촉을 제어 단계 끝에서 관찰한 횟수.
simultaneous_contact_and_rise_frames는 동시 접촉과 2cm 초과 높이 증가가 함께 나온 단계 수.
이 진단은 지속적 안정 파지의 완전한 증명이나 공식 작업 성공률을 대신하지 않음.
공식 성공 판정은 LIBERO goal predicate이며, 성공 방식은 영상으로 함께 확인해야 함.\n{contact_note}

이번에 비교한 것은 무엇인가?
데이터 개수나 LoRA 범위를 바꾸지 않고 같은 학습을 네 번 더 이어갔다.
가중치가 바뀌지 않은 1회 모델과, 총 5회 학습한 최종 모델을 같은 입력·초기 상태에서 비교했다.
추가 학습 중 검증을 매 epoch 수행했으나 주 비교에는 최저 loss 모델을 몰래 고르지 않고 epoch5를 사용했다.
최저 teacher validation loss 체크포인트:
epoch {s['training']['best_teacher_validation_epoch']}: {s['training']['best_teacher_validation_checkpoint']}
최종 epoch5 체크포인트:
{s['training']['final_checkpoint']}

해석할 때 구분할 것
- loss/사진별 행동 오차 감소는 실제 작업 성공률 증가와 다르다.
- 같은 시범 40개를 반복한 결과이지 데이터가 더 다양해진 실험이 아니다.
- 초기 상태 5개, 특정 작업 1개 결과이므로 LIBERO 전체 성능이나 보지 않은 상태 일반화를 보장하지 않는다.
- 학습 시범과 시뮬레이터 초기 상태의 완전한 분리는 확인되지 않았다.
- 각 방향의 행동 bin 정확도와 집게 열기·닫기 정확도는 서로 다른 지표다.
- 이번 결과만으로 모든 실패 원인을 학습 횟수 하나에 귀속할 수는 없다.

영상 보기
- 왼쪽 1회 모델 / 오른쪽 5회 모델. 같은 초기 상태와 같은 정책 단계를 나란히 비교.
- 비교 영상은 초당 10장으로 재생. 실제 계산 시간 영상은 아님.
- 한 모델이 먼저 성공하면 그쪽 마지막 화면을 유지하고 다른 모델은 계속 재생.
- 대표 비교 영상: {representative}
  이 영상은 결과를 본 뒤 대표 사례로 선택한 것이며 무작위 표본은 아님.
- 원본영상 폴더에는 각각의 20fps 원본 10개가 있음.

저장·검증
- epoch2·3·4·5 체크포인트와 optimizer 상태를 별도 폴더에 저장.
- 5회 모델을 새 Python 프로세스에서 다시 로드, 동일 validation batch loss, (1,7) 행동, NaN/무한대 없음 확인.
- 추가 4회 각각 4,025장면이 빠짐없이 정확히 한 번씩 사용됐는지 확인.
- 기존 프로젝트 코드와 openvla310·lerobot312·eval310 패키지 목록 변경 없음.
- 원본 1회 체크포인트·데이터 보호 해시 변경 없음.
- 원본 영상 10개와 비교 영상 5개 프레임 수·첫/마지막 디코드 확인.
- 추가 학습 시간 {s['training']['additional_training_wall_seconds']:.1f}초.
- 학습 과정 GPU 최대 할당량 {s['training']['gpu_peak_allocated_gib']:.2f}GiB (프로세스 기준).
  학습 중 1회 기준 평가를 병행했으므로 벽시계 학습 시간은 이전 단독 실행 속도와 직접 비교하면 안 됨.
- W&B는 오프라인 기록. 원격 CSV·PNG·체크포인트·영상이 먼저 저장됨.

두 개념
1) epoch: 같은 학습 데이터 전체를 한 번씩 사용하는 단위.
2) 공정한 대조: epoch 외 조건을 맞춰, 반복 학습 증가와 결과 변화의 관련성을 확인.

원격 핵심 파일
학습·검증·평가 코드: {RUN/'five_epoch_experiment.py'}
단계 실행 드라이버: {RUN/'run_pipeline.py'}
시뮬레이션 물체 진단: {RUN/'motion_worker.py'}
학습 설정: {RUN/'training_config.json'}
재시작 상태 확인: {RUN/'resume_check.json'}
매 회차 coverage: {RUN/'training_coverage.json'}
teacher validation history: {RUN/'validation_history.json'}
사진별 자유 예측: {RUN/'validation_free_predictions.jsonl'}
전체 결과: {RUN/'experiment_summary.json'}
기존 환경·코드 보존: {RUN/'preservation_report.json'}

결과 다시 확인
cat {RUN/'experiment_summary.json'}

최종 체크포인트 새 프로세스 재확인
conda activate lerobot312
python {RUN/'five_epoch_experiment.py'} check

새 폴더에서 전체 실험 재현 (기존 결과 보존)
conda activate lerobot312
NEW_RUN="/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/$(date -u +%Y%m%dT%H%M%S)_manual"
mkdir "$NEW_RUN"
cp "{RUN/'five_epoch_experiment.py'}" "{RUN/'run_pipeline.py'}" "{RUN/'motion_worker.py'}" "{RUN/'build_report.py'}" "$NEW_RUN/"
python "$NEW_RUN/run_pipeline.py"
/home/jovyan/anaconda3/envs/openvla310/bin/python "$NEW_RUN/build_report.py"
"""
(bundle/"00_먼저 읽기_총5회 학습 결과.txt").write_text(text,encoding="utf-8")
status=json.loads((RUN/"wandb_status.json").read_text())
if status.get("directory"):
 offline=Path(status["directory"]).parent
 (bundle/"WandB_수동연결명령.txt").write_text("conda activate lerobot312\nwandb login\nwandb sync "+str(offline)+"\n",encoding="utf-8")
hashes={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob("*") if p.is_file()}
write(RUN/"viewing_hashes.json",hashes)
write(RUN/"delivery_metadata.json",{"representative_state":selection,"representative_video":representative,
 "epoch1_successes":a["successes"],"epoch5_successes":b["successes"],
 "epoch1_free_validation_l1":v1["free_generation_normalized_action_l1"],
 "epoch5_free_validation_l1":v5["free_generation_normalized_action_l1"],"files":len(hashes)})
write(RUN/"complete.json",{"complete":True,"core_evidence":str(RUN/"core_complete.json"),
      "viewing_bundle":str(bundle),"original_videos_verified":10,"comparison_videos_verified":5})
print("REPORT_AND_VIDEOS_COMPLETE",bundle,flush=True)
