"""Render the completed adaptive epoch search, including explicit unevaluated entries."""
from pathlib import Path
import csv,hashlib,json,shutil
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
RUN=Path(__file__).parent
OLD=Path("/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z")
assert json.loads((RUN/"core_complete.json").read_text())["complete"]
s=json.loads((RUN/"summary.json").read_text())
cohorts={int(k):v for k,v in s["cohorts"].items()}
minimum=s["minimum_epoch_matching_5epoch_performance"]
first_any=s["minimum_epoch_at_least_one_success"]
bundle=RUN/"viewing_bundle";bundle.mkdir(exist_ok=False)
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
epochs=np.arange(1,6)
values=[cohorts[e]["success_rate"] if e in cohorts else np.nan for e in epochs]
colors=["#4a9c78" if e==minimum else "#548cb5" for e in epochs]
axes[0].bar(epochs,values,color=colors)
axes[0].axhline(.6,linestyle="--",color="#c46950",label="Target: at least 3/5")
for e,v in zip(epochs,values):
 if np.isfinite(v):axes[0].text(e,v+.025,f"{cohorts[e]['successes']}/5",ha="center")
 else:axes[0].text(e,.02,"Not\nevaluated",ha="center",color="gray",fontsize=8)
axes[0].set(xlabel="Total training epochs",ylabel="Success on the same five states",ylim=(0,1.12),xticks=epochs,
 title=f"Minimum meeting target: epoch {minimum}")
axes[0].legend(loc="upper left",fontsize=8)
matrix=np.full((5,5),np.nan)
for e,c in cohorts.items():
 for trial in c["trials"]:matrix[trial["init_state_index"],e-1]=int(trial["success"])
cmap=ListedColormap(["#dfb2a3","#84b49b"]);cmap.set_bad("#e5e5e5")
axes[1].imshow(matrix,cmap=cmap,vmin=0,vmax=1,aspect="auto")
axes[1].set_xticks(np.arange(5),[f"Epoch {e}" for e in epochs])
axes[1].set_yticks(np.arange(5),[f"State {i}" for i in range(5)])
axes[1].set_title("1 = goal success, 0 = failure, NA = not evaluated")
for i in range(5):
 for j in range(5):axes[1].text(j,i,str(int(matrix[i,j])) if np.isfinite(matrix[i,j]) else "NA",ha="center",va="center")
fig.tight_layout();fig.savefig(bundle/"01_epoch별_성공횟수와 같은상태 비교.png",dpi=150);plt.close(fig)
def make_pair(left,right,state,destination,left_label,right_label):
 ta=cohorts[left]["trials"][state];tb=cohorts[right]["trials"][state]
 assert ta["init_state_index"]==tb["init_state_index"]==state
 ra=imageio.get_reader(ta["video"]);rb=imageio.get_reader(tb["video"])
 ca=ra.count_frames();cb=rb.count_frames()
 assert ca==ta["policy_steps"]+1 and cb==tb["policy_steps"]+1
 with imageio.get_writer(str(destination),fps=10,codec="libx264",quality=8) as writer:
  for step in range(max(ca,cb)):
   ia=min(step,ca-1);ib=min(step,cb-1)
   panel=Image.new("RGB",(512,288),"white");draw=ImageDraw.Draw(panel)
   panel.paste(Image.fromarray(ra.get_data(ia)),(0,32));panel.paste(Image.fromarray(rb.get_data(ib)),(256,32))
   draw.text((5,9),left_label+f" | step {ia}",fill="black")
   draw.text((261,9),right_label+f" | step {ib}",fill="black")
   writer.append_data(np.asarray(panel))
 ra.close();rb.close()
 with imageio.get_reader(str(destination)) as reader:
  assert reader.count_frames()==max(ca,cb);reader.get_data(0);reader.get_data(max(ca,cb)-1)
 return {"file":destination.name,"left_epoch":left,"right_epoch":right,"state":state,"frames":max(ca,cb),"fps":10}
common=[i for i in range(5) if cohorts[minimum]["trials"][i]["success"] and cohorts[5]["trials"][i]["success"]]
state=common[0] if common else next(i for i,t in enumerate(cohorts[minimum]["trials"]) if t["success"])
if minimum<5:contrast=minimum
else:
 contrast=max(e for e in cohorts if e<5)
 changed=[i for i in range(5) if cohorts[5]["trials"][i]["success"] and not cohorts[contrast]["trials"][i]["success"]]
 if changed:state=changed[0]
filename=f"02_{contrast}회vs5회_같은시작상태{state}.mp4"
comparison=make_pair(contrast,5,state,bundle/filename,f"{contrast} epochs","5 epochs")
baseline_name=f"03_1회vs최소{minimum}회_같은시작상태{state}.mp4"
baseline_comparison=make_pair(1,minimum,state,bundle/baseline_name,"1 epoch",f"{minimum} epochs")
first_any_video=baseline_name
first_any_comparison=None
if first_any!=minimum:
 any_state=next(i for i,t in enumerate(cohorts[first_any]["trials"]) if t["success"])
 first_any_video=f"05_1회vs최초성공{first_any}회_같은시작상태{any_state}.mp4"
 first_any_comparison=make_pair(1,first_any,any_state,bundle/first_any_video,"1 epoch",f"{first_any} epochs")

raw=bundle/"새로평가한원본영상";raw.mkdir()
trial_rows=[]
for e,c in sorted(cohorts.items()):
 for t in c["trials"]:
  trial_rows.append({"epoch":e,"state":t["init_state_index"],"success":t["success"],"policy_steps":t["policy_steps"],
                     "result_reused":e in [1,5],"source_video":t["video"]})
  if e in s["evaluated_new_epochs"]:shutil.copyfile(t["video"],raw/f"{e}회_시작상태{t['init_state_index']}.mp4")
with (bundle/"04_모든평가_시도별성공결과.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(trial_rows[0]));w.writeheader();w.writerows(trial_rows)
shutil.copyfile(RUN/"summary.json",bundle/"최소epoch_원본집계.json")
lines=[]
for e in range(1,6):
 if e in cohorts:
  c=cohorts[e];lines.append(f"{e}회: {c['successes']}/5, 성공률 {100*c['success_rate']:.1f}%, 평균 {c['mean_policy_steps']:.1f}단계"+(" (기존 동일조건 결과 재사용)" if e in [1,5] else " (이번에 새로 평가)"))
 else:lines.append(f"{e}회: 추가 평가하지 않음. 더 작은 epoch에서 목표 3/5를 이미 달성해 최소값 판정에 불필요.")
text=f"""최소 epoch 확인 — 2026-11-03

먼저 볼 답
- 5회 모델 수준인 3/5 이상 성공을 처음 달성한 정수 epoch: {minimum}회.
- 같은 5개 초기 상태 중 한 번이라도 성공한 최초 epoch: {first_any}회.
- '최소'는 이번 저장 학습 경로의 1~5회 정수 체크포인트와 고정된 초기 상태 5개에 한정한다.
- 다른 데이터·seed·초기 상태나 1.5회 같은 중간 학습 시점까지 포함한 보편적 최소값은 아니다.

실제 결과
{chr(10).join(lines)}

무엇을 했나?
저장해둔 2회 모델부터 5개 초기 상태에서 차례대로 실행했다.
각 후보에서 5회 모두 끝내고 결과를 집계했다.
3/5 이상이 처음 나온 후보 이후의 더 높은 epoch 추가 평가는 멈췄다.
1회 0/5와 5회 3/5는 앞선 실험의 동일 입력·동일 상태 결과를 재사용했다.
실제로 이번에 새로 평가한 epoch: {s['evaluated_new_epochs']}.
이번에 추가 학습은 하지 않았다. 각 epoch가 끝났을 때의 LoRA 저장본을 불러와 시험한 것이다.

공정한 비교 조건
같은 공개 LIBERO 실험용 분할(학습 전체 시범 40개·4,025장면, 검증 전체 시범 5개·462장면)에서 얻은 체크포인트.
같은 기본 OpenVLA와 LoRA 구조·학습률·batch·정규화. 바뀐 것은 저장 시점(epoch).
같은 초기 상태 0~4, env seed 0, global seed 7, warmup 10, 최대 정책 단계 220.
같은 원본 upright 256 RGB -> native HF 사진 처리.
시작 이미지와 초기 상태 SHA가 앞선 기준 평가와 정확히 같은지 확인.
새 Python 프로세스에서 각 LoRA를 정확히 불러와 같은 검증 batch loss와 (1,7) 유한 행동을 확인.
기술 오류를 작업 실패로 세지 않았으며, 이번 최종 집계에 기술 오류는 없다.

성공의 의미
LIBERO가 목표 그릇이 접시 위에 놓였다고 판정한 경우다.
모든 방향에서 안정적인 파지나 두 손가락으로 마주 집는 자세를 따로 보장하는 지표는 아니다.
각 초기 상태는 한 번씩 실행했다. 3/5 같은 값은 이 다섯 사례의 관측 결과이며 전체 환경 성공 확률의 보장은 아니다.
학습 시범과 평가 초기 상태의 완전한 분리는 확인되지 않았다.
성공률이 epoch에 따라 반드시 계속 증가한다고 가정하지 않았다. 실제 관측값으로 최소를 정했다.

영상 보기
02: 왼쪽 {contrast}회 / 오른쪽 5회, 같은 초기 상태 {state}.
03: 왼쪽 1회 / 오른쪽 기준을 충족한 최소 {minimum}회, 같은 초기 상태 {state}.
처음 한 번이라도 성공한 모델의 비교 영상: {first_any_video}
원본영상 폴더: 이번에 새로 실행한 후보들의 원본 영상.
1회·5회 전체 원본은 앞선 08_총5회학습_1회와비교_20261103 폴더에 있다.
비교 영상은 초당 10장으로 재생한다. 실제 계산 시간과 다르다.
한쪽이 먼저 종료되면 그쪽의 마지막 화면을 유지하고 다른 쪽은 계속 재생한다.
대표 상태는 두 모델의 공통 성공 사례를 우선 선택한 것이며 무작위 표본이 아니다.
전체 판단은 대표 영상 하나가 아니라 다섯 상태의 결과표를 사용했다.

남은 일
다른 초기 상태와 seed에서도 같은 epoch가 충분한지 확인하는 일반화·반복성 검증은 이번 범위에 포함하지 않았다.
더 큰 epoch가 더 높은 최고 성능을 내는지 찾는 실험과, 주어진 기준을 충족하는 최소 epoch를 찾는 실험은 다르다.

이해할 개념 두 가지
1) 체크포인트: 각 학습 시점의 가중치를 저장해둔 것. 재학습하지 않고도 그 시점의 실력을 시험할 수 있다.
2) 조건부 최소값: 특정 시험 조건과 기준을 만족한 가장 작은 값. 모든 상황에서의 최소값이라는 뜻은 아니다.

실제 원격 코드·결과
검사 코드: {RUN/'search_epochs.py'}
사전 기준: {RUN/'plan.json'}
체크포인트 목록: {RUN/'checkpoints.json'}
시도별 폴더: {RUN/'simulation'}
최종 요약: {RUN/'summary.json'}
영상 검사: {RUN/'video_verification.json'}
소스·환경 보존: {RUN/'preservation_report.json'}
원본 체크포인트 보존: {RUN/'protected_preservation.json'}

결과 확인 명령
cat {RUN/'summary.json'}

같은 검색을 새 폴더에서 재실행
conda activate lerobot312
NEW_RUN="/home/jovyan/projects/lerobot-openvla/outputs/minimum_epoch_search/$(date -u +%Y%m%dT%H%M%S)_manual"
mkdir "$NEW_RUN"
cp "{RUN/'search_epochs.py'}" "{RUN/'build_report.py'}" "$NEW_RUN/"
python "$NEW_RUN/search_epochs.py"
/home/jovyan/anaconda3/envs/openvla310/bin/python "$NEW_RUN/build_report.py"

기존 프로젝트 코드·세 환경·원본 체크포인트·이전 결과를 보존했다.
"""
(bundle/"00_먼저 읽기_최소epoch 결과.txt").write_text(text,encoding="utf-8")
metadata={"minimum_any":first_any,"minimum_matching_3_of_5":minimum,
 "representative_video":filename,"baseline_comparison_video":baseline_name,"first_success_comparison_video":first_any_video,"state":state,"evaluated_epochs":s["evaluated_new_epochs"],
 "results":{str(e):c["successes"] for e,c in cohorts.items()},"comparison_videos":[comparison,baseline_comparison]+([first_any_comparison] if first_any_comparison else [])}
(RUN/"delivery_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
hashes={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob("*") if p.is_file()}
(RUN/"viewing_hashes.json").write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
(RUN/"complete.json").write_text(json.dumps({"complete":True,"minimum_target":minimum,"minimum_any":first_any,"bundle_files":len(hashes),"comparison_videos_verified":2+(1 if first_any_comparison else 0)},indent=2))
print("MINIMUM_REPORT_READY",json.dumps(metadata),flush=True)
