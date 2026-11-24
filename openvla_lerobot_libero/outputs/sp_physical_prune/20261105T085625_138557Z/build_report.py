"""CPU-only result tables, figures and paired videos from actually executed trials."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
R=Path(__file__).resolve().parent
OUT=R/"viewing_bundle";OUT.mkdir(exist_ok=True)
assert json.loads((R/"evaluation_complete.json").read_text())["complete"]
def read(p):return json.loads(Path(p).read_text())
config=read(R/"experiment_config.json")
rows=[];simulations={}
for tag in ["sp_full128","sp_pruned96","sp_pruned64"]:
 b=read(R/"benchmarks"/tag/"result.json")
 sim=read(R/("simulation_"+tag+".json"));simulations[tag]=sim
 rows.append({"variant":tag,"physical_rank":b["physical_bank_rank"],"router_width":128,
    "adapter_parameters":b["adapter_parameters"],"adapter_mb":b["adapter_file_bytes"]/1e6,
    "inference_gpu_gib":b["peak_allocated_gib"],"load_peak_gpu_gib":b["load_peak_allocated_gib"],
    "inference_ms":1000*b["mean_action_seconds"],"round_mean_std_ms":1000*b["round_mean_std_seconds"],
    "rounds":3,"images_per_round":32,"validation_loss":b["validation"]["loss"],
    "validation_action_token_accuracy":b["validation"]["action_token_accuracy"],
    "validation_normalized_action_l1":b["validation"]["normalized_action_l1"],
    "mean_selected_kept_directions":b["rank_statistics"]["mean_active_rank"],
    "successes":sim["successes"],"attempts":5,"success_rate":sim["success_rate"],
    "mean_episode_steps_including_failures":sim["mean_policy_steps"],
    "new_robot_trials":0 if tag=="sp_full128" else 5,
    "fresh_process_reload":b["fresh_process_reload"],"adapter_hash":b["adapter_hash"],
    "matrix_example":b["matrix_shapes"][0],"checkpoint":b["checkpoint"]})
base=rows[0]
for row in rows:
 row.update({"adapter_reduction_percent":100*(1-row["adapter_mb"]/base["adapter_mb"]),
 "gpu_memory_reduction_mib":1024*(base["inference_gpu_gib"]-row["inference_gpu_gib"]),
 "latency_change_percent":100*(row["inference_ms"]/base["inference_ms"]-1),
 "A_B_theoretical_multiplications_fraction":row["physical_rank"]/128})
assert rows[1]["adapter_parameters"]<rows[0]["adapter_parameters"] and rows[2]["adapter_parameters"]<rows[1]["adapter_parameters"]
assert all(r["fresh_process_reload"] for r in rows)
summary={"config":config,"results":rows,"calibration":read(R/"calibration_complete.json"),
    "unit_checks":read(R/"unit_checks.json"),"preservation":read(R/"preservation_report.json"),
    "interpretation":{"new_pruning_extension_not_exact_paper_reproduction":True,
    "static_removal_changes_policy":True,"original_router_width_retained":128,
    "no_retraining_or_validation_based_rank_selection":True,
    "only_adapter_bank_reduced_base_OpenVLA_unchanged":True,
    "five_previously_used_states_not_independent_generalization":True,
    "small_timing_changes_need_more_repetitions":True}}
(R/"summary.json").write_text(json.dumps(summary,indent=2))
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
flat=[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in rows]
with (OUT/"02_comparison.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
trials=[]
for tag,sim in simulations.items():
 for t in sim["trials"]:trials.append({"variant":tag,"initial_state":t["init_state_index"],"success":t["success"],
    "policy_steps":t["policy_steps"],"reason":t["termination_reason"],"source_video":t["video"]})
with (OUT/"03_trials.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(trials[0]));w.writeheader();w.writerows(trials)
labels=["Full SP128","Pruned SP96","Pruned SP64"];colors=["#8597ad","#4c86bc","#5a9c80"]
fig,axes=plt.subplots(2,2,figsize=(11,7))
for ax,key,title,unit in zip(axes.flat,
 ["adapter_mb","inference_gpu_gib","inference_ms","successes"],
 ["Saved adapter size","Inference peak allocated GPU memory","Mean action prediction latency","Task success on same five states"],
 ["MB","GiB","ms","successes / 5"]):
 vals=[r[key] for r in rows];bars=ax.bar(labels,vals,color=colors)
 ax.set_title(title,fontsize=11);ax.set_ylabel(unit);ax.set_ylim(0,max(vals)*1.25 if max(vals)>0 else 1)
 if key=="successes":ax.set_ylim(0,5.8);ax.set_yticks(range(6))
 for bar,v in zip(bars,vals):
  label=f"{v}/5" if key=="successes" else (f"{v:.3f}" if key=="inference_gpu_gib" else f"{v:.1f}")
  ax.text(bar.get_x()+bar.get_width()/2,v+max(max(vals),1)*.025,label,ha="center",fontsize=10)
fig.suptitle("Physical SP bank pruning | Same trained checkpoint | No additional training",fontsize=12)
fig.text(.5,.012,"A/B directions removed permanently; original 128-score router retained. Baseline trials reused; five states only.",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/"04_results.png",dpi=160);plt.close(fig)
# Shapes prove physical removal, not just zero-valued masks.
shape_lines=["Actual adapter matrix dimensions (first of 64 language q/v modules)"]
for row in rows:
 x=row["matrix_example"]
 shape_lines.append(f"{row['variant']}: A={x['A']}, B={x['B']}, router={x['router_out']}; params={row['adapter_parameters']:,}")
(OUT/"05_physical_shapes.txt").write_text("\n".join(shape_lines))
videos=[]
for state in range(5):
 ts=[simulations[tag]["trials"][state] for tag in ["sp_full128","sp_pruned96","sp_pruned64"]]
 seq=[]
 for t in ts:
  reader=imageio.get_reader(t["video"]);frames=[frame for frame in reader];reader.close()
  assert len(frames)==t["policy_steps"]+1
  seq.append(frames)
 dest=OUT/f"06_compare_state_{state:02d}.mp4"
 with imageio.get_writer(str(dest),fps=20,codec="libx264",quality=7,macro_block_size=16) as writer:
  for k in range(max(map(len,seq))):
   canvas=Image.new("RGB",(768,336),"#182131");draw=ImageDraw.Draw(canvas)
   for j,(frames,t,label) in enumerate(zip(seq,ts,labels)):
    canvas.paste(Image.fromarray(frames[min(k,len(frames)-1)]).resize((256,256)),(256*j,48))
    draw.text((256*j+6,5),label,fill="white")
    draw.text((256*j+6,21),f"state {state} | {'SUCCESS' if t['success'] else 'FAILURE'} | {t['policy_steps']} steps",
       fill="#8aefac" if t["success"] else "#ffb09e")
    text=f"step {k}" if k<len(frames) else f"done at {len(frames)-1}; last frame held"
    draw.text((256*j+6,324),text,fill="white")
   draw.text((8,307),"20fps replay aligned by control step; not prediction speed.",fill="white")
   writer.append_data(np.asarray(canvas))
 n=0
 reader=imageio.get_reader(str(dest))
 for frame in reader:
  assert frame.shape[:2]==(336,768);n+=1
 reader.close();assert n==max(map(len,seq))
 videos.append({"file":dest.name,"decoded_frames":n,"successes":[t["success"] for t in ts]})
text=[
"SP 실제 방향 제거 실험 결과",
"",
"기존 학습된 SP128 모델에서 출발. 추가 학습 없음.",
"학습 시범 40개에서 4장면씩 총 160장면으로 제거 후보를 정함.",
"검증 자료나 로봇 시험 결과로 남길 방향을 고르지 않음.",
"모든 64개 q/v 모듈에서 A의 행과 B의 열을 128 -> 96 / 64로 실제 축소.",
"원래 128개 점수를 만드는 router와 eta=0.9 선택 규칙은 유지.",
"따라서 A/B 저장 공간과 해당 행렬 곱셈은 감소하지만, 본체와 router 비용은 유지.",
"이는 기존 SP에 정적 방향 제거를 추가한 실험. 원논문 전체 재현이 아님.",
"입력마다 필요 방향이 다르므로 영구 제거가 행동을 바꿀 수 있음.",
"",
"모델 | adapter MB | GPU GiB | 예측 ms | 작업 성공 | 검증 loss",
]
for row in rows:text.append(f"{row['variant']} | {row['adapter_mb']:.1f} | {row['inference_gpu_gib']:.3f} | {row['inference_ms']:.1f} | {row['successes']}/5 | {row['validation_loss']:.3f}")
text+=["","정밀한 차이"]
for row in rows[1:]:
 text.append(f"{row['variant']}: adapter {row['adapter_reduction_percent']:.1f}% 감소, GPU {row['gpu_memory_reduction_mib']:.1f} MiB 감소, 시간 변화 {row['latency_change_percent']:+.1f}% (음수면 빨라짐)")
text+=["",
"성공률은 같은 기존 시작 상태 5개에서 측정. 독립적인 새 상태 성능 검증이 아님.",
"기존 full SP 5회 결과 재사용, 작은 모델은 각각 5회 새로 실행.",
"시간은 모델별 단독 실행, 같은 32개 사진을 3번 반복. 작은 차이는 재측정 필요.",
"GPU 수치는 PyTorch peak allocated이며 전체 GPU 사용량과 구분.",
"예측 오차 감소와 실제 작업 성공은 다른 결과.",
"",
"코드 흐름:",
"run_phase.py calibrate -> 학습 사진으로 방향별 기여량 측정 -> export_compact()로 작은 파일 저장",
"run_phase.py benchmark -> 새 프로세스 재로딩 -> 검증 462장면 -> 행동 32장면 x3 속도/메모리 측정",
"run_phase.py simulate -> 같은 시작 상태 5개에서 최대 220번 행동",
"",
"실제로 크기를 줄이는 코드(pruned_policy.py):",
'if suffix=="sp_A.weight": value=value.index_select(0,keep)',
'elif suffix=="sp_B.weight": value=value.index_select(1,keep)',
"index_select(0, ...): 남길 행만 가져온다. A의 방향 수를 실제로 줄인다.",
"index_select(1, ...): 남길 열만 가져온다. B에서 같은 방향만 남긴다.",
"",
"실제 예측에 쓰는 코드:",
"projected = self.sp_A(x.to(self.sp_A.weight.dtype))",
"update = self.sp_B(projected * selected)",
"A/B가 작아진 상태에서 계산한다. 단순히 큰 행렬의 일부 값을 0으로 만드는 것과 구분.",
"",
"다시 한 장면 예측하기 (원격 Linux):",
"/home/jovyan/anaconda3/envs/lerobot312/bin/python "+str(R/"replay_action.py")+" --variant sp_pruned64 --index 0",
"",
"결과 파일: 02_comparison.csv, 03_trials.csv, 04_results.png, 06_compare_state_00..04.mp4",
"REMOTE RUN: "+str(R),
]
(OUT/"01_RESULTS_KO.txt").write_text("\n".join(text),encoding="utf-8")
for file in ["pruned_policy.py","run_phase.py","check_compact.py","driver.py","replay_action.py","experiment_config.json","calibration_complete.json","unit_checks.json","preservation_report.json"]:
 (OUT/file).write_bytes((R/file).read_bytes())
manifest={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json"}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
(R/"report_complete.json").write_text(json.dumps({"complete":True,"files":len(manifest),"videos":videos},indent=2))
print("PRUNING_REPORT_COMPLETE",json.dumps(rows),flush=True)

