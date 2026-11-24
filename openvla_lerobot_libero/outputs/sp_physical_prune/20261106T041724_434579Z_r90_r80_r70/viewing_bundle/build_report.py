"""Combine preserved and new physical ranks with explicit measurement provenance."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
R=Path(__file__).resolve().parent
O=R/"viewing_bundle";O.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
assert read(R/"evaluation_complete.json")["complete"]
cfg=read(R/"experiment_config.json");old=Path(cfg["parent_pruning_run"])
rows=[];sims={}
for rank in [128,96,90,80,70,64]:
    tag="sp_full128" if rank==128 else f"sp_pruned{rank}"
    source=old if rank in [96,64] else R
    b=read(source/"benchmarks"/tag/"result.json")
    s=read(source/("simulation_"+tag+".json"));sims[tag]=s
    assert b["fresh_process_reload"] and b["adapter_hash_unchanged"]
    assert all(t["finite"] and t["shape"]==[1,7] for t in b["rows"])
    assert all(x["A"][0]==rank and x["B"][1]==rank for x in b["matrix_shapes"])
    rows.append({"variant":tag,"physical_rank":rank,"adapter_mb":b["adapter_file_bytes"]/1e6,
        "adapter_parameters":b["adapter_parameters"],"gpu_gib":b["peak_allocated_gib"],
        "inference_ms":b["mean_action_seconds"]*1000,"round_std_ms":b["round_mean_std_seconds"]*1000,
        "validation_loss":b["validation"]["loss"],"action_token_accuracy":b["validation"]["action_token_accuracy"],
        "normalized_action_l1":b["validation"]["normalized_action_l1"],
        "mean_active_rank":b["rank_statistics"]["mean_active_rank"],
        "successes":s["successes"],"attempts":5,"success_rate":s["success_rate"],
        "mean_episode_steps":s["mean_policy_steps"],"checkpoint":b["checkpoint"],
        "benchmark_source":str(source),"benchmark_reused":rank in [96,64],
        "simulation_reused":rank in [128,96,64],"new_trials":5 if rank in [90,80,70] else 0})
summary={"config":cfg,"results":rows,"preservation":read(R/"preservation_report.json"),
    "new_robot_trials":15,"reused_robot_trials":15,"single_seed":True,
    "previously_used_states_not_new_generalization":True,"static_pruning_changes_policy":True,
    "rank96_64_timing_from_previous_run_not_simultaneous":True}
(R/"summary.json").write_text(json.dumps(summary,indent=2))
(O/"summary.json").write_text(json.dumps(summary,indent=2))
with (O/"02_rank_comparison.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
trial_rows=[]
for tag,s in sims.items():
    for t in s["trials"]:
        trial_rows.append({"variant":tag,"state":t["init_state_index"],"success":t["success"],
                           "steps":t["policy_steps"],"reason":t["termination_reason"],"video":t["video"]})
with (O/"03_trials.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(trial_rows[0]));w.writeheader();w.writerows(trial_rows)
labels=[str(x["physical_rank"])+( "*" if x["benchmark_reused"] else "") for x in rows]
fig,axs=plt.subplots(2,2,figsize=(12,7))
for ax,key,title,unit in zip(axs.flat,["adapter_mb","gpu_gib","inference_ms","successes"],
    ["Saved adapter size","Inference peak allocated GPU memory","Mean action prediction latency","Task success on same five states"],
    ["MB","GiB","ms","successes / 5"]):
    values=[x[key] for x in rows];bars=ax.bar(labels,values,color=["#8998ac","#4e99a8","#5a8ec4","#5a8ec4","#5a8ec4","#8998ac"])
    ax.set_title(title);ax.set_ylabel(unit);ax.set_xlabel("Physical bank rank")
    ax.set_ylim(0,5.8 if key=="successes" else max(values)*1.25)
    for bar,v in zip(bars,values):
        label=f"{v}/5" if key=="successes" else f"{v:.3f}" if key=="gpu_gib" else f"{v:.1f}"
        ax.text(bar.get_x()+bar.get_width()/2,v+max(values)*.02,label,ha="center",fontsize=9)
fig.suptitle("SP physical rank sweep | No retraining | Same original checkpoint")
fig.text(.5,.012,"* Rank96/64 resources reused from prior run. Robot trials: 90/80/70 new; 128/96/64 reused. Five old states only.",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(O/"04_rank_results.png",dpi=160);plt.close(fig)
videos=[]
for state in range(5):
    tags=["sp_pruned96","sp_pruned90","sp_pruned80","sp_pruned70"]
    ts=[sims[tag]["trials"][state] for tag in tags];seq=[]
    for t in ts:
        reader=imageio.get_reader(t["video"]);frames=[x for x in reader];reader.close()
        assert len(frames)==t["policy_steps"]+1;seq.append(frames)
    dest=O/f"05_compare_state_{state:02d}.mp4"
    with imageio.get_writer(str(dest),fps=20,codec="libx264",quality=7,macro_block_size=16) as writer:
        for step in range(max(map(len,seq))):
            canvas=Image.new("RGB",(1024,336),"#182131");draw=ImageDraw.Draw(canvas)
            for j,(frames,t,label) in enumerate(zip(seq,ts,["SP96 (previous)","SP90","SP80","SP70"])):
                x=j*256;canvas.paste(Image.fromarray(frames[min(step,len(frames)-1)]).resize((256,256)),(x,48))
                draw.text((x+6,5),label,fill="white")
                draw.text((x+6,22),f"state {state} | {'SUCCESS' if t['success'] else 'FAILURE'} | {t['policy_steps']} steps",fill="#8aefac" if t["success"] else "#ffb09e")
                draw.text((x+6,324),f"step {step}" if step<len(frames) else "Finished; last frame held",fill="white")
            draw.text((8,307),"20fps replay aligned by control step; not prediction latency.",fill="white")
            writer.append_data(np.asarray(canvas))
    count=0;reader=imageio.get_reader(str(dest))
    for frame in reader:assert frame.shape[:2]==(336,1024);count+=1
    reader.close();assert count==max(map(len,seq))
    videos.append({"file":dest.name,"decoded_frames":count})
guide=["SP rank 90 / 80 / 70 자동 비교 결과","","같은 학습된 SP128 원본에서 각각 제거. 추가 학습 없음.",
 "학습용 시범 40개의 160장면으로 측정했던 중요도를 재사용. 검증 자료로 방향을 고르지 않음.",
 "rank | 어댑터 MB | GPU GiB | 예측 ms | 성공/5 | 검증 loss"]
for x in rows:guide.append(f"{x['physical_rank']} | {x['adapter_mb']:.1f} | {x['gpu_gib']:.3f} | {x['inference_ms']:.1f} | {x['successes']}/5 | {x['validation_loss']:.3f}")
guide += ["","90/80/70은 각각 5번 새로 시험. 기존 128/96/64 로봇 결과는 재사용.",
 "128/90/80/70 속도는 이번에 단독 실행으로 새로 측정. 96/64 속도는 이전 측정이므로 작은 속도 차이 해석에 주의.",
 "같은 기존 시작 상태 5개와 학습 seed 1개. 독립적인 새 상태 일반화 검증이 아님.",
 "본체 7B와 router128은 유지. 줄어든 것은 A/B 방향이며 전체 연산·메모리가 같은 비율로 줄지는 않음.",
 "검증 loss 감소와 실제 작업 성공은 따로 읽기.","",
 "영상 순서: 왼쪽부터 이전 SP96 / 새 SP90 / 새 SP80 / 새 SP70.",
 "모든 5개 영상은 20fps로 재생하고 끝난 영상의 마지막 사진을 유지함. 실제 예측 속도는 표의 ms로 비교.",
 "핵심 코드: export_ranks.py(작은 checkpoint 저장), run_phase.py(검증·예측·로봇 시험), driver.py(자동 순서).",
 "재실행: /home/jovyan/anaconda3/envs/lerobot312/bin/python "+str(R/"replay_action.py")+" --variant sp_pruned70 --index 0",
 "REMOTE RUN: "+str(R)]
(O/"01_RESULTS_KO.txt").write_text("\n".join(guide),encoding="utf-8")
for name in ["experiment_config.json","pruned_policy.py","export_ranks.py","run_phase.py","driver.py","replay_action.py","unit_checks.json","preservation_report.json","build_report.py","upload_results.py"]:
    (O/name).write_bytes((R/name).read_bytes())
manifest={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in O.iterdir() if p.is_file() and p.name!="manifest.json"}
(O/"manifest.json").write_text(json.dumps(manifest,indent=2))
(R/"report_complete.json").write_text(json.dumps({"complete":True,"videos":videos,"files":len(manifest)},indent=2))
print("REPORT_COMPLETE",json.dumps(rows),flush=True)
