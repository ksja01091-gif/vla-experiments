from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw
R=Path(__file__).parent;OUT=R/"viewing_bundle"
s=json.loads((R/"summary.json").read_text());rows=s["methods"];etas=s["eta_inference_ablation"]
labels=["LoRA 8","LoRA 128","SP 128"];colors=["#8396ad","#4378bf","#d88b32"]
fig,axes=plt.subplots(2,3,figsize=(13,7.5))
fields=[("isolated_train_frames_per_second","Training throughput (128 frames)","frames/s"),
("isolated_train_peak_gib","Training peak allocated GPU memory","GiB"),
("mean_inference_ms","Action prediction latency (32 images)","ms"),
("inference_peak_gib","Inference peak allocated GPU memory","GiB"),
("adapter_file_mb","Saved adapter size","MB"),
("successes","Same five initial states","successes / 5")]
for ax,(key,title,unit) in zip(axes.flat,fields):
 vals=[r[key] for r in rows];bars=ax.bar(labels,vals,color=colors)
 ax.set_title(title,fontsize=10);ax.set_ylabel(unit);ax.set_ylim(0,max(vals)*1.25 if max(vals)>0 else 1)
 if key=="successes":ax.set_ylim(0,5.8);ax.set_yticks(range(6))
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(max(vals),1)*.025,f"{v:.2f}" if key!="successes" else f"{v}/5",ha="center",fontsize=9)
fig.suptitle("OpenVLA | Same 40 demonstrations, 5 epochs, q/v targets | SP eta=0.9",fontsize=12)
fig.text(.5,.012,"Isolated resource benchmarks. SP is a dense full-bank OpenVLA adaptation; five reused states do not establish generalization.",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/"05_method_resources.png",dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,4,figsize=(13,3.8))
for ax,key,title,unit in zip(axes,["mean_active_rank","inference_peak_gib","mean_inference_ms","successes"],
["Mean active basis count","Inference GPU memory","Prediction latency","Task success"],["directions","GiB","ms","successes / 5"]):
 vals=[r[key] for r in etas];bars=ax.bar([str(r["eta"]) for r in etas],vals,color=["#d88b32","#65a089","#6688b5"])
 ax.set_title(title,fontsize=10);ax.set_ylabel(unit);ax.set_xlabel("Inference eta");ax.set_ylim(0,max(max(vals),1)*1.25)
 if key=="successes":ax.set_ylim(0,5.8);ax.set_yticks(range(6))
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(max(vals),1)*.025,f"{v:.2f}" if key!="successes" else f"{v}/5",ha="center",fontsize=8)
fig.suptitle("Same SP checkpoint; inference-only threshold changes; no retraining",fontsize=12)
fig.tight_layout(rect=(0,0,1,.9));fig.savefig(OUT/"06_eta_effects.png",dpi=160);plt.close(fig)

def triples(tags,family):
 data=[json.loads((R/("simulation_"+t+".json")).read_text())["trials"] for t in tags]
 reports=[]
 for state in range(5):
  trials=[d[state] for d in data]
  readers=[imageio.get_reader(t["video"]) for t in trials]
  sequences=[[frame for frame in reader] for reader in readers]
  for reader in readers:reader.close()
  assert all(len(frames)==t["policy_steps"]+1 for frames,t in zip(sequences,trials))
  dest=OUT/f"{family}_state_{state:02d}.mp4"
  with imageio.get_writer(str(dest),fps=20,codec="libx264",quality=7,macro_block_size=16) as writer:
   for k in range(max(map(len,sequences))):
    canvas=Image.new("RGB",(768,336),"#182131");draw=ImageDraw.Draw(canvas)
    for j,(frames,t,tag) in enumerate(zip(sequences,trials,tags)):
     canvas.paste(Image.fromarray(frames[min(k,len(frames)-1)]).resize((256,256)),(256*j,48))
     draw.text((256*j+6,5),tag,fill="white")
     draw.text((256*j+6,21),f"state {state} | {'SUCCESS' if t['success'] else 'FAILURE'} | {t['policy_steps']} steps",
       fill="#8aefac" if t["success"] else "#ffb09e")
     status=f"step {k}" if k<len(frames) else f"done at {len(frames)-1}; last frame held"
     draw.text((256*j+6,324),status,fill="white")
    draw.text((8,307),"20fps replay aligned by control step; not wall-clock inference speed.",fill="white")
    writer.append_data(np.asarray(canvas))
  n=0
  for frame in imageio.get_reader(str(dest)):
   assert frame.shape[:2]==(336,768);n+=1
  assert n==max(map(len,sequences))
  reports.append({"file":dest.name,"state":state,"decoded_frames":n,"successes":[t["success"] for t in trials]})
 return reports
videos=triples(["lora8","lora128","sp128_eta_090"],"07_three_methods")
videos+=triples(["sp128_eta_090","sp128_eta_080","sp128_eta_070"],"08_eta_comparison")
# Learning curves compare task-only CE; SP auxiliary losses excluded.
old=Path("/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z")
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for method,label,color in zip(["lora8","lora128","sp128"],labels,colors):
 hist=json.loads(((old if method=="lora8" else R/method)/"validation_history.json").read_text())
 if method=="lora8":
  origin=Path("/home/jovyan/projects/lerobot-openvla/outputs/bowl40_experiment/20261103T051914_711739Z")
  metric_rows=[]
  for log in [origin/"training_metrics.jsonl",old/"training_metrics.jsonl"]:
   metric_rows.extend(json.loads(line) for line in log.read_text().splitlines() if line)
  for h in hist:
   matches=[x for x in metric_rows if x["epoch"]==h["epoch"]]
   assert sum(x["batch_frames"] for x in matches)==4025
   h["training_loss_frame_weighted"]=sum(x["loss"]*x["batch_frames"] for x in matches)/4025
 epochs=[h["epoch"] for h in hist]
 axes[0].plot(epochs,[h["training_loss_frame_weighted"] for h in hist],marker="o",color=color,label=label)
 axes[1].plot(epochs,[h["loss"] for h in hist],marker="o",color=color,label=label)
for ax,title in zip(axes,["Training task CE (40 demonstrations)","Validation task CE (5 demonstrations)"]):
 ax.set_title(title,fontsize=10);ax.set_xlabel("Epoch");ax.set_ylabel("Task CE, lower is better");ax.set_xticks(range(1,6));ax.legend();ax.grid(alpha=.2)
fig.suptitle("Prediction error is not robot task success",fontsize=12)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/"09_learning_curves.png",dpi=160);plt.close(fig)

manifest={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json"}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
(R/"report_complete.json").write_text(json.dumps({"complete":True,"files":len(manifest),"videos":videos},indent=2))
print("COMPARISON_VISUALS_COMPLETE",len(videos),flush=True)
