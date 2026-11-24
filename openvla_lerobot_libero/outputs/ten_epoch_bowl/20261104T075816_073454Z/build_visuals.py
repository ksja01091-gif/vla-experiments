from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw
RUN=Path(__file__).parent;OUT=RUN/"viewing_bundle"
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
OLD=ROOT/"outputs/five_epoch_bowl/20261103T064207_586519Z"
s=json.loads((RUN/"summary.json").read_text());rows=s["rows"]
epochs=[r["epoch"] for r in rows];success=[r["successes"] for r in rows]
fig,axes=plt.subplots(2,2,figsize=(11,7.5))
ax=axes[0,0];ax.bar(epochs,success,color=["#7d94ad" if e<=5 else "#cf842c" for e in epochs])
ax.set_ylim(0,5.8);ax.set_yticks(range(6));ax.set_xticks(epochs);ax.set_ylabel("Successful trials / 5");ax.set_xlabel("Epoch")
ax.set_title("Same five initial states; 220-action horizon")
for e,v in zip(epochs,success):ax.text(e,v+.1,f"{v}/5",ha="center",fontsize=9)
ax=axes[0,1];ax.plot(epochs,[r["training_loss"] if r["training_loss"] is not None else np.nan for r in rows],"o-",label="Train")
ax.plot(epochs,[r["validation_loss"] for r in rows],"o-",label="Validation (462 frames)")
ax.set_title("Teacher-forced cross entropy");ax.set_xlabel("Epoch");ax.set_xticks(epochs);ax.legend()
ax=axes[1,0];ax.plot(epochs,[r["validation_action_token_accuracy"]*100 for r in rows],"o-",color="#4378bf")
ax.set_title("Validation action-token accuracy (teacher-forced)");ax.set_ylabel("%");ax.set_xlabel("Epoch");ax.set_xticks(epochs)
ax=axes[1,1];ax.plot(epochs,[r["validation_normalized_action_l1_teacher_forced"] for r in rows],"o-",color="#368c69")
ax.set_title("Validation normalized action L1 (teacher-forced)");ax.set_xlabel("Epoch");ax.set_xticks(epochs)
fig.suptitle("OpenVLA | LoRA rank 8 | Same data, batch 4, LR 1e-4 | BF16 backbone",fontsize=12)
fig.text(.5,.012,"Previously used states: model selection on these trials does not establish unseen-state generalization.",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(OUT/"03_epoch_curves.png",dpi=160);plt.close(fig)
old=json.loads((OLD/"simulation_epoch5.json").read_text())["trials"]
new=json.loads((RUN/"simulation_epoch10.json").read_text())["trials"]
verified=[]
for state in range(5):
 trials=[old[state],new[state]]
 readers=[imageio.get_reader(t["video"]) for t in trials]
 sequences=[[frame for frame in reader] for reader in readers]
 for reader in readers:reader.close()
 assert all(sequences)
 video=OUT/f"04_epoch5_vs_epoch10_state_{state:02d}.mp4"
 with imageio.get_writer(str(video),fps=20,codec="libx264",quality=7,macro_block_size=16) as writer:
  for k in range(max(map(len,sequences))):
   canvas=Image.new("RGB",(512,320),"#182131");draw=ImageDraw.Draw(canvas)
   for j,(frames,t,epoch) in enumerate(zip(sequences,trials,[5,10])):
    canvas.paste(Image.fromarray(frames[min(k,len(frames)-1)]).resize((256,256)),(256*j,40))
    draw.text((256*j+8,5),f"Epoch {epoch} | BF16 | LoRA rank 8",fill="white")
    draw.text((256*j+8,21),f"state {state} | {'SUCCESS' if t['success'] else 'FAILURE'} | {t['policy_steps']} steps",fill="#8aefac" if t["success"] else "#ffb09e")
   draw.text((8,303),f"Control-step aligned replay at 20fps; not wall-clock time. step {k}",fill="white")
   writer.append_data(np.asarray(canvas))
 count=0
 for frame in imageio.get_reader(str(video)):
  assert frame.shape[:2]==(320,512);count+=1
 assert count==max(map(len,sequences))
 verified.append({"state":state,"video":str(video),"decoded_frames":count,"epoch5_success":old[state]["success"],"epoch10_success":new[state]["success"]})
for name in ["ten_epoch_experiment.py","training_source_snapshot.py","pipeline.py","reproduce.py","parallel_eval_scheduler.py","code_locations.json"]:
 (OUT/name).write_text((RUN/name).read_text())
manifest={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json"}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
(RUN/"report_complete.json").write_text(json.dumps({"complete":True,"files":len(manifest),"videos":verified},indent=2))
print("VISUALS_COMPLETE",verified,flush=True)
