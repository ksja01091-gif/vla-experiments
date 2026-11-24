from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
RUN=Path(__file__).parent;OUT=RUN/"viewing_bundle"
OLD=Path("/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z")
s=json.loads((RUN/"summary.json").read_text())
fig,axes=plt.subplots(2,2,figsize=(10,7))
for ax,title,labels,values,unit in [
(axes[0,0],"Training throughput (128 measured frames)",["Batch 4","Batch 8"],[s["batch_size_4"]["frames_per_second"],s["batch_size_8"]["frames_per_second"]],"frames/s"),
(axes[0,1],"Training peak allocated GPU memory",["Batch 4","Batch 8"],[s["batch_size_4"]["peak_allocated_gib"],s["batch_size_8"]["peak_allocated_gib"]],"GiB"),
(axes[1,0],"Inference peak allocated GPU memory",["BF16","INT8"],[s["inference"]["bf16"]["peak_allocated_gib"],s["inference"]["int8"]["peak_allocated_gib"]],"GiB"),
(axes[1,1],"Action prediction latency (32 images)",["BF16","INT8"],[s["inference"]["bf16"]["mean_action_seconds"]*1000,s["inference"]["int8"]["mean_action_seconds"]*1000],"ms/action")]:
 bars=ax.bar(labels,values,color=["#4378bf","#e29439"])
 ax.set_title(title,fontsize=10);ax.set_ylabel(unit);ax.set_ylim(0,max(values)*1.2)
 for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+max(values)*.025,f"{value:.2f}",ha="center")
fig.suptitle(f"Same epoch-5 LoRA | Simulator: BF16 {s['simulation']['bf16']['successes']}/5, INT8 {s['simulation']['int8']['successes']}/5",fontsize=12)
fig.text(.5,.015,"Short throughput test; not batch-8 model quality. Simulation uses the same five previously tested states.",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/"01_comparison.png",dpi=160);plt.close(fig)
old=json.loads((OLD/"simulation_epoch5.json").read_text())["trials"]
new=json.loads((RUN/"simulation_int8.json").read_text())["trials"]
changed=[n for n in range(5) if old[n]["success"]!=new[n]["success"]]
def make_video(state):
 paths=[Path(old[state]["video"]),Path(new[state]["video"])]
 readers=[imageio.get_reader(str(p)) for p in paths]
 sequences=[[frame for frame in reader] for reader in readers]
 for r in readers:r.close()
 assert all(sequences)
 video=OUT/f"02_BF16_vs_INT8_state_{state:02d}.mp4"
 with imageio.get_writer(str(video),fps=20,codec="libx264",quality=7,macro_block_size=16) as w:
  for k in range(max(map(len,sequences))):
   canvas=Image.new("RGB",(512,320),"#182131");d=ImageDraw.Draw(canvas)
   for j,(frames,trial,label) in enumerate(zip(sequences,[old[state],new[state]],["BF16","INT8"])):
    frame=Image.fromarray(frames[min(k,len(frames)-1)]).resize((256,256))
    canvas.paste(frame,(256*j,40))
    d.text((256*j+8,5),f"{label} | SAME epoch-5 LoRA",fill="white")
    d.text((256*j+8,21),f"state {state} | {'SUCCESS' if trial['success'] else 'FAILURE'} | {trial['policy_steps']} steps",fill="#8aefac" if trial["success"] else "#ffb09e")
   d.text((8,303),f"Control-step aligned replay at 20 fps; not wall-clock latency. step {k}",fill="white")
   w.append_data(np.asarray(canvas))
 # Verify actual decoding of every generated frame.
 count=0
 for frame in imageio.get_reader(str(video)):
  assert frame.shape[:2]==(320,512);count+=1
 assert count==max(map(len,sequences))
 return {"video":str(video),"decoded_frames":count,"state":state}
comparisons=[make_video(state) for state in (changed if changed else [0])]
manifest={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file()}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
(RUN/"report_complete.json").write_text(json.dumps({"complete":True,"comparisons":comparisons,"files":len(manifest)},indent=2))
print("VISUAL_REPORT_COMPLETE",comparisons,flush=True)
