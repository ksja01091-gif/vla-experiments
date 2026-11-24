"""Build an epoch1-10 comparison from executed results; no model runs."""
from pathlib import Path
import json,csv,hashlib,math,subprocess
RUN=Path(__file__).parent
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
OLD=ROOT/"outputs/five_epoch_bowl/20261103T064207_586519Z"
MINIMUM=ROOT/"outputs/minimum_epoch_search/20261103T083647_304730Z"
def read(p):return json.loads(Path(p).read_text())
assert read(RUN/"pipeline_status.json")["status"]=="complete"
out=RUN/"viewing_bundle";out.mkdir(exist_ok=True)
hist={int(r["epoch"]):r for r in read(OLD/"validation_history.json")}
hist.update({int(r["epoch"]):r for r in read(RUN/"validation_history.json")})
summaries={};rows=[];trials=[]
for epoch in range(1,11):
 path=(OLD/f"simulation_epoch{epoch}.json") if epoch in [1,5] else (MINIMUM/f"simulation_epoch{epoch}.json" if epoch<5 else RUN/f"simulation_epoch{epoch}.json")
 s=read(path);summaries[epoch]=s
 assert s["attempts"]==5
 v=hist[epoch]
 rows.append({"epoch":epoch,"successes":s["successes"],"attempts":5,"success_rate":s["success_rate"],
 "mean_policy_steps":s["mean_policy_steps"],"training_loss":v.get("training_loss_frame_weighted"),
 "validation_loss":v["loss"],"validation_action_token_accuracy":v["action_token_accuracy"],
 "validation_normalized_action_l1_teacher_forced":v["normalized_action_l1"],
 "checkpoint":s["checkpoint"],"evaluation_source":str(path),"newly_evaluated":epoch>=6})
 for t in s["trials"]:
  assert t["trial_executed"]
  trials.append({"epoch":epoch,"state":t["init_state_index"],"success":t["success"],"policy_steps":t["policy_steps"],
  "termination_reason":t["termination_reason"],"video":t["video"],"initial_image_sha256":t["initial_image_sha256"]})
with (out/"01_epoch_comparison.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (out/"02_all_trials.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(trials[0]));w.writeheader();w.writerows(trials)
training=read(RUN/"training_result.json");inference=read(RUN/"epoch10_inference_benchmark.json")
complete=[r["epoch"] for r in rows if r["successes"]==5]
best=max(r["successes"] for r in rows)
summary={"training_completed":True,"epochs_completed":10,"resumed_from_epoch":5,
 "additional_epochs":5,"new_training_frames_seen":20125,"unique_train_frames_per_epoch":4025,
 "validation_frames_per_epoch":462,"new_simulator_trials":25,"reused_simulator_trials":25,
 "epoch5_successes":summaries[5]["successes"],"epoch10_successes":summaries[10]["successes"],
 "first_observed_5of5_epoch":min(complete) if complete else None,
 "best_observed_successes":best,"best_observed_epochs":[r["epoch"] for r in rows if r["successes"]==best],
 "rows":rows,"additional_training_wall_seconds":training["additional_training_wall_seconds"],
 "training_wall_time_is_not_an_isolated_speed_benchmark":True,
 "training_peak_allocated_gib":training["gpu_peak_allocated_gib"],
 "epoch10_inference":{"samples":inference["samples"],"mean_action_seconds":inference["mean_action_seconds"],
 "peak_allocated_gib":inference["peak_allocated_gib"],"peak_reserved_gib":inference["peak_reserved_gib"]},
 "fresh_process_checkpoint_check":read(RUN/"checkpoint_reload_check.json"),
 "source_preservation":read(RUN/"preservation_report.json"),"protected_preservation":read(RUN/"protected_preservation.json"),
 "lora_sp_comparison_executed":False,"evaluation_schedule":read(RUN/"parallel_eval_scheduler.json") if (RUN/"parallel_eval_scheduler.json").exists() else {"used":False},"lora_sp_preparation":str(RUN/"lora_sp_preparation.json"),
 "caveat":"Same five previously used states, checkpoint selection may overfit them. Not an independent benchmark success estimate."}
(RUN/"summary.json").write_text(json.dumps(summary,indent=2))
(out/"summary.json").write_text(json.dumps(summary,indent=2))
for name in ["training_config.json","lora_sp_preparation.json","lora_sp_threshold_boundary_check.json","phase_source_provenance.json","paper_provenance.json","05_NEXT_LORA_SP_PLAN_KO.txt"]:
 (out/name).write_text((RUN/name).read_text())
header=[
"\ucd1d 10 epoch \uc5f0\uc7a5 \uc2e4\ud5d8 \uacb0\uacfc",
"",
"\uac19\uc740 \uc870\uac74: OpenVLA-7B / BF16 / LoRA rank8 / batch4 / LR 0.0001.",
"\uacf5\uac1c LIBERO \uc2dc\ubc94\uc5d0\uc11c \uc120\ud0dd\u00b7\uc804\ucc98\ub9ac\ud55c \ud559\uc2b5 40\uac1c \uc804\uccb4 \uc2dc\ubc94(4025\uc7a5\uba74), \uac80\uc99d 5\uac1c \uc804\uccb4 \uc2dc\ubc94(462\uc7a5\uba74).",
"\uae30\uc874 5 epoch\uc758 \uac00\uc911\uce58/AdamW \uae30\ub85d/\ub09c\uc218 \uc0c1\ud0dc\ub97c \uc774\uc5b4\uc11c 5\ubc88 \ub354 \ud559\uc2b5.",
"\ub9e4 epoch 6~10\uc758 \uccb4\ud06c\ud3ec\uc778\ud2b8 \uc800\uc7a5, \uc804\uccb4 \uac80\uc99d 462\uc7a5\uba74, \uac19\uc740 \ucd08\uae30 \uc0c1\ud0dc 5\uac1c \ud3c9\uac00.",
"",
"epoch | success | validation loss | action token accuracy",
]
for r in rows:header.append(f"{r['epoch']:>5} | {r['successes']}/5     | {r['validation_loss']:.4f}          | {r['validation_action_token_accuracy']*100:.2f}%")
header += ["",(f"\ucd5c\ucd08 5/5 \uad00\ucc30: {summary['first_observed_5of5_epoch']} epoch" if summary["first_observed_5of5_epoch"] is not None else "\ucd1d 10 epoch\uae4c\uc9c0 5/5\ub294 \uad00\ucc30\ub418\uc9c0 \uc54a\uc558\uc74c."),
f"Epoch5 -> epoch10 successes: {summary['epoch5_successes']}/5 -> {summary['epoch10_successes']}/5",
f"Additional training wall minutes: {summary['additional_training_wall_seconds']/60:.2f}",
f"Training peak allocated GPU memory: {summary['training_peak_allocated_gib']:.2f} GiB",
f"Epoch10 prediction: {inference['mean_action_seconds']*1000:.1f} ms/action, peak {inference['peak_allocated_gib']:.2f} GiB",
"",
"\ud559\uc2b5\uacfc \ud3c9\uac00 \uc77c\ubd80\ub97c \uac19\uc740 GPU\uc5d0\uc11c \uacb9\uccd0 \uc2e4\ud589\ud588\uc73c\ubbc0\ub85c \ud559\uc2b5 \uc2dc\uac04\uc740 \ub2e8\ub3c5 \uc18d\ub3c4 \ube44\uad50\uac12\uc774 \uc544\ub2d8.",
"\ucd5c\uc885 \ud589\ub3d9 \uc18d\ub3c4\ub294 \ud559\uc2b5\u00b7\uc2dc\ubbac\ub808\uc774\ud130 \uc885\ub8cc \ud6c4 \ub2e8\ub3c5\uc73c\ub85c \uce21\uc815.",
"\uc804\uccb4 \uac80\uc99d loss\ub294 \uc2dc\ubc94 \ud589\ub3d9 \uc608\uce21 \uc624\ucc28, success\ub294 \uc2dc\ubbac\ub808\uc774\ud130\uc5d0\uc11c \uc811\uc2dc \uc704\uc5d0 \uadf8\ub987\uc744 \ub193\uc558\ub294\uc9c0\uc758 \ud310\uc815.",
"\uac80\uc99d \uc2dc\ubc94 5\uac1c\uc640 \uc2dc\ubbac\ub808\uc774\ud130 \ucd08\uae30 \uc0c1\ud0dc 5\uac1c\ub294 \ub2e4\ub978 \ub2e8\uc704.",
"\uac19\uc740 \ub2e4\uc12f \uc0c1\ud0dc\ub97c \ubc18\ubcf5 \ud3c9\uac00\ud588\uc73c\ubbc0\ub85c 5/5 \ub2ec\uc131 \uc5ec\ubd80\uc640 \ubcc4\uac1c\ub85c \uc0c8 \uc0c1\ud0dc \uac80\uc99d\uc774 \ud544\uc694.",
"",
"LoRA-SP: \ucc28\ud6c4 \ube44\uad50 \uc900\ube44\ub9cc \ud588\uc74c. \ubaa8\ub378 \ud559\uc2b5\u00b7\uc18d\ub3c4\u00b7\uba54\ubaa8\ub9ac \ube44\uad50\ub294 \uc544\uc9c1 \uc2e4\ud589\ud558\uc9c0 \uc54a\uc74c.",
"\ub17c\ubb38: Adaptive Capacity Allocation for Vision Language Action Fine-tuning (Select-Prune).",
"\ub17c\ubb38\uc758 \u03c0\u2080/SmolVLA\u00b7rank128\uacfc \uc6b0\ub9ac OpenVLA\u00b7rank8\uc740 \ub2e4\ub974\ubbc0\ub85c \uc218\uc815 \ub300\uc0c1\uacfc \uc6a9\ub7c9 \uc608\uc0b0\uc744 \ub9de\ucdb0 \ube44\uad50 \uc124\uacc4 \ud544\uc694.",
"\ud65c\uc131 rank \uac10\uc18c\uac00 \uc2e4\uc81c GPU \uba54\ubaa8\ub9ac/\uc2dc\uac04 \uac10\uc18c\ub97c \ubcf4\uc7a5\ud558\uc9c0\ub294 \uc54a\uc74c.",
"",
"REMOTE RUN: "+str(RUN),
"CODE: ten_epoch_experiment.py (train, simulate_all, check, benchmark_final, audit)",
"RERUN REPORT: /home/jovyan/anaconda3/envs/lerobot312/bin/python "+str(RUN/"build_report.py"),
"RERUN EXPERIMENT IN NEW FOLDER: /home/jovyan/anaconda3/envs/lerobot312/bin/python "+str(RUN/"reproduce.py")+" --run",
"Reproduction helper: syntax/--help checked; full helper not rerun. The reported train/eval phases actually executed.",
"READ RESULTS: 01_epoch_comparison.csv; 02_all_trials.csv; 03_epoch_curves.png; paired videos",
"Original checkpoint/environment preservation: see preservation_report in summary.json.",
"Source launch snapshot: training_source_snapshot.py; final check adds isolated inference benchmark.",
]
(out/"00_START_HERE_KO.txt").write_text("\n".join(header),encoding="utf-8")
print("REPORT_TABLE_READY",summary["epoch10_successes"],summary["first_observed_5of5_epoch"],flush=True)
