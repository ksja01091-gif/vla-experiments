"""Executed-result report and beginner-readable tables. No model training here."""
from pathlib import Path
import json,csv,hashlib,math
R=Path(__file__).parent
OLD=Path("/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z")
def read(p):return json.loads(Path(p).read_text())
assert read(R/"status.json")["status"]=="complete"
out=R/"viewing_bundle";out.mkdir(exist_ok=True)
rows=[]
for method in ["lora8","lora128","sp128"]:
 tag=method if method!="sp128" else "sp128_eta_090"
 profile=read(R/method/"profile.json")
 training=read(OLD/"training_result.json") if method=="lora8" else read(R/method/"training_result.json")
 v=training["final_validation"]
 infer=read(R/"inference"/(tag+".json"))
 sim=read(R/("simulation_"+tag+".json"))
 rows.append({"method":method,"nominal_initial_rank":8 if method=="lora8" else 128,
 "trainable_parameters":profile["trainable_parameters"],"isolated_train_frames_per_second":profile["frames_per_second"],
 "isolated_train_peak_gib":profile["peak_allocated_gib"],"isolated_train_128_frame_seconds":profile["seconds"],
 "epochs":5,"validation_loss_task_only":v["loss"],"action_token_accuracy_teacher_forced":v["action_token_accuracy"],
 "normalized_action_l1_teacher_forced":v["normalized_action_l1"],
 "mean_inference_ms":infer["mean_action_seconds"]*1000,"inference_peak_gib":infer["peak_allocated_gib"],
 "adapter_file_mb":infer["adapter_file_bytes"]/1e6,
 "mean_active_rank":infer["rank_statistics"]["mean_active_rank"] if infer["rank_statistics"] else (8 if method=="lora8" else 128),
 "successes":sim["successes"],"attempts":5,"success_rate":sim["success_rate"],"mean_episode_steps_including_failures":sim["mean_policy_steps"]})
eta_rows=[]
for eta in [.9,.8,.7]:
 tag=f"sp128_eta_{round(eta*100):03d}"
 infer=read(R/"inference"/(tag+".json"));sim=read(R/("simulation_"+tag+".json"))
 eta_rows.append({"eta":eta,"initial_bank_rank":128,"mean_active_rank":infer["rank_statistics"]["mean_active_rank"],
 "inference_peak_gib":infer["peak_allocated_gib"],"mean_inference_ms":infer["mean_action_seconds"]*1000,
 "adapter_file_mb":infer["adapter_file_bytes"]/1e6,"adapter_hash":infer["adapter_hash"],
 "successes":sim["successes"],"attempts":5,"success_rate":sim["success_rate"],"mean_steps":sim["mean_policy_steps"]})
assert len({r["adapter_hash"] for r in eta_rows})==1
for name,data in [("02_method_comparison.csv",rows),("03_eta_comparison.csv",eta_rows)]:
 with (out/name).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
trials=[]
for tag in ["lora8","lora128","sp128_eta_090","sp128_eta_080","sp128_eta_070"]:
 for t in read(R/("simulation_"+tag+".json"))["trials"]:
  trials.append({"tag":tag,"state":t["init_state_index"],"success":t["success"],"policy_steps":t["policy_steps"],
 "reason":t["termination_reason"],"video":t["video"]})
with (out/"04_all_trials.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(trials[0]));w.writeheader();w.writerows(trials)
source_hashes=read(R/"source_hashes_at_training_start.json")
changed=[p for p,h in source_hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed,changed
sources={m:read(R/m/"profile_initial/base_model.json")["checkpoint"] for m in ["lora8","lora128","sp128"]}
assert len(set(sources.values()))==1,sources
assert next(iter(sources.values()))==read(OLD/"training_config.json")["base_snapshot"]
summary={"resolved_base_snapshots":sources,"methods":rows,"eta_inference_ablation":eta_rows,"same_sp_weights_all_eta":True,
 "new_simulator_trials":20,"reused_baseline_trials":5,"training_epochs":5,"one_training_seed":7,
 "conditions":"same 40 train demos/4025 frames, 5 validation demos/462 frames; same native processor/q-v targets/normalization/batch4/LR1e-4/BF16+FP32 adapters",
 "speed_measurement":"128-frame isolated initial-training profile and 32-image isolated final-inference benchmark; training wall time under overlap is NOT a speed comparison",
 "sp_implementation":"OpenVLA application of paper select-prune equations, hidden32 SiLU/Softplus router; dense full bank; no true-SVD error guarantee",
 "capacity_control":"same bank128, additional SP router parameters still differ; trainable counts reported",
 "eta_ablation":"inference-only eta0.9/0.8/0.7 on same SP5 checkpoint; not per-eta retraining",
 "generalization":"same five previously used initialization states, no independent held-out-state success claim",
 "preservation":read(R/"preservation_report.json"),"training_sources_unchanged":not changed,
 "unit_checks":read(R/"sp_unit_check.json")}
(R/"summary.json").write_text(json.dumps(summary,indent=2));(out/"summary.json").write_text(json.dumps(summary,indent=2))
text=[
"\uc138 \ubaa8\ub378 \ube44\uad50 + SP \uc120\ud0dd \uae30\uc900 \uc2e4\ud5d8 \uacb0\uacfc",
"",
"\uc870\uac74: OpenVLA-7B, \uac19\uc740 LIBERO \uadf8\ub987 \uc62e\uae30 1\uac1c \uc791\uc5c5, 5 epoch, batch4, LR0.0001, \uae30\ubcf8 BF16 / \ud559\uc2b5 \uac12 FP32.",
"\uacf5\uac1c \uc2dc\ubc94\uc5d0\uc11c \uc120\ud0dd\u00b7\uc804\ucc98\ub9ac\ud55c \ud559\uc2b5 \uc804\uccb4 \uc2dc\ubc94 40\uac1c(4025\uc7a5\uba74) / \uac80\uc99d \uc804\uccb4 \uc2dc\ubc94 5\uac1c(462\uc7a5\uba74).",
"\uae30\uc874 LoRA8 \ud559\uc2b5 \uacb0\uacfc\ub97c \uc7ac\uc0ac\uc6a9. LoRA128\uacfc SP128\uc740 \uac19\uc740 \uc0ac\uc804\ud559\uc2b5 \ubaa8\ub378\uc5d0\uc11c \uc0c8\ub85c 5\ubc88 \ud559\uc2b5.",
"",
"\ubaa8\ub378 / \uc791\uc5c5 \uc131\uacf5 / \ud559\uc2b5 \uba54\ubaa8\ub9ac / \uc2e4\ud589 \uba54\ubaa8\ub9ac / \ud589\ub3d9 \uc608\uce21 \uc2dc\uac04",
]
for x in rows:text.append(f"{x['method']} | {x['successes']}/5 | {x['isolated_train_peak_gib']:.2f} GiB | {x['inference_peak_gib']:.2f} GiB | {x['mean_inference_ms']:.1f} ms")
text+=["","\uc120\ud0dd \uae30\uc900 eta / \ud3c9\uade0 \uc720\ud6a8 rank / \uc791\uc5c5 \uc131\uacf5 / \uc2e4\ud589 \uba54\ubaa8\ub9ac"]
for x in eta_rows:text.append(f"{x['eta']} | {x['mean_active_rank']:.2f} | {x['successes']}/5 | {x['inference_peak_gib']:.2f} GiB")
text+=["",
"\uc720\ud6a8 rank\ub294 \uc120\ud0dd\ud574 \uc4f0\ub294 \ubc29\ud5a5 \uc218. \uae30\ubcf8 \ud30c\ub77c\ubbf8\ud130 \uc218\ub098 \uc800\uc7a5 \ud30c\uc77c \ud06c\uae30\uc640 \ub2e4\ub974\ub2e4.",
"\uc774 \uad6c\ud604\uc740 \uc804\uccb4 \ubc29\ud5a5\uc744 \ubcf4\uad00\ud558\uace0 dense \ud589\ub82c \uacc4\uc0b0\uc744 \ud55c\ub2e4. \ud65c\uc131 rank \uac10\uc18c\ub9cc\uc73c\ub85c \uba54\ubaa8\ub9ac/\uc2dc\uac04 \uc808\uc57d\uc744 \ub73b\ud558\uc9c0 \uc54a\ub294\ub2e4.",
"\uc120\ud0dd \uae30\uc900 \uc2e4\ud5d8\uc740 \uac19\uc740 SP \uccb4\ud06c\ud3ec\uc778\ud2b8\uc5d0\uc11c \ucd94\ub860\ub9cc \ubc14\uafbc \uac83\uc774\ub2e4. \uac01 eta\ub85c \ub2e4\uc2dc \ud559\uc2b5\ud55c \uc2e4\ud5d8\uc740 \uc544\ub2c8\ub2e4.",
"\ud559\uc2b5 \uc18d\ub3c4\ub294 \ub2e8\ub3c5 128\uc7a5\uba74 \uce21\uc815, \ud589\ub3d9 \uc18d\ub3c4\ub294 \ud559\uc2b5/\ub2e4\ub978 \ud3c9\uac00 \uc5c6\uc774 \uac19\uc740 32\uc7a5\uba74\uc5d0\uc11c \uce21\uc815.",
"\uba54\ubaa8\ub9ac\ub294 PyTorch peak allocated, GPU \uc804\uccb4 \uc0ac\uc6a9\ub7c9\uacfc \ub2e4\ub974\ub2e4.",
"\uac19\uc740 \ucd08\uae30 \uc0c1\ud0dc 5\uac1c, \ucd5c\ub300 220\ubc88 \ud589\ub3d9\uc73c\ub85c \ud3c9\uac00. \uc0c8\ub85c\uc6b4 \uc0c1\ud0dc\uc758 \uc131\ub2a5\uc744 \ubcf4\uc7a5\ud558\uc9c0 \uc54a\ub294\ub2e4.",
"",
"\ub17c\ubb38\uc758 \ud575\uc2ec \uaddc\uce59\uc744 OpenVLA\uc5d0 \uc801\uc6a9\ud55c \uc2e4\ud5d8. pi0/SmolVLA/PiPER \uc804\uccb4 \uc7ac\ud604\uc740 \uc544\ub2d8.",
"\uad6c\uccb4\uc801 router \ud06c\uae30/\ud65c\uc131\ud568\uc218/\ucd08\uae30\ud654/\uc815\ubc00\ub3c4\ub294 experiment_config.json\uc5d0 \uae30\ub85d.",
"\ud45c\uc758 \uac80\uc99d loss\ub294 \ubaa8\ub450 task CE. SP \ubcf4\uc870 loss\ub97c \ub354\ud55c total\uacfc \uc77c\ubc18 LoRA CE\ub97c \uc11e\uc5b4 \ube44\uad50\ud558\uc9c0 \uc54a\uc74c.",
"",
"\ucf54\ub4dc \ud750\ub984\uc740 00_CODE_FLOW_KO.txt\ub97c \uba3c\uc800 \uc77d\uae30.",
"REMOTE RUN: "+str(R),
"CODE: train_and_profile.py -> comparison_common.step -> policy.forward -> backward/optimizer.step",
"EVAL: evaluate.py infer/simulate",
"RAW LOCAL LOGS: training_metrics.jsonl, validation_history.json, inference/*.json, simulation/*/state_*/result.json",
"W&B: offline logs only; not uploaded.",
"REPRODUCE (new folder, remote Linux only): /home/jovyan/anaconda3/envs/lerobot312/bin/python "+str(R/"reproduce.py")+" --run",
"Reproduction helper: syntax/--help checked; full helper not rerun. Main reported phases actually executed.",
"",
"\uc624\ub958 \uc218\uc815: \ucd08\uae30 \uc791\uc740 \uac80\uc0ac\uc5d0\uc11c 100% \uacbd\uacc4 FP32 \ubc18\uc62c\ub9bc \ubb38\uc81c\ub97c \ubc1c\uacac\ud574 \uc120\ud0dd \ub204\uc801 \uacc4\uc0b0\uc744 FP64\ub85c \uc548\uc815\ud654.",
"\uac80\uc0ac\uc6a9 deepcopy \uc804\uc5d0 \uc784\uc2dc \uacc4\uc0b0 \uae30\ub85d\uc744 \uc9c0\uc6cc \ud574\uacb0. \uc791\uc740 \uac80\uc0ac \ubc0f \uc2e4\uc81c \ubaa8\ub378 \uc800\uc7a5/\uc7ac\ub85c\ub529 \ud1b5\uacfc \ud6c4 \ubcf8 \ud559\uc2b5 \uc2dc\uc791.",
]
(out/"01_RESULTS_KO.txt").write_text("\n".join(text),encoding="utf-8")
for name in ["00_CODE_FLOW_KO.txt","one_demo_frame.png","one_sample_flow.json","experiment_config.json","code_locations.json","sp_unit_check.json"]:
 (out/name).write_bytes((R/name).read_bytes())
for name in ["sp_policy.py","comparison_common.py","train_and_profile.py","evaluate.py","driver.py","trace_one_sample.py","reproduce.py"]:
 (out/name).write_bytes((R/name).read_bytes())
reference_loop=Path("/home/jovyan/projects/lerobot-openvla/outputs/input_pipeline_check/20261103T055533_073692Z/matched_rollout_function.py")
(out/"simulation_loop_reference.py").write_bytes(reference_loop.read_bytes())
print("REPORT_TABLES_COMPLETE",json.dumps(rows),json.dumps(eta_rows),flush=True)
