from pathlib import Path
import json,csv,hashlib,subprocess,sys
RUN=Path(__file__).parent
OLD=Path("/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z")
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
assert read(RUN/"pipeline_status.json")["status"]=="complete"
out=RUN/"viewing_bundle";out.mkdir(exist_ok=True)
b=[read(RUN/f"batch_{n}.json") for n in [4,8]]
assert b[0]["starting_adapter_hash"]==b[1]["starting_adapter_hash"]
assert b[0]["same_measured_frame_indices"]==b[1]["same_measured_frame_indices"]
i=[read(RUN/f"inference_{n}.json") for n in ["bf16","int8"]]
s=[read(OLD/"simulation_epoch5.json"),read(RUN/"simulation_int8.json")]
assert read(RUN/"preservation_report.json")["changed_files"]==[]
summary={"batch_size_4":{k:b[0][k] for k in ["wall_seconds","frames_per_second","peak_allocated_gib","estimated_five_epoch_train_minutes_excluding_validation"]},
"batch_size_8":{k:b[1][k] for k in ["wall_seconds","frames_per_second","peak_allocated_gib","estimated_five_epoch_train_minutes_excluding_validation"]},
"batch8_throughput_ratio":b[1]["frames_per_second"]/b[0]["frames_per_second"],
"batch_measurement_caveat":"128 measured training frames after 16 warmup frames, one run per batch; not full retraining or quality assessment",
"inference":{r["mode"]:{k:r[k] for k in ["mean_action_seconds","median_action_seconds","allocated_gib","peak_allocated_gib","peak_reserved_gib","model_tensor_footprint_bytes","quantized_linear_count","sample_count","all_actions_finite"]} for r in i},
"int8_peak_allocated_reduction_fraction":1-i[1]["peak_allocated_gib"]/i[0]["peak_allocated_gib"],
"int8_latency_ratio":i[1]["mean_action_seconds"]/i[0]["mean_action_seconds"],
"simulation":{tag:{"successes":r["successes"],"attempts":r["attempts"],"success_rate":r["success_rate"],"mean_policy_steps":r["mean_policy_steps"]} for tag,r in zip(["bf16","int8"],s)},
"quantization_scope":i[1]["scope"],"base_files_compressed_or_overwritten":False,
"parameter_count_reduction":False,"adapter_checkpoint_unchanged":True,
"baseline_simulation_reused_with_initial_image_state_reference_version_checks":True,
"generalization":"same five previously used initial states; not an independent held-out simulator test"}
write(RUN/"summary.json",summary);write(out/"summary.json",summary)
with (out/"batch_comparison.csv").open("w",newline="") as f:
 keys=["batch_size","measured_frames","optimizer_updates","wall_seconds","frames_per_second","peak_allocated_gib","estimated_five_epoch_train_minutes_excluding_validation"]
 w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in b)
with (out/"inference_comparison.csv").open("w",newline="") as f:
 keys=["mode","sample_count","mean_action_seconds","peak_allocated_gib","peak_reserved_gib","model_tensor_footprint_bytes","quantized_linear_count"]
 w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in i)
with (out/"simulation_comparison.csv").open("w",newline="") as f:
 keys=["mode","state","success","steps","termination_reason","failure_detail"]
 w=csv.DictWriter(f,keys);w.writeheader()
 for tag,r in zip(["bf16","int8"],s):
  for t in r["trials"]:w.writerow({"mode":tag,"state":t["init_state_index"],"success":t["success"],"steps":t["policy_steps"],
    "termination_reason":t["termination_reason"],"failure_detail":t.get("failure_detail")})
lines=[
"\uac19\uc740 5 epoch \ubaa8\ub378: batch \uc18d\ub3c4 \ube44\uad50 + 8\ube44\ud2b8 \ud3c9\uac00",
"",
"1. Batch size \ube44\uad50",
"   \uac01\uac01 \uac19\uc740 \ud559\uc2b5 \uc7a5\uba74 128\uac1c. 16\uac1c\ub294 \uc900\ube44 \uc6a9\ub3c4. \ubcf8 \ubaa8\ub378\uc5d0 \ubcc0\uacbd\uc744 \uc800\uc7a5\ud558\uc9c0 \uc54a\uc74c.",
f"   batch 4: {b[0]['frames_per_second']:.2f} frames/s, peak {b[0]['peak_allocated_gib']:.2f} GiB",
f"   batch 8: {b[1]['frames_per_second']:.2f} frames/s, peak {b[1]['peak_allocated_gib']:.2f} GiB",
f"   \ucc98\ub9ac\ub7c9 \ube44\uc728: {summary['batch8_throughput_ratio']:.3f}x",
"   \uc18d\ub3c4 \uc2e4\ud5d8\uc774\uba70 batch 8 \ucd5c\uc885 \uc131\ub2a5\uc740 \ud655\uc778\ud558\uc9c0 \uc54a\uc74c.",
"",
"2. \ubaa8\ub378 \uc2e4\ud589 \ube44\uad50",
"   \uac19\uc740 5 epoch LoRA. \uc5b8\uc5b4\ubaa8\ub378 \uc8fc\uc694 \uc120\ud615\uce35\ub9cc INT8. \uc0ac\uc9c4 \ucc98\ub9ac/\uc5f0\uacb0\uce35/lm_head\ub294 BF16, LoRA FP32.",
f"   BF16: peak {i[0]['peak_allocated_gib']:.2f} GiB, mean action {i[0]['mean_action_seconds']*1000:.1f} ms",
f"   INT8: peak {i[1]['peak_allocated_gib']:.2f} GiB, mean action {i[1]['mean_action_seconds']*1000:.1f} ms",
f"   BF16 success {s[0]['successes']}/5; INT8 success {s[1]['successes']}/5",
"   \uac19\uc740 \ucd08\uae30 \uc0c1\ud0dc 5\uac1c, \ucd5c\ub300 220 step. \ub3c5\ub9bd\uc801 \uc77c\ubc18\ud654 \uac80\uc99d\uc740 \uc544\ub2d8.",
"   \uc608\uce21 \uc2dc\uac04\uc740 \uc0ac\uc9c4 \uc804\ucc98\ub9ac \ud3ec\ud568. \uc2dc\ubbac\ub808\uc774\ud130 \uc2dc\uac04 \uc81c\uc678.",
"   \uba54\ubaa8\ub9ac\ub294 PyTorch peak allocated. reserved/GPU \uc804\uccb4 \uc0ac\uc6a9\ub7c9\uacfc \ub2e4\ub984.",
"   \uc6d0\ub798 15.08 GB \ud30c\uc77c\uc740 \ubcf4\uc874. \uc774\ubc88\uc740 \ub85c\ub529 \uc2dc \uc591\uc790\ud654, \ud30c\uc77c \uc555\ucd95 \uacb0\uacfc\ub85c \ud574\uc11d\ud558\uc9c0 \uc54a\uc74c.",
"",
"3. \ubcf4\uc874 \ud655\uc778",
"   \uae30\uc874 \uc18c\uc2a4/\uccb4\ud06c\ud3ec\uc778\ud2b8 \ud574\uc2dc \ub3d9\uc77c, openvla310/lerobot312/eval310 pip freeze \ub3d9\uc77c.",
"   bitsandbytes 0.45.5\ub294 \uc774 \uc2e4\ud5d8 deps \ud3f4\ub354\uc5d0\ub9cc \uc124\uce58.",
"",
"REMOTE FOLDER: "+str(RUN),
"CODE: experiment.py",
"  line 33 batch_benchmark: same 5-epoch start, disposable training speed test.",
"  line 68 make_inference: load same model as BF16 or quantized language layers + unchanged LoRA.",
"  line 125 infer: 32-image output/timing/memory check, then INT8 simulator trials.",
"  line 182 audit: original source/checkpoint hashes and three environments.",
"  Quantization uses the ORIGINAL saved epoch-5 adapter, not the disposable batch-8 updates.",
"RE-RUN BATCH ONLY (new output folder): /home/jovyan/anaconda3/envs/lerobot312/bin/python "+str(RUN/"reproduce.py")+" --batch-only",
"RE-RUN FULL (new output folder): /home/jovyan/anaconda3/envs/lerobot312/bin/python "+str(RUN/"reproduce.py"),
"reproduce.py: syntax and --help checked; original experiment stages actually executed. Full fresh-output wrapper has not been rerun.",
"",
"4. Errors and fixes",
"Transformers dispatched quantized model through forbidden .to(): enabled Accelerate force_hooks only during this model load.",
"PEFT read removed memory_efficient_backward metadata: added unused legacy flag False to this process only.",
"Both handled in experiment.py; package files and original sources unchanged.",
"ERROR RECORDS: infer_mode_int8.log; infer_mode_int8_retry1.log; successful retry2 in infer_mode_int8_retry2.log.",
"",
"5. Concepts",
"Batch size: frames grouped before one optimizer update; affects training throughput and update count.",
"Quantization: store/calculate main weights with fewer bits; parameter count stays the same.",
"Here memory saving and prediction latency are different outcomes. Simulation success is a third outcome.",
]
(out/"00_READ_ME.txt").write_text("\n".join(lines),encoding="utf-8")
print("REPORT_JSON_COMPLETE",json.dumps(summary),flush=True)
