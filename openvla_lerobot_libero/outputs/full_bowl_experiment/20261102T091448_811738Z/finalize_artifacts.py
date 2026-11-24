"""Validate completed experiment artifacts; create user-readable result and manual W&B sync commands."""
import json,time,traceback,hashlib,shlex
from pathlib import Path
import imageio.v2 as imageio
RUN=Path("/home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z")
ROOT=Path("/home/jovyan/projects/lerobot-openvla")
def main():
    deadline=time.monotonic()+900
    while not (RUN/"pipeline_complete.json").exists():
        if (RUN/"pipeline_error.json").exists():raise RuntimeError((RUN/"pipeline_error.json").read_text())
        if time.monotonic()>deadline:raise TimeoutError("Waiting for remaining experiment phases; no completion claimed")
        time.sleep(5)
    summary=json.loads((RUN/"experiment_summary.json").read_text())
    assert all(summary["completion_evidence"].values())
    assert summary["paired_initial_states_and_initial_images"]
    coverage=json.loads((RUN/"training_coverage.json").read_text())
    assert all(sorted(row["indices"])==list(range(1928)) for row in coverage)
    videos=[]
    for tag in ["base","lora","reference"]:
        cohort=json.loads((RUN/f"simulation_{tag}.json").read_text())
        assert cohort["attempts"]==5 and cohort["technical_errors"]==0
        for trial in cohort["trials"]:
            path=Path(trial["video"])
            reader=imageio.get_reader(str(path))
            try:
                metadata=reader.get_meta_data()
                count=reader.count_frames()
                first=reader.get_data(0)
                last=reader.get_data(count-1)
            finally:reader.close()
            assert count==trial["policy_steps"]+1,(path,count,trial["policy_steps"])
            assert abs(metadata["fps"]-20)<1e-6
            assert first.shape[:2]==last.shape[:2]==(256,256)
            videos.append({"model":tag,"state":trial["init_state_index"],"path":str(path),"frames":count,"fps":metadata["fps"],
                           "first_last_frame_decode":bool(first.size and last.size),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    diagnostics=[json.loads(p.read_text()) for p in (RUN/"simulation").glob("*/state_*/replay_diagnostics.json")]
    diagnosed={r["model_tag"]:r for r in diagnostics if r["init_state_index"]==0}
    (RUN/"motion_diagnostics_final.json").write_text(json.dumps(diagnostics,indent=2))
    for r in diagnostics:
        assert r["maximum_eef_replay_error_m"]<1e-7 and r["success_flags_match_original"]
    commands=["conda activate lerobot312","wandb login"]
    offline_dirs=sorted((RUN/"wandb").glob("offline-run-*"))
    commands += ["wandb sync "+shlex.quote(str(p)) for p in offline_dirs if p.is_dir()]
    (RUN/"WANDB_SYNC_COMMANDS.txt").write_text(chr(10).join(commands)+chr(10))
    check={"pipeline_marker_present":(RUN/"pipeline_complete.json").exists(),"completion_evidence":summary["completion_evidence"],
           "paired_initial_states_and_images":summary["paired_initial_states_and_initial_images"],
           "videos_checked":len(videos),"all_video_frame_counts_match":all(v["frames"]>0 for v in videos),
           "all_video_fps_20":all(v["fps"]==20 for v in videos),"motion_replays_checked":len(diagnostics),
           "offline_wandb_directories":[str(p) for p in offline_dirs],"wandb_upload_performed":False}
    (RUN/"video_verification.json").write_text(json.dumps(videos,indent=2))
    (RUN/"final_verification.json").write_text(json.dumps(check,indent=2))
    train=summary["training"];models=summary["cohorts"]
    lines=["# OpenVLA \uadf8\ub987 \uc62e\uae30\uae30 \ubcf8 \uc2e4\ud5d8 \uacb0\uacfc","",
           "\uacf5\uac1c LIBERO \ub370\uc774\ud130\uc5d0\uc11c \uc120\ud0dd\u00b7\uc804\ucc98\ub9ac\ud55c \uc2e4\ud5d8\uc6a9 \ubd84\ud560\ub85c \uc2e4\ud589\ud55c \uacb0\uacfc\uc785\ub2c8\ub2e4.",
           "\uc2dc\ubc94 20\uac1c \uc804\uccb4\uc758 1,928\uc7a5\uba74\uc744 \ud55c \ubc88\uc529 \ud559\uc2b5\ud588\uace0, \ubcc4\ub3c4 \uc2dc\ubc94 5\uac1c\uc758 462\uc7a5\uba74 \uc804\uccb4\ub97c \uac80\uc99d\ud588\uc2b5\ub2c8\ub2e4.","",
           f"\uc124\uc815: 1 epoch, batch 4, \ud559\uc2b5\ub960 0.0001, LoRA rank 8 (q_proj/v_proj), {train['optimizer_steps']}\ud68c \uc5c5\ub370\uc774\ud2b8.",
           f"\ud559\uc2b5 \ub8e8\ud504 \uc2dc\uac04(\uc911\uac04\u00b7\ub9c8\uc9c0\ub9c9 \uac80\uc99d \ud3ec\ud568): {train['training_loop_wall_seconds']/60:.2f}\ubd84. GPU \ucd5c\ub300 allocated: {train['gpu_peak_allocated_gib']:.2f} GiB.","",
           "| \uc804\uccb4 \uac80\uc99d \uc9c0\ud45c | \ud559\uc2b5 \uc804 | \ud559\uc2b5 \ud6c4 |","|---|---:|---:|",
           f"| loss | {train['baseline_validation']['loss']:.4f} | {train['final_validation']['loss']:.4f} |",
           f"| \ud589\ub3d9 \ud1a0\ud070 \uc815\ud655\ub3c4 | {train['baseline_validation']['action_token_accuracy']*100:.2f}% | {train['final_validation']['action_token_accuracy']*100:.2f}% |",
           f"| \uc815\uaddc\ud654 \ud589\ub3d9 L1 | {train['baseline_validation']['normalized_action_l1']:.4f} | {train['final_validation']['normalized_action_l1']:.4f} |","",
           "| \ubaa8\ub378 | \uc131\uacf5/\uc2dc\ub3c4 | \uc131\uacf5\ub960 | \ud3c9\uade0 episode step |","|---|---:|---:|---:|"]
    labels={"base":"\uae30\ubcf8 OpenVLA","lora":"\uc774\ubc88 LoRA \ud559\uc2b5 \ud6c4","reference":"\uacf5\uac1c Spatial \ubbf8\uc138\uc870\uc815 \ubaa8\ub378 (\ucc38\uace0)"}
    for tag in ["base","lora","reference"]:
        c=models[tag];lines.append(f"| {labels[tag]} | {c['successes']}/{c['attempts']} | {c['success_rate']*100:.1f}% | {c['mean_policy_steps']:.1f} |")
    lines += ["","\uac80\uc99d \uc624\ucc28 \uac10\uc18c\uc640 \uc2dc\ubbac\ub808\uc774\ud130 \uc131\uacf5\ub960\uc740 \ubcc4\uac1c\uc758 \uacb0\uacfc\uc785\ub2c8\ub2e4. \uc774\ubc88 \uc124\uc815\uc5d0\uc11c \uc2e4\uc81c\ub85c \ud655\uc778\ud55c \uacb0\uacfc\ub9cc \uc704\uc5d0 \uae30\ub85d\ud588\uc2b5\ub2c8\ub2e4.",
              "\uc138 \ubaa8\ub378\uc758 \ucd08\uae30 \uc0c1\ud0dc\u00b7\ucd08\uae30 \uc774\ubbf8\uc9c0 \uc77c\uce58\ub97c \ud655\uc778\ud588\uace0 \uae30\uc220\uc801 \uc624\ub958\ub294 \uc5c6\uc5c8\uc2b5\ub2c8\ub2e4. \ucd08\uae30 \uc0c1\ud0dc 0~4, seed 7, \ud658\uacbd seed 0, \uc900\ube44 10\ud68c, \ucd5c\ub300 220\ud68c \ud589\ub3d9\uc785\ub2c8\ub2e4.",
              "\uae30\ubcf8 \ubaa8\ub378\uacfc LoRA\ub294 \uac19\uc740 \ud559\uc2b5\uc6a9 \ud589\ub3d9 \uc815\uaddc\ud654 \ud1b5\uacc4\ub97c \uc0ac\uc6a9\ud588\uc2b5\ub2c8\ub2e4. \uacf5\uac1c Spatial \ubaa8\ub378\uc758 \uacb0\uacfc\ub97c \uae30\ubcf8 \ubaa8\ub378 \uacb0\uacfc\ub85c \uc11e\uc9c0 \uc54a\uc558\uc2b5\ub2c8\ub2e4.","",
              "## \uc2e4\ud328\uc758 \uad00\ucc30 \uac00\ub2a5\ud55c \uc591\uc0c1",""]
    for tag in ["base","lora"]:
        if tag in diagnosed:
            r=diagnosed[tag]
            lines.append(f"- {labels[tag]}, \ub300\ud45c \ucd08\uae30 \uc0c1\ud0dc 0: \uc9d1\uac8c\uc640 \ubaa9\ud45c \uadf8\ub987 \uc911\uc2ec\uc758 \ucd5c\ub2e8\uac70\ub9ac {r['minimum_eef_to_bowl_center_distance_m']*100:.1f}cm, \uadf8\ub987 \ucd5c\ub300 \ub192\uc774 \uc99d\uac00 {r['maximum_bowl_height_increase_m']*1000:.3f}mm.")
    lines += ["\uc7ac\uc0dd\ud55c \uc81c\uc5b4 \uba85\ub839\uc758 \ub85c\ubd07 \uada4\uc801\uacfc \uc131\uacf5 \ud310\uc815\uc740 \uc6d0\ub798 \uae30\ub85d\uacfc \uc77c\uce58\ud588\uc2b5\ub2c8\ub2e4. \uc774 \uc218\uce58\ub294 \ubb3c\uccb4 \uc6c0\uc9c1\uc784\uc758 \uad00\ucc30\uc774\uba70, \uc2e0\uacbd\ub9dd\uc758 \uc5b4\ub5a4 \ubd80\ubd84\uc774 \uadfc\ubcf8 \uc6d0\uc778\uc778\uc9c0\ub294 \ud655\uc815\ud558\uc9c0 \uc54a\uc558\uc2b5\ub2c8\ub2e4.",
              "\ub300\ud45c \uc7a5\uba74\uc5d0\uc11c\ub294 \ud559\uc2b5 \ud6c4 \uc9d1\uac8c\uac00 \uadf8\ub987 \uadfc\ucc98\uc5d0\uc11c \uc811\uc2dc \ucabd\uc73c\ub85c \uc6c0\uc9c1\uc5ec\ub3c4 \uadf8\ub987\uc774 \ud14c\uc774\ube14\uc5d0 \ub0a8\ub294 \ubaa8\uc2b5\uc774 \ubcf4\uc785\ub2c8\ub2e4.","",
              "## \uc800\uc7a5\uacfc \ud655\uc778","",
              f"- \uc120\ud0dd checkpoint: {train['selected_checkpoint']}",
              f"- \uc0c8 Python \ud504\ub85c\uc138\uc2a4\uc758 \uac00\uc911\uce58 \uc7ac\ub85c\ub529\u00b7\ub3d9\uc77c \uc785\ub825 loss \ub300\uc870\u00b7\uc720\ud6a8\ud55c (1,7) \ud589\ub3d9 \ucd9c\ub825 \ud655\uc778.",
              f"- \uc601\uc0c1 {len(videos)}\uac1c\ub97c \uc9c1\uc811 \uc5f4\uc5b4 20 FPS, \uccab\u00b7\ub9c8\uc9c0\ub9c9 \ud504\ub808\uc784 \ubc0f \uc2e4\uc81c \ud504\ub808\uc784 \uc218\ub97c \ud655\uc778.",
              "- \uae30\uc874 Python \uc18c\uc2a4\uc640 openvla310\u00b7lerobot312\u00b7eval310\uc758 \ud328\ud0a4\uc9c0 \ubaa9\ub85d\uc744 \ub300\uc870\ud588\uc2b5\ub2c8\ub2e4.",
              "- W&B\ub294 \uc778\uc99d \uc815\ubcf4\uac00 \uc5c6\uc5b4 offline \uae30\ub85d\uc744 \uc900\ube44\ud588\uc73c\uba70 \uc5c5\ub85c\ub4dc\ud558\uc9c0 \uc54a\uc558\uc2b5\ub2c8\ub2e4. WANDB_SYNC_COMMANDS.txt\ub97c \ub85c\uadf8\uc778 \ud6c4 \uc9c1\uc811 \uc2e4\ud589\ud558\uba74 \ub429\ub2c8\ub2e4.","",
              "## \ub2e4\uc2dc \uc2e4\ud589","",
              "```bash","conda activate lerobot312",
              "python /home/jovyan/projects/lerobot-openvla/examples/run_openvla_bowl_full.py","```",
              "\uc0c8 \ucd9c\ub825 \ud3f4\ub354\uc5d0 \uc0c8\ub85c\uc6b4 \uc2e4\ud5d8\uc744 \uc2e4\ud589\ud558\ub294 \uba85\ub839\uc785\ub2c8\ub2e4. \uae30\uc874 \uacb0\uacfc\ub294 \ub36e\uc5b4\uc4f0\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4.","",
              "```bash",
              f"python /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl_experiment.py check --run {RUN}","```",
              "\uc704 \uba85\ub839\uc740 \uc774\ubc88 \uc800\uc7a5 checkpoint\ub97c \uc0c8 \ud504\ub85c\uc138\uc2a4\uc5d0\uc11c \ub2e4\uc2dc \ud655\uc778\ud569\ub2c8\ub2e4.","",
              "## \ud575\uc2ec \ud30c\uc77c\uacfc \uac1c\ub150","",
              "- /home/jovyan/projects/lerobot-openvla/src/lerobot/policies/openvla/full_experiment_openvla.py: \ubc30\uce58 \uad6c\uc131, forward, LoRA \uc800\uc7a5\u00b7\uc7ac\ub85c\ub529.",
              "- /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl_experiment.py: \uc804\uccb4 \ud559\uc2b5, \uc804\uccb4 \uac80\uc99d, \uac19\uc740 \uc870\uac74\uc758 \uc2dc\ubbac\ub808\uc774\uc158.",
              f"- \uc804\uccb4 \uacb0\uacfc: {RUN}",
              f"- \uc218\uce58\uc640 \uc6d0\ubcf8 \ub85c\uadf8: {RUN}/experiment_summary.json, training_metrics.csv, validation_metrics.csv, simulation_summary.csv.",
              f"- \uacb0\uacfc \uadf8\ub798\ud504: {RUN}/experiment_summary.png",
              f"- \ucd5c\uc885 \uc601\uc0c1 \uc7a5\uba74: {RUN}/rollout_endpoints.png","",
              "epoch\ub294 \ud559\uc2b5 \uc7a5\uba74 \uc804\uccb4\ub97c \ud55c \ubc88 \uc0ac\uc6a9\ud558\ub294 \ub2e8\uc704\uc785\ub2c8\ub2e4. loss\ub294 \uc2dc\ubc94 \ud589\ub3d9 \uc608\uce21 \uc624\ucc28\uc774\uace0 \uc131\uacf5\ub960\uc740 \uc2e4\uc81c \uc81c\uc5b4 \uc2dc\ub3c4\uc758 \ubaa9\ud45c \ub2ec\uc131 \ube44\uc728\uc785\ub2c8\ub2e4.","",
              "## \ub0a8\uc740 \uc77c\uacfc \ud574\uc11d \ubc94\uc704","",
              "W&B \uc628\ub77c\uc778 \uc5f0\uacb0\uc5d0\ub294 \uc0ac\uc6a9\uc790 \ub85c\uadf8\uc778\uc774 \ud544\uc694\ud569\ub2c8\ub2e4. \uc2e4\ub825 \uac1c\uc120\uc744 \uc8fc\uc7a5\ud558\ub824\uba74 \ub2e4\uc74c \uc124\uc815 \uc2e4\ud5d8\uacfc \ub354 \ub9ce\uc740 \ucd08\uae30 \uc0c1\ud0dc \ud3c9\uac00\uac00 \ud544\uc694\ud569\ub2c8\ub2e4.",
              "\uc774\ubc88\uc740 \ud55c \uc791\uc5c5\u00b7\uc791\uc740 \uacf5\uac1c \ub370\uc774\ud130 \ubd84\ud560\u00b71 epoch\uc758 \ucd5c\uc18c \ubca0\uc774\uc2a4\ub77c\uc778\uc785\ub2c8\ub2e4. \uc804\uccb4 LIBERO \ub17c\ubb38 \uc218\uce58 \uc7ac\ud604\uc774\ub098 \ud1b5\uacc4\uc801\uc73c\ub85c \ud655\uc815\ub41c \uc77c\ubc18\ud654 \uc131\ub2a5\uc774\ub77c\uace0 \ud574\uc11d\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4.",
              "\ud559\uc2b5 \uc0ac\uc9c4 \ucc98\ub9ac\ub294 \uae30\uc874 HF processor\ub97c \uc720\uc9c0\ud588\uc2b5\ub2c8\ub2e4. \uacf5\uc2dd RLDS \uc804\uccb4 \ud559\uc2b5 \ub808\uc2dc\ud53c\uc758 \uc7ac\ud604\uacfc\ub294 \uad6c\ubd84\ud569\ub2c8\ub2e4.",""]
    text=chr(10).join(lines)
    (RUN/"FINAL_RESULTS_KO.md").write_text(text)
    (ROOT/"OPENVLA_BOWL_FULL_RESULT.md").write_text(text)
    print("FINAL_ARTIFACT_CHECK_COMPLETE",json.dumps(check),flush=True)
    print("FINAL_RESULT",ROOT/"OPENVLA_BOWL_FULL_RESULT.md",flush=True)
try:main()
except Exception as error:
    (RUN/"finalization_error.json").write_text(json.dumps({"error":repr(error),"traceback":traceback.format_exc()},indent=2))
    raise
