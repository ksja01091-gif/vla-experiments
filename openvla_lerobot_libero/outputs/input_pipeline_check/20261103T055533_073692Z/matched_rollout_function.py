def simulate_trial(policy,tag,state_index,args):
    horizon,warmup=reference_settings();set_reference_seed()
    output=args.run/"simulation"/tag/f"state_{state_index:02d}"
    output.mkdir(parents=True,exist_ok=False)
    environment=os.environ.copy()
    environment.update({"MUJOCO_GL":"egl","TF_CPP_MIN_LOG_LEVEL":"2","TF_NUM_INTRAOP_THREADS":"2","TF_NUM_INTEROP_THREADS":"1"})
    environment["PYTHONPATH"]="/home/jovyan/projects/LIBERO"+(os.pathsep+environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    worker=ROOT/"examples/libero_eval_worker.py"
    process=None;records=[];initial=None;current=None;started=time.perf_counter()
    with (output/"simulator.log").open("w") as log:
        try:
            process=subprocess.Popen(["/home/jovyan/projects/venvs/openvla_eval310/bin/python","-u",str(worker),
                "--output",str(output),"--task-id","0","--init-state-index",str(state_index)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=environment)
            initial=current=read_observation(process,timeout=240)
            assert initial["type"]=="ready" and initial["task"]==TASK
            assert initial["actual_initial_image_reference_check"]["max_pixel_error"]==0
            write_json(output/"initialization.json",initial)
            initial_hash=hashlib.sha256(Path(initial["frame"]).read_bytes()).hexdigest()
            baseline_file=args.run/"simulation/base"/f"state_{state_index:02d}"/"result.json"
            if tag!="base" and baseline_file.exists():
                baseline=json.loads(baseline_file.read_text())
                if baseline.get("trial_executed"):
                    assert baseline["selected_init_state_sha256"]==initial["selected_init_state_sha256"]
                    assert baseline["initial_image_sha256"]==initial_hash
            low,high=np.asarray(initial["action_low"]),np.asarray(initial["action_high"])
            frames=[initial["frame"]];policy.reset()
            for step in range(1,horizon+1):
                with Image.open(current["frame"]) as source:image=source.convert("RGB")
                pixels=np.asarray(image).copy()
                tensor=torch.from_numpy(pixels).permute(2,0,1).float().div(255)
                assert np.array_equal(np.rint(tensor.permute(1,2,0).numpy()*255).astype(np.uint8),pixels)
                action=policy.select_action({policy.config.image_key:tensor.unsqueeze(0),"task":[TASK]})
                assert action.shape==(1,7) and bool(torch.isfinite(action).all())
                raw=action[0].cpu().numpy()
                converted,command=controller_action(raw,low,high)
                assert np.array_equal(converted,command),"Action would require clipping"
                before=current;send(process,{"type":"step","action":command.tolist()});current=read_observation(process)
                assert current["type"]=="observation" and current["step"]==step
                row={"step":step,"raw_action":raw.tolist(),"controller_action":command.tolist(),"before":before,"after":current}
                records.append(row);append_json(output/"steps.jsonl",row);frames.append(current["frame"])
                if step==1 or step%25==0 or current["task_success"]:
                    print("SIM_STEP",tag,state_index,step,"success",current["task_success"],flush=True)
                if current["task_success"]:break
            success=bool(current["task_success"])
            save_media(frames,output,success)
            # The old media helper labels state 0; correct that label in this new output only.
            comparison=Image.open(output/"before_after.png").convert("RGB")
            draw=ImageDraw.Draw(comparison);draw.rectangle((0,0,255,23),fill="white")
            draw.text((6,6),f"Initial state {state_index}",fill="black");comparison.save(output/"before_after.png")
            report={"trial_executed":True,"model_tag":tag,"task":TASK,"task_id":0,"init_state_index":state_index,
                "success":success,"termination_reason":"task_success" if success else "horizon_exhausted",
                "failure_detail":None if success else "LIBERO success predicate not reached within the fixed 220-action horizon; see video for motion-level diagnosis",
                "policy_steps":len(records),"max_policy_steps":horizon,"warmup_steps":warmup,
                "environment_seed":0,"global_seed":7,"action_shape":[1,7],"all_actions_finite":True,"action_clipping_used":False,
                "actual_policy_preprocessing":"raw upright 256px RGB to native HF processor, exactly matching training","worker_original_reference_check_passed":True,"initial_image_sha256":initial_hash,
                "selected_init_state_sha256":initial["selected_init_state_sha256"],"init_state_file_sha256":initial["init_state_file_sha256"],
                "reference_sha256":initial["reference_sha256"],"simulator_versions":initial["simulator_versions"],
                "initial_eef_position":initial["eef_position"],"final_eef_position":current["eef_position"],
                "wall_seconds":time.perf_counter()-started,"video":str(output/"episode.mp4"),"overview":str(output/"episode_overview.png")}
        except Exception as error:
            report={"trial_executed":False,"model_tag":tag,"init_state_index":state_index,"success":False,
                    "termination_reason":"technical_error","error":repr(error),"policy_steps":len(records),"wall_seconds":time.perf_counter()-started}
            print("SIM_TECHNICAL_ERROR",tag,state_index,repr(error),flush=True)
        finally:
            if process is not None and process.poll() is None:
                try:send(process,{"type":"close"});process.wait(timeout=15)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
    write_json(output/"result.json",report)
    print("SIM_TRIAL_COMPLETE",json.dumps(report),flush=True)
    return report