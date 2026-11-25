# Panda can pick-and-place with MolmoAct2

This study compares 100, 200, and 400 successful scripted demonstrations for LoRA adaptation. One demonstration starts at a reset Panda scene and ends after the can is released and stable in its assigned bin. The scene is XenoArm's robosuite Pick Place scene, including the three other objects. The task instruction names only the can.

The teacher is the centered-grasp controller from the Panda ACT diagnosis. It uses the simulator's object position to align the finger pads, grasp, lift, transfer, and release. The MolmoAct2 input consists of exterior and wrist RGB views, robot state, and the instruction. The object position is used by the demonstration generator and scoring code.

## Collection and checks

- Two 128 × 128 RGB views, robot state, and the OSC command are saved together before each 20 Hz action. The existing XenoArm converter maps the recovered commands to its 7-component action format and builds the 8-component robot state. The seed and simulator-state hash are saved in the manifest.
- The pilot records eight successful trials, reads their stored commands back from the native and LeRobot files, and replays both sets from the corresponding reset states. Bulk collection starts after this gate passes. The datasets are nested in the same collection order, with distinct can start positions at millimeter precision and separate evaluation positions.
- Evaluation uses seeds 30000–30049 across all three models, with up to 1,000 control steps and a fresh prediction after five actions. Metrics report a grasped 4 cm lift held for three steps and placement after release inside the can's bin, stable at under 2 cm/s for ten steps. Retraction is not part of the placement criterion. Exterior and wrist footage is stored side by side with the trial result and step counter.

## Training and outputs

Each condition starts independently from `allenai/MolmoAct2-LIBERO-LeRobot` and uses seed 777, rank-32 LoRA in the VLM and action expert, batch size 4, and 4,000 optimizer steps. The same update budget compares data quantity at fixed compute. It gives larger datasets fewer passes per episode. Use the observed trial rates to interpret the result at this training budget.

On HPC, this directory is staged beside `source`, `full_train_src`, and the previous `convert_full.py`. `collect.py` uses the simulator environment, and `convert.py` uses the existing LeRobot environment and `VideoWriter`. `run.sh` coordinates the data and GPU jobs. W&B records training and assessment in `gist-infonet/xenoarm-proposal` under `MolmoAct2 Panda Can Scripted Data Scale`.

The first completed outputs are `pilot_replay_verified.json` and per-size `conversion_verified.json` files. The final output includes `RESULTS.md`, `results.json`, and the per-seed evaluations. The GPU job waits for earlier scheduled experiments on the shared A100.
