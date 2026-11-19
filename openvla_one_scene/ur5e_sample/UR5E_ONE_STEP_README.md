# UR5e one-step OpenVLA bridge check

This folder separates three things:

`ur5e_with_gripper.py` is the earlier interactive assembly check of the UR5e, Robotiq gripper, and a test block. It is preserved as the first scene-building step.

1. `ur5e_auto_grasp.py`: human-scripted block grasp (does **not** use OpenVLA).
2. `record_ur5e_rollout.py`: repeats that script and saves 0.1 s samples in `trajectory/`.
3. `ur5e_openvla_one_step.py`: applies **one bounded candidate movement** from a previously recorded base OpenVLA prediction, with an identical zero-command control.

`scene.png` was rendered by `ur5e_camera_capture.py`. The `prior_colab_prediction.json` values were transcribed from an earlier Colab output in the conversation; no new inference was run for this step. **Correction:** token 31744 is a valid OpenVLA action token. The earlier out-of-range report counted 255 bin centers instead of 256 action tokens. This test holds the gripper open because the OpenVLA-to-UR5e gripper command mapping is uncalibrated, not because of an invalid token. The first six values are treated as a candidate xyz/rotation-vector delta with explicit gains of 0.01 m and 5 degrees per normalized unit. Those gains are **not calibrated to UR5e data**. This test checks software plumbing and a tiny simulated movement; it does not validate grasping, a full seven-value controller, or task success. See `ur5e_token_audit.json` for the offline check of six saved predictions.

Run in Ubuntu with the existing virtual environment and cloned MuJoCo Menagerie:

```bash
source /home/jw/ur5e-venv/bin/activate
cd /mnt/c/Users/User/Desktop/vla-experiments/openvla_one_scene/ur5e_sample
python record_ur5e_rollout.py --seed 7
python ur5e_openvla_one_step.py --prediction prior_colab_prediction.json --output one_step_result.json
```

Read `one_step_result.json` in this order: `requested_ee_delta_m` (candidate target), `actual_ee_delta_m` (what happened in 0.1 s), and `difference_vs_zero_action_control_m` (effect after subtracting natural startup motion). `block_lifted` is only a 5 cm height check for this one step; task success was not evaluated.

The seed is recorded as 7. This scene has fixed initial conditions and no enabled randomization, so changing the seed alone does not create new trials.

## World-axis and gripper command check

`verify_ur5e_controls.py` runs seven fresh MuJoCo scenes for the arm: one zero-command control and six `+x/-x/+y/-y/+z/-z` requests of 10 mm each. It also runs two fresh scenes for gripper commands 0 and 255. Every case lasts 1 s, keeps the other controls fixed, and uses seed 7. It does **not** call OpenVLA.

```bash
source /home/jw/ur5e-venv/bin/activate
cd /mnt/c/Users/User/Desktop/vla-experiments/openvla_one_scene/ur5e_sample
python verify_ur5e_controls.py --seed 7
```

Read `control_check/arm_directions.csv` for each requested direction. The `baseline_corrected_*_mm` columns subtract the motion seen in the zero-command case, so they isolate the extra movement caused by the command. The coordinates are MuJoCo **world** axes, not left/right/up in the camera image. Read `control_check/control_check.json` for the gripper result: `pad_center_distance_mm` is the distance between two pad **centers**, not a directly measured free opening width. A smaller center distance with command 255 means the pads closed. This checks controller semantics, not grasp success or OpenVLA prediction quality.

## W&B diagnostic record

`upload_ur5e_checks_wandb.py` reads the saved CSV/JSON/image files and creates a **post-hoc diagnostic run** in the separate project `openvla-ur5e-control-diagnostics`. It records the scripted block-height curve, six arm-direction checks, two gripper settings, and the bounded prior-OpenVLA one-step check. It labels task success as **not measured**. It does not train or rerun OpenVLA.

The current offline run ID and exact folder are in `wandb_offline_manifest.json`. The offline run is stored under `/home/jw/wandb_offline_runs/ur5e` and is not visible on the W&B website until an authenticated `wandb sync` succeeds. The raw CSV, JSON, image, and this upload script are committed to Git; the offline run can be recreated from them.
