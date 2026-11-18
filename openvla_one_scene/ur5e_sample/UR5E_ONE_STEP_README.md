# UR5e one-step OpenVLA bridge check

This folder separates three things:

1. `ur5e_auto_grasp.py`: human-scripted block grasp (does **not** use OpenVLA).
2. `record_ur5e_rollout.py`: repeats that script and saves 0.1 s samples in `trajectory/`.
3. `ur5e_openvla_one_step.py`: applies **one bounded candidate movement** from a previously recorded base OpenVLA prediction, with an identical zero-command control.

`scene.png` was rendered by `ur5e_camera_capture.py`. The `prior_colab_prediction.json` values were transcribed from an earlier Colab output in the conversation; no new inference was run for this step. The seventh token was previously reported outside the nominal action-token range, so this test holds the gripper open. The first six values are treated as a candidate xyz/rotation-vector delta with explicit gains of 0.01 m and 5 degrees per normalized unit. Those gains are **not calibrated to UR5e data**. This test checks software plumbing and a tiny simulated movement; it does not validate grasping, a full seven-value controller, or task success.

Run in Ubuntu with the existing virtual environment and cloned MuJoCo Menagerie:

```bash
source /home/jw/ur5e-venv/bin/activate
cd /mnt/c/Users/User/Desktop/vla-experiments/openvla_one_scene/ur5e_sample
python record_ur5e_rollout.py --seed 7
python ur5e_openvla_one_step.py --prediction prior_colab_prediction.json --output one_step_result.json
```

Read `one_step_result.json` in this order: `requested_ee_delta_m` (candidate target), `actual_ee_delta_m` (what happened in 0.1 s), and `difference_vs_zero_action_control_m` (effect after subtracting natural startup motion). `block_lifted` is only a 5 cm height check for this one step; task success was not evaluated.

The seed is recorded as 7. This scene has fixed initial conditions and no enabled randomization, so changing the seed alone does not create new trials.
