# UR5e scripted grasp demonstrations

This folder contains the code used to generate 40 simulated block-grasp demonstrations with a UR5e arm and Robotiq 2F-85 gripper in MuJoCo. A scripted controller chooses the movements. **OpenVLA did not generate these actions and has not been trained on this dataset.**

## What each file does

| File | Purpose |
| --- | --- |
| `ur5e_auto_grasp.py` | Builds the MuJoCo scene and scripted grasp controller used by the other scripts. |
| `record_ur5e_visual_demo.py` | Records one grasp demonstration with camera frames and action values. |
| `profile_ur5e_frames.py` | Measures camera capture speed before collecting many episodes. |
| `collect_ur5e_demos.py` | Runs the controller at 40 fixed block positions and saves one episode per position. |
| `verify_ur5e_demos.py` | Checks the saved episodes and builds `all_samples.csv`. |
| `verification_40_demos.json` | Small report from the completed 40-episode run. |

## Completed run

The block started at 40 distinct positions in a fixed 8-by-5 grid. Each episode produced 90 image/action pairs, for 3,600 pairs in total. The scripted grasp/lift/hold check succeeded in 40/40 episodes. These figures describe **scripted simulation**, not OpenVLA task success. The saved action is six UR5e joint actuator targets plus one gripper target. It still needs conversion and normalization before OpenVLA training. No random seed was used for the fixed position grid.

The full episode images and CSVs are stored outside this Git repository at `C:\Users\User\Documents\Codex\2026-10-30\new-chat\ur5e_demos_40_parallel_20261119`. They are not included in this code commit.

## Reproduce

On Ubuntu, install MuJoCo, NumPy, and Pillow, and place the `universal_robots_ur5e` and `robotiq_2f85` model directories under `~/mujoco_menagerie`. From this folder, run:

```bash
python collect_ur5e_demos.py --output /path/to/new_output --workers 1
python verify_ur5e_demos.py /path/to/new_output
```

`--output` must point to a new directory; the collector refuses to overwrite existing episodes. Use a larger worker count only if your machine can run multiple MuJoCo renderers.

Related OpenVLA one-scene prediction and gripper mapping experiments remain in `openvla_one_scene/ur5e_sample/`.
