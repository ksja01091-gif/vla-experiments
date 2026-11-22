# π0.5 UR5e experiment

This folder contains the Colab notebook and the recorded result of **one** MuJoCo rollout. It is a record of the completed experiment, not a ready-to-run package for a fresh Colab runtime: the notebook refers to data and model files in `/content` and the research lab Google Drive.

## What was done

1. Collected **40 complete scripted UR5e demonstrations** (3,600 image/action frames) and kept 5 demonstrations (450 frames) for a separate validation split. These demonstrations were generated in our MuJoCo scene, not downloaded from LIBERO.
2. Trained `lerobot/pi05_base` for 3,600 optimizer steps (one pass through the 3,600 training frames), batch size 1, seed 7, BF16, `train_expert_only=true`, `freeze_vision_encoder=true`. `use_peft=false`: this run did **not** use LoRA. The resulting checkpoint lives in the research lab Drive at `VLA_experiments/UR5e_Pi05_40demo_1epoch_seed7/pretrained_model/` (~8.9 GB), outside Git.
3. Reloaded the checkpoint in a fresh Python process and obtained a finite `(1, 7)` action.
4. In one MuJoCo start state, fed a new camera image and robot state to the policy, predicted an action, applied it, and repeated 90 times. Each decision advanced 50 physics steps, or 0.1 simulated seconds.

## Recorded rollout result

| Measure | Result |
|---|---:|
| Start states / episodes | 1 / 1 |
| Action decisions | 90 |
| Simulated time | 9.0 s |
| Wall time for rollout | 31.8 s |
| Gripper-close decisions | 43 |
| First gripper-close decision | 48 |
| Closest recorded gripper-to-block distance | 0.0855 m |
| Initial block height | 0.0500 m |
| Highest block height | 0.0509 m |
| Lift by 5 cm | No |

**Interpretation:** This run confirms that a π0.5 prediction was repeatedly connected to the UR5e MuJoCo controls. The gripper approached the block and later closed, but this one rollout did not lift the block. It is not a success-rate estimate for multiple start states.

## Files

- `Pi05_UR5e_40demo_training.ipynb`: Colab source cells. Saved cell outputs and execution counters were removed before committing; the rollout results remain in separate files below.
- `rollout_state0/actions.csv`: 90 action and robot-state records. One row corresponds to one policy decision.
- `rollout_state0/summary.json`: measured rollout summary.
- `rollout_state0/start.png`, `final.png`: first and last frames.
- `rollout_state0/pi05_ur5e_state0.mp4`: rollout video.

Training images/actions and model weights remain outside Git because of their size. The training command and the closed-loop simulation code are in the notebook. The notebook includes exploratory and troubleshooting cells, so read the final training command and the 90-step rollout cell first.