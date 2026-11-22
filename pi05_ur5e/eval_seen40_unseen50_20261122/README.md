# UR5e π0.5: 40 seen and 50 new starts

This is a completed MuJoCo evaluation of the π0.5 model trained on 40 scripted UR5e demonstrations. Each episode starts with the block at one position and runs 90 policy decisions; every decision advances 50 physics steps.

| Start positions | Lifted block by at least 5 cm | Rate |
|---|---:|---:|
| 40 positions used for training demonstrations | 1 / 40 | 2.5% |
| 50 new positions in the same sampled region | 0 / 50 | 0% |

The five held-out validation demonstration positions were excluded. The 50 new positions test new locations within the sampled region, not new tasks or robots. **A 5 cm lift is the score here; this is not a full pick-and-place success rate.** One observed lift happened at `seen/34/rollout.mp4`.

- `manifest.json`: exact positions, random seeds, episode length, and score definition.
- `report.json`: split-level counts and mean minimum gripper-to-block distance.
- `all_90_trials.csv`: one row per episode, convenient for spreadsheets.
- `seen/00` through `seen/39` and `new/00` through `new/49`: each contains `summary.json`, `actions.csv`, `start.png`, `final.png`, and `rollout.mp4`.
- `../evaluate_pi05_ur5e_90.py`: evaluator source; it requires a Colab session where the model, simulator, and preprocessors are already loaded.

[W&B evaluation run](https://forge.coreweave.com/wandb/ksja01091-gwangju-institute-of-science-and-technology/pi05-ur5e-mujoco/runs/pi05ur5e90nov22) contains the aggregate metrics, a 90-row table, example videos, and an artifact with every evaluation file.
