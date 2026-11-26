# MolmoAct-7B-D-0812 — Panda and UR5e LoRA pilot

This experiment uses Ai2's `allenai/MolmoAct-7B-D-0812` with the MimicGen Pick Place D0 task. Each robot uses its own rank-8 action-only LoRA adapter.

| Step | Entry point |
| --- | --- |
| Download the public Panda and UR5e demonstrations | `download_mimicgen_pair.py` |
| Download the model | `download_molmoact_checkpoint.py` |
| Build the training and validation manifests | `build_pickplace_sample_manifest.py` |
| Compute action statistics on training episodes | `train_action_stats.py` |
| Inspect input images and action tokens | `molmoact_train_preflight.py` |
| Train a separate adapter for each robot | `molmoact_lora_pilot.py` |
| Run the simulator | `molmoact_pickplace_rollout.py` |

`data_audit.json` records the episode IDs from each public dataset and the split used by this experiment. Each robot starts from 500 complete demonstrations, split into 400 training, 50 validation and 50 test demonstrations. The pilot uses 10 sampled decision points per training demonstration with a target of 8 consecutive actions at each point.

After preparing the dataset, model and statistics, run the following from this directory in the MolmoAct inference and training environment. The rollout also needs robosuite and the MimicGen checkout at `mimicgen/`.

```bash
python molmoact_lora_pilot.py --robot panda --steps 4000 --run-name panda_full1 --save-interval 1000
python molmoact_lora_pilot.py --robot ur5e --steps 4000 --run-name ur5e_full1 --save-interval 1000
python molmoact_pickplace_rollout.py --help
```

The full-length trained pilot completed the task in 0 of 1 recorded rollout per robot. This is a connection and training pilot with limited evaluation coverage.
