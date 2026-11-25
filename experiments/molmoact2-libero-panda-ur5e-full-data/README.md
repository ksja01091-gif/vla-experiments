# MolmoAct2-LIBERO-LeRobot — Panda and UR5e full demonstration LoRA

This experiment uses `allenai/MolmoAct2-LIBERO-LeRobot` and the XenoArm four-object `pp4` task. The existing 7B experiment's public MimicGen D0 split supplies 400 training, 50 validation and 50 test episodes per robot. The converter retains every decision point from the selected episodes.

| Entry point | Purpose |
| --- | --- |
| `convert_full.py` | Verifies split membership, camera alignment, robot state and action conventions, then writes XenoArm's MolmoAct2 LeRobot v3 layout |
| `run_full_data.sh` | Converts each split and reopens the dataset to verify the first and last frames |
| `prepare_full_train.py` | Creates an isolated copy of XenoArm's training code, adapts the LeRobot import and video backend, retains the last batch and enables intermediate saves |
| `run_full_lora.sh` | Checks both robots with three GPU updates each, then trains one pass through each verified 400-episode training split |

Use a work directory containing these scripts and a XenoArm checkout at `source/`. The HPC run uses XenoArm `5f1efe4` with the MolmoAct2 configuration loader fix `bfe697f`. Prepare the simulation environment at `source/.venv` and the MolmoAct2 environment at `source/.external/lerobot/.venv`. Install LeRobot's `dataset` extra and verify that PyTorch can use the server GPU. The HPC environment uses Python 3.12 and CUDA 12.8 PyTorch.

Reuse the datasets and `data_audit.json` from the 7B experiment directory. Run the commands below from the work directory.

```bash
export MIMICGEN_SOURCE=/path/to/molmoact-7b-experiment
bash run_full_data.sh > full_data.log 2>&1 &
conversion_pid=$!
source/.venv/bin/python prepare_full_train.py
bash run_full_lora.sh "$conversion_pid" > full_lora.log 2>&1
```

The training launch uses rank 32, learning rate `5e-5`, seed 777 and batch size 4, subject to the GPU check. Both the VLM and action expert attention and feed-forward projections receive LoRA adapters. The large data run has its own output folders. The full runs and their evaluation are pending.
