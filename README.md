# VLA experiments

This repository records the UR5e MuJoCo demonstration collection, the π0.5 UR5e experiment, and a separate π0.5 Panda training connection test. The 40 scripted UR5e demonstrations were generated in our simulation, not taken from LIBERO. The Panda connection test uses public MimicGen demonstrations.

## Start here

- [`ur5e_scripted_demos/README.md`](ur5e_scripted_demos/README.md): how the 40 demonstrations were collected and verified.
- [`UR5e_Pi05_학습전_과정_설명.md`](UR5e_Pi05_학습전_과정_설명.md): Korean explanation of the simulation and data preparation before training.
- [`pi05_ur5e/README.md`](pi05_ur5e/README.md): π0.5 training settings and the first rollout and the later 90-position evaluation.
- [`pi05_ur5e/Pi05_UR5e_40demo_training.ipynb`](pi05_ur5e/Pi05_UR5e_40demo_training.ipynb): Colab notebook source.
- [`pi05_panda_mimicgen_d0_l4_2step_smoke_20261123/README.md`](pi05_panda_mimicgen_d0_l4_2step_smoke_20261123/README.md): Panda 공개 시범으로 π0.5 학습이 Colab L4에서 두 번 업데이트되는지 확인한 시험. 로봇 성공률 평가는 아직 하지 않았습니다.

The demonstration images/actions and the trained model weights are stored outside Git. The π0.5 checkpoint is in the research lab Drive. Earlier OpenVLA prediction experiments were removed from this repository.
## Latest result

The π0.5 UR5e model lifted the block by at least 5 cm in **1/40 training-grid starts** and **0/50 new starts**. This is a lift test, not a completed pick-and-place test. Read [`pi05_ur5e/eval_seen40_unseen50_20261122/README.md`](pi05_ur5e/eval_seen40_unseen50_20261122/README.md) for all episodes and videos, or see the [W&B evaluation dashboard](https://forge.coreweave.com/wandb/ksja01091-gwangju-institute-of-science-and-technology/pi05-ur5e-mujoco/runs/pi05ur5e90nov22).
