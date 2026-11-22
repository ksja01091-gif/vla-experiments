# VLA experiments

This repository records the UR5e MuJoCo demonstration collection and the π0.5 experiment. The 40 scripted UR5e demonstrations were generated in our simulation, not taken from LIBERO.

## Start here

- [`ur5e_scripted_demos/README.md`](ur5e_scripted_demos/README.md): how the 40 demonstrations were collected and verified.
- [`UR5e_Pi05_학습전_과정_설명.md`](UR5e_Pi05_학습전_과정_설명.md): Korean explanation of the simulation and data preparation before training.
- [`pi05_ur5e/README.md`](pi05_ur5e/README.md): π0.5 training settings and the one recorded MuJoCo rollout.
- [`pi05_ur5e/Pi05_UR5e_40demo_training.ipynb`](pi05_ur5e/Pi05_UR5e_40demo_training.ipynb): Colab notebook source.

The demonstration images/actions and the trained model weights are stored outside Git. The π0.5 checkpoint is in the research lab Drive. Earlier OpenVLA prediction experiments were removed from this repository.