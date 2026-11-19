# VLA experiments

Code and small examples from my robot learning experiments. The current UR5e training direction uses Pi 0.5 and the 40 scripted MuJoCo demonstrations described below.

## OpenVLA: one LIBERO scene

The earlier OpenVLA Colab notebook has been removed from this repository. The remaining scripts in `openvla_one_scene/` document the earlier OpenVLA prediction and robot-control checks; they are not Pi 0.5 training code.

The full model weights, public datasets, checkpoints, and GPU outputs are kept outside Git.

## UR5e scripted demonstration collection

See [`ur5e_scripted_demos/README.md`](ur5e_scripted_demos/README.md) for the code and result of the 40-position MuJoCo block-grasp demonstration run. This is scripted data generation; the data has not yet been converted for OpenVLA training.

The 40 demonstration episodes are stored outside this Git repository. The Pi 0.5 data conversion is also outside this repository; neither the demonstration images nor a trained Pi 0.5 checkpoint are included here.
