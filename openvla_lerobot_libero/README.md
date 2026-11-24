# OpenVLA + LeRobot + LIBERO experiments

This directory preserves the **earlier OpenVLA-7B experiments** from the remote HPC folder `/home/jovyan/projects/lerobot-openvla`. It is separate from the newer π0.5 UR5e and Panda work at the root of this repository.

## What is here

| Location | Meaning |
|---|---|
| `examples/openvla_*.py`, `examples/run_openvla_*.py` | Data preparation, training, evaluation, reporting, and W&B summary entry points |
| `src/lerobot/policies/openvla/` | Local OpenVLA policy implementation and training connection |
| `outputs/` | Lightweight experiment settings, logs, metrics, tables, and MP4 comparison videos |
| `OPENVLA_*.md`, `LIBERO_*.md` | Korean experiment notes and result explanations |
| `EXPORT_OVERVIEW.json` | File counts and output folder index |
| `EXPORT_MANIFEST.json.gz` | Complete path and size inventory, including omitted large files |

The underlying project was based on the upstream [LeRobot repository](https://github.com/huggingface/lerobot). The HPC checkout was at commit `f25ac02` when this export was prepared. This folder includes the experiment-specific code and results; use that upstream commit for the rest of the LeRobot package.

## Dataset and experiment meaning

The bowl demonstrations came from **public LIBERO data selected and preprocessed for these experiments**, not from new demonstrations created by this repository. The source dataset was `openvla/modified_libero_rlds`, `libero_spatial_no_noops/1.0.0`. The documented 40-demo split means 40 complete trajectories, not 40 images. Validation demonstrations and simulator start states are different evaluation units. Prediction loss and simulator task success are different metrics.

## What is not on GitHub

The remote `outputs/` directory held about 200 GB of checkpoints, frames, arrays, and generated files. Ordinary GitHub files have size limits, and publishing the raw artifacts would make the repository impractical. This export includes source, documents, settings, text results, and MP4 videos. Large model weights, raw PNG frame sequences, arrays, and W&B local state remain on HPC. The exact paths and byte sizes of omitted files are in `EXPORT_MANIFEST.json.gz`; this repository is **not a complete backup of those bytes**.

The modified upstream `src/lerobot/policies/{groot,pi0,pi05}/__init__.py` files are included for traceability. They should be compared with upstream before reuse.

## Where to start

1. Read `OPENVLA_BOWL_FULL_RESULT.md` and `OPENVLA_BOWL_FULL_EXPERIMENT.md` for the bowl run.
2. Look in `outputs/` for each named run's metrics, settings, and `viewing_bundle` videos.
3. Read `examples/prepare_libero_bowl_data.py` for the public-data selection and `examples/run_openvla_bowl_full.py` for the experiment sequence.
