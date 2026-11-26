#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/source/src" CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
trap 'echo "FAILED line $LINENO (exit $?)"' ERR
for split in train validation test; do
  source/.external/lerobot/.venv/bin/python convert_full.py \
    --source /home/jovyan/projects/molmoact_dualrobot --robot ur5e --split "$split" --output full_data
done
echo "UR5E_ALL_SPLITS_VERIFIED"
