#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/source/src" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python=source/.external/lerobot/.venv/bin/python
input=${MIMICGEN_SOURCE:-/home/jovyan/projects/molmoact_dualrobot}
trap 'echo "FAILED line $LINENO (exit $?)"' ERR
source/.venv/bin/python - <<'PY'
import json
from pathlib import Path
from xenoarm.data.mimicgen import REPOS, reference_env

for robot in REPOS:
    ref = reference_env(robot, 128)
    for key in ("agentview", "wrist"):
        del ref[key]
    Path(f"full_data_reference_{robot}.json").write_text(json.dumps(ref, default=lambda x: x.tolist()))
PY
for robot in panda ur5e; do
  "$python" convert_full.py --source "$input" --robot "$robot" --split train \
    --output full_data_probe --limit 1
done
echo "BOTH_ROBOT_READBACK_PASSED"
for split in train validation test; do
  for robot in panda ur5e; do
    "$python" convert_full.py --source "$input" --robot "$robot" --split "$split" --output full_data
  done
done
echo "ALL_SIX_SPLITS_VERIFIED"
