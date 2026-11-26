#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
root="$PWD"
conversion_pid=${1:?Pass the running data conversion PID, or 0 if all data is verified}
export PYTHONPATH="$root/full_train_src" XA_WANDB_PROJECT=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0 XA_PROGRESS=lines
python="$root/source/.external/lerobot/.venv/bin/python"
base=allenai/MolmoAct2-LIBERO-LeRobot
batch=4
trap 'echo "FAILED line $LINENO (exit $?)"' ERR

wait_gpu() {
  while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) >= 1000 )); do
    echo "WAIT_GPU $(date -Is)"
    sleep 45
  done
}

train() {
  local robot=$1 path=$2 shots=$3 steps=$4 warmup=$5 output=$6
  "$python" -m xenoarm.teacher.train --backbone molmoact2 --base "$base" \
    --dataset "xenoarm_molmo_full/${robot}_train" --dataset-root "$path" \
    --robot "$robot" --shots "$shots" --rank 32 --mode lora \
    --scopes vlm_attn,vlm_ffn,expert_attn,expert_ffn \
    --steps "$steps" --batch-size "$batch" --lr 5e-5 --warmup "$warmup" \
    --num-workers 2 --seed 777 --output "$output"
}

for robot in panda ur5e; do
  wait_gpu
  echo "SMOKE $robot batch=$batch"
  train "$robot" "$root/full_data_probe/$robot/train" 1 3 0 "$root/full_lora_smoke"
  test -s "$root/full_lora_smoke/molmoact2__teacher__${robot}_k1_r32/seed777/lora.safetensors"
done
echo "BOTH_TRAINING_SMOKES_PASSED"

for robot in panda ur5e; do
  dataset="$root/full_data/$robot/train"
  manifest="$dataset/conversion_verified.json"
  while [[ ! -s "$manifest" ]]; do
    if [[ "$conversion_pid" == 0 ]] || ! ps -p "$conversion_pid" -o stat= | grep -q '^[^Z]'; then
      echo "Dataset conversion stopped before verification. See $root/full_data.log"
      exit 1
    fi
    sleep 30
  done
  steps=$("$python" - "$manifest" "$batch" <<'PY'
import json
import math
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
assert data["episodes"] == 400 and data["split"] == "train", data
print(math.ceil(data["frames"] / int(sys.argv[2])))
PY
)
  wait_gpu
  echo "FULL_TRAIN $robot batch=$batch steps=$steps"
  train "$robot" "$dataset" 400 "$steps" 200 "$root/full_lora"
done
echo "BOTH_FULL_TRAINING_RUNS_COMPLETE"
