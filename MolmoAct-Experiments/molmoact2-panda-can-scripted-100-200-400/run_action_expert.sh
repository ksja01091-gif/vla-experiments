#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
run="$PWD"
root="$(dirname "$run")"
sim="$root/source/.venv/bin/python"
trainer="$root/source/.external/lerobot/.venv/bin/python"
train_code="$root/full_train_src"
base=allenai/MolmoAct2-LIBERO-LeRobot
output="$run/evaluation/k400_action_expert_only"
training="$run/training_action_expert_only"
adapter="$training/molmoact2__teacher__panda_k400_r32/seed777"
port=8296
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false XA_PROGRESS=lines
export XA_WANDB_PROJECT= WANDB_ERROR_REPORTING=false
export PYTHONPATH="$root/source/src:$root"

server_pid=
stop_server() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
}
cleanup() {
    local status=$?
    stop_server
    if (( status != 0 )); then
        echo "ACTION_EXPERT_FAILED exit=$status time=$(date -Is)" >&2
    fi
}
trap cleanup EXIT

# Keep a queued rerun from holding GPU access while the same run is already queued.
exec 8>>"$run/run-action-expert.lock"
flock -n 8
echo "WAIT_K400_BASELINE $(date -Is)"
exec 9>>"$run/run-gpu.lock"
flock 9
test -s evaluation/k400/summary.json
test -s evaluation/k400/training_seconds.txt
test -s converted/k400/conversion_verified.json
"$sim" compare_lora_scope.py --check-baseline
while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) >= 1000 )); do
    echo "WAIT_GPU $(date -Is)"
    sleep 30
done
free=$(df -BG --output=avail "$run" | tail -1 | tr -cd '0-9')
test "$free" -ge 16
mkdir -p "$output"

if [[ ! -s "$adapter/lora.safetensors" ]]; then
    if [[ -e "$adapter/xenoarm_run.json" ]]; then
        echo "Incomplete training at $adapter" >&2
        exit 1
    fi
    echo "ACTION_EXPERT_TRAIN_START $(date -Is)"
    started=$(date +%s)
    PYTHONPATH="$train_code" CUDA_VISIBLE_DEVICES=0 "$trainer" -m xenoarm.teacher.train \
        --backbone molmoact2 --base "$base" --dataset xenoarm_panda_scripted/can_k400 \
        --dataset-root "$run/converted/k400" --robot panda --shots 400 --rank 32 --mode lora \
        --scopes expert_attn,expert_ffn --steps 4000 --batch-size 4 \
        --lr 5e-5 --warmup 200 --num-workers 2 --seed 777 --output "$training"
    echo $(( $(date +%s) - started )) > "$output/training_seconds.txt"
fi
test -s "$adapter/policy_preprocessor.json"
test -s "$adapter/policy_postprocessor.json"
test -s "$output/training_seconds.txt"

if [[ ! -s "$output/summary.json" ]]; then
    echo "ACTION_EXPERT_EVAL_START $(date -Is)"
    PYTHONPATH="$train_code" CUDA_VISIBLE_DEVICES=0 "$trainer" -m xenoarm.serving \
        --backbone molmoact2 --base "$base" --lora "$adapter" --processors "$adapter" \
        --action-space ee_delta_si --host 127.0.0.1 --port "$port" > "$output/server.log" 2>&1 &
    server_pid=$!
    "$sim" -u evaluate.py --layouts raw/evaluation_layouts.json --output "$output" --port "$port"
    stop_server
fi
"$sim" compare_lora_scope.py
echo "ACTION_EXPERT_COMPARISON_COMPLETE $(date -Is)"
