#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
run="$PWD"
root="$(dirname "$run")"
sim="$root/source/.venv/bin/python"
trainer="$root/source/.external/lerobot/.venv/bin/python"
base=allenai/MolmoAct2-LIBERO-LeRobot
steps=4000
port=8296
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false XA_PROGRESS=lines
export XA_WANDB_PROJECT= WANDB_ERROR_REPORTING=false
sim_code="$root/source/src"
train_code="$root/full_train_src"
export PYTHONPATH="$sim_code:$root"
exec 9>"$run/run-${1:?Use data or gpu}.lock"
flock -n 9
trap 'echo "FAILED stage=$1 line=$LINENO time=$(date -Is)" >&2' ERR

running() {
    local pid=$1 status
    status=$(ps -p "$pid" -o stat=) || return 1
    [[ -n "$status" && "$status" != Z* ]]
}

if [[ $1 == data ]]; then
    while running "$(cat pilot.pid)"; do sleep 10; done
    "$trainer" convert.py --raw raw --output converted/k8 --count 8 --extract-actions pilot_exported_actions
    "$sim" audit.py --raw raw --exported pilot_exported_actions --out pilot_replay_verified.json
    for count in 100 200 400; do
        "$sim" -u collect.py --output raw --target "$count" --workers 6
        "$trainer" convert.py --raw raw --output "converted/k$count" --count "$count"
    done
    echo 'DATA_COLLECTION_COMPLETE'
    exit
fi

[[ $1 == gpu ]]
wait_gpu() {
    while :; do
        local prior=0 memory
        for pid in "$@"; do
            if running "$pid"; then prior=1; fi
        done
        memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if (( prior == 0 && memory < 1000 )); then return; fi
        echo "WAIT_GPU $(date -Is) prior_jobs_active=$prior used_mib=$memory"
        sleep 90
    done
}

train() {
    local count=$1 train_steps=$2 output=$3 warmup=$4
    PYTHONPATH="$train_code" CUDA_VISIBLE_DEVICES=0 "$trainer" -m xenoarm.teacher.train \
        --backbone molmoact2 --base "$base" --dataset "xenoarm_panda_scripted/can_k$count" \
        --dataset-root "$run/converted/k$count" --robot panda --shots "$count" --rank 32 --mode lora \
        --scopes vlm_attn,vlm_ffn,expert_attn,expert_ffn --steps "$train_steps" --batch-size 4 \
        --lr 5e-5 --warmup "$warmup" --num-workers 2 --seed 777 --output "$output"
}

server_pid=
stop_server() {
    if [[ -n $server_pid ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
}
trap stop_server EXIT

for count in 100 200 400; do
    while [[ ! -s converted/k$count/conversion_verified.json ]]; do
        running "$(cat data.pid)" || { echo 'Dataset collection ended before conversion.' >&2; exit 1; }
        sleep 30
    done
    wait_gpu "${@:2}"
    free=$(df -BG --output=avail "$run" | tail -1 | tr -cd '0-9')
    test "$free" -ge 16
    adapter="$run/training/molmoact2__teacher__panda_k${count}_r32/seed777"
    output="$run/evaluation/k$count"
    mkdir -p "$output"
    if [[ ! -s $adapter/lora.safetensors ]]; then
        if [[ $count == 100 && ! -s $run/smoke/molmoact2__teacher__panda_k100_r32/seed777/lora.safetensors ]]; then
            echo 'TRAIN_SMOKE k100 three updates'
            train 100 3 "$run/smoke" 0
        fi
        [[ ! -e $adapter/xenoarm_run.json ]] || { echo "Incomplete adapter at $adapter" >&2; exit 1; }
        echo "TRAIN_START k$count $(date -Is)"
        started=$(date +%s)
        train "$count" "$steps" "$run/training" 200
        echo $(( $(date +%s) - started )) > "$output/training_seconds.txt"
    fi
    test -s "$adapter/lora.safetensors"
    test -s "$adapter/policy_preprocessor.json"
    test -s "$adapter/policy_postprocessor.json"
    if [[ ! -s $output/summary.json ]]; then
        echo "EVAL_START k$count $(date -Is)"
        PYTHONPATH="$train_code" CUDA_VISIBLE_DEVICES=0 "$trainer" -m xenoarm.serving \
            --backbone molmoact2 --base "$base" --lora "$adapter" --processors "$adapter" \
            --action-space ee_delta_si --host 127.0.0.1 --port "$port" > "$output/server.log" 2>&1 &
        server_pid=$!
        "$sim" -u evaluate.py --layouts raw/evaluation_layouts.json --output "$output" --port "$port"
        stop_server
    fi
    if ! "$sim" publish_wandb.py "$adapter" "$output" "$count" "$(cat "$output/training_seconds.txt")"; then
        echo "WANDB_PENDING k$count"
    fi
done
"$sim" report.py
echo 'PANDA_CAN_STUDY_COMPLETE'
