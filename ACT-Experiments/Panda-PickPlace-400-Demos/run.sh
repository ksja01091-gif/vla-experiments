#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
exec 9>run.lock
flock -n 9
trap 'echo "FAILED line $LINENO, exit $?" >&2' ERR
export CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 XA_PROGRESS=lines WANDB_MODE=offline WANDB_ENTITY=gist-infonet
export WANDB_PROJECT=xenoarm-proposal WANDB_DIR="$root/tracking"
export WANDB_NAME=panda-act-pp4-k400-seed777-3000step
export PYTHONPATH="$root/source/src"
mkdir -p "$WANDB_DIR"
train="$root/train-env/bin/python"
sim="$root/sim-env/bin/python"
output="$root/training"
port=18131
batch=4

"$train" - <<'PY'
import json
from pathlib import Path
import torch
source = json.loads(Path("dataset/conversion_verified.json").read_text())
assert source["robot"] == "panda" and source["split"] == "train"
assert source["episodes"] == len(set(source["source_ids"])) == 400
assert torch.cuda.device_count() == 1 and "TITAN RTX" in torch.cuda.get_device_name(0)
print(f'Data verified: {source["episodes"]} demonstrations, {source["frames"]} frames')
print(f"Training GPU: {torch.cuda.get_device_name(0)}")
PY

# Check the simulator before committing GPU time to training.
"$sim" - <<'PY'
from xenoarm.envs.robosuite_multi import RobosuiteMultiRobotEnv
env = RobosuiteMultiRobotEnv(dict(robot="panda", task="pick_place", resolution=256))
try:
    obs = env.reset(0)
    assert obs["agentview_image"].shape == (256, 256, 3)
    assert obs["robot0_eye_in_hand_image"].shape == (256, 256, 3)
finally:
    env.close()
print("Panda pp4 and both camera views verified")
PY

if ! test -s training_seconds.txt; then
    echo "TRAINING START | seed 777 | 3000 optimizer steps | batch $batch"
    start=$(date +%s)
    "$root/train-env/bin/lerobot-train" \
        --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
        --dataset.repo_id=xenoarm_molmo_full/panda_train --dataset.root="$root/dataset" \
        --dataset.video_backend=pyav --batch_size="$batch" --steps=3000 --seed=777 \
        --num_workers=2 --log_freq=25 --save_freq=1000 --eval_freq=0 \
        --output_dir="$output" --job_name="$WANDB_NAME" \
        --wandb.enable=true --wandb.mode=offline --wandb.project=xenoarm-proposal \
        --wandb.entity=gist-infonet --wandb.disable_artifact=true
    checkpoint="$output/checkpoints/last/pretrained_model"
    test -s "$checkpoint/model.safetensors"
    echo "$(($(date +%s) - start))" >training_seconds.txt
    echo "TRAINING COMPLETE | $(cat training_seconds.txt) seconds"
fi

if ! test -s "$root/evaluation/summary.json"; then
    checkpoint="$output/checkpoints/last/pretrained_model"
    "$train" "$root/code/serve_act.py" "$checkpoint" --port "$port" >policy.log 2>&1 &
    server=$!
    trap 'kill "$server" 2>/dev/null || true' EXIT
    echo "EVALUATION START | seeds 0-49"
    for ((attempt=0; attempt<120; attempt++)); do
        if curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then break; fi
        kill -0 "$server" 2>/dev/null || { cat policy.log; exit 1; }
        sleep 1
    done
    curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null
    cd "$root/source"
    "$sim" scripts/parallel_run.py --config configs/robosuite_pickplace4_openpi_eval.json \
        --set "policy.port=$port" --set policy.resize=256 --set policy.checkpoint_label=panda-act-k400-seed777 \
        --seeds 0:50 --workers 2 --output "$root/evaluation"
    cd "$root"
fi
"$sim" "$root/code/summarize.py" "$root"
echo "EXPERIMENT COMPLETE"
