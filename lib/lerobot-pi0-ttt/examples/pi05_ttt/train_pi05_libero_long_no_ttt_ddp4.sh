#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/test-TTT-eval-git/lib/lerobot-pi0-ttt}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/venv/main/bin/accelerate}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/artifacts/models/pi05_libero_finetuned}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/artifacts/datasets/lerobot_libero}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/train/pi05_libero_long_no_ttt_action_head_s5000_seed1000}"
STEPS="${STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
SEED="${SEED:-1000}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

if (( NUM_PROCESSES != 4 )); then
  echo "The formal PI0.5 no-TTT ablation requires NUM_PROCESSES=4" >&2
  exit 2
fi

if [[ ! -f "${MODEL_ROOT}/model.safetensors" ]]; then
  echo "PI0.5 base checkpoint is missing: ${MODEL_ROOT}" >&2
  exit 3
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite an existing output directory: ${OUTPUT_DIR}" >&2
  exit 4
fi

EPISODES="$(${PYTHON_BIN} -c 'import json; print(json.dumps(list(range(379))))')"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/artifacts/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/src"
export LEROBOT_VERIFY_DDP_SYNC="${LEROBOT_VERIFY_DDP_SYNC:-1}"

cd "${REPO_ROOT}"

exec "${ACCELERATE_BIN}" launch \
  --num_machines=1 \
  --num_processes="${NUM_PROCESSES}" \
  --multi_gpu \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.episodes="${EPISODES}" \
  --dataset.video_backend=pyav \
  --policy.path="${MODEL_ROOT}" \
  --policy.dtype=bfloat16 \
  --policy.n_action_steps=10 \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=false \
  --policy.gradient_checkpointing=false \
  --policy.compile_model=false \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_weight_decay=1e-5 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --num_workers=2 \
  --prefetch_factor=1 \
  --persistent_workers=true \
  --steps="${STEPS}" \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --eval_freq=0 \
  --wandb.enable=false \
  --seed="${SEED}" \
  --output_dir="${OUTPUT_DIR}"
