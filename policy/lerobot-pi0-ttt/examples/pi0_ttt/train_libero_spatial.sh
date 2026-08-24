#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/lerobot-pi0-ttt}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lerobot/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/hf_libero/models/pi0_libero_finetuned}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/hf_libero/lerobot/lerobot/libero}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/train/pi0_ttt_libero_spatial_t128_v1}"
STEPS="${STEPS:-3000}"
SAVE_FREQ="${SAVE_FREQ:-${STEPS}}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"

EPISODES="$(${PYTHON_BIN} -c 'import json; print(json.dumps(list(range(1261, 1693))))')"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/hf_libero/lerobot}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}/src"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.episodes="${EPISODES}" \
  --dataset.video_backend=pyav \
  --policy.type=pi0_ttt \
  --policy.pretrained_path="${MODEL_ROOT}" \
  --policy.dtype=bfloat16 \
  --policy.empty_cameras=1 \
  --policy.n_action_steps=10 \
  --policy.sequence_length=128 \
  --policy.sequence_stride=32 \
  --policy.tbptt_segment_length=8 \
  --policy.ttt_hidden_dim=4096 \
  --policy.ttt_layer_indices='[14,15,16,17]' \
  --policy.ttt_base_inner_lr=0.1 \
  --policy.ttt_freeze_base=true \
  --policy.gradient_checkpointing=false \
  --policy.compile_model=false \
  --policy.optimizer_lr=2e-5 \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --num_workers=2 \
  --prefetch_factor=1 \
  --persistent_workers=true \
  --steps="${STEPS}" \
  --log_freq=10 \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --save_freq="${SAVE_FREQ}" \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir="${OUTPUT_DIR}"
