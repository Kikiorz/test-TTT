#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/lerobot-pi0-ttt}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/venv/main/bin/accelerate}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/artifacts/models/pi05_libero_finetuned}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/artifacts/datasets/lerobot_libero}"
PHASE1_OUTPUT="${PHASE1_OUTPUT:-/workspace/outputs/train/pi05_ttt_libero_long_c256_stage1_s5000_seed1000}"
PHASE2_OUTPUT="${PHASE2_OUTPUT:-/workspace/outputs/train/pi05_ttt_libero_long_c256_stage2_s5000_seed1000}"
PHASE1_STEPS="${PHASE1_STEPS:-5000}"
PHASE2_STEPS="${PHASE2_STEPS:-5000}"
PHASE1_CHECKPOINT_OVERRIDE="${PHASE1_CHECKPOINT_OVERRIDE:-}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
SEED="${SEED:-1000}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PHASE1_TBPTT_SEGMENT_LENGTH="${PHASE1_TBPTT_SEGMENT_LENGTH:-8}"
PHASE2_TBPTT_SEGMENT_LENGTH="${PHASE2_TBPTT_SEGMENT_LENGTH:-4}"

if (( NUM_PROCESSES != 4 )); then
  echo "The formal PI0.5-TTT recipe requires NUM_PROCESSES=4" >&2
  exit 2
fi

EPISODES="$(${PYTHON_BIN} -c 'import json; print(json.dumps(list(range(379))))')"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/artifacts/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}/src"

cd "${REPO_ROOT}"

LAUNCH=(
  "${ACCELERATE_BIN}" launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --multi_gpu
  --mixed_precision=bf16
  --dynamo_backend=no
  -m
)

run_stage() {
  local pretrained_path="$1"
  local training_stage="$2"
  local optimizer_lr="$3"
  local scheduler_decay_lr="$4"
  local steps="$5"
  local output_dir="$6"
  local tbptt_segment_length="$7"

  "${LAUNCH[@]}" lerobot.scripts.lerobot_train \
    --dataset.repo_id=lerobot/libero \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episodes="${EPISODES}" \
    --dataset.video_backend=pyav \
    --policy.type=pi05_ttt \
    --policy.pretrained_path="${pretrained_path}" \
    --policy.dtype=bfloat16 \
    --policy.n_action_steps=10 \
    --policy.sequence_length=256 \
    --policy.sequence_stride=256 \
    --policy.tbptt_segment_length="${tbptt_segment_length}" \
    --policy.ttt_hidden_dim=4096 \
    --policy.ttt_layer_indices='[14,15,16,17]' \
    --policy.ttt_base_inner_lr=0.1 \
    --policy.ttt_effective_gate_init=0.05 \
    --policy.ttt_training_stage="${training_stage}" \
    --policy.gradient_checkpointing=false \
    --policy.compile_model=false \
    --policy.optimizer_lr="${optimizer_lr}" \
    --policy.optimizer_weight_decay=1e-5 \
    --policy.scheduler_decay_lr="${scheduler_decay_lr}" \
    --policy.push_to_hub=false \
    --batch_size=1 \
    --num_workers=2 \
    --prefetch_factor=1 \
    --persistent_workers=true \
    --steps="${steps}" \
    --log_freq=10 \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq=0 \
    --wandb.enable=false \
    --seed="${SEED}" \
    --output_dir="${output_dir}"
}

if [[ -n "${PHASE1_CHECKPOINT_OVERRIDE}" ]]; then
  PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT_OVERRIDE%/}"
else
  run_stage "${MODEL_ROOT}" ttt_only 2e-5 2e-6 "${PHASE1_STEPS}" "${PHASE1_OUTPUT}" \
    "${PHASE1_TBPTT_SEGMENT_LENGTH}"
  PHASE1_CHECKPOINT="${PHASE1_OUTPUT}/checkpoints/$(printf '%06d' "${PHASE1_STEPS}")/pretrained_model"
fi

if [[ ! -f "${PHASE1_CHECKPOINT}/model.safetensors" ]]; then
  echo "Stage-1 checkpoint is missing: ${PHASE1_CHECKPOINT}" >&2
  exit 3
fi

run_stage "${PHASE1_CHECKPOINT}" action_head 5e-5 5e-6 "${PHASE2_STEPS}" "${PHASE2_OUTPUT}" \
  "${PHASE2_TBPTT_SEGMENT_LENGTH}"
