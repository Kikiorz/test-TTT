#!/usr/bin/env bash

# Reproducible MIKASA HD-TTT training recipe.
#
# The default recipe is episode-balanced: four evenly spaced 64-frame windows
# per long episode and 150 passes over the 250 demonstrations.  This keeps the
# recurrent update causal while covering the beginning, middle, and terminal
# phases of the Long task. Set MAX_WINDOWS_PER_EPISODE=none to use every
# tail-preserving window (the strict full-coverage setting; more expensive).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/test-TTT/policy/HD-TTT}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/MIKASA-Robo/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/workspace/MIKASA-Robo/.venv/bin/accelerate}"
DATASET_REPO_ID="${DATASET_REPO_ID:?set DATASET_REPO_ID}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/smolvla_base}"
LABEL_PATH="${LABEL_PATH:-}"
EPOCHS="${EPOCHS:-150}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-64}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-64}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-4}"
TBPTT_SEGMENT_LENGTH="${TBPTT_SEGMENT_LENGTH:-32}"
HISTORY_WARMUP_LENGTH="${HISTORY_WARMUP_LENGTH:-64}"
if [[ "${HISTORY_WARMUP_LENGTH}" == "full" ]]; then
  HISTORY_WARMUP_ARG="None"
else
  HISTORY_WARMUP_ARG="${HISTORY_WARMUP_LENGTH}"
fi
TTT_HIDDEN_DIM="${TTT_HIDDEN_DIM:-1024}"
TTT_LAYERS="${TTT_LAYERS:-[12,13,14,15]}"
REGISTER_TOKENS="${REGISTER_TOKENS:-16}"
RESIZE="${RESIZE:-[224,224]}"
TRAINING_STAGE="${TRAINING_STAGE:-ttt_only}"
HD_ENABLED="${HD_ENABLED:-false}"
SAVE_FREQ="${SAVE_FREQ:-500}"
LOG_FREQ="${LOG_FREQ:-50}"
SEED="${SEED:-1000}"
# Optional comma-free JSON/list syntax understood by draccus, e.g. ``[0]`` or
# ``[0,1,2]``.  This is useful for a bounded smoke run and for reproducible
# per-shard experiments; when unset all 250 demonstrations are used.
DATASET_EPISODES="${DATASET_EPISODES:-}"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/data_mikasa_robo}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${MAX_WINDOWS_PER_EPISODE}" == "none" ]]; then
  MAX_WINDOWS_ARG="None"
else
  MAX_WINDOWS_ARG="${MAX_WINDOWS_PER_EPISODE}"
fi

# Count the exact sequence windows consumed by TailPreservingSequenceDataset
# so ``steps`` really denotes the requested number of dataset epochs.
WINDOWS="$(${PYTHON_BIN} - "${DATASET_ROOT}" "${SEQUENCE_LENGTH}" "${SEQUENCE_STRIDE}" "${MAX_WINDOWS_ARG}" "${DATASET_EPISODES}" <<'PY'
import json
import math
import sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

root = Path(sys.argv[1])
length = int(sys.argv[2])
stride = int(sys.argv[3])
cap = None if sys.argv[4] == "None" else int(sys.argv[4])
selected = None
if sys.argv[5]:
    selected = {int(value) for value in json.loads(sys.argv[5].replace("'", '"'))}
meta = LeRobotDatasetMetadata(root.name, root=root)
episodes = meta.episodes
total = 0
for episode_index, row in enumerate(episodes):
    if selected is not None and episode_index not in selected:
        continue
    n = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
    windows = math.ceil(n / stride)
    if cap is not None:
        windows = min(windows, cap)
    total += windows
print(total)
PY
)"
STEPS_PER_EPOCH=$(( (WINDOWS + NUM_PROCESSES - 1) / NUM_PROCESSES ))
STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))

echo "MIKASA HD-TTT: windows=${WINDOWS}, steps/epoch=${STEPS_PER_EPOCH}, epochs=${EPOCHS}, steps=${STEPS}"

COMMON_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.video_backend=pyav
  --dataset.return_uint8=true
  --policy.type=smolvla_ttt
  --policy.pretrained_path="${PRETRAINED_PATH}"
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.sequence_length="${SEQUENCE_LENGTH}"
  --policy.sequence_stride="${SEQUENCE_STRIDE}"
  --policy.max_windows_per_episode="${MAX_WINDOWS_ARG}"
  --policy.tbptt_segment_length="${TBPTT_SEGMENT_LENGTH}"
  --policy.ttt_history_warmup_length="${HISTORY_WARMUP_ARG}"
  --policy.ttt_hidden_dim="${TTT_HIDDEN_DIM}"
  --policy.ttt_second_order=false
  --policy.ttt_layer_indices="${TTT_LAYERS}"
  --policy.ttt_num_register_tokens="${REGISTER_TOKENS}"
  --policy.ttt_training_stage="${TRAINING_STAGE}"
  --policy.resize_imgs_with_padding="${RESIZE}"
  --policy.hd_ttt_enabled="${HD_ENABLED}"
  --batch_size=1
  --num_workers="${NUM_WORKERS:-4}"
  --prefetch_factor="${PREFETCH_FACTOR:-2}"
  --persistent_workers=true
  --steps="${STEPS}"
  --log_freq="${LOG_FREQ}"
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
  --eval_freq=0
  --wandb.enable=false
  --seed="${SEED}"
  --output_dir="${OUTPUT_DIR}"
)

if [[ -n "${LABEL_PATH}" ]]; then
  COMMON_ARGS+=(--dataset.hd_label_path="${LABEL_PATH}")
fi
if [[ -n "${DATASET_EPISODES}" ]]; then
  COMMON_ARGS+=(--dataset.episodes="${DATASET_EPISODES}")
fi

LAUNCH=(
  "${ACCELERATE_BIN}" launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --mixed_precision=bf16
  --dynamo_backend=no
)
if (( NUM_PROCESSES > 1 )); then
  LAUNCH+=(--multi_gpu)
fi
LAUNCH+=(-m lerobot.scripts.lerobot_train)

cd "${REPO_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${LAUNCH[@]}" "${COMMON_ARGS[@]}"
  printf '\n'
  exit 0
fi
exec "${LAUNCH[@]}" "${COMMON_ARGS[@]}"
