#!/usr/bin/env bash

# Two-stage Method1 SmolVLA-TTT training for the two selected MIKASA tasks.
#
# The canonical defaults use all 250 official episodes, tail-preserving
# 256-frame selected sequences, two GPUs, and per-device batch size 8. Every
# window starts from W0 and carries fast state only across its internal TBPTT
# segments; each global minibatch of windows produces one outer step. Stage 1 trains the
# TTT/register parameters and learned gate for 50 complete sequence epochs.
# Stage 2 loads the final stage-1 pretrained model into a fresh run and trains
# the action head together with the TTT parameters for another
# 50 complete sequence epochs. Each stage saves only its final checkpoint.
#
# With no argument (or ``plan``), this script only prints the resolved protocol
# and commands. Execute explicitly with ``EXECUTE=1 ... run``.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python3}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/venv/main/bin/accelerate}"
MODE="${1:-plan}"
EXECUTE="${EXECUTE:-0}"

case "${MODE}" in
  plan|run|help|-h|--help) ;;
  *) echo "Unknown mode '${MODE}'. Use plan or run." >&2; exit 2 ;;
esac

usage() {
  sed -n '3,16p' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Usage:
  train_mikasa_two_stage.sh plan
  EXECUTE=1 train_mikasa_two_stage.sh run

Canonical task choices:
  TASK_ID=remember_shape5
  TASK_ID=shuffle_touch

Useful overrides:
  BASE_CHECKPOINT=/path/to/native_smolvla/checkpoints/last/pretrained_model
  DATASET_ROOT=/path/to/data_lerobot/<dataset>
  OUTPUT_ROOT=/path/to/method1_smolvla_ttt_sequence_outer_v1_50x50/<task>
  BATCH_SIZE=8, NUM_PROCESSES=2, MAIN_PROCESS_PORT=29500
  STAGE1_EPOCHS=50, STAGE2_EPOCHS=50, SEQUENCE_LENGTH=256
  NUM_WORKERS=4, MIXED_PRECISION=bf16, SEED=1000

The run mode is deliberately fresh-only: it never resumes optimizer state.
Both stage output directories must not exist before launch.
EOF
}

if [[ "${MODE}" == "help" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

die() {
  echo "MIKASA SmolVLA-TTT preflight: $*" >&2
  exit 2
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_positive_int() {
  local name=$1 value=$2
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer; got '${value}'"
}

require_nonnegative_int() {
  local name=$1 value=$2
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer; got '${value}'"
}

TASK_ID="${TASK_ID:-remember_shape5}"
case "${TASK_ID}" in
  remember_shape5)
    DATASET_REPO_ID="${DATASET_REPO_ID:-remember_shape_5_vla_v0}"
    ENV_ID="RememberShape5-VLA-v0"
    ;;
  shuffle_touch)
    DATASET_REPO_ID="${DATASET_REPO_ID:-shell_game_shuffle_touch_vla_v0}"
    ENV_ID="ShellGameShuffleTouch-VLA-v0"
    ;;
  *) die "TASK_ID must be remember_shape5 or shuffle_touch; got '${TASK_ID}'" ;;
esac

DATASET_ROOT="${DATASET_ROOT:-/workspace/data_mikasa_robo/data_lerobot/${DATASET_REPO_ID}}"
BASELINE_ROOT="${BASELINE_ROOT:-/workspace/experiments/native_smolvla_150}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${BASELINE_ROOT}/${TASK_ID}/checkpoints/last/pretrained_model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiments/method1_smolvla_ttt_sequence_outer_v1_50x50/${TASK_ID}}"
STAGE1_OUTPUT="${STAGE1_OUTPUT:-${OUTPUT_ROOT}/stage1_ttt_only}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-${OUTPUT_ROOT}/stage2_action_head}"
STAGE1_FINAL="${STAGE1_OUTPUT}/checkpoints/last/pretrained_model"
STAGE2_FINAL="${STAGE2_OUTPUT}/checkpoints/last/pretrained_model"

EXPECTED_EPISODES=250
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-256}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-256}"
TBPTT_SEGMENT_LENGTH="${TBPTT_SEGMENT_LENGTH:-4}"
TTT_HIDDEN_DIM="${TTT_HIDDEN_DIM:-4096}"
TTT_LAYERS="${TTT_LAYERS:-[12,13,14,15]}"
REGISTER_TOKENS="${REGISTER_TOKENS:-16}"
EFFECTIVE_GATE_INIT="${EFFECTIVE_GATE_INIT:-0.001}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
LOG_FREQ="${LOG_FREQ:-10}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cuda}"

require_positive_int EXPECTED_EPISODES "${EXPECTED_EPISODES}"
require_positive_int STAGE1_EPOCHS "${STAGE1_EPOCHS}"
require_positive_int STAGE2_EPOCHS "${STAGE2_EPOCHS}"
require_positive_int BATCH_SIZE "${BATCH_SIZE}"
require_positive_int NUM_PROCESSES "${NUM_PROCESSES}"
require_positive_int SEQUENCE_LENGTH "${SEQUENCE_LENGTH}"
require_positive_int SEQUENCE_STRIDE "${SEQUENCE_STRIDE}"
require_positive_int TBPTT_SEGMENT_LENGTH "${TBPTT_SEGMENT_LENGTH}"
require_positive_int TTT_HIDDEN_DIM "${TTT_HIDDEN_DIM}"
require_nonnegative_int REGISTER_TOKENS "${REGISTER_TOKENS}"
require_positive_int N_ACTION_STEPS "${N_ACTION_STEPS}"
require_nonnegative_int NUM_WORKERS "${NUM_WORKERS}"
require_positive_int PREFETCH_FACTOR "${PREFETCH_FACTOR}"
require_positive_int MAIN_PROCESS_PORT "${MAIN_PROCESS_PORT}"
require_positive_int LOG_FREQ "${LOG_FREQ}"
require_nonnegative_int SEED "${SEED}"
(( SEQUENCE_STRIDE <= SEQUENCE_LENGTH )) \
  || die "SEQUENCE_STRIDE cannot exceed SEQUENCE_LENGTH"
(( TBPTT_SEGMENT_LENGTH <= SEQUENCE_LENGTH )) \
  || die "TBPTT_SEGMENT_LENGTH cannot exceed SEQUENCE_LENGTH"
(( N_ACTION_STEPS <= 50 )) || die "N_ACTION_STEPS cannot exceed the default chunk size (50)"
(( MAIN_PROCESS_PORT <= 65535 )) || die "MAIN_PROCESS_PORT must be in 1..65535"
[[ "${STAGE1_OUTPUT}" != "${STAGE2_OUTPUT}" ]] || die "stage output directories must differ"
[[ "${EFFECTIVE_GATE_INIT}" == "0.001" ]] \
  || die "the canonical two-stage recipe requires EFFECTIVE_GATE_INIT=0.001"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/data_mikasa_robo}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

VALIDATION_PYTHON=""
if command_exists "${PYTHON_BIN}"; then
  VALIDATION_PYTHON="${PYTHON_BIN}"
elif command_exists python3; then
  VALIDATION_PYTHON="$(command -v python3)"
fi

validate_checkpoint() {
  local checkpoint=$1 expected_type=$2 expected_stage=$3
  local required_file
  [[ -d "${checkpoint}" ]] || die "checkpoint directory does not exist: ${checkpoint}"
  for required_file in \
    config.json \
    model.safetensors \
    train_config.json \
    policy_preprocessor.json \
    policy_postprocessor.json; do
    [[ -s "${checkpoint}/${required_file}" ]] \
      || die "checkpoint is missing non-empty ${required_file}: ${checkpoint}"
  done
  [[ -n "${VALIDATION_PYTHON}" ]] || die "python3 is required to validate ${checkpoint}"

  "${VALIDATION_PYTHON}" - "${checkpoint}" "${expected_type}" "${expected_stage}" \
      "${EFFECTIVE_GATE_INIT}" "${N_ACTION_STEPS}" <<'PY'
import json
import math
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
expected_type = sys.argv[2]
expected_stage = sys.argv[3]
expected_gate = float(sys.argv[4])
expected_action_steps = int(sys.argv[5])

config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
actual_type = config.get("type")
if actual_type != expected_type:
    raise SystemExit(
        f"checkpoint {checkpoint} has type={actual_type!r}, expected {expected_type!r}"
    )
if expected_stage:
    actual_stage = config.get("ttt_training_stage")
    if actual_stage != expected_stage:
        raise SystemExit(
            f"checkpoint {checkpoint} has ttt_training_stage={actual_stage!r}, "
            f"expected {expected_stage!r}"
        )
    gate = float(config.get("ttt_effective_gate_init", float("nan")))
    if not math.isclose(gate, expected_gate, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"checkpoint {checkpoint} has ttt_effective_gate_init={gate!r}, "
            f"expected {expected_gate!r}"
        )
    if int(config.get("n_action_steps", -1)) != expected_action_steps:
        raise SystemExit(
            f"checkpoint {checkpoint} has n_action_steps={config.get('n_action_steps')!r}, "
            f"expected {expected_action_steps}"
        )
    if config.get("ttt_sequence_state_semantics") != "sequence_outer_step_v1":
        raise SystemExit(
            f"checkpoint {checkpoint} is not marked with selected-sequence outer-step semantics"
        )

for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
    processor = json.loads((checkpoint / processor_name).read_text(encoding="utf-8"))
    for step in processor.get("steps", []):
        state_file = step.get("state_file")
        if state_file:
            state_path = checkpoint / state_file
            if not state_path.is_file() or state_path.stat().st_size == 0:
                raise SystemExit(
                    f"processor {processor_name} references missing or empty state file "
                    f"{state_file!r}"
                )
PY
}

validate_stage_output() {
  local output_dir=$1 expected_stage=$2 expected_steps=$3
  local checkpoint_root="${output_dir}/checkpoints/last"
  local pretrained_model="${checkpoint_root}/pretrained_model"
  local training_step_file="${checkpoint_root}/training_state/training_step.json"

  [[ -d "${checkpoint_root}" ]] \
    || die "stage did not create checkpoints/last: ${output_dir}"
  validate_checkpoint "${pretrained_model}" smolvla_ttt "${expected_stage}"
  [[ -s "${training_step_file}" ]] \
    || die "stage checkpoint is missing training_step.json: ${training_step_file}"
  "${VALIDATION_PYTHON}" - "${training_step_file}" \
      "${pretrained_model}/train_config.json" "${expected_steps}" <<'PY'
import json
import sys
from pathlib import Path

step_file = Path(sys.argv[1])
train_config_file = Path(sys.argv[2])
expected = int(sys.argv[3])
actual = int(json.loads(step_file.read_text(encoding="utf-8"))["step"])
train_config = json.loads(train_config_file.read_text(encoding="utf-8"))
configured = int(train_config["steps"])
if actual != expected or configured != expected:
    raise SystemExit(
        f"final checkpoint step mismatch: training_state={actual}, "
        f"train_config={configured}, expected={expected}"
    )
PY
}

if [[ "${MODE}" == "run" ]]; then
  [[ "${EXECUTE}" == "1" ]] \
    || die "refusing to execute; review plan first and rerun with EXECUTE=1"
  command_exists "${PYTHON_BIN}" || die "PYTHON_BIN not found: ${PYTHON_BIN}"
  command_exists "${ACCELERATE_BIN}" || die "ACCELERATE_BIN not found: ${ACCELERATE_BIN}"
  [[ -f "${REPO_ROOT}/src/lerobot/scripts/lerobot_train.py" ]] \
    || die "REPO_ROOT is not a Method1 LeRobot checkout: ${REPO_ROOT}"
  [[ -d "${DATASET_ROOT}" && -f "${DATASET_ROOT}/meta/info.json" ]] \
    || die "dataset root is incomplete: ${DATASET_ROOT}"
  validate_checkpoint "${BASE_CHECKPOINT}" smolvla ""
  for output_dir in "${STAGE1_OUTPUT}" "${STAGE2_OUTPUT}"; do
    [[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] \
      || die "fresh stage output already exists: ${output_dir}"
  done
elif [[ -d "${BASE_CHECKPOINT}" ]]; then
  if [[ -n "${VALIDATION_PYTHON}" ]]; then
    validate_checkpoint "${BASE_CHECKPOINT}" smolvla ""
  else
    echo "Plan warning: checkpoint exists but python3 is unavailable for JSON validation." >&2
  fi
else
  echo "Plan note: baseline checkpoint is not present locally; run mode will require it: ${BASE_CHECKPOINT}" >&2
fi

RESOLVED_STATS=""
if [[ -d "${DATASET_ROOT}" && -f "${DATASET_ROOT}/meta/info.json" ]] \
    && command_exists "${PYTHON_BIN}"; then
  RESOLVED_STATS="$("${PYTHON_BIN}" - \
      "${DATASET_REPO_ID}" \
      "${DATASET_ROOT}" \
      "${EXPECTED_EPISODES}" \
      "${SEQUENCE_LENGTH}" \
      "${SEQUENCE_STRIDE}" \
      "${BATCH_SIZE}" \
      "${NUM_PROCESSES}" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

repo_id, root_raw, expected_raw, length_raw, stride_raw, batch_raw, world_raw = sys.argv[1:]
root = Path(root_raw)
expected = int(expected_raw)
sequence_length = int(length_raw)
sequence_stride = int(stride_raw)
batch_size = int(batch_raw)
world_size = int(world_raw)

meta = LeRobotDatasetMetadata(repo_id, root=root)
episodes = meta.episodes
if episodes is None:
    raise SystemExit("dataset metadata does not contain episode boundaries")
if isinstance(episodes, dict):
    rows = [
        (index, int(start), int(end))
        for index, (start, end) in enumerate(
            zip(episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True)
        )
    ]
else:
    rows = [
        (
            int(row.get("episode_index", index)),
            int(row["dataset_from_index"]),
            int(row["dataset_to_index"]),
        )
        for index, row in enumerate(episodes)
    ]

rows.sort(key=lambda item: item[0])
indices = [row[0] for row in rows]
if indices != list(range(expected)):
    raise SystemExit(
        f"canonical run requires exactly episode indices 0..{expected - 1}; got "
        f"{len(indices)} indices spanning {indices[:1]}..{indices[-1:] if indices else []}"
    )

lengths = [end - start for _, start, end in rows]
if any(length <= 0 for length in lengths):
    raise SystemExit("all selected episodes must have positive length")

if sequence_stride > sequence_length:
    raise SystemExit(
        "sequence_stride must not exceed sequence_length"
    )
windows_per_episode = [
    (episode_length + sequence_stride - 1) // sequence_stride for episode_length in lengths
]
windows = sum(windows_per_episode)
global_batch_size = batch_size * world_size
window_count_histogram = {
    window_count: windows_per_episode.count(window_count)
    for window_count in sorted(set(windows_per_episode))
}
steps_per_epoch = (windows + global_batch_size - 1) // global_batch_size
effective_slots = steps_per_epoch * global_batch_size

print(
    json.dumps(
        {
            "schema": "method1_mikasa_smolvla_ttt_sequence_windows_v1",
            "dataset_repo_id": repo_id,
            "dataset_root": str(root),
            "episode_indices": indices,
            "episodes_json": json.dumps(indices, separators=(",", ":")),
            "num_episodes": len(indices),
            "num_frames": sum(lengths),
            "min_episode_length": min(lengths),
            "max_episode_length": max(lengths),
            "sequence_length": sequence_length,
            "sequence_stride": sequence_stride,
            "num_windows": windows,
            "window_count_histogram": window_count_histogram,
            "steps_per_epoch": steps_per_epoch,
            "effective_window_slots": effective_slots,
            # These are fully masked dummy lanes, never repeated real windows.
            "unused_batch_slots": effective_slots - windows,
        },
        separators=(",", ":"),
    )
)
PY
  )" || die "could not resolve the canonical 250-episode sequence epoch"
fi

if [[ -n "${RESOLVED_STATS}" ]]; then
  IFS=$'\t' read -r \
    NUM_EPISODES \
    NUM_FRAMES \
    MIN_EPISODE_LENGTH \
    MAX_EPISODE_LENGTH \
    NUM_WINDOWS \
    STEPS_PER_EPOCH \
    UNUSED_BATCH_SLOTS \
    EPISODES_JSON \
    <<< "$("${VALIDATION_PYTHON}" -c '
import json, sys
x = json.loads(sys.argv[1])
print(x["num_episodes"], x["num_frames"], x["min_episode_length"],
      x["max_episode_length"], x["num_windows"], x["steps_per_epoch"],
      x["unused_batch_slots"], x["episodes_json"], sep="\t")
' "${RESOLVED_STATS}")"
  STAGE1_STEPS=$((STEPS_PER_EPOCH * STAGE1_EPOCHS))
  STAGE2_STEPS=$((STEPS_PER_EPOCH * STAGE2_EPOCHS))
else
  if [[ "${MODE}" == "run" ]]; then
    die "run mode requires locally resolvable dataset metadata"
  fi
  NUM_EPISODES="<250>"
  NUM_FRAMES="<N_FRAMES>"
  MIN_EPISODE_LENGTH="<MIN_EPISODE_LENGTH>"
  MAX_EPISODE_LENGTH="<MAX_EPISODE_LENGTH>"
  NUM_WINDOWS="<SUM_CEIL(EPISODE_LENGTH/SEQUENCE_STRIDE)>"
  STEPS_PER_EPOCH="<CEIL(NUM_WINDOWS/GLOBAL_BATCH)>"
  UNUSED_BATCH_SLOTS="<UNUSED_FINAL_BATCH_SLOTS>"
  EPISODES_JSON="<ALL_250_EPISODE_INDICES>"
  STAGE1_STEPS="<STEPS_PER_EPOCH*STAGE1_EPOCHS>"
  STAGE2_STEPS="<STEPS_PER_EPOCH*STAGE2_EPOCHS>"
fi

LAUNCH=(
  "${ACCELERATE_BIN}" launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --multi_gpu
  --main_process_port="${MAIN_PROCESS_PORT}"
  --mixed_precision="${MIXED_PRECISION}"
  --dynamo_backend=no
  -m lerobot.scripts.lerobot_train
)

build_stage_command() {
  local -n command_out=$1
  local pretrained_path=$2 training_stage=$3 steps=$4 output_dir=$5
  command_out=(
    "${LAUNCH[@]}"
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.episodes="${EPISODES_JSON}"
    --dataset.video_backend=pyav
    --dataset.return_uint8=true
    --policy.type=smolvla_ttt
    --policy.pretrained_path="${pretrained_path}"
    --policy.device="${DEVICE}"
    --policy.push_to_hub=false
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.sequence_length="${SEQUENCE_LENGTH}"
    --policy.sequence_stride="${SEQUENCE_STRIDE}"
    --policy.tbptt_segment_length="${TBPTT_SEGMENT_LENGTH}"
    --policy.ttt_hidden_dim="${TTT_HIDDEN_DIM}"
    --policy.ttt_layer_indices="${TTT_LAYERS}"
    --policy.ttt_base_inner_lr=0.1
    --policy.ttt_effective_gate_init="${EFFECTIVE_GATE_INIT}"
    --policy.ttt_second_order=true
    --policy.ttt_num_register_tokens="${REGISTER_TOKENS}"
    --policy.ttt_training_stage="${training_stage}"
    --policy.compile_model=false
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --prefetch_factor="${PREFETCH_FACTOR}"
    --persistent_workers=true
    --steps="${steps}"
    --log_freq="${LOG_FREQ}"
    --save_checkpoint=true
    --save_freq="${steps}"
    --eval_freq=0
    --resume=false
    --wandb.enable=false
    --seed="${SEED}"
    --output_dir="${output_dir}"
  )
}

build_stage_command STAGE1_COMMAND "${BASE_CHECKPOINT}" ttt_only "${STAGE1_STEPS}" "${STAGE1_OUTPUT}"
build_stage_command STAGE2_COMMAND "${STAGE1_FINAL}" action_head "${STAGE2_STEPS}" "${STAGE2_OUTPUT}"

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

echo "Method1 SmolVLA-TTT two-stage protocol"
echo "task=${TASK_ID} env=${ENV_ID} dataset=${DATASET_REPO_ID}"
echo "dataset_root=${DATASET_ROOT} baseline=${BASE_CHECKPOINT}"
echo "episodes=${NUM_EPISODES} frames=${NUM_FRAMES} episode_length=${MIN_EPISODE_LENGTH}..${MAX_EPISODE_LENGTH}"
echo "selected_sequence_max=${SEQUENCE_LENGTH} stride=${SEQUENCE_STRIDE} windows=${NUM_WINDOWS} unused_batch_slots=${UNUSED_BATCH_SLOTS}"
echo "batch/device=${BATCH_SIZE} processes=${NUM_PROCESSES} steps/sequence_epoch=${STEPS_PER_EPOCH}"
echo "inference_action_steps=${N_ACTION_STEPS} fast_updates=once_per_action_chunk_inference"
echo "stage1=ttt_only gate_learned_init=${EFFECTIVE_GATE_INIT} epochs=${STAGE1_EPOCHS} steps=${STAGE1_STEPS}"
echo "stage2=action_head fresh_warm_start=true epochs=${STAGE2_EPOCHS} steps=${STAGE2_STEPS}"
echo "stage1_output=${STAGE1_OUTPUT}"
echo "stage2_output=${STAGE2_OUTPUT}"
echo "stage 1 command:"
print_command "${STAGE1_COMMAND[@]}"
echo "stage 2 command (runs only after stage-1 final checkpoint validation):"
print_command "${STAGE2_COMMAND[@]}"

if [[ "${MODE}" == "plan" ]]; then
  echo "Plan only: no output directory, checkpoint, or training process was created."
  exit 0
fi

cd "${REPO_ROOT}"
"${STAGE1_COMMAND[@]}"
validate_stage_output "${STAGE1_OUTPUT}" ttt_only "${STAGE1_STEPS}"

# This is intentionally a new run: only pretrained weights/processors flow
# across the boundary. Stage-1 optimizer, scheduler, RNG, and global step are
# not resumed into the action-head stage.
"${STAGE2_COMMAND[@]}"
validate_stage_output "${STAGE2_OUTPUT}" action_head "${STAGE2_STEPS}"

echo "Completed ${TASK_ID}:"
echo "  stage1_final=${STAGE1_FINAL}"
echo "  stage2_final=${STAGE2_FINAL}"
