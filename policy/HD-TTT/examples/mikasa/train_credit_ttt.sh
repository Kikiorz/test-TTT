#!/usr/bin/env bash

# Reproducible CreditTTT (V3) training/evaluation recipe for MIKASA-Robo-VLA.
#
# This launcher is intentionally independent from train_hd_ttt.sh.  It does
# not change the legacy/V2 recipe and it never fabricates an evaluation JSON.
# With no argument (or with ``plan``) it only prints the frozen commands.  A
# stage is executed only after the caller explicitly sets EXECUTE=1.
#
# The stages are deliberately separated:
#
#   teacher  -> full-history causal action teacher + prefix feature cache
#   labels   -> event/future pair labels (hindsight event-write deletion)
#   student  -> SmolVLA-TTT QH2L student, initialized from the base checkpoint
#   baselines -> print the official Native-SmolVLA/Clean-TTT/CreditTTT eval commands
#   all      -> teacher, labels, and student in that order
#
# Full-history contract (paper protocol):
#   * feature extraction covers complete episodes [FEATURE_EPISODE_START,
#     FEATURE_EPISODE_END), with validation episodes separated before labels;
#   * the student uses one complete episode window per selected train episode
#     (sequence_length >= the longest selected episode, stride=length,
#     max_windows_per_episode=1, history_warmup_length=null);
#   * the dataset emits episode-local ``_lerobot_sequence_offset=0`` for each
#     full window.  TBPTT segment ``s`` is passed to the policy as
#     ``sequence_offset=window_offset+s``; the offset is reset at each episode
#     and is never a global concatenated-dataset index;
#   * no window crosses an episode boundary and no test-seed result is used to
#     choose sequence/TTT hyperparameters.
#
# ``plan`` is safe on a workstation without the MIKASA environment.  Runtime
# stages perform a metadata preflight and fail before launching distributed
# training when the full-history contract cannot be satisfied.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

STAGE="${1:-plan}"
EXECUTE="${EXECUTE:-0}"

usage() {
  sed -n '2,55p' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Usage:
  train_credit_ttt.sh plan
  EXECUTE=1 train_credit_ttt.sh teacher|labels|student|all
  train_credit_ttt.sh baselines

Required for executable stages:
  DATASET_REPO_ID, DATASET_ROOT, BASE_CHECKPOINT

Useful overrides:
  TASK_ID=color|shuffle_long|shell_touch|intercept_medium|remember_color3|remember_color9,
  TRAIN_EPISODE_END=250 (canonical: all 250 demos),
  FEATURE_EPISODE_END=250, OUTPUT_ROOT=..., SEED=1000,
  EPOCHS=150 (canonical student minimum; one sequence epoch = one pass over
  the selected episode windows),
  STEPS=<optional override> (canonical runs must contain complete sequence
  epochs and satisfy MIN_SEQUENCE_EPOCHS; partial/short values require
  ALLOW_SHORT_RUN=1),
  BATCH_SIZE=1 (per-device; set to 2 or 4 with EQUAL_LENGTH_BATCHING=1 for
  exact-length trajectory buckets),
  GRADIENT_ACCUMULATION_STEPS=1 (sequence windows averaged per optimizer step;
  default 1, fast state resets at every window),
  NUM_PROCESSES=4 (use all four GPUs on the reference server),
  TEACHER_EPISODE_BATCH_SIZE=1 (legacy default; set >1 to right-pad cached
  feature episodes with a valid_mask for higher teacher-fit throughput),
  CREDIT_TTT_REPLAY_PAIR_CHUNK_SIZE=4 (execution-only replay chunk; hardware
  bound, every pair is still evaluated),
  CREDIT_TTT_REPLAY_SAVE_ON_CPU=0 (execution-only host-offload switch),
  EQUAL_LENGTH_BATCHING=0 (opt-in; never pads time or mixes offset domains),
  TRAINING_METADATA_PATH=... (default: <student_output>/training_metadata.json),
  NATIVE_CHECKPOINT=..., CLEAN_CHECKPOINT=...

The V3 loss weights are exposed for explicitly named ablations/smoke tests
(`HD_V3_LOCAL_WEIGHT`, `HD_V3_CMD_WEIGHT`, `HD_V3_NULL_WEIGHT`).  The
canonical recipe keeps local=1, CMD=1, null=0.25; changing them does not
silently alter the benchmark manifest and must be recorded in the output
directory/experiment name.

  V3_ABLATION=full|qh2l_only|cmd_only (sets the objective family and weights)
  HD_V3_GLOBAL_PAIR_NORMALIZATION=1 (all-rank pair denominators for DDP;
  default enabled, set 0 only for a named compatibility ablation)
EOF
}

case "${STAGE}" in
  plan|teacher|labels|student|all|baselines|baseline_commands|help|-h|--help) ;;
  *) echo "Unknown stage '${STAGE}'. Use plan, teacher, labels, student, all, or baselines." >&2; exit 2 ;;
esac
if [[ "${STAGE}" == "help" || "${STAGE}" == "-h" || "${STAGE}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${STAGE}" != "plan" && "${STAGE}" != "baselines" && "${STAGE}" != "baseline_commands" && "${EXECUTE}" != "1" ]]; then
  echo "Refusing to execute '${STAGE}'. Review the plan first, then rerun with EXECUTE=1." >&2
  exit 2
fi

# Task defaults are only convenience labels; DATASET_ROOT remains explicit at
# runtime because a local staging path is not part of dataset identity.
TASK_ID="${TASK_ID:-color}"
case "${TASK_ID}" in
  color)
    DEFAULT_DATASET_REPO_ID="shell_game_color_lamp_touch_vla_v0"
    DEFAULT_ENV_ID="ShellGameColorLampTouch-VLA-v0"
    ;;
  shuffle_long)
    DEFAULT_DATASET_REPO_ID="shell_game_shuffle_color_lamp_touch_long_vla_v0"
    DEFAULT_ENV_ID="ShellGameShuffleColorLampTouch-Long-VLA-v0"
    ;;
  shell_touch|shell_game_touch)
    TASK_ID="shell_touch"
    DEFAULT_DATASET_REPO_ID="shell_game_touch_vla_v0"
    DEFAULT_ENV_ID="ShellGameTouch-VLA-v0"
    ;;
  intercept_medium|intercept)
    TASK_ID="intercept_medium"
    DEFAULT_DATASET_REPO_ID="intercept_medium_vla_v0"
    DEFAULT_ENV_ID="InterceptMedium-VLA-v0"
    ;;
  remember_color3|remember_color_3)
    TASK_ID="remember_color3"
    DEFAULT_DATASET_REPO_ID="remember_color_3_vla_v0"
    DEFAULT_ENV_ID="RememberColor3-VLA-v0"
    ;;
  remember_color9|remember_color_9)
    TASK_ID="remember_color9"
    DEFAULT_DATASET_REPO_ID="remember_color_9_vla_v0"
    DEFAULT_ENV_ID="RememberColor9-VLA-v0"
    ;;
  *)
    echo "TASK_ID must be one of color, shuffle_long, shell_touch, intercept_medium, remember_color3, remember_color9; got '${TASK_ID}'" >&2
    exit 2
    ;;
esac
DATASET_REPO_ID="${DATASET_REPO_ID:-${DEFAULT_DATASET_REPO_ID}}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/data_mikasa_robo/data_lerobot/${DATASET_REPO_ID}}"
ENV_ID="${ENV_ID:-${DEFAULT_ENV_ID}}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-<BASE_CHECKPOINT>}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/credit_ttt_v3/${TASK_ID}}"
FEATURES_PATH="${FEATURES_PATH:-${OUTPUT_ROOT}/full_history_features.pt}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${OUTPUT_ROOT}/full_history_teacher.pt}"
LABEL_PATH="${LABEL_PATH:-${OUTPUT_ROOT}/credit_pairs.pt}"
STUDENT_OUTPUT_DIR="${STUDENT_OUTPUT_DIR:-${OUTPUT_ROOT}/student}"
# Keep provenance beside (rather than inside) the trainer output directory.
# lerobot_train intentionally refuses to start in a non-empty output directory
# when resume=false; placing this sidecar under ``student/`` would therefore
# make every fresh canonical run fail during its own preflight.
TRAINING_METADATA_PATH="${TRAINING_METADATA_PATH:-${OUTPUT_ROOT}/training_metadata.json}"

FEATURE_EPISODE_START="${FEATURE_EPISODE_START:-0}"
FEATURE_EPISODE_END="${FEATURE_EPISODE_END:-250}"
TRAIN_EPISODE_START="${TRAIN_EPISODE_START:-0}"
# Canonical V3 uses every official demonstration (250/250) for feature,
# teacher, label, and student fitting.  Validation is optional diagnostics
# only and is placed after the dataset by default, so it cannot remove demos
# from training or influence optimization.  A smaller split remains possible
# only as an explicitly named smoke/ablation override.
TRAIN_EPISODE_END="${TRAIN_EPISODE_END:-250}"
VALIDATION_EPISODE_START="${VALIDATION_EPISODE_START:-250}"

# A comma-separated list is accepted for non-contiguous paper shards.  The
# default is exactly the declared train split [0, TRAIN_EPISODE_END).
TRAIN_EPISODES="${TRAIN_EPISODES:-}"
if [[ -z "${TRAIN_EPISODES}" ]]; then
  if [[ "${TRAIN_EPISODE_END}" =~ ^[0-9]+$ && "${TRAIN_EPISODE_START}" =~ ^[0-9]+$ ]]; then
    TRAIN_EPISODES="$(seq -s, "${TRAIN_EPISODE_START}" $((TRAIN_EPISODE_END - 1)))"
  else
    TRAIN_EPISODES="<TRAIN_EPISODE_INDICES>"
  fi
fi
if [[ "${TRAIN_EPISODES}" == \[*\] ]]; then
  TRAIN_EPISODES_JSON="${TRAIN_EPISODES}"
else
  TRAIN_EPISODES_JSON="[${TRAIN_EPISODES}]"
fi

# Full-history window settings are structural protocol fields.  ``auto`` is
# resolved from episode metadata before an executable student stage; a plan
# without the dataset prints the symbolic value rather than inventing a
# sequence length.
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-auto}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-auto}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-1}"
HISTORY_WARMUP_LENGTH="${HISTORY_WARMUP_LENGTH:-null}"
TBPTT_SEGMENT_LENGTH="${TBPTT_SEGMENT_LENGTH:-32}"

# V3 structural choices.  These defaults are shared across tasks and are not
# selected from test performance; an ablation must be named in its output
# directory/manifest instead of silently changing this recipe.
PAIR_K="${PAIR_K:-5}"
EVENT_BLOCK_SIZE="${EVENT_BLOCK_SIZE:-1}"
POSITIVE_THRESHOLD="${POSITIVE_THRESHOLD:-0.05}"
# Event-write deletion is the preregistered CreditTTT intervention: it exactly
# matches the student's traced W_i^- -> W_i^+ effect.  Content replacement is
# retained only as an explicitly named offline ablation and is rejected by the
# canonical student provenance validator.
INTERVENTION="${INTERVENTION:-delete}"
TTT_HIDDEN_DIM="${TTT_HIDDEN_DIM:-1024}"
TTT_LAYERS="${TTT_LAYERS:-[12,13,14,15]}"
REGISTER_TOKENS="${REGISTER_TOKENS:-16}"
_HD_V3_LOCAL_WEIGHT_SET="${HD_V3_LOCAL_WEIGHT+x}"
_HD_V3_CMD_WEIGHT_SET="${HD_V3_CMD_WEIGHT+x}"
HD_V3_LOCAL_WEIGHT="${HD_V3_LOCAL_WEIGHT:-1.0}"
HD_V3_CMD_WEIGHT="${HD_V3_CMD_WEIGHT:-1.0}"
HD_V3_NULL_WEIGHT="${HD_V3_NULL_WEIGHT:-0.25}"
HD_V3_GLOBAL_PAIR_NORMALIZATION="${HD_V3_GLOBAL_PAIR_NORMALIZATION:-1}"
V3_ABLATION="${V3_ABLATION:-full}"
case "${V3_ABLATION}" in
  full)
    [[ -n "${_HD_V3_LOCAL_WEIGHT_SET}" ]] || HD_V3_LOCAL_WEIGHT=1.0
    [[ -n "${_HD_V3_CMD_WEIGHT_SET}" ]] || HD_V3_CMD_WEIGHT=1.0
    ;;
  qh2l_only)
    [[ -n "${_HD_V3_LOCAL_WEIGHT_SET}" ]] || HD_V3_LOCAL_WEIGHT=1.0
    [[ -n "${_HD_V3_CMD_WEIGHT_SET}" ]] || HD_V3_CMD_WEIGHT=0.0
    ;;
  cmd_only)
    [[ -n "${_HD_V3_LOCAL_WEIGHT_SET}" ]] || HD_V3_LOCAL_WEIGHT=0.0
    [[ -n "${_HD_V3_CMD_WEIGHT_SET}" ]] || HD_V3_CMD_WEIGHT=1.0
    ;;
  *)
    echo "V3_ABLATION must be full, qh2l_only, or cmd_only; got '${V3_ABLATION}'" >&2
    exit 2
    ;;
esac
# The paper recipe trains each task-specific student for at least 150 complete
# passes over the selected episode windows.  A shorter run is allowed only
# when the caller explicitly marks it as a smoke/pilot run; this prevents a
# 3k-step diagnostic from being accidentally promoted to a paper checkpoint.
EPOCHS="${EPOCHS:-150}"
MIN_SEQUENCE_EPOCHS="${MIN_SEQUENCE_EPOCHS:-150}"
ALLOW_SHORT_RUN="${ALLOW_SHORT_RUN:-0}"
# Preserve whether the caller supplied STEPS explicitly.  The canonical
# default is resolved from the exact sampler arithmetic below; an explicit
# value is accepted only when it denotes complete sequence epochs (unless the
# caller has explicitly marked a smoke/pilot run).
if [[ "${STEPS+x}" == "x" && -n "${STEPS}" ]]; then
  STEPS_WAS_EXPLICIT=1
else
  STEPS_WAS_EXPLICIT=0
fi
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
EQUAL_LENGTH_BATCHING="${EQUAL_LENGTH_BATCHING:-0}"
BATCHING_METADATA_JSON="{}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-20}"
TEACHER_HIDDEN_DIM="${TEACHER_HIDDEN_DIM:-256}"
TEACHER_LR="${TEACHER_LR:-0.001}"
TEACHER_EPISODE_BATCH_SIZE="${TEACHER_EPISODE_BATCH_SIZE:-1}"
SEED="${SEED:-1000}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-500}"
RESUME="${RESUME:-false}"
DEVICE="${DEVICE:-cuda}"
TEACHER_DEVICE="${TEACHER_DEVICE:-${DEVICE}}"
SIM_BACKEND="${SIM_BACKEND:-gpu}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_START_SEED="${EVAL_START_SEED:-4242424242}"
EVAL_TORCH_SEED="${EVAL_TORCH_SEED:-7000}"
RESULTS_ROOT="${RESULTS_ROOT:-${OUTPUT_ROOT}/benchmark_results}"
NATIVE_CHECKPOINT="${NATIVE_CHECKPOINT:-<CHECKPOINT_NATIVE_SMOLVLA>}"
CLEAN_CHECKPOINT="${CLEAN_CHECKPOINT:-<CHECKPOINT_CLEAN_TTT>}"
CREDIT_CHECKPOINT="${CREDIT_CHECKPOINT:-${STUDENT_OUTPUT_DIR}/checkpoints/last/pretrained_model}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/data_mikasa_robo}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

print_cmd() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

require_runtime_inputs() {
  local missing=()
  [[ -n "${DATASET_REPO_ID}" ]] || missing+=(DATASET_REPO_ID)
  case "${STAGE}" in
    teacher|student|all)
      [[ -n "${DATASET_ROOT}" && -d "${DATASET_ROOT}" ]] || missing+=("DATASET_ROOT(existing directory)")
      [[ -n "${BASE_CHECKPOINT}" && "${BASE_CHECKPOINT}" != "<BASE_CHECKPOINT>" ]] || missing+=(BASE_CHECKPOINT)
      ;;
    labels)
      # Label construction is artifact-only and can use the dataset/fps
      # provenance already embedded in the feature cache; it does not need to
      # instantiate the physical dataset or a base policy.
      ;;
  esac
  if ((${#missing[@]})); then
    printf 'Missing runtime input(s): %s\n' "${missing[*]}" >&2
    return 2
  fi
  if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer; got '${BATCH_SIZE}'" >&2
    return 2
  fi
  if ! [[ "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GRADIENT_ACCUMULATION_STEPS must be a positive integer; got '${GRADIENT_ACCUMULATION_STEPS}'" >&2
    return 2
  fi
  if [[ "${HD_V3_GLOBAL_PAIR_NORMALIZATION}" != "0" && "${HD_V3_GLOBAL_PAIR_NORMALIZATION}" != "1" ]]; then
    echo "HD_V3_GLOBAL_PAIR_NORMALIZATION must be 0 or 1; got '${HD_V3_GLOBAL_PAIR_NORMALIZATION}'" >&2
    return 2
  fi
  if [[ "${EQUAL_LENGTH_BATCHING}" != "0" && "${EQUAL_LENGTH_BATCHING}" != "1" ]]; then
    echo "EQUAL_LENGTH_BATCHING must be 0 or 1; got '${EQUAL_LENGTH_BATCHING}'" >&2
    return 2
  fi
  if ! [[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer; got '${NUM_PROCESSES}'" >&2
    return 2
  fi
  if ! [[ "${TEACHER_EPISODE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TEACHER_EPISODE_BATCH_SIZE must be a positive integer; got '${TEACHER_EPISODE_BATCH_SIZE}'" >&2
    return 2
  fi
  if (( BATCH_SIZE > 1 && EQUAL_LENGTH_BATCHING != 1 )); then
    echo "BATCH_SIZE>1 requires EQUAL_LENGTH_BATCHING=1 for SmolVLA-TTT" >&2
    return 2
  fi
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "PYTHON_BIN not found: ${PYTHON_BIN}" >&2; return 2; }
  if [[ "${STAGE}" != "labels" ]]; then
    command -v "${ACCELERATE_BIN}" >/dev/null 2>&1 || { echo "ACCELERATE_BIN not found: ${ACCELERATE_BIN}" >&2; return 2; }
  fi
}

resolve_full_history_window() {
  # Resolve and validate the sequence capacity using the same episode metadata
  # consumed by TailPreservingSequenceDataset.  This function intentionally
  # prints only JSON so shell parsing cannot confuse a warning with a count.
  local stats
  stats="$(${PYTHON_BIN} - "${DATASET_REPO_ID}" "${DATASET_ROOT}" "${TRAIN_EPISODES_JSON}" "${SEQUENCE_LENGTH}" "${SEQUENCE_STRIDE}" "${MAX_WINDOWS_PER_EPISODE}" "${HISTORY_WARMUP_LENGTH}" "${BATCH_SIZE}" "${EQUAL_LENGTH_BATCHING}" "${NUM_PROCESSES}" <<'PY'
import json
import sys
from pathlib import Path

repo_id, root, selected_raw, length_raw, stride_raw, cap_raw, warmup_raw, batch_raw, equal_raw, replicas_raw = sys.argv[1:]
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

selected = json.loads(selected_raw)
if not isinstance(selected, list) or not selected:
    raise SystemExit("TRAIN_EPISODES must be a non-empty JSON list")
selected = [int(value) for value in selected]
if len(set(selected)) != len(selected) or min(selected) < 0:
    raise SystemExit("TRAIN_EPISODES must contain unique non-negative indices")
meta = LeRobotDatasetMetadata(repo_id, root=Path(root))
episodes = meta.episodes
if isinstance(episodes, dict):
    starts, ends = episodes["dataset_from_index"], episodes["dataset_to_index"]
    count = len(starts)
    lengths = {i: int(ends[i]) - int(starts[i]) for i in range(count)}
else:
    lengths = {
        i: int(episodes[i]["dataset_to_index"]) - int(episodes[i]["dataset_from_index"])
        for i in range(len(episodes))
    }
unknown = [i for i in selected if i not in lengths]
if unknown:
    raise SystemExit(f"TRAIN_EPISODES out of range: {unknown[:8]}")
chosen = [lengths[i] for i in selected]
max_len = max(chosen)
if length_raw == "auto":
    sequence_length = max_len
else:
    sequence_length = int(length_raw)
if sequence_length < max_len:
    raise SystemExit(
        f"full-history contract violated: sequence_length={sequence_length} < "
        f"longest selected episode={max_len}"
    )
if stride_raw == "auto":
    sequence_stride = sequence_length
else:
    sequence_stride = int(stride_raw)
if sequence_stride != sequence_length:
    raise SystemExit(
        "full-history contract requires sequence_stride == sequence_length "
        f"(got {sequence_stride} vs {sequence_length})"
    )
if int(cap_raw) != 1:
    raise SystemExit("full-history contract requires max_windows_per_episode=1")
if warmup_raw not in {"null", "None", "none", ""}:
    raise SystemExit("full-history contract requires ttt_history_warmup_length=null")
batch_size = int(batch_raw)
equal_length = equal_raw == "1"
num_replicas = int(replicas_raw)
if batch_size <= 0 or num_replicas <= 0:
    raise SystemExit("BATCH_SIZE and NUM_PROCESSES must be positive")
if batch_size > 1 and not equal_length:
    raise SystemExit("BATCH_SIZE>1 requires EQUAL_LENGTH_BATCHING=1")
# Full-history windows have offset zero.  In equal-length mode the runtime
# sampler groups exact physical lengths (and repeats only an index from the
# same bucket to complete a final group), then pads the *number of groups* for
# DDP synchronization.  This mirrors EqualLengthBatchSampler.__len__.
from collections import Counter
bucket_rows = []
if equal_length:
    # Keep this arithmetic aligned with EqualLengthBatchSampler: each bucket
    # is keyed by (T, offset), and canonical full-history windows have offset
    # zero.
    bucket_counts = Counter((length, 0) for length in chosen)
    for (length, offset), count in sorted(bucket_counts.items()):
        groups = (count + batch_size - 1) // batch_size
        slots = groups * batch_size
        bucket_rows.append({
            "length": int(length),
            "offset": int(offset),
            "count": int(count),
            "groups": int(groups),
            "slots": int(slots),
            "fill_repeated_rows": int(slots - count),
        })
    groups_before_ddp = sum(row["groups"] for row in bucket_rows)
    bucket_fill_repeated_rows = sum(row["fill_repeated_rows"] for row in bucket_rows)
else:
    # B>1 is rejected above for the ordinary ragged path. At B=1 this is the
    # same sample-count arithmetic used by Accelerate's distributed loader.
    bucket_counts = Counter()
    groups_before_ddp = len(chosen)
    bucket_fill_repeated_rows = 0
ddp_repeated_groups = (-groups_before_ddp) % num_replicas if groups_before_ddp else 0
total_groups = groups_before_ddp + ddp_repeated_groups
effective_rows = total_groups * batch_size
ddp_repeated_rows = ddp_repeated_groups * batch_size
total_repeated_rows = effective_rows - len(chosen)
batches_per_rank = total_groups // num_replicas
print(json.dumps({
    "schema": "credit_ttt_training_batch_v1",
    "windows": len(selected),
    "raw_windows": len(selected),
    "min_episode_length": min(chosen),
    "max_episode_length": max_len,
    "sequence_length": sequence_length,
    "sequence_stride": sequence_stride,
    "fps": int(getattr(meta, "fps", 0) or 0),
    "total_batches": total_groups,
    "batches_per_rank": batches_per_rank,
    "batch_size": batch_size,
    "per_device_batch_size": batch_size,
    "world_size": num_replicas,
    "global_batch_size": batch_size * num_replicas,
    "equal_length_batching": equal_length,
    "sampler": "EqualLengthBatchSampler" if equal_length else "accelerate_default",
    "ddp_flow_weighting": (
        "valid_action_slots"
        if equal_length and batch_size > 1 and num_replicas > 1
        else "historical_rank_mean"
    ),
    "no_temporal_padding": equal_length or batch_size == 1,
    "stats_available": True,
    "offset_domains": 1,
    "bucket_count": len(bucket_rows) if equal_length else None,
    "buckets": bucket_rows,
    "groups_before_ddp": groups_before_ddp,
    "ddp_repeated_groups": ddp_repeated_groups,
    "total_groups": total_groups,
    "bucket_fill_repeated_rows": bucket_fill_repeated_rows,
    "ddp_repeated_rows": ddp_repeated_rows,
    "total_repeated_rows": total_repeated_rows,
    "effective_rows": effective_rows,
    "repeat_rate": total_repeated_rows / len(chosen),
    "steps_per_epoch_per_rank": batches_per_rank,
}))
PY
  )"
  WINDOWS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["windows"])' "${stats}")"
  MIN_EPISODE_LENGTH="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["min_episode_length"])' "${stats}")"
  MAX_EPISODE_LENGTH="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["max_episode_length"])' "${stats}")"
  SEQUENCE_LENGTH="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["sequence_length"])' "${stats}")"
  SEQUENCE_STRIDE="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["sequence_stride"])' "${stats}")"
  DATASET_FPS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["fps"])' "${stats}")"
  TOTAL_BATCHES="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["total_batches"])' "${stats}")"
  BATCHES_PER_RANK="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["batches_per_rank"])' "${stats}")"
  BATCHING_METADATA_JSON="${stats}"
  if [[ "${DATASET_FPS}" == "0" ]]; then
    echo "Dataset metadata did not expose a positive fps" >&2
    return 2
  fi
  # One optimizer update consumes exactly GRADIENT_ACCUMULATION_STEPS
  # independent windows. Require an integral number of groups so every
  # declared sequence epoch remains a complete sampler pass; carrying a
  # partial group across epochs would make the epoch/step provenance unclear.
  if (( BATCHES_PER_RANK % GRADIENT_ACCUMULATION_STEPS != 0 )); then
    echo "BATCHES_PER_RANK=${BATCHES_PER_RANK} is not divisible by GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}; choose a divisor to preserve complete sequence epochs" >&2
    return 2
  fi
  MICRO_BATCHES_PER_EPOCH="${BATCHES_PER_RANK}"
  STEPS_PER_EPOCH="$((BATCHES_PER_RANK / GRADIENT_ACCUMULATION_STEPS))"
  if (( STEPS_WAS_EXPLICIT == 0 )); then
    STEPS="$((STEPS_PER_EPOCH * EPOCHS))"
  fi
}

validate_student_step_contract() {
  if ! [[ "${MIN_SEQUENCE_EPOCHS}" =~ ^[0-9]+$ && "${EPOCHS}" =~ ^[0-9]+$ ]]; then
    echo "MIN_SEQUENCE_EPOCHS and EPOCHS must be non-negative integers" >&2
    return 2
  fi
  if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer; got '${STEPS}'" >&2
    return 2
  fi
  if (( STEPS_PER_EPOCH <= 0 )); then
    echo "Resolved STEPS_PER_EPOCH must be positive; got '${STEPS_PER_EPOCH}'" >&2
    return 2
  fi
  if (( STEPS_WAS_EXPLICIT == 1 && ALLOW_SHORT_RUN != 1 \
        && STEPS % STEPS_PER_EPOCH != 0 )); then
    echo "Refusing explicit STEPS=${STEPS}: it is not a complete number of sequence epochs (${STEPS_PER_EPOCH} optimizer steps/epoch). Set ALLOW_SHORT_RUN=1 only for a named smoke/pilot run." >&2
    return 2
  fi
  if [[ "${ALLOW_SHORT_RUN}" != "1" ]]; then
    if (( EPOCHS < MIN_SEQUENCE_EPOCHS )); then
      echo "Refusing a canonical student run with EPOCHS=${EPOCHS}; minimum is "\
           "${MIN_SEQUENCE_EPOCHS}. Set ALLOW_SHORT_RUN=1 only for a named "\
           "smoke/pilot experiment." >&2
      return 2
    fi
    local minimum_steps=$((STEPS_PER_EPOCH * MIN_SEQUENCE_EPOCHS))
    if (( STEPS < minimum_steps )); then
      echo "Refusing a canonical student run with STEPS=${STEPS}; minimum is "\
           "${minimum_steps} (${MIN_SEQUENCE_EPOCHS} sequence epochs x "\
           "${STEPS_PER_EPOCH} optimizer steps/sequence epoch). Set "\
           "ALLOW_SHORT_RUN=1 only for a named smoke/pilot experiment." >&2
      return 2
    fi
  fi
}

write_training_metadata() {
  # This sidecar is provenance only. It is written after the same metadata
  # preflight that determines STEPS_PER_EPOCH and is never read by the model
  # or sampler, so enabling it cannot alter training behavior.
  local metadata_path="${TRAINING_METADATA_PATH}"
  "${PYTHON_BIN}" - \
    "${BATCHING_METADATA_JSON}" \
    "${metadata_path}" \
    "${TASK_ID}" \
    "${DATASET_REPO_ID}" \
    "${DATASET_ROOT}" \
    "${TRAIN_EPISODES_JSON}" \
    "${SEED}" \
    "${EPOCHS}" \
    "${STEPS}" \
    "${STEPS_PER_EPOCH}" \
    "${STEPS_WAS_EXPLICIT}" \
    "${SEQUENCE_LENGTH}" \
    "${SEQUENCE_STRIDE}" \
    "${MIN_EPISODE_LENGTH}" \
    "${MAX_EPISODE_LENGTH}" \
    "${DATASET_FPS}" \
    "${GRADIENT_ACCUMULATION_STEPS}" \
    "${HD_V3_GLOBAL_PAIR_NORMALIZATION}" \
    "${V3_ABLATION}" \
    "${INTERVENTION}" \
    "${TTT_LAYERS}" \
    "${TTT_HIDDEN_DIM}" \
    "${REGISTER_TOKENS}" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    batching_raw,
    output_raw,
    task_id,
    repo_id,
    dataset_root,
    episodes_raw,
    seed,
    epochs,
    steps,
    steps_per_epoch,
    steps_was_explicit,
    sequence_length,
    sequence_stride,
    min_episode_length,
    max_episode_length,
    fps,
    gradient_accumulation_steps,
    global_pair_normalization,
    ablation,
    intervention,
    ttt_layers,
    ttt_hidden_dim,
    register_tokens,
) = sys.argv[1:]
batching = json.loads(batching_raw)
episodes = json.loads(episodes_raw)
if not isinstance(batching, dict) or not isinstance(episodes, list):
    raise SystemExit("internal metadata preflight produced malformed JSON")
payload = {
    "schema": "credit_ttt_training_metadata_v1",
    "created_by": "train_credit_ttt.sh",
    "task_id": task_id,
    "dataset_repo_id": repo_id,
    "dataset_root": dataset_root,
    "train_episode_indices": [int(value) for value in episodes],
    "seed": int(seed),
    "epochs": int(epochs),
    "steps": int(steps),
    "steps_per_epoch": int(steps_per_epoch),
    "steps_was_explicit": bool(int(steps_was_explicit)),
    "complete_sequence_epochs": (
        int(steps) // int(steps_per_epoch)
        if int(steps) % int(steps_per_epoch) == 0
        else None
    ),
    "gradient_accumulation_steps": int(gradient_accumulation_steps),
    "effective_batch_size": int(batching["global_batch_size"] * int(gradient_accumulation_steps)),
    "hd_v3_global_pair_normalization": bool(int(global_pair_normalization)),
    "sequence_length": int(sequence_length),
    "sequence_stride": int(sequence_stride),
    "min_episode_length": int(min_episode_length),
    "max_episode_length": int(max_episode_length),
    "dataset_fps": int(fps),
    "student_protocol": "creditttt_qh2l_v3",
    "v3_ablation": ablation,
    "intervention": intervention,
    "ttt_layer_indices": ttt_layers,
    "ttt_hidden_dim": int(ttt_hidden_dim),
    "register_tokens": int(register_tokens),
    # Execution-only replay bounds are deliberately recorded separately from
    # the scientific V3 hyperparameters.  The model reads these environment
    # variables at replay time; preserving the raw values here makes a
    # hardware-profiled run reproducible without implying a changed target.
    "execution_bounds": {
        "credit_ttt_replay_pair_chunk_size": os.environ.get(
            "CREDIT_TTT_REPLAY_PAIR_CHUNK_SIZE", "default:4"
        ),
        "credit_ttt_replay_save_on_cpu": os.environ.get(
            "CREDIT_TTT_REPLAY_SAVE_ON_CPU", "default:0"
        ),
    },
    "batching": batching,
}
output = Path(output_raw)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "Wrote training metadata: ${metadata_path}"
}

teacher_command() {
  local -n out=$1
  out=(
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/mikasa/train_full_history_teacher.py"
    --dataset-repo-id "${DATASET_REPO_ID}"
    --dataset-root "${DATASET_ROOT}"
    --base-checkpoint "${BASE_CHECKPOINT}"
    --features-output "${FEATURES_PATH}"
    --output "${TEACHER_CHECKPOINT}"
    --episode-start "${FEATURE_EPISODE_START}"
    --episode-end "${FEATURE_EPISODE_END}"
    --validation-episode-start "${VALIDATION_EPISODE_START}"
    --epochs "${TEACHER_EPOCHS}"
    --episode-batch-size "${TEACHER_EPISODE_BATCH_SIZE}"
    --hidden-dim "${TEACHER_HIDDEN_DIM}"
    --lr "${TEACHER_LR}"
    --seed "${SEED}"
    --device "${TEACHER_DEVICE}"
  )
  if [[ "${REUSE_FEATURES:-0}" == "1" ]]; then
    out+=(--features-input "${FEATURES_PATH}")
  fi
  if [[ "${DOWNLOAD_VIDEOS:-0}" == "1" ]]; then
    out+=(--download-videos)
  fi
}

labels_command() {
  local -n out=$1
  out=(
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/mikasa/build_credit_labels.py"
    --features-input "${FEATURES_PATH}"
    --teacher-checkpoint "${TEACHER_CHECKPOINT}"
    --output "${LABEL_PATH}"
    --pair-k "${PAIR_K}"
    --intervention "${INTERVENTION}"
    --event-block-size "${EVENT_BLOCK_SIZE}"
    --positive-threshold "${POSITIVE_THRESHOLD}"
    --seed "${SEED}"
    --dataset-repo-id "${DATASET_REPO_ID}"
    --device "${LABEL_DEVICE:-cpu}"
  )
  if [[ -n "${FPS:-}" ]]; then
    out+=(--fps "${FPS}")
  fi
  # Query columns are optional in the direct action-teacher adapter.  Do not
  # claim that this recipe produced them; a future flow-query cache may opt in
  # explicitly with REQUIRE_QUERY_FEATURES=1.
  if [[ "${REQUIRE_QUERY_FEATURES:-0}" == "1" ]]; then
    out+=(--require-query-features)
  fi
}

student_command() {
  local -n out=$1
  local max_windows_arg="${MAX_WINDOWS_PER_EPISODE}"
  local warmup_arg="null"
  local equal_length_arg="false"
  [[ "${EQUAL_LENGTH_BATCHING}" == "1" ]] && equal_length_arg="true"
  out=(
    "${ACCELERATE_BIN}" launch
    --num_machines=1
    --num_processes="${NUM_PROCESSES}"
    --mixed_precision="${MIXED_PRECISION}"
    --dynamo_backend=no
    -m lerobot.scripts.lerobot_train
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.episodes="${TRAIN_EPISODES_JSON}"
    --dataset.hd_label_path="${LABEL_PATH}"
    --dataset.video_backend=pyav
    --dataset.return_uint8=true
    --policy.type=smolvla_ttt
    --policy.pretrained_path="${BASE_CHECKPOINT}"
    --policy.device="${DEVICE}"
    --policy.push_to_hub=false
    --policy.sequence_length="${SEQUENCE_LENGTH}"
    --policy.sequence_stride="${SEQUENCE_STRIDE}"
    --policy.max_windows_per_episode="${max_windows_arg}"
    --policy.ttt_history_warmup_length="${warmup_arg}"
    --policy.equal_length_batching="${equal_length_arg}"
    --policy.tbptt_segment_length="${TBPTT_SEGMENT_LENGTH}"
    --policy.ttt_hidden_dim="${TTT_HIDDEN_DIM}"
    --policy.ttt_second_order=true
    --policy.ttt_stable_inner_update=true
    --policy.ttt_layer_indices="${TTT_LAYERS}"
    --policy.ttt_num_register_tokens="${REGISTER_TOKENS}"
    --policy.ttt_writer_mode=prefix_only
    --policy.ttt_training_stage=ttt_only
    --policy.hd_ttt_enabled=true
    --policy.hd_attribution_protocol=credit_ttt_v3_query_effect
    --policy.hd_effect_weight=0.0
    --policy.hd_learned_write_gate=false
    --policy.hd_phase_mode=deployment
    --policy.hd_v3_pair_k="${PAIR_K}"
    --policy.hd_v3_local_weight="${HD_V3_LOCAL_WEIGHT}"
    --policy.hd_v3_cmd_weight="${HD_V3_CMD_WEIGHT}"
    --policy.hd_v3_global_pair_normalization="$([[ "${HD_V3_GLOBAL_PAIR_NORMALIZATION}" == "1" ]] && echo true || echo false)"
    --policy.hd_v3_ablation="${V3_ABLATION}"
    --policy.hd_v3_cmd_margin=0.05
    --policy.hd_v3_null_weight="${HD_V3_NULL_WEIGHT}"
    --policy.hd_v3_null_threshold="${POSITIVE_THRESHOLD}"
    --policy.hd_v3_include_previous_action=true
    --policy.hd_v3_intervention="${INTERVENTION}"
    --policy.hd_v3_effect_layer=last_selected
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
    --num_workers="${NUM_WORKERS}"
    --prefetch_factor="${PREFETCH_FACTOR}"
    --persistent_workers=true
    --steps="${STEPS}"
    --log_freq="${LOG_FREQ}"
    --save_checkpoint=true
    --save_freq="${SAVE_FREQ}"
    --eval_freq=0
    --resume="${RESUME}"
    --wandb.enable=false
    --seed="${SEED}"
    --output_dir="${STUDENT_OUTPUT_DIR}"
  )
}

baseline_commands() {
  # Match benchmark_credit_ttt_v3.py's frozen result layout so aggregation
  # can recover training replicates from the path without guessing.
  local native_output="${RESULTS_ROOT}/native_smolvla/train_seed_fixed/${TASK_ID}/eval.json"
  local native_k1_output="${RESULTS_ROOT}/native_smolvla_k1/train_seed_fixed/${TASK_ID}/eval.json"
  local clean_output="${RESULTS_ROOT}/clean_ttt/train_seed_${SEED}/${TASK_ID}/eval.json"
  local credit_output="${RESULTS_ROOT}/credit_ttt/train_seed_${SEED}/${TASK_ID}/eval.json"
  local native_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/mikasa/evaluate_smolvla_baseline.py"
    --checkpoint "${NATIVE_CHECKPOINT}" --dataset-repo-id "${DATASET_REPO_ID}"
    --dataset-root "${DATASET_ROOT}" --task "${ENV_ID}"
    --num-episodes "${EVAL_EPISODES}" --start-seed "${EVAL_START_SEED}"
    --torch-seed "${EVAL_TORCH_SEED}" --sim-backend "${SIM_BACKEND}"
    --device cuda --output "${native_output}"
  )
  local native_k1_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/mikasa/evaluate_smolvla_baseline.py"
    --checkpoint "${NATIVE_CHECKPOINT}" --dataset-repo-id "${DATASET_REPO_ID}"
    --dataset-root "${DATASET_ROOT}" --task "${ENV_ID}"
    --num-episodes "${EVAL_EPISODES}" --start-seed "${EVAL_START_SEED}"
    --torch-seed "${EVAL_TORCH_SEED}" --sim-backend "${SIM_BACKEND}"
    --device cuda --execution-action-steps 1 --output "${native_k1_output}"
  )
  local clean_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/mikasa/evaluate_smolvla_ttt.py"
    --checkpoint "${CLEAN_CHECKPOINT}" --dataset-repo-id "${DATASET_REPO_ID}"
    --dataset-root "${DATASET_ROOT}" --task "${ENV_ID}"
    --num-episodes "${EVAL_EPISODES}" --start-seed "${EVAL_START_SEED}"
    --torch-seed "${EVAL_TORCH_SEED}" --sim-backend "${SIM_BACKEND}"
    --device cuda --no-hd-ttt-enabled --no-hd-learned-write-gate
    --hd-v3-include-previous-action
    --output "${clean_output}"
  )
  local credit_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/mikasa/evaluate_smolvla_ttt.py"
    --checkpoint "${CREDIT_CHECKPOINT}" --dataset-repo-id "${DATASET_REPO_ID}"
    --dataset-root "${DATASET_ROOT}" --task "${ENV_ID}"
    --num-episodes "${EVAL_EPISODES}" --start-seed "${EVAL_START_SEED}"
    --torch-seed "${EVAL_TORCH_SEED}" --sim-backend "${SIM_BACKEND}"
    --device cuda --output "${credit_output}"
  )
  echo "Official MIKASA baseline/evaluation commands (not executed by this stage):"
  echo "# Freeze the benchmark envelope/protocol before evaluating any checkpoint:"
  local manifest_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/mikasa/benchmark_credit_ttt_v3.py" manifest
    --output "${RESULTS_ROOT}/manifest.json" --repo-root "${REPO_ROOT}"
    --python-bin "${PYTHON_BIN}" --results-root "${RESULTS_ROOT}"
    --native-checkpoint "${NATIVE_CHECKPOINT}" --clean-checkpoint "${CLEAN_CHECKPOINT}"
    --credit-checkpoint "${CREDIT_CHECKPOINT}"
    --per-device-batch-size "${BATCH_SIZE}" --world-size "${NUM_PROCESSES}"
  )
  if [[ "${EQUAL_LENGTH_BATCHING}" == "1" ]]; then
    manifest_cmd+=(--equal-length-batching)
  else
    manifest_cmd+=(--no-equal-length-batching)
  fi
  # The sidecar is produced by an executable student preflight.  Keep plan
  # mode usable before that file exists, while embedding exact bucket/repeat
  # statistics whenever baselines are frozen after training setup.
  if [[ -f "${TRAINING_METADATA_PATH}" ]]; then
    manifest_cmd+=(--training-metadata-json "${TRAINING_METADATA_PATH}")
  fi
  print_cmd "${manifest_cmd[@]}"
  echo "# Native-SmolVLA (canonical native action chunk K=50):"
  print_cmd "${native_cmd[@]}"
  echo "# Native-SmolVLA-K1 (same native checkpoint/model horizon, matched receding cadence):"
  print_cmd "${native_k1_cmd[@]}"
  echo "# Clean-TTT (K=1, HD explicitly disabled, previous-action schema matched):"
  print_cmd "${clean_cmd[@]}"
  echo "# CreditTTT (K=1, canonical V3 identity emitted by evaluator):"
  print_cmd "${credit_cmd[@]}"
  echo "# Aggregate only after all per-episode JSON files are real outputs:"
  local aggregate_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/mikasa/benchmark_credit_ttt_v3.py" aggregate
    --manifest "${RESULTS_ROOT}/manifest.json" --results-root "${RESULTS_ROOT}"
    --output "${RESULTS_ROOT}/aggregate.json"
  )
  print_cmd "${aggregate_cmd[@]}"
  echo "No result JSON is created by train_credit_ttt.sh itself."
}

print_protocol() {
  echo "CreditTTT V3 protocol: creditttt_qh2l_v3"
  echo "canonical identity: format=credit_ttt_v3 version=3 pair_schema=event_future_control_pair_v3"
  echo "target=final_slot0_action intervention=event_write_deletion state=causal_fast_weights causal=true"
  echo "task=${TASK_ID} dataset=${DATASET_REPO_ID} root=${DATASET_ROOT}"
  echo "episodes: features=${FEATURE_EPISODE_START}..$((FEATURE_EPISODE_END - 1)), train=${TRAIN_EPISODES_JSON}, validation_start=${VALIDATION_EPISODE_START}"
  echo "full-history window: sequence_length=${SEQUENCE_LENGTH} sequence_stride=${SEQUENCE_STRIDE} max_windows_per_episode=${MAX_WINDOWS_PER_EPISODE} history_warmup_length=${HISTORY_WARMUP_LENGTH}"
  echo "trajectory batch: per_device=${BATCH_SIZE} equal_length=${EQUAL_LENGTH_BATCHING} processes=${NUM_PROCESSES} accumulation=${GRADIENT_ACCUMULATION_STEPS} (no temporal padding; fast state resets per window)"
  echo "V3 DDP pair normalization: global_pair_weighted=${HD_V3_GLOBAL_PAIR_NORMALIZATION} (denominator all-reduced; B1/single-process unchanged)"
  echo "execution-only replay bounds: pair_chunk=${CREDIT_TTT_REPLAY_PAIR_CHUNK_SIZE:-default:4} save_on_cpu=${CREDIT_TTT_REPLAY_SAVE_ON_CPU:-default:0}"
  echo "teacher fit: episode_batch_size=${TEACHER_EPISODE_BATCH_SIZE} (right-padded valid_mask; checkpoint execution provenance records this value)"
  echo "offset contract: episode-local origin; full window offset=0; TBPTT segment offset=window_offset+segment_start; reset at episode boundary"
  echo "stages: full-history teacher -> pair labels -> QH2L student; baseline commands are evaluation-only"
  echo "outputs: features=${FEATURES_PATH} teacher=${TEACHER_CHECKPOINT} labels=${LABEL_PATH} student=${STUDENT_OUTPUT_DIR} training_metadata=${TRAINING_METADATA_PATH}"
}

if [[ "${STAGE}" == "plan" ]]; then
  print_protocol
  echo
  echo "Commands below are a plan only; no files or result JSON are written."
  teacher=(); labels=(); student=()
  teacher_command teacher
  labels_command labels
  # A plan may not have dataset metadata, so retain symbolic auto values.
  if [[ "${SEQUENCE_LENGTH}" == "auto" ]]; then
    SEQUENCE_LENGTH="<MAX_SELECTED_EPISODE_LENGTH>"
  fi
  if [[ "${SEQUENCE_STRIDE}" == "auto" ]]; then
    SEQUENCE_STRIDE="${SEQUENCE_LENGTH}"
  fi
  WINDOWS="<N_SELECTED_EPISODES>"
  STEPS="<N_SELECTED_EPISODES*EPOCHS/(world_size*gradient_accumulation_steps)>"
  student_command student
  echo "# teacher (stage 1)"
  print_cmd "${teacher[@]}"
  echo "# labels (stage 2)"
  print_cmd "${labels[@]}"
  echo "# student (stage 3; execute only after labels exist)"
  print_cmd "${student[@]}"
  echo
  baseline_commands
  exit 0
fi

if [[ "${STAGE}" == "baselines" || "${STAGE}" == "baseline_commands" ]]; then
  print_protocol
  baseline_commands
  exit 0
fi

require_runtime_inputs
if [[ "${STAGE}" == "student" || "${STAGE}" == "all" ]]; then
  resolve_full_history_window
  validate_student_step_contract || exit $?
else
  # Teacher/label stages do not launch the sequence trainer.  Keep symbolic
  # values available for a concise status line without pretending that a
  # full-history capacity was measured.
  WINDOWS="unknown"
  MIN_EPISODE_LENGTH="unknown"
  MAX_EPISODE_LENGTH="unknown"
  STEPS_PER_EPOCH="unknown"
  STEPS="unknown"
fi
mkdir -p "${OUTPUT_ROOT}"
if [[ "${STAGE}" == "student" || "${STAGE}" == "all" ]]; then
  write_training_metadata
fi
echo "Resolved full-history windows=${WINDOWS}, episode_length=${MIN_EPISODE_LENGTH}..${MAX_EPISODE_LENGTH}, sequence=${SEQUENCE_LENGTH}, batch/device=${BATCH_SIZE}, processes=${NUM_PROCESSES}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, micro-batches/epoch=${MICRO_BATCHES_PER_EPOCH:-unknown}, optimizer-steps/epoch=${STEPS_PER_EPOCH}, total_steps=${STEPS}"

run_cmd() {
  local label=$1
  shift
  echo "[${label}]"
  print_cmd "$@"
  if [[ "${EXECUTE}" == "1" ]]; then
    "$@"
  fi
}

case "${STAGE}" in
  teacher)
    teacher=(); teacher_command teacher
    run_cmd teacher "${teacher[@]}"
    [[ -f "${TEACHER_CHECKPOINT}" ]] || { echo "Teacher stage finished without checkpoint: ${TEACHER_CHECKPOINT}" >&2; exit 3; }
    [[ -f "${FEATURES_PATH}" ]] || { echo "Teacher stage finished without feature cache: ${FEATURES_PATH}" >&2; exit 3; }
    ;;
  labels)
    [[ -f "${FEATURES_PATH}" ]] || { echo "Missing feature cache: ${FEATURES_PATH}; run teacher first" >&2; exit 3; }
    [[ -f "${TEACHER_CHECKPOINT}" ]] || { echo "Missing teacher checkpoint: ${TEACHER_CHECKPOINT}; run teacher first" >&2; exit 3; }
    labels=(); labels_command labels
    run_cmd labels "${labels[@]}"
    [[ -f "${LABEL_PATH}" ]] || { echo "Label stage finished without artifact: ${LABEL_PATH}" >&2; exit 3; }
    ;;
  student)
    [[ -f "${LABEL_PATH}" ]] || { echo "Missing CreditTTT labels: ${LABEL_PATH}; run labels first" >&2; exit 3; }
    student=(); student_command student
    run_cmd student "${student[@]}"
    ;;
  all)
    teacher=(); teacher_command teacher
    run_cmd teacher "${teacher[@]}"
    [[ -f "${TEACHER_CHECKPOINT}" && -f "${FEATURES_PATH}" ]] || { echo "Teacher stage did not produce both artifacts" >&2; exit 3; }
    labels=(); labels_command labels
    run_cmd labels "${labels[@]}"
    [[ -f "${LABEL_PATH}" ]] || { echo "Label stage did not produce artifact" >&2; exit 3; }
    student=(); student_command student
    run_cmd student "${student[@]}"
    ;;
esac

echo "Completed requested stage '${STAGE}'. Evaluation is separate: run the printed official commands and aggregate only real outputs."
