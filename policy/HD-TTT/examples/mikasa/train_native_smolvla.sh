#!/usr/bin/env bash

# Reproducible task-trained Native SmolVLA baseline for MIKASA-Robo-VLA.
#
# This launcher is deliberately separate from CreditTTT: it invokes only the
# standard ``smolvla`` policy.  No fast weights, register tokens, HD labels,
# or sequence-TTT options are passed to lerobot-train.  The default protocol
# uses every episode in the dataset and at least 150 complete frame epochs.
# A sidecar manifest records the exact episode/frame/batch arithmetic.
#
# With no argument (or ``plan``), this file only prints the protocol and
# command.  Execute explicitly with ``EXECUTE=1 ... run``.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python3}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/venv/main/bin/accelerate}"
STAGE="${1:-plan}"
EXECUTE="${EXECUTE:-0}"

case "${STAGE}" in
  plan|run|help|-h|--help) ;;
  *) echo "Unknown stage '${STAGE}'. Use plan or run." >&2; exit 2 ;;
esac

usage() {
  sed -n '3,16p' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Usage:
  train_native_smolvla.sh plan
  EXECUTE=1 train_native_smolvla.sh run

Useful overrides:
  TASK_ID=color|shuffle_long|shell_touch|intercept_medium|remember_color3|remember_color9
  DATASET_ROOT=..., BASE_CHECKPOINT=..., OUTPUT_DIR=...
  EPOCHS=150, BATCH_SIZE=6 (per device), NUM_PROCESSES=4
  TRAIN_EPISODE_START=0, TRAIN_EPISODE_END=auto (all episodes by default)
  TRAIN_EPISODES=... requires ALLOW_PARTIAL=1 and a named ablation
  STEPS=... (complete epoch override; short runs require ALLOW_SHORT_RUN=1)
  RESUME=false, SAVE_FREQ=1000, LOG_FREQ=100

The source checkpoint must declare policy type ``smolvla``.  A
``smolvla_ttt`` checkpoint is rejected so Native cannot be confused with
Clean-TTT or CreditTTT.
EOF
}

if [[ "${STAGE}" == "help" || "${STAGE}" == "-h" || "${STAGE}" == "--help" ]]; then
  usage
  exit 0
fi

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
  *) echo "Unsupported TASK_ID='${TASK_ID}'" >&2; exit 2 ;;
esac

DATASET_REPO_ID="${DATASET_REPO_ID:-${DEFAULT_DATASET_REPO_ID}}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/data_mikasa_robo/data_lerobot/${DATASET_REPO_ID}}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-lerobot/smolvla_base}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/experiments/native_smolvla_150/${TASK_ID}}"
TRAINING_METADATA_PATH="${TRAINING_METADATA_PATH:-${OUTPUT_DIR%/}/../${TASK_ID}_training_metadata.json}"

TRAIN_EPISODE_START="${TRAIN_EPISODE_START:-0}"
TRAIN_EPISODE_END="${TRAIN_EPISODE_END:-auto}"
TRAIN_EPISODES="${TRAIN_EPISODES:-}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
EPOCHS="${EPOCHS:-150}"
MIN_EPOCHS="${MIN_EPOCHS:-150}"
ALLOW_SHORT_RUN="${ALLOW_SHORT_RUN:-0}"
STEPS_WAS_EXPLICIT=0
if [[ "${STEPS+x}" == "x" && -n "${STEPS}" ]]; then
  STEPS_WAS_EXPLICIT=1
else
  STEPS=""
fi
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
SEED="${SEED:-1000}"
RESUME="${RESUME:-false}"
DEVICE="${DEVICE:-cuda}"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/data_mikasa_robo}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

die() { echo "Native SmolVLA preflight: $*" >&2; exit 2; }
require_nonnegative_int() {
  local name=$1 value=$2
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer; got '${value}'"
}

if [[ "${STAGE}" == "run" && "${EXECUTE}" != "1" ]]; then
  die "refusing to execute; review 'plan' first and set EXECUTE=1"
fi
if [[ "${RESUME}" != "true" && "${RESUME}" != "false" ]]; then
  die "RESUME must be true or false; got '${RESUME}'"
fi
if [[ "${ALLOW_PARTIAL}" != "0" && "${ALLOW_PARTIAL}" != "1" ]]; then
  die "ALLOW_PARTIAL must be 0 or 1"
fi
require_nonnegative_int BATCH_SIZE "${BATCH_SIZE}"
require_nonnegative_int NUM_PROCESSES "${NUM_PROCESSES}"
require_nonnegative_int NUM_WORKERS "${NUM_WORKERS}"
require_nonnegative_int EPOCHS "${EPOCHS}"
require_nonnegative_int MIN_EPOCHS "${MIN_EPOCHS}"
require_nonnegative_int SEED "${SEED}"
[[ "${BATCH_SIZE}" -gt 0 && "${NUM_PROCESSES}" -gt 0 ]] || die "batch/process counts must be positive"
[[ "${EPOCHS}" -gt 0 ]] || die "EPOCHS must be positive"
if [[ -n "${TRAIN_EPISODES}" && "${ALLOW_PARTIAL}" != "1" ]]; then
  die "TRAIN_EPISODES is a subset override; set ALLOW_PARTIAL=1 for a named ablation"
fi
if [[ "${STAGE}" == "run" ]]; then
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "PYTHON_BIN not found: ${PYTHON_BIN}"
  command -v "${ACCELERATE_BIN}" >/dev/null 2>&1 || die "ACCELERATE_BIN not found: ${ACCELERATE_BIN}"
  [[ -d "${DATASET_ROOT}" ]] || die "DATASET_ROOT does not exist: ${DATASET_ROOT}"
fi

# Resolve the selected episodes and exact frame/batch arithmetic using the
# same metadata class used by lerobot-train.  In plan mode, a missing dataset
# leaves symbolic values and never pretends that an epoch count was measured.
RESOLVED_STATS=""
if [[ -d "${DATASET_ROOT}" && -f "${DATASET_ROOT}/meta/info.json" ]] \
    && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  RESOLVED_STATS="$(${PYTHON_BIN} - "${DATASET_REPO_ID}" "${DATASET_ROOT}" "${TRAIN_EPISODE_START}" "${TRAIN_EPISODE_END}" "${TRAIN_EPISODES}" "${NUM_PROCESSES}" "${BATCH_SIZE}" "${ALLOW_PARTIAL}" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

repo_id, root_raw, start_raw, end_raw, explicit_raw, world_raw, batch_raw, allow_partial = sys.argv[1:]
root = Path(root_raw)
start = int(start_raw)
if start < 0:
    raise SystemExit("TRAIN_EPISODE_START must be non-negative")
meta = LeRobotDatasetMetadata(repo_id, root=root)
episodes = meta.episodes
if isinstance(episodes, dict):
    rows = [(i, int(frm), int(to)) for i, (frm, to) in enumerate(
        zip(episodes["dataset_from_index"], episodes["dataset_to_index"])
    )]
else:
    rows = [(int(row.get("episode_index", i)), int(row["dataset_from_index"]),
             int(row["dataset_to_index"])) for i, row in enumerate(episodes)]
available = {index: (frm, to) for index, frm, to in rows}
all_indices = sorted(available)
if explicit_raw:
    try:
        selected = [int(x.strip()) for x in explicit_raw.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"TRAIN_EPISODES must be comma-separated integers: {exc}")
else:
    end = (max(all_indices) + 1) if end_raw == "auto" and all_indices else int(end_raw)
    if end <= start:
        raise SystemExit(f"TRAIN_EPISODE_END must be greater than start ({start}); got {end}")
    selected = list(range(start, end))
if not selected:
    raise SystemExit("selected episode set is empty")
if len(set(selected)) != len(selected) or min(selected) < 0:
    raise SystemExit("selected episodes must be unique non-negative indices")
unknown = [i for i in selected if i not in available]
if unknown:
    raise SystemExit(f"selected episodes out of range: {unknown[:8]}")
all_official = selected == all_indices
if not all_official and allow_partial != "1":
    raise SystemExit("canonical Native protocol requires every official episode")
frames = sum(available[i][1] - available[i][0] for i in selected)
if frames <= 0:
    raise SystemExit("selected episodes contain no frames")
batch = int(batch_raw)
world = int(world_raw)
if batch <= 0 or world <= 0:
    raise SystemExit("batch and world size must be positive")
# The standard map-style path uses drop_last=False.  Accelerate shards
# complete local batches and pads the batch count across ranks; this gives the
# exact per-rank optimizer-step count for one frame epoch.
raw_batches = (frames + batch - 1) // batch
steps_per_epoch = (raw_batches + world - 1) // world
effective_slots = steps_per_epoch * world * batch
payload = {
    "schema": "native_smolvla_training_batch_v1",
    "dataset_repo_id": repo_id,
    "dataset_root": str(root),
    "available_episode_indices": all_indices,
    "train_episode_indices": selected,
    "num_episodes": len(selected),
    "num_frames": frames,
    "fps": int(getattr(meta, "fps", 0) or 0),
    "per_device_batch_size": batch,
    "world_size": world,
    "global_batch_size": batch * world,
    "raw_batches": raw_batches,
    "steps_per_epoch": steps_per_epoch,
    "effective_frame_slots_per_epoch": effective_slots,
    "sampler_repeated_slots_per_epoch": effective_slots - frames,
    "all_official_demos": all_official,
}
print(json.dumps(payload, separators=(",", ":")))
PY
  )" || die "could not resolve dataset metadata"
fi

if [[ -n "${RESOLVED_STATS}" ]]; then
  RESOLVED_EPISODES_JSON="$(${PYTHON_BIN} -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1])["train_episode_indices"], separators=(",", ":")))' "${RESOLVED_STATS}")"
  NUM_EPISODES="$(${PYTHON_BIN} -c 'import json,sys; print(json.loads(sys.argv[1])["num_episodes"])' "${RESOLVED_STATS}")"
  NUM_FRAMES="$(${PYTHON_BIN} -c 'import json,sys; print(json.loads(sys.argv[1])["num_frames"])' "${RESOLVED_STATS}")"
  STEPS_PER_EPOCH="$(${PYTHON_BIN} -c 'import json,sys; print(json.loads(sys.argv[1])["steps_per_epoch"])' "${RESOLVED_STATS}")"
  ALL_OFFICIAL="$(${PYTHON_BIN} -c 'import json,sys; print("1" if json.loads(sys.argv[1])["all_official_demos"] else "0")' "${RESOLVED_STATS}")"
  [[ "${ALL_OFFICIAL}" == "1" || "${ALLOW_PARTIAL}" == "1" ]] || die "canonical Native run must use every official episode"
  [[ -n "${STEPS}" ]] || STEPS=$((STEPS_PER_EPOCH * EPOCHS))
else
  RESOLVED_EPISODES_JSON="<ALL_DATASET_EPISODES>"
  NUM_EPISODES="<N_EPISODES>"
  NUM_FRAMES="<N_FRAMES>"
  STEPS_PER_EPOCH="<CEIL(N_FRAMES/(BATCH_SIZE*WORLD_SIZE))>"
  [[ -n "${STEPS}" ]] || STEPS="<STEPS_PER_EPOCH*EPOCHS>"
fi

if [[ "${STEPS}" =~ ^[0-9]+$ ]]; then
  [[ "${STEPS}" -gt 0 ]] || die "STEPS must be positive"
  if [[ "${STEPS_WAS_EXPLICIT}" == "1" && "${ALLOW_SHORT_RUN}" != "1" && "${STEPS_PER_EPOCH}" =~ ^[0-9]+$ ]]; then
    (( STEPS % STEPS_PER_EPOCH == 0 )) || die "explicit STEPS=${STEPS} is not a complete frame-epoch count (steps/epoch=${STEPS_PER_EPOCH})"
  fi
  if [[ "${ALLOW_SHORT_RUN}" != "1" && "${EPOCHS}" -lt "${MIN_EPOCHS}" ]]; then
    die "canonical Native run requires at least ${MIN_EPOCHS} epochs; got ${EPOCHS}"
  fi
fi

# Reject a local TTT checkpoint before model construction.  Hub IDs are
# checked by the policy loader; this local check provides an early, readable
# failure for accidental cross-baseline evaluation.
if [[ -d "${BASE_CHECKPOINT}" && -f "${BASE_CHECKPOINT}/config.json" ]] \
    && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  CHECKPOINT_TYPE="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("type", ""))' "${BASE_CHECKPOINT}/config.json")"
  [[ "${CHECKPOINT_TYPE}" == "smolvla" ]] || die "Native baseline requires config type=smolvla, got '${CHECKPOINT_TYPE}'"
fi

write_metadata() {
  [[ -n "${RESOLVED_STATS}" ]] || return 0
  "${PYTHON_BIN}" - "${TRAINING_METADATA_PATH}" "${RESOLVED_STATS}" "${BASE_CHECKPOINT}" "${OUTPUT_DIR}" "${TASK_ID}" "${EPOCHS}" "${STEPS}" "${STEPS_PER_EPOCH}" "${BATCH_SIZE}" "${NUM_PROCESSES}" "${SEED}" "${STEPS_WAS_EXPLICIT}" <<'PY'
import json
import sys
from pathlib import Path

out, stats_raw, checkpoint, output_dir, task_id, epochs, steps, spe, batch, world, seed, explicit = sys.argv[1:]
stats = json.loads(stats_raw)
payload = {
    "schema": "native_smolvla_training_metadata_v1",
    "protocol": "native_smolvla_frame_epoch",
    "task_id": task_id,
    "dataset_repo_id": stats["dataset_repo_id"],
    "dataset_root": stats["dataset_root"],
    "train_episode_indices": stats["train_episode_indices"],
    "all_official_demos": bool(stats["all_official_demos"]),
    "num_episodes": int(stats["num_episodes"]),
    "num_frames": int(stats["num_frames"]),
    "dataset_fps": int(stats["fps"]),
    "epochs": int(epochs),
    "steps": int(steps),
    "steps_per_epoch": int(spe),
    "complete_frame_epochs": int(steps) // int(spe) if int(steps) % int(spe) == 0 else None,
    "steps_was_explicit": bool(int(explicit)),
    "per_device_batch_size": int(batch),
    "world_size": int(world),
    "global_batch_size": int(batch) * int(world),
    "effective_batch_size": int(batch) * int(world),
    "sampler_repeated_slots_per_epoch": int(stats["sampler_repeated_slots_per_epoch"]),
    "policy_type": "smolvla",
    "pretrained_checkpoint": checkpoint,
    "output_dir": output_dir,
    "seed": int(seed),
    "chunk_size": 50,
    "n_action_steps": 50,
    "ttt_enabled": False,
    "hd_ttt_enabled": False,
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

LAUNCH=(
  "${ACCELERATE_BIN}" launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --mixed_precision="${MIXED_PRECISION}"
  --dynamo_backend=no
  -m lerobot.scripts.lerobot_train
)
ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.episodes="${RESOLVED_EPISODES_JSON}"
  --dataset.video_backend=pyav
  --dataset.return_uint8=true
  --policy.type=smolvla
  --policy.pretrained_path="${BASE_CHECKPOINT}"
  --policy.device="${DEVICE}"
  --policy.push_to_hub=false
  --policy.chunk_size=50
  --policy.n_action_steps=50
  --policy.rtc_config=null
  --policy.compile_model=false
  --batch_size="${BATCH_SIZE}"
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
  --output_dir="${OUTPUT_DIR}"
)

echo "Native SmolVLA protocol: policy.type=smolvla (no fast weights, registers, HD labels, or TTT flags)"
echo "task=${TASK_ID} dataset=${DATASET_REPO_ID} root=${DATASET_ROOT}"
echo "episodes=${NUM_EPISODES} frames=${NUM_FRAMES} all_official=${ALL_OFFICIAL:-unknown}"
echo "per_device_batch=${BATCH_SIZE} processes=${NUM_PROCESSES} global_batch=$((BATCH_SIZE * NUM_PROCESSES))"
echo "steps_per_frame_epoch=${STEPS_PER_EPOCH} epochs=${EPOCHS} total_steps=${STEPS} resume=${RESUME}"
echo "metadata=${TRAINING_METADATA_PATH} output=${OUTPUT_DIR}"
echo "command:"
print_command "${LAUNCH[@]}" "${ARGS[@]}"

if [[ "${STAGE}" == "plan" ]]; then
  echo "Plan only: no metadata, checkpoint, or training process was created."
  exit 0
fi

if [[ "${RESUME}" == "false" && -d "${OUTPUT_DIR}" ]]; then
  # Never delete user data implicitly; lerobot-train rejects a non-empty fresh
  # output directory, so require a new path instead.
  if find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "OUTPUT_DIR is non-empty while RESUME=false: ${OUTPUT_DIR}"
  fi
fi
mkdir -p "${OUTPUT_DIR}"
write_metadata
cd "${REPO_ROOT}"
exec "${LAUNCH[@]}" "${ARGS[@]}"
