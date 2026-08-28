#!/usr/bin/env bash

# Train the Native SmolVLA baseline on all 50 RMBench cover_blocks/demo_clean
# demonstrations for exactly 150 complete frame epochs.
#
# `plan` performs the complete dataset/checkpoint/output preflight and prints
# the exact command without writing anything. `run` additionally requires
# EXECUTE=1, writes a sidecar training manifest, and starts Accelerate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
ACCELERATE_BIN="${VENV_DIR}/bin/accelerate"

MODE="${1:-plan}"
EXECUTE="${EXECUTE:-0}"

readonly TASK_ID="cover_blocks"
readonly DATASET_REPO_ID="rmbench/cover_blocks_demo_clean"
readonly EXPECTED_EPISODES=50
readonly EXPECTED_FRAMES=51077
readonly EPOCHS=150

DATASET_ROOT="${DATASET_ROOT:-/workspace/data_rmbench_lerobot/cover_blocks_demo_clean}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/models/smolvla_base}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/experiments/rmbench_cover_blocks/native_smolvla_150ep}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
MANIFEST_PATH="${OUTPUT_DIR}.training_manifest.json"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"
LOG_FREQ="${LOG_FREQ:-50}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
SEED="${SEED:-1000}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/data_rmbench_lerobot}"

usage() {
  cat <<'EOF'
Usage:
  examples/rmbench/train_cover_blocks_smolvla.sh plan
  EXECUTE=1 examples/rmbench/train_cover_blocks_smolvla.sh run

The canonical defaults are:
  dataset repo id:  rmbench/cover_blocks_demo_clean
  dataset root:     /workspace/data_rmbench_lerobot/cover_blocks_demo_clean
  source model:     /workspace/models/smolvla_base
  output:           /workspace/experiments/rmbench_cover_blocks/native_smolvla_150ep
  duration:         150 complete frame epochs over all 50 demos / 51,077 transitions
  distributed:      4 GPUs, per-device batch 32, bf16 (global batch 128)

Safe path/hardware overrides:
  DATASET_ROOT=... BASE_CHECKPOINT=... OUTPUT_DIR=...
  NUM_PROCESSES=4 BATCH_SIZE=32 MIXED_PRECISION=bf16
  NUM_WORKERS=4 PREFETCH_FACTOR=2 MAIN_PROCESS_PORT=29500
  SAVE_EVERY_EPOCHS=10 LOG_FREQ=50 SEED=1000

The launcher never resumes and refuses an existing output directory or
training manifest. Python and Accelerate are always taken from this Method1
checkout's .venv.
EOF
}

die() {
  echo "RMBench Native SmolVLA preflight: $*" >&2
  exit 2
}

require_positive_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ && "${value}" -gt 0 ]] \
    || die "${name} must be a positive integer; got '${value}'"
}

path_exists() {
  [[ -e "$1" || -L "$1" ]]
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

case "${MODE}" in
  plan|run) ;;
  help|-h|--help)
    usage
    exit 0
    ;;
  *)
    die "unknown mode '${MODE}'; use plan or run"
    ;;
esac

if [[ "${MODE}" == "run" && "${EXECUTE}" != "1" ]]; then
  die "run mode requires EXECUTE=1 after reviewing plan mode"
fi

require_positive_int NUM_PROCESSES "${NUM_PROCESSES}"
require_positive_int BATCH_SIZE "${BATCH_SIZE}"
require_positive_int NUM_WORKERS "${NUM_WORKERS}"
require_positive_int PREFETCH_FACTOR "${PREFETCH_FACTOR}"
require_positive_int SAVE_EVERY_EPOCHS "${SAVE_EVERY_EPOCHS}"
require_positive_int LOG_FREQ "${LOG_FREQ}"
require_positive_int MAIN_PROCESS_PORT "${MAIN_PROCESS_PORT}"
require_positive_int SEED "${SEED}"
[[ "${MAIN_PROCESS_PORT}" -le 65535 ]] \
  || die "MAIN_PROCESS_PORT must be at most 65535; got '${MAIN_PROCESS_PORT}'"
case "${MIXED_PRECISION}" in
  bf16|fp16|no) ;;
  *) die "MIXED_PRECISION must be bf16, fp16, or no; got '${MIXED_PRECISION}'" ;;
esac

[[ -n "${OUTPUT_DIR}" && "${OUTPUT_DIR}" != "/" ]] || die "unsafe OUTPUT_DIR '${OUTPUT_DIR}'"
[[ "${DATASET_ROOT}" == /* ]] || die "DATASET_ROOT must be absolute"
[[ "${BASE_CHECKPOINT}" == /* ]] || die "BASE_CHECKPOINT must be absolute"
[[ "${OUTPUT_DIR}" == /* ]] || die "OUTPUT_DIR must be absolute"

[[ -x "${PYTHON_BIN}" ]] \
  || die "Method1 virtualenv Python is missing or not executable: ${PYTHON_BIN}"
[[ -d "${DATASET_ROOT}" ]] || die "dataset directory does not exist: ${DATASET_ROOT}"
[[ -f "${DATASET_ROOT}/meta/info.json" ]] \
  || die "LeRobot metadata is missing: ${DATASET_ROOT}/meta/info.json"
[[ -d "${BASE_CHECKPOINT}" ]] || die "base checkpoint directory does not exist: ${BASE_CHECKPOINT}"
[[ -f "${BASE_CHECKPOINT}/config.json" ]] \
  || die "base checkpoint config is missing: ${BASE_CHECKPOINT}/config.json"

if path_exists "${OUTPUT_DIR}"; then
  die "refusing to overwrite existing output path: ${OUTPUT_DIR}"
fi
if path_exists "${MANIFEST_PATH}"; then
  die "refusing to overwrite existing training manifest: ${MANIFEST_PATH}"
fi

CHECKPOINT_TYPE="$("${PYTHON_BIN}" - "${BASE_CHECKPOINT}/config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"cannot read checkpoint config {path}: {exc}") from exc
print(payload.get("type", ""))
PY
)" || die "could not inspect base checkpoint type"
[[ "${CHECKPOINT_TYPE}" == "smolvla" ]] \
  || die "Native baseline requires checkpoint config type=smolvla; got '${CHECKPOINT_TYPE:-<missing>}'"

# The values below come only from LeRobot's own metadata loader. In addition
# to counting episodes, validate contiguous frame bounds and the exact feature
# schema emitted by convert_cover_blocks_to_lerobot.py. This prevents a smoke
# subset or a differently converted dataset from being mistaken for the full
# baseline corpus.
if ! METADATA_OUTPUT="$("${PYTHON_BIN}" - \
    "${DATASET_REPO_ID}" \
    "${DATASET_ROOT}" \
    "${EXPECTED_EPISODES}" \
    "${EXPECTED_FRAMES}" \
    "${BATCH_SIZE}" \
    "${NUM_PROCESSES}" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

repo_id, root_raw, expected_raw, expected_frames_raw, batch_raw, world_raw = sys.argv[1:]
root = Path(root_raw)
expected = int(expected_raw)
expected_frames = int(expected_frames_raw)
batch = int(batch_raw)
world = int(world_raw)

meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
episodes = meta.episodes
columns = set(episodes.column_names)
required_columns = {
    "episode_index",
    "length",
    "dataset_from_index",
    "dataset_to_index",
}
missing_columns = sorted(required_columns - columns)
if missing_columns:
    raise SystemExit(f"episode metadata is missing columns: {missing_columns}")

indices = [int(value) for value in episodes["episode_index"]]
lengths = [int(value) for value in episodes["length"]]
from_indices = [int(value) for value in episodes["dataset_from_index"]]
to_indices = [int(value) for value in episodes["dataset_to_index"]]

if int(meta.total_episodes) != expected or len(episodes) != expected:
    raise SystemExit(
        "full cover_blocks dataset requires exactly "
        f"{expected} demos; info.json reports {meta.total_episodes}, "
        f"episode metadata contains {len(episodes)}"
    )
expected_indices = list(range(expected))
if indices != expected_indices:
    raise SystemExit(
        f"episode indices must be contiguous 0..{expected - 1}; got {indices}"
    )

cursor = 0
for episode_index, length, frame_from, frame_to in zip(
    indices, lengths, from_indices, to_indices, strict=True
):
    if length <= 0:
        raise SystemExit(f"episode {episode_index} has non-positive length {length}")
    if frame_from != cursor or frame_to != frame_from + length:
        raise SystemExit(
            f"episode {episode_index} has inconsistent frame bounds: "
            f"length={length}, from={frame_from}, to={frame_to}, expected_from={cursor}"
        )
    cursor = frame_to

frames = sum(lengths)
if frames != cursor or frames != int(meta.total_frames):
    raise SystemExit(
        "frame metadata is inconsistent: "
        f"sum(length)={frames}, final_to={cursor}, info.total_frames={meta.total_frames}"
    )
if frames <= 0:
    raise SystemExit("dataset contains no training frames")
if frames != expected_frames:
    raise SystemExit(
        "full 50-demo cover_blocks conversion requires exactly "
        f"{expected_frames} transitions; metadata contains {frames}. "
        "Do not train from a max-frames/debug conversion."
    )

if int(meta.fps) != 30:
    raise SystemExit(f"cover_blocks conversion must be 30 FPS; got {meta.fps}")
if meta.robot_type != "aloha_agilex":
    raise SystemExit(
        f"cover_blocks conversion must declare robot_type=aloha_agilex; got {meta.robot_type!r}"
    )

required_features = {
    "observation.state": ("float32", (14,)),
    "action": ("float32", (14,)),
    "observation.images.head_camera": ("video", None),
}
for key, (expected_dtype, expected_shape) in required_features.items():
    if key not in meta.features:
        raise SystemExit(f"dataset is missing required feature {key!r}")
    feature = meta.features[key]
    if feature.get("dtype") != expected_dtype:
        raise SystemExit(
            f"feature {key!r} must have dtype={expected_dtype}; got {feature.get('dtype')!r}"
        )
    shape = tuple(int(value) for value in feature.get("shape", ()))
    if expected_shape is not None and shape != expected_shape:
        raise SystemExit(f"feature {key!r} must have shape {expected_shape}; got {shape}")
    if key == "observation.images.head_camera" and (len(shape) != 3 or shape[-1] != 3):
        raise SystemExit(f"head-camera feature must have HWC RGB shape; got {shape}")

# DataLoader(drop_last=False) first forms per-device batches, then Accelerate
# shards those batches across ranks with even_batches=True. Thus this is the
# exact number of optimizer updates in one complete distributed frame epoch.
raw_batches = (frames + batch - 1) // batch
steps_per_epoch = (raw_batches + world - 1) // world
global_batch = batch * world
effective_slots = steps_per_epoch * global_batch

payload = {
    "repo_id": repo_id,
    "root": str(root),
    "robot_type": meta.robot_type,
    "fps": int(meta.fps),
    "episode_indices": indices,
    "episode_lengths": lengths,
    "num_episodes": len(indices),
    "num_frames": frames,
    "features": {
        key: {
            "dtype": feature["dtype"],
            "shape": list(feature["shape"]),
        }
        for key, feature in meta.features.items()
    },
    "per_device_batch_size": batch,
    "world_size": world,
    "global_batch_size": global_batch,
    "raw_batches_before_sharding": raw_batches,
    "steps_per_epoch": steps_per_epoch,
    "effective_frame_slots_per_epoch": effective_slots,
    "accelerate_repeated_slots_per_epoch": effective_slots - frames,
}

print(json.dumps(payload, separators=(",", ":")))
print(json.dumps(indices, separators=(",", ":")))
print(len(indices))
print(frames)
print(raw_batches)
print(steps_per_epoch)
print(global_batch)
print(effective_slots - frames)
print(meta.fps)
PY
)"; then
  die "dataset metadata validation failed"
fi

mapfile -t METADATA_LINES <<<"${METADATA_OUTPUT}"
[[ "${#METADATA_LINES[@]}" -eq 9 ]] \
  || die "internal metadata resolver returned ${#METADATA_LINES[@]} lines instead of 9"
METADATA_JSON="${METADATA_LINES[0]}"
EPISODES_JSON="${METADATA_LINES[1]}"
NUM_EPISODES="${METADATA_LINES[2]}"
NUM_FRAMES="${METADATA_LINES[3]}"
RAW_BATCHES="${METADATA_LINES[4]}"
STEPS_PER_EPOCH="${METADATA_LINES[5]}"
GLOBAL_BATCH_SIZE="${METADATA_LINES[6]}"
REPEATED_SLOTS="${METADATA_LINES[7]}"
DATASET_FPS="${METADATA_LINES[8]}"

TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
SAVE_FREQ=$((STEPS_PER_EPOCH * SAVE_EVERY_EPOCHS))

LAUNCH=(
  "${ACCELERATE_BIN}" launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --main_process_port="${MAIN_PROCESS_PORT}"
  --mixed_precision="${MIXED_PRECISION}"
  --dynamo_backend=no
  -m lerobot.scripts.lerobot_train
)
TRAIN_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.episodes="${EPISODES_JSON}"
  --dataset.video_backend=pyav
  --dataset.return_uint8=true
  --policy.type=smolvla
  --policy.pretrained_path="${BASE_CHECKPOINT}"
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.chunk_size=50
  --policy.n_action_steps=50
  --policy.compile_model=false
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --prefetch_factor="${PREFETCH_FACTOR}"
  --persistent_workers=true
  --steps="${TOTAL_STEPS}"
  --log_freq="${LOG_FREQ}"
  --save_checkpoint=true
  --save_freq="${SAVE_FREQ}"
  --eval_freq=0
  --resume=false
  --wandb.enable=false
  --seed="${SEED}"
  --output_dir="${OUTPUT_DIR}"
)

echo "RMBench Native SmolVLA baseline (fresh run only)"
echo "task=${TASK_ID} dataset=${DATASET_REPO_ID} root=${DATASET_ROOT}"
echo "episodes=${NUM_EPISODES} frames=${NUM_FRAMES} fps=${DATASET_FPS}"
echo "per_device_batch=${BATCH_SIZE} processes=${NUM_PROCESSES} global_batch=${GLOBAL_BATCH_SIZE}"
echo "raw_batches=${RAW_BATCHES} steps_per_epoch=${STEPS_PER_EPOCH} repeated_slots_per_epoch=${REPEATED_SLOTS}"
echo "epochs=${EPOCHS} total_steps=${TOTAL_STEPS} precision=${MIXED_PRECISION}"
echo "checkpoint=${BASE_CHECKPOINT} checkpoint_type=${CHECKPOINT_TYPE}"
echo "output=${OUTPUT_DIR}"
echo "manifest=${MANIFEST_PATH}"
echo "command:"
print_command "${LAUNCH[@]}" "${TRAIN_ARGS[@]}"

if [[ "${MODE}" == "plan" ]]; then
  if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Plan warning: Method1 Accelerate is not installed yet at ${ACCELERATE_BIN}." >&2
  fi
  echo "Plan only: no manifest, output directory, checkpoint, or process was created."
  exit 0
fi

[[ -x "${ACCELERATE_BIN}" ]] \
  || die "Method1 virtualenv Accelerate is missing or not executable: ${ACCELERATE_BIN}"

"${PYTHON_BIN}" - "${NUM_PROCESSES}" "${MIXED_PRECISION}" <<'PY' \
  || die "GPU/bf16 preflight failed"
import sys

import torch

required = int(sys.argv[1])
precision = sys.argv[2]
available = torch.cuda.device_count()
if available < required:
    raise SystemExit(f"need {required} visible CUDA devices, but torch sees {available}")
if precision == "bf16":
    for index in range(required):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise SystemExit(f"CUDA device {index} does not support bf16")
PY

# Re-check immediately before committing the manifest, narrowing the race with
# another launch targeting the same experiment directory.
if path_exists "${OUTPUT_DIR}"; then
  die "refusing to overwrite output path created after preflight: ${OUTPUT_DIR}"
fi
if path_exists "${MANIFEST_PATH}"; then
  die "refusing to overwrite manifest created after preflight: ${MANIFEST_PATH}"
fi

"${PYTHON_BIN}" - \
  "${MANIFEST_PATH}" \
  "${METADATA_JSON}" \
  "${BASE_CHECKPOINT}" \
  "${OUTPUT_DIR}" \
  "${EPOCHS}" \
  "${TOTAL_STEPS}" \
  "${STEPS_PER_EPOCH}" \
  "${BATCH_SIZE}" \
  "${NUM_PROCESSES}" \
  "${MIXED_PRECISION}" \
  "${SAVE_FREQ}" \
  "${LOG_FREQ}" \
  "${SEED}" \
  "${SCRIPT_DIR}/train_cover_blocks_smolvla.sh" \
  "${REPO_ROOT}" \
  "${LAUNCH[@]}" \
  "${TRAIN_ARGS[@]}" <<'PY'
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

(
    manifest_raw,
    metadata_raw,
    checkpoint_raw,
    output_raw,
    epochs_raw,
    steps_raw,
    steps_per_epoch_raw,
    batch_raw,
    world_raw,
    precision,
    save_freq_raw,
    log_freq_raw,
    seed_raw,
    script_raw,
    repo_root_raw,
    *command,
) = sys.argv[1:]

manifest = Path(manifest_raw)
checkpoint = Path(checkpoint_raw)
repo_root = Path(repo_root_raw)
try:
    git_commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    git_commit = None

config_bytes = (checkpoint / "config.json").read_bytes()
dataset = json.loads(metadata_raw)
payload = {
    "schema": "rmbench_native_smolvla_training_manifest_v1",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "protocol": "rmbench_cover_blocks_native_smolvla_full50_frame_epochs",
    "task_id": "cover_blocks",
    "dataset": dataset,
    "training": {
        "epochs": int(epochs_raw),
        "steps_per_epoch": int(steps_per_epoch_raw),
        "total_steps": int(steps_raw),
        "step_formula": "ceil(ceil(num_frames/per_device_batch)/world_size) * epochs",
        "per_device_batch_size": int(batch_raw),
        "world_size": int(world_raw),
        "global_batch_size": int(batch_raw) * int(world_raw),
        "mixed_precision": precision,
        "save_freq_steps": int(save_freq_raw),
        "log_freq_steps": int(log_freq_raw),
        "seed": int(seed_raw),
        "resume": False,
    },
    "policy": {
        "type": "smolvla",
        "pretrained_path": str(checkpoint),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "ttt_enabled": False,
    },
    "output_dir": output_raw,
    "launcher": script_raw,
    "repo_root": str(repo_root),
    "git_commit": git_commit,
    "command": command,
}

manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(f"Wrote training manifest: {manifest}")
PY

cd "${REPO_ROOT}"
exec "${LAUNCH[@]}" "${TRAIN_ARGS[@]}"
