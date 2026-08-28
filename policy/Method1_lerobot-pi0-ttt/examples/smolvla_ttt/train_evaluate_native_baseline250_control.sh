#!/usr/bin/env bash

# Train a strong native SmolVLA duration control from the task's 150-epoch
# checkpoint for 100 additional frame epochs, then run the canonical MIKASA
# 50-episode evaluation. The original 150-epoch run is never modified.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD_REPO_ROOT="${METHOD_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
POLICY_ROOT="$(dirname -- "${METHOD_REPO_ROOT}")"
HD_REPO_ROOT="${HD_REPO_ROOT:-${POLICY_ROOT}/HD-TTT}"
MODE="${1:-plan}"
EXECUTE="${EXECUTE:-0}"
TASK_ID="${TASK_ID:-remember_shape5}"

die() {
  echo "Native SmolVLA 250-epoch control: $*" >&2
  exit 2
}

case "${MODE}" in
  plan|run) ;;
  *) die "mode must be plan or run" ;;
esac

case "${TASK_ID}" in
  remember_shape5)
    DATASET_REPO_ID=remember_shape_5_vla_v0
    ENV_ID=RememberShape5-VLA-v0
    DEFAULT_GPU_A=2
    DEFAULT_GPU_B=3
    STEPS_PER_EPOCH=59
    EXTRA_STEPS=5900
    SOURCE_STEPS=8850
    SAVE_INTERVAL=2950
    DEFAULT_PORT=29652
    DEFAULT_WAIT_SERVICE=""
    DEFAULT_WAIT_LOG=""
    ;;
  shuffle_touch)
    DATASET_REPO_ID=shell_game_shuffle_touch_vla_v0
    ENV_ID=ShellGameShuffleTouch-VLA-v0
    DEFAULT_GPU_A=0
    DEFAULT_GPU_B=1
    STEPS_PER_EPOCH=165
    EXTRA_STEPS=16500
    SOURCE_STEPS=24750
    SAVE_INTERVAL=8250
    DEFAULT_PORT=29653
    DEFAULT_WAIT_SERVICE=smolvla_ttt_shuffle_touch_official_eval3
    DEFAULT_WAIT_LOG=/workspace/logs/smolvla_ttt_shuffle_touch_official_eval3.log
    ;;
  *) die "TASK_ID must be remember_shape5 or shuffle_touch" ;;
esac

GPU_A="${GPU_A:-${DEFAULT_GPU_A}}"
GPU_B="${GPU_B:-${DEFAULT_GPU_B}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${DEFAULT_PORT}}"
WAIT_FOR_SERVICE="${WAIT_FOR_SERVICE-${DEFAULT_WAIT_SERVICE}}"
WAIT_FOR_LOG="${WAIT_FOR_LOG-${DEFAULT_WAIT_LOG}}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"

SOURCE_ROOT="${SOURCE_ROOT:-/workspace/experiments/native_smolvla_short_memory_b32_150ep_20260827/${TASK_ID}}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SOURCE_ROOT}/checkpoints/last/pretrained_model}"
SOURCE_METADATA="${SOURCE_METADATA:-$(dirname -- "${SOURCE_ROOT}")/${TASK_ID}_training_metadata.json}"
OUTPUT_PARENT="${OUTPUT_PARENT:-/workspace/experiments/native_smolvla_short_memory_b32_150plus100ep_control_20260827}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_PARENT}/${TASK_ID}}"
OUTPUT_CHECKPOINT="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
OUTPUT_METADATA="${OUTPUT_PARENT}/${TASK_ID}_training_metadata.json"
CONTROL_MANIFEST="${OUTPUT_DIR}/baseline250_control_manifest.json"

EVALUATION_ROOT="${EVALUATION_ROOT:-/workspace/evaluations/mikasa_official50_20260827/${TASK_ID}}"
EVALUATION_OUTPUT="${EVALUATION_ROOT}/baseline_250ep_warmstart_k50/eval.json"
OFFICIAL_OUTPUT_DIR="${EVALUATION_ROOT}/baseline_250ep_warmstart_k50/official"
COMPARISON_INPUT="${EVALUATION_ROOT}/comparison.json"
COMPARISON_OUTPUT="${EVALUATION_ROOT}/comparison_with_baseline250.json"

DATASET_ROOT="${DATASET_ROOT:-/workspace/data_mikasa_robo/data_lerobot/${DATASET_REPO_ID}}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${HD_REPO_ROOT}/examples/mikasa/train_native_smolvla.sh}"
EVALUATOR="${EVALUATOR:-${SCRIPT_DIR}/evaluate_mikasa.py}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/venv/main/bin/python3}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/venv/main/bin/accelerate}"
EVAL_PYTHON="${EVAL_PYTHON:-/workspace/MIKASA-Robo/.venv/bin/python}"

for integer_name in GPU_A GPU_B MAIN_PROCESS_PORT WAIT_SECONDS; do
  integer_value="${!integer_name}"
  [[ "${integer_value}" =~ ^[0-9]+$ ]] || die "${integer_name} must be a non-negative integer"
done
[[ "${GPU_A}" != "${GPU_B}" ]] || die "GPU_A and GPU_B must differ"
(( WAIT_SECONDS > 0 )) || die "WAIT_SECONDS must be positive"

echo "Native SmolVLA staged duration control"
echo "task=${TASK_ID} env=${ENV_ID} dataset=${DATASET_REPO_ID}"
echo "source=${SOURCE_CHECKPOINT} source_epochs=150 source_steps=${SOURCE_STEPS}"
echo "extra_epochs=100 steps_per_epoch=${STEPS_PER_EPOCH} extra_steps=${EXTRA_STEPS}"
echo "optimizer=fresh scheduler=fresh per_device_batch=32 processes=2 global_batch=64"
echo "output=${OUTPUT_DIR} total_demonstration_epochs=250"
echo "gpu_pair=${GPU_A},${GPU_B} wait_for_service=${WAIT_FOR_SERVICE:-none}"
echo "evaluation=${EVALUATION_OUTPUT}"

if [[ "${MODE}" == plan ]]; then
  echo "Plan only: no process or output was created."
  exit 0
fi

[[ "${EXECUTE}" == 1 ]] || die "run mode requires EXECUTE=1"
[[ -x "${TRAIN_PYTHON}" ]] || die "training Python not found: ${TRAIN_PYTHON}"
[[ -x "${ACCELERATE_BIN}" ]] || die "accelerate not found: ${ACCELERATE_BIN}"
[[ -x "${EVAL_PYTHON}" ]] || die "MIKASA Python not found: ${EVAL_PYTHON}"
[[ -x "${TRAIN_LAUNCHER}" ]] || die "native training launcher not found: ${TRAIN_LAUNCHER}"
[[ -f "${EVALUATOR}" ]] || die "MIKASA evaluator not found: ${EVALUATOR}"
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || die "dataset metadata missing: ${DATASET_ROOT}"
command -v supervisorctl >/dev/null || die "supervisorctl is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/data_mikasa_robo}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

validate_source() {
  "${TRAIN_PYTHON}" - \
    "${SOURCE_CHECKPOINT}" "${SOURCE_METADATA}" "${TASK_ID}" \
    "${SOURCE_STEPS}" "${STEPS_PER_EPOCH}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
metadata_path = Path(sys.argv[2])
task_id = sys.argv[3]
source_steps = int(sys.argv[4])
steps_per_epoch = int(sys.argv[5])
for filename in (
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
):
    path = checkpoint / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing source checkpoint file: {path}")
config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
train = json.loads((checkpoint / "train_config.json").read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
checks = {
    "policy type": config.get("type") == "smolvla",
    "source steps": int(train.get("steps", -1)) == source_steps,
    "source batch": int(train.get("batch_size", -1)) == 32,
    "task id": metadata.get("task_id") == task_id,
    "source epochs": int(metadata.get("complete_frame_epochs", -1)) == 150,
    "steps per epoch": int(metadata.get("steps_per_epoch", -1)) == steps_per_epoch,
    "world size": int(metadata.get("world_size", -1)) == 2,
    "all demonstrations": metadata.get("all_official_demos") is True,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"source control validation failed: {failed}")
PY
}

wait_for_dependency() {
  [[ -n "${WAIT_FOR_SERVICE}" ]] || return 0
  while true; do
    local status
    status="$(supervisorctl status "${WAIT_FOR_SERVICE}" 2>&1 || true)"
    echo "dependency_status=${status}"
    if grep -Eq '\b(RUNNING|STARTING|STOPPING)\b' <<<"${status}"; then
      sleep "${WAIT_SECONDS}"
      continue
    fi
    if grep -Eq '\b(EXITED|STOPPED)\b' <<<"${status}"; then
      break
    fi
    die "dependency entered an unsafe state: ${status}"
  done
  [[ -f "${COMPARISON_INPUT}" ]] \
    || die "dependency stopped without completed three-model comparison: ${COMPARISON_INPUT}"
  [[ -f "${WAIT_FOR_LOG}" ]] \
    && grep -Fq "All three canonical evaluations complete: ${COMPARISON_INPUT}" "${WAIT_FOR_LOG}" \
    || die "dependency stopped without its canonical completion marker"
  "${EVAL_PYTHON}" - "${EVALUATION_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for label in ("baseline_native_k50", "stage1_ttt_only", "stage2_action_head"):
    path = root / label / "eval.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["results"][0]
    if len(result["successes"]) != 50 or result["start_seed"] != 4242424242:
        raise SystemExit(f"dependency result is incomplete or non-canonical: {path}")
PY
}

wait_for_gpu_pair_idle() {
  local uuid_a uuid_b active
  uuid_a="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU_A}")"
  uuid_b="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU_B}")"
  while true; do
    active="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null || true)"
    if ! grep -Fq "${uuid_a}" <<<"${active}" && ! grep -Fq "${uuid_b}" <<<"${active}"; then
      return 0
    fi
    echo "Waiting for GPU pair ${GPU_A},${GPU_B} to become compute-idle."
    sleep "${WAIT_SECONDS}"
  done
}

validate_output() {
  "${TRAIN_PYTHON}" - "${OUTPUT_DIR}" "${OUTPUT_CHECKPOINT}" "${EXTRA_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
expected_steps = int(sys.argv[3])
for filename in (
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
):
    path = checkpoint / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing output checkpoint file: {path}")
config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
train = json.loads((checkpoint / "train_config.json").read_text(encoding="utf-8"))
if config.get("type") != "smolvla":
    raise SystemExit(f"output policy type is {config.get('type')!r}, expected 'smolvla'")
if int(train.get("steps", -1)) != expected_steps or int(train.get("batch_size", -1)) != 32:
    raise SystemExit("output training config does not match the 100-epoch B32 control")
last = (output_dir / "checkpoints" / "last").resolve()
state = json.loads((last / "training_state" / "training_step.json").read_text(encoding="utf-8"))
step = state.get("step", state.get("training_step"))
if int(step) != expected_steps:
    raise SystemExit(f"output training step is {step}, expected {expected_steps}")
PY
}

checkpoint_step() {
  "${TRAIN_PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

last = (Path(sys.argv[1]) / "checkpoints" / "last").resolve()
state = json.loads((last / "training_state" / "training_step.json").read_text(encoding="utf-8"))
print(int(state.get("step", state.get("training_step"))))
PY
}

write_control_manifest() {
  "${TRAIN_PYTHON}" - \
    "${CONTROL_MANIFEST}" "${SOURCE_CHECKPOINT}" "${RESOLVED_OUTPUT_CHECKPOINT}" \
    "${TASK_ID}" "${STEPS_PER_EPOCH}" "${EXTRA_STEPS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

target, source_raw, output_raw, task_id, steps_per_epoch, extra_steps = sys.argv[1:]
source = Path(source_raw).resolve()
output = Path(output_raw).resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema": "native_smolvla_staged_duration_control_v1",
    "task_id": task_id,
    "policy_type": "smolvla",
    "source_checkpoint": str(source),
    "source_model_sha256": sha256(source / "model.safetensors"),
    "source_complete_epochs": 150,
    "extra_training_checkpoint": str(output),
    "extra_training_model_sha256": sha256(output / "model.safetensors"),
    "extra_complete_epochs": 100,
    "total_demonstration_epochs": 250,
    "steps_per_extra_epoch": int(steps_per_epoch),
    "extra_optimizer_steps": int(extra_steps),
    "optimizer_transition": "fresh AdamW at 150-epoch warm-start boundary",
    "scheduler_transition": "fresh warmup+cosine at 150-epoch warm-start boundary",
    "per_device_batch_size": 32,
    "world_size": 2,
    "global_batch_size": 64,
}
path = Path(target)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

validate_source
wait_for_dependency
wait_for_gpu_pair_idle

if [[ -f "${OUTPUT_CHECKPOINT}/model.safetensors" ]]; then
  CURRENT_STEP="$(checkpoint_step)"
  if (( CURRENT_STEP == EXTRA_STEPS )); then
    echo "Validated output checkpoint already exists; skipping the 100-epoch training stage."
    validate_output
  elif (( CURRENT_STEP > 0 && CURRENT_STEP < EXTRA_STEPS )); then
    RESUME_CONFIG="$(readlink -f "${OUTPUT_DIR}/checkpoints/last")/pretrained_model/train_config.json"
    echo "Resuming interrupted duration-control training from step ${CURRENT_STEP}/${EXTRA_STEPS}."
    (
      cd "${HD_REPO_ROOT}"
      CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" env \
        EXECUTE=1 \
        TASK_ID="${TASK_ID}" \
        EPOCHS=100 \
        MIN_EPOCHS=100 \
        NUM_PROCESSES=2 \
        BATCH_SIZE=32 \
        NUM_WORKERS=4 \
        PREFETCH_FACTOR=2 \
        MIXED_PRECISION=bf16 \
        MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
        SAVE_FREQ="${SAVE_INTERVAL}" \
        LOG_FREQ=100 \
        SEED=1000 \
        RESUME=true \
        CONFIG_PATH="${RESUME_CONFIG}" \
        BASE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
        OUTPUT_DIR="${OUTPUT_DIR}" \
        TRAINING_METADATA_PATH="${OUTPUT_METADATA}" \
        PYTHON_BIN="${TRAIN_PYTHON}" \
        ACCELERATE_BIN="${ACCELERATE_BIN}" \
        "${TRAIN_LAUNCHER}" run
    )
    validate_output
  else
    die "output checkpoint has unsafe step ${CURRENT_STEP}; expected 1..${EXTRA_STEPS}"
  fi
elif [[ -e "${OUTPUT_DIR}" ]]; then
  die "partial output exists without a validated final checkpoint: ${OUTPUT_DIR}"
else
  echo "Starting native SmolVLA extra 100 epochs on physical GPUs ${GPU_A},${GPU_B}."
  (
    cd "${HD_REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}" env \
      EXECUTE=1 \
      TASK_ID="${TASK_ID}" \
      EPOCHS=100 \
      MIN_EPOCHS=100 \
      NUM_PROCESSES=2 \
      BATCH_SIZE=32 \
      NUM_WORKERS=4 \
      PREFETCH_FACTOR=2 \
      MIXED_PRECISION=bf16 \
      MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
      SAVE_FREQ="${SAVE_INTERVAL}" \
      LOG_FREQ=100 \
      SEED=1000 \
      RESUME=false \
      BASE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
      OUTPUT_DIR="${OUTPUT_DIR}" \
      TRAINING_METADATA_PATH="${OUTPUT_METADATA}" \
      PYTHON_BIN="${TRAIN_PYTHON}" \
      ACCELERATE_BIN="${ACCELERATE_BIN}" \
      "${TRAIN_LAUNCHER}" run
  )
  validate_output
fi

RESOLVED_OUTPUT_CHECKPOINT="$(readlink -f "${OUTPUT_CHECKPOINT}")"
write_control_manifest
wait_for_gpu_pair_idle

mkdir -p "$(dirname -- "${EVALUATION_OUTPUT}")"
echo "Starting canonical baseline-250 evaluation on physical GPU ${GPU_A}."
PYTHONPATH="${METHOD_REPO_ROOT}/src:/workspace/MIKASA-Robo${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${GPU_A}" "${EVAL_PYTHON}" "${EVALUATOR}" \
  --policy-kind baseline \
  --checkpoint "${RESOLVED_OUTPUT_CHECKPOINT}" \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --dataset-root "${DATASET_ROOT}" \
  --task "${ENV_ID}" \
  --num-episodes 50 \
  --start-seed 4242424242 \
  --torch-seed 7000 \
  --sim-backend gpu \
  --device cuda \
  --execution-action-steps 50 \
  --output "${EVALUATION_OUTPUT}" \
  --official-output-dir "${OFFICIAL_OUTPUT_DIR}"

[[ -f "${COMPARISON_INPUT}" ]] || die "three-model comparison missing: ${COMPARISON_INPUT}"
"${EVAL_PYTHON}" - \
  "${COMPARISON_INPUT}" "${EVALUATION_OUTPUT}" "${CONTROL_MANIFEST}" \
  "${COMPARISON_OUTPUT}" "${ENV_ID}" <<'PY'
import json
import sys
from pathlib import Path

comparison_path, control_path, manifest_path, output_path = map(Path, sys.argv[1:5])
env_id = sys.argv[5]
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
control = json.loads(control_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
result = control["results"][0]
if (
    result["env_id"] != env_id
    or result["n_episodes"] != 50
    or result["start_seed"] != 4242424242
    or result["torch_seed"] != 7000
    or result["execution_action_steps"] != 50
    or len(result["successes"]) != 50
):
    raise SystemExit("baseline-250 result does not match the canonical K50 protocol")
expected_existing = {
    "baseline_native_k50": 50,
    "stage1_ttt_only": 1,
    "stage2_action_head": 1,
}
for label, cadence in expected_existing.items():
    existing_path = comparison_path.parent / label / "eval.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    existing_result = existing["results"][0]
    if (
        existing_result["env_id"] != env_id
        or existing_result["n_episodes"] != 50
        or existing_result["start_seed"] != 4242424242
        or existing_result["torch_seed"] != 7000
        or existing_result["execution_action_steps"] != cadence
        or len(existing_result["successes"]) != 50
    ):
        raise SystemExit(f"existing comparison member is non-canonical: {existing_path}")
row = {
    "label": "baseline_250ep_warmstart_k50",
    "checkpoint": control["evaluation_identity"]["checkpoint"],
    "checkpoint_model_sha256": control["evaluation_identity"]["checkpoint_model_sha256"],
    "sr": result["sr"],
    "mean_return": result["mean_return"],
    "n_episodes": result["n_episodes"],
    "start_seed": result["start_seed"],
    "execution_action_steps": result["execution_action_steps"],
    "training_control": manifest,
}
models = [item for item in comparison["models"] if item.get("label") != row["label"]]
models.append(row)
payload = {
    "models": models,
    "control_note": (
        "baseline_250ep_warmstart_k50 is the native 150-epoch checkpoint followed by a "
        "fresh-optimizer 100-epoch stage; it is not a from-scratch single-stage 250-epoch run."
    ),
}
temporary = output_path.with_name(f".{output_path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(output_path)
print(f"Extended comparison complete: {output_path}")
PY
