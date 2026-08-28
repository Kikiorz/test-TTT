#!/usr/bin/env bash

# Wait for one task's two-stage training service, then evaluate its native
# baseline, stage-1 TTT, and stage-2 joint checkpoint on the canonical MIKASA
# 50-episode seed stream.  plan is read-only; run requires EXECUTE=1.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MODE="${1:-plan}"
EXECUTE="${EXECUTE:-0}"
TASK_ID="${TASK_ID:-remember_shape5}"

die() {
  echo "MIKASA three-checkpoint evaluation: $*" >&2
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
    DEFAULT_TRAIN_SERVICE=smolvla_ttt_remember_shape5_b8_50x50
    DEFAULT_TRAIN_LOG=/workspace/logs/smolvla_ttt_remember_shape5_b8_50x50.log
    ;;
  shuffle_touch)
    DATASET_REPO_ID=shell_game_shuffle_touch_vla_v0
    ENV_ID=ShellGameShuffleTouch-VLA-v0
    DEFAULT_GPU_A=0
    DEFAULT_GPU_B=1
    DEFAULT_TRAIN_SERVICE=smolvla_ttt_shuffle_touch_b8_50x50
    DEFAULT_TRAIN_LOG=/workspace/logs/smolvla_ttt_shuffle_touch_b8_50x50.log
    ;;
  *) die "TASK_ID must be remember_shape5 or shuffle_touch" ;;
esac

GPU_A="${GPU_A:-${DEFAULT_GPU_A}}"
GPU_B="${GPU_B:-${DEFAULT_GPU_B}}"
[[ "${GPU_A}" != "${GPU_B}" ]] || die "GPU_A and GPU_B must differ"

TRAIN_SERVICE="${TRAIN_SERVICE:-${DEFAULT_TRAIN_SERVICE}}"
TRAIN_LOG="${TRAIN_LOG:-${DEFAULT_TRAIN_LOG}}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/data_mikasa_robo/data_lerobot/${DATASET_REPO_ID}}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-/workspace/experiments/method1_smolvla_ttt_sequence_outer_v1_50x50/${TASK_ID}}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/experiments/native_smolvla_short_memory_b32_150ep_20260827/${TASK_ID}/checkpoints/last/pretrained_model}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${TRAIN_OUTPUT_ROOT}/stage1_ttt_only/checkpoints/last/pretrained_model}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${TRAIN_OUTPUT_ROOT}/stage2_action_head/checkpoints/last/pretrained_model}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/evaluations/mikasa_official50_20260827/${TASK_ID}}"
EVALUATOR="${EVALUATOR:-${SCRIPT_DIR}/evaluate_mikasa.py}"
EVAL_PYTHON="${EVAL_PYTHON:-/workspace/MIKASA-Robo/.venv/bin/python}"

NUM_EPISODES=50
START_SEED=4242424242
TORCH_SEED="${TORCH_SEED:-7000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"

for integer_name in GPU_A GPU_B TORCH_SEED WAIT_SECONDS; do
  integer_value="${!integer_name}"
  [[ "${integer_value}" =~ ^[0-9]+$ ]] || die "${integer_name} must be a non-negative integer"
done
(( WAIT_SECONDS > 0 )) || die "WAIT_SECONDS must be positive"

BASE_OUTPUT="${RESULTS_ROOT}/baseline_native_k50/eval.json"
STAGE1_OUTPUT="${RESULTS_ROOT}/stage1_ttt_only/eval.json"
STAGE2_OUTPUT="${RESULTS_ROOT}/stage2_action_head/eval.json"

echo "MIKASA official three-checkpoint evaluation"
echo "task=${TASK_ID} env=${ENV_ID} dataset=${DATASET_REPO_ID}"
echo "episodes=${NUM_EPISODES} env_seeds=${START_SEED}..$((START_SEED + NUM_EPISODES - 1)) torch_seeds=${TORCH_SEED}..$((TORCH_SEED + NUM_EPISODES - 1))"
echo "train_service=${TRAIN_SERVICE} gpu_pair=${GPU_A},${GPU_B}"
echo "baseline=${BASE_CHECKPOINT} cadence=K50"
echo "stage1=${STAGE1_CHECKPOINT} cadence=K1"
echo "stage2=${STAGE2_CHECKPOINT} cadence=K1"
echo "results_root=${RESULTS_ROOT}"

if [[ "${MODE}" == plan ]]; then
  echo "Plan only: no process or output was created."
  exit 0
fi

[[ "${EXECUTE}" == 1 ]] || die "run mode requires EXECUTE=1"
[[ -x "${EVAL_PYTHON}" ]] || die "MIKASA Python not found: ${EVAL_PYTHON}"
[[ -f "${EVALUATOR}" ]] || die "Method1 evaluator not found: ${EVALUATOR}"
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || die "dataset metadata missing: ${DATASET_ROOT}"
command -v supervisorctl >/dev/null || die "supervisorctl is required for the automatic handoff"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required for GPU ownership checks"

export PYTHONPATH="${REPO_ROOT}/src:/workspace/MIKASA-Robo${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/data_mikasa_robo}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

wait_for_training() {
  while true; do
    local status
    status="$(supervisorctl status "${TRAIN_SERVICE}" 2>&1 || true)"
    echo "training_status=${status}"
    if grep -Eq '\b(RUNNING|STARTING|STOPPING)\b' <<<"${status}"; then
      sleep "${WAIT_SECONDS}"
      continue
    fi
    if grep -Eq '\b(EXITED|STOPPED)\b' <<<"${status}"; then
      break
    fi
    die "training service entered an unsafe state: ${status}"
  done

  [[ -f "${TRAIN_LOG}" ]] || die "training log missing: ${TRAIN_LOG}"
  grep -Fq "Completed ${TASK_ID}:" "${TRAIN_LOG}" \
    || die "training service stopped without the two-stage completion marker"
}

validate_checkpoint() {
  local checkpoint=$1 expected_type=$2 expected_stage=$3
  "${EVAL_PYTHON}" - "${checkpoint}" "${expected_type}" "${expected_stage}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
expected_type = sys.argv[2]
expected_stage = sys.argv[3]
for filename in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
    path = checkpoint / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing checkpoint file: {path}")
config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
if config.get("type") != expected_type:
    raise SystemExit(f"{checkpoint}: type={config.get('type')!r}, expected {expected_type!r}")
if expected_stage and config.get("ttt_training_stage") != expected_stage:
    raise SystemExit(
        f"{checkpoint}: stage={config.get('ttt_training_stage')!r}, expected {expected_stage!r}"
    )
if expected_type == "smolvla_ttt":
    semantics = config.get("ttt_sequence_state_semantics")
    if semantics != "sequence_outer_step_v1":
        raise SystemExit(
            f"{checkpoint}: ttt_sequence_state_semantics={semantics!r}, "
            "expected 'sequence_outer_step_v1'"
        )
    if int(config.get("n_action_steps", -1)) != 1:
        raise SystemExit(f"{checkpoint}: canonical MIKASA evaluation requires n_action_steps=1")
PY
}

wait_for_gpu_pair_idle() {
  local uuid_a uuid_b active
  uuid_a="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU_A}")"
  uuid_b="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU_B}")"
  while true; do
    active="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null || true)"
    if ! grep -Fq "${uuid_a}" <<<"${active}" && ! grep -Fq "${uuid_b}" <<<"${active}"; then
      return
    fi
    echo "Waiting for GPU pair ${GPU_A},${GPU_B} to become compute-idle."
    sleep "${WAIT_SECONDS}"
  done
}

run_eval() {
  local label=$1 policy_kind=$2 checkpoint=$3 gpu=$4 output=$5
  local official_dir="${output%/eval.json}/official"
  local log_path="${output%/eval.json}/eval.log"
  mkdir -p "$(dirname "${output}")"
  echo "Starting ${label} on physical GPU ${gpu}"

  local command=(
    "${EVAL_PYTHON}" "${EVALUATOR}"
    --policy-kind "${policy_kind}"
    --checkpoint "${checkpoint}"
    --dataset-repo-id "${DATASET_REPO_ID}"
    --dataset-root "${DATASET_ROOT}"
    --task "${ENV_ID}"
    --num-episodes "${NUM_EPISODES}"
    --start-seed "${START_SEED}"
    --torch-seed "${TORCH_SEED}"
    --sim-backend gpu
    --device cuda
    --output "${output}"
    --official-output-dir "${official_dir}"
  )
  if [[ "${policy_kind}" == baseline ]]; then
    command+=(--execution-action-steps 50)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}" 2>&1 | tee -a "${log_path}"
  "${EVAL_PYTHON}" - "${output}" "${checkpoint}" "${NUM_EPISODES}" <<'PY'
import json
import math
import sys
from pathlib import Path

output, checkpoint, expected_n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
payload = json.loads(output.read_text(encoding="utf-8"))
result = payload["results"][0]
if payload["evaluation_identity"]["checkpoint"] != checkpoint:
    raise SystemExit(f"checkpoint provenance mismatch in {output}")
if len(result["successes"]) != expected_n or len(result["episode_seeds"]) != expected_n:
    raise SystemExit(f"episode count mismatch in {output}")
if not math.isfinite(float(result["sr"])) or not math.isfinite(float(result["mean_return"])):
    raise SystemExit(f"non-finite metric in {output}")
print(f"Completed {output}: SR={result['sr']:.4f}, mean_return={result['mean_return']:.4f}")
PY
}

wait_for_training
validate_checkpoint "${BASE_CHECKPOINT}" smolvla ""
validate_checkpoint "${STAGE1_CHECKPOINT}" smolvla_ttt ttt_only
validate_checkpoint "${STAGE2_CHECKPOINT}" smolvla_ttt action_head
wait_for_gpu_pair_idle

# Keep both GPUs useful: GPU_A runs the quick native baseline and then stage 2;
# GPU_B evaluates stage 1 concurrently.  Every individual evaluation remains
# canonical num_envs=1 and the three runs use the identical seed stream.
(
  run_eval baseline_native_k50 baseline "${BASE_CHECKPOINT}" "${GPU_A}" "${BASE_OUTPUT}"
  run_eval stage2_action_head ttt "${STAGE2_CHECKPOINT}" "${GPU_A}" "${STAGE2_OUTPUT}"
) &
chain_a_pid=$!
run_eval stage1_ttt_only ttt "${STAGE1_CHECKPOINT}" "${GPU_B}" "${STAGE1_OUTPUT}" &
chain_b_pid=$!

status=0
wait "${chain_a_pid}" || status=1
wait "${chain_b_pid}" || status=1
(( status == 0 )) || die "one or more official evaluations failed; completed JSON files are restart-safe"

"${EVAL_PYTHON}" - "${RESULTS_ROOT}" "${BASE_OUTPUT}" "${STAGE1_OUTPUT}" "${STAGE2_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = ("baseline_native_k50", "stage1_ttt_only", "stage2_action_head")
paths = [Path(value) for value in sys.argv[2:]]
rows = []
for label, path in zip(labels, paths, strict=True):
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["results"][0]
    rows.append(
        {
            "label": label,
            "checkpoint": payload["evaluation_identity"]["checkpoint"],
            "checkpoint_model_sha256": payload["evaluation_identity"]["checkpoint_model_sha256"],
            "sr": result["sr"],
            "mean_return": result["mean_return"],
            "n_episodes": result["n_episodes"],
            "start_seed": result["start_seed"],
            "execution_action_steps": result["execution_action_steps"],
        }
    )
target = root / "comparison.json"
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(".comparison.json.tmp")
temporary.write_text(json.dumps({"models": rows}, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
print(f"All three canonical evaluations complete: {target}")
PY
