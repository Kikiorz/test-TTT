#!/usr/bin/env bash
set -Eeuo pipefail

readonly RMBENCH_ROOT="/workspace/RMBench"
readonly METHOD1_ROOT="/workspace/test-TTT/policy/Method1_lerobot-pi0-ttt"
readonly METHOD1_ENV="${METHOD1_ROOT}/.venv"
readonly SIM_ENV="${RMBENCH_ROOT}/.venv-sim"
readonly METHOD1_PYTHON="${METHOD1_ENV}/bin/python"
readonly SIM_PYTHON="${SIM_ENV}/bin/python"
readonly INHERITED_PATH="${PATH}"

readonly DATASET_ROOT="/workspace/data_rmbench_lerobot/cover_blocks_demo_clean"
readonly DATASET_REPO_ID="rmbench/cover_blocks_demo_clean"
readonly BASELINE_MODEL="/workspace/experiments/rmbench_cover_blocks/native_smolvla_150ep/checkpoints/060000/pretrained_model"
readonly STAGE1_CHECKPOINT="/workspace/experiments/rmbench_cover_blocks/smolvla_ttt_50x50/stage1_ttt_only/checkpoints/000350"
readonly STAGE1_MODEL="${STAGE1_CHECKPOINT}/pretrained_model"
readonly STAGE2_CHECKPOINT="/workspace/experiments/rmbench_cover_blocks/smolvla_ttt_50x50/stage2_action_head/checkpoints/000350"
readonly STAGE2_MODEL="${STAGE2_CHECKPOINT}/pretrained_model"

readonly MODEL_SERVER_SCRIPT="${RMBENCH_ROOT}/policy/smolvla_ttt/model_server.sh"
readonly MODEL_CONFIG="${RMBENCH_ROOT}/policy/smolvla_ttt/deploy_policy.yml"
readonly EVAL_CLIENT="${RMBENCH_ROOT}/script/eval_policy_client.py"
readonly OFFICIAL_TASK_CONFIG="${RMBENCH_ROOT}/task_config/demo_clean.yml"
readonly NO_VIDEO_TASK_NAME="demo_clean_no_video"
readonly NO_VIDEO_TASK_CONFIG="${RMBENCH_ROOT}/task_config/${NO_VIDEO_TASK_NAME}.yml"

readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RUN_META_DIR="/workspace/experiments/rmbench_cover_blocks/eval_three_models_no_video/${RUN_ID}"
readonly BASELINE_SETTING="baseline_150ep_official100_novideo_${RUN_ID}"
readonly STAGE1_SETTING="stage1_ttt_only_50ep_official100_novideo_${RUN_ID}"
readonly STAGE2_SETTING="stage2_action_head_50ep_official100_novideo_${RUN_ID}"

readonly BASELINE_PORT="10001"
readonly STAGE1_PORT="10002"
readonly STAGE2_PORT="10003"
readonly SERVER_HOST="127.0.0.1"
readonly SERVER_READY_TIMEOUT_SECONDS="600"

declare -A server_pids=()
declare -A client_pids=()
declare -A server_logs=(
  [baseline]="/workspace/logs/rmbench_cover_blocks_baseline_server_no_video.log"
  [stage1]="/workspace/logs/rmbench_cover_blocks_stage1_server_no_video.log"
  [stage2]="/workspace/logs/rmbench_cover_blocks_stage2_server_no_video.log"
)
declare -A client_logs=(
  [baseline]="/workspace/logs/rmbench_cover_blocks_baseline_eval_no_video.log"
  [stage1]="/workspace/logs/rmbench_cover_blocks_stage1_eval_no_video.log"
  [stage2]="/workspace/logs/rmbench_cover_blocks_stage2_eval_no_video.log"
)
declare -A result_settings=(
  [baseline]="${BASELINE_SETTING}"
  [stage1]="${STAGE1_SETTING}"
  [stage2]="${STAGE2_SETTING}"
)

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

stop_processes() {
  local label pid
  for label in baseline stage1 stage2; do
    pid="${client_pids[${label}]:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for label in baseline stage1 stage2; do
    pid="${server_pids[${label}]:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for label in baseline stage1 stage2; do
    pid="${client_pids[${label}]:-}"
    [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    client_pids[${label}]=""
  done
  for label in baseline stage1 stage2; do
    pid="${server_pids[${label}]:-}"
    [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    server_pids[${label}]=""
  done
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_processes
  if [[ "${status}" -eq 0 ]]; then
    log "Three-model no-video evaluation completed successfully (run_id=${RUN_ID})"
  else
    log "Three-model no-video evaluation failed with status ${status} (run_id=${RUN_ID})"
  fi
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_nonempty_file() {
  local path="$1"
  [[ -s "${path}" ]] || die "Required non-empty file is missing: ${path}"
}

tcp_ready() {
  local port="$1"
  "${METHOD1_PYTHON}" - "${SERVER_HOST}" "${port}" <<'PY'
import socket
import sys

try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
}

assert_gpu_idle() {
  local gpu_id="$1" processes
  processes="$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
  [[ -z "${processes}" ]] || die "GPU ${gpu_id} is not idle: ${processes}"
}

validate_inputs() {
  local executable
  for executable in "${METHOD1_PYTHON}" "${SIM_PYTHON}" "${MODEL_SERVER_SCRIPT}"; do
    [[ -x "${executable}" ]] || die "Required executable is missing: ${executable}"
  done
  for executable in "${MODEL_CONFIG}" "${EVAL_CLIENT}" "${OFFICIAL_TASK_CONFIG}" "${NO_VIDEO_TASK_CONFIG}"; do
    require_nonempty_file "${executable}"
  done
  [[ -d "${DATASET_ROOT}" ]] || die "Dataset is missing: ${DATASET_ROOT}"

  "${METHOD1_PYTHON}" - \
    "${BASELINE_MODEL}" \
    "${STAGE1_CHECKPOINT}" \
    "${STAGE2_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

baseline = Path(sys.argv[1])
stage1 = Path(sys.argv[2])
stage2 = Path(sys.argv[3])

checks = (
    ("baseline", baseline, baseline.parent / "training_state" / "training_step.json", "smolvla", None, 60000),
    ("stage1", stage1 / "pretrained_model", stage1 / "training_state" / "training_step.json", "smolvla_ttt", "ttt_only", 350),
    ("stage2", stage2 / "pretrained_model", stage2 / "training_state" / "training_step.json", "smolvla_ttt", "action_head", 350),
)
for label, model_dir, step_path, expected_type, expected_stage, expected_step in checks:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    for path in (config_path, weights_path, step_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"{label}: required checkpoint file is missing or empty: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = json.loads(step_path.read_text(encoding="utf-8"))
    if config.get("type") != expected_type:
        raise SystemExit(f"{label}: type={config.get('type')!r}, expected {expected_type!r}")
    if expected_stage is not None and config.get("ttt_training_stage") != expected_stage:
        raise SystemExit(
            f"{label}: ttt_training_stage={config.get('ttt_training_stage')!r}, "
            f"expected {expected_stage!r}"
        )
    if int(state.get("step", -1)) != expected_step:
        raise SystemExit(f"{label}: step={state.get('step')!r}, expected {expected_step}")
    print(f"Validated {label}: type={expected_type} stage={expected_stage} step={expected_step}")
PY

  env PATH="${SIM_ENV}/bin:${INHERITED_PATH}" PYTHONDONTWRITEBYTECODE=1 \
    "${SIM_PYTHON}" - "${OFFICIAL_TASK_CONFIG}" "${NO_VIDEO_TASK_CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

official = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
no_video = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
if official.get("eval_video_log") is not True:
    raise SystemExit("Official demo_clean config no longer has eval_video_log=true")
if no_video.get("eval_video_log") is not False:
    raise SystemExit("No-video config must have eval_video_log=false")
official_without_flag = dict(official)
no_video_without_flag = dict(no_video)
official_without_flag.pop("eval_video_log", None)
no_video_without_flag.pop("eval_video_log", None)
if official_without_flag != no_video_without_flag:
    raise SystemExit("No-video task config differs from official demo_clean beyond eval_video_log")
print("Validated no-video task config: sole difference is eval_video_log true -> false")
PY

  local gpu_id port
  for gpu_id in 0 1 2 3; do
    assert_gpu_idle "${gpu_id}"
  done
  for port in "${BASELINE_PORT}" "${STAGE1_PORT}" "${STAGE2_PORT}"; do
    if tcp_ready "${port}"; then
      die "TCP ${SERVER_HOST}:${port} is already accepting connections"
    fi
  done
}

start_server() {
  local label="$1" gpu="$2" port="$3" checkpoint="$4"
  local server_log="${server_logs[${label}]}"
  printf '\n[%s] run_id=%s label=%s checkpoint=%s gpu=%s port=%s\n' \
    "$(date --iso-8601=seconds)" "${RUN_ID}" "${label}" "${checkpoint}" "${gpu}" "${port}" >>"${server_log}"
  (
    cd -- "${RMBENCH_ROOT}"
    exec env \
      PATH="${METHOD1_ENV}/bin:${INHERITED_PATH}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONDONTWRITEBYTECODE=1 \
      bash "${MODEL_SERVER_SCRIPT}" \
        "${checkpoint}" \
        "${DATASET_ROOT}" \
        "${DATASET_REPO_ID}" \
        "${gpu}" \
        "${port}" \
        "${METHOD1_ROOT}"
  ) >>"${server_log}" 2>&1 &
  server_pids[${label}]=$!
  log "Started ${label} model server pid=${server_pids[${label}]} GPU=${gpu} port=${port}"
}

wait_for_server() {
  local label="$1" port="$2" pid
  pid="${server_pids[${label}]}"
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))
  while ! tcp_ready "${port}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 100 "${server_logs[${label}]}" >&2 || true
      die "${label} model server exited before becoming ready"
    fi
    if ((SECONDS >= deadline)); then
      tail -n 100 "${server_logs[${label}]}" >&2 || true
      die "${label} model server did not become ready within ${SERVER_READY_TIMEOUT_SECONDS}s"
    fi
    sleep 2
  done
  log "${label} model server is ready on ${SERVER_HOST}:${port}"
}

start_client() {
  local label="$1" gpu="$2" port="$3" setting
  setting="${result_settings[${label}]}"
  local client_log="${client_logs[${label}]}"
  printf '\n[%s] run_id=%s label=%s gpu=%s port=%s setting=%s video=false\n' \
    "$(date --iso-8601=seconds)" "${RUN_ID}" "${label}" "${gpu}" "${port}" "${setting}" >>"${client_log}"
  (
    cd -- "${RMBENCH_ROOT}"
    exec env \
      PATH="${SIM_ENV}/bin:${INHERITED_PATH}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONFAULTHANDLER=1 \
      PYTHONWARNINGS=ignore::UserWarning \
      TORCH_EXTENSIONS_DIR="/workspace/torch_extensions" \
      "${SIM_PYTHON}" "${EVAL_CLIENT}" \
        --port "${port}" \
        --config "${MODEL_CONFIG}" \
        --overrides \
        --task_name cover_blocks \
        --task_config "${NO_VIDEO_TASK_NAME}" \
        --ckpt_setting "${setting}" \
        --seed 0 \
        --instruction_type unseen \
        --policy_name smolvla_ttt \
        --rpc_timeout 60 \
        --policy_episode_attempts 3
  ) >>"${client_log}" 2>&1 &
  client_pids[${label}]=$!
  log "Started ${label} official100 client pid=${client_pids[${label}]} GPU=${gpu}; video=false"
}

validate_result() {
  local label="$1" setting
  setting="${result_settings[${label}]}"
  local root="${RMBENCH_ROOT}/eval_result/cover_blocks/smolvla_ttt/${NO_VIDEO_TASK_NAME}/${setting}"
  local result_file progress_file run_dir
  result_file="$(find "${root}" -mindepth 2 -maxdepth 2 -type f -name '_result.txt' -print -quit 2>/dev/null || true)"
  [[ -n "${result_file}" ]] || die "${label}: final result file is missing under ${root}"
  run_dir="$(dirname -- "${result_file}")"
  progress_file="${run_dir}/progress.jsonl"
  require_nonempty_file "${progress_file}"
  "${METHOD1_PYTHON}" - "${label}" "${progress_file}" "${result_file}" <<'PY'
import json
import math
import sys
from pathlib import Path

label = sys.argv[1]
progress_path = Path(sys.argv[2])
result_path = Path(sys.argv[3])
records = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line]
if len(records) != 100:
    raise SystemExit(f"{label}: expected 100 progress records, got {len(records)}")
last = records[-1]
if int(last.get("completed_episodes", -1)) != 100:
    raise SystemExit(f"{label}: final completed_episodes is not 100: {last}")
successes = int(last.get("successes", -1))
if not 0 <= successes <= 100:
    raise SystemExit(f"{label}: invalid final success count: {successes}")
result_lines = [line.strip() for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
reported_rate = float(result_lines[-1])
if not math.isclose(reported_rate, successes / 100.0, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(
        f"{label}: result rate {reported_rate} disagrees with progress {successes}/100"
    )
print(f"Validated {label}: success={successes}/100 rate={reported_rate:.4f} path={result_path}")
PY
  if find "${run_dir}" -type f -name '*.mp4' -print -quit | grep -q .; then
    die "${label}: video file was created despite no-video configuration: ${run_dir}"
  fi
  log "Validated ${label} result and confirmed no videos: ${run_dir}"
}

log "Starting Cover Blocks three-model official100 evaluation with video disabled (run_id=${RUN_ID})"
validate_inputs

mkdir -p -- "${RUN_META_DIR}"
{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'video=false\n'
  printf 'seed_selector=0\n'
  printf 'episodes_per_model=100\n'
  printf 'baseline_setting=%s\n' "${BASELINE_SETTING}"
  printf 'stage1_setting=%s\n' "${STAGE1_SETTING}"
  printf 'stage2_setting=%s\n' "${STAGE2_SETTING}"
} >"${RUN_META_DIR}/manifest.env"

# Three full official runs execute concurrently.  The lightweight baseline and
# Stage 1 server/client pairs share one 32 GiB GPU each; the heavier Stage 2
# server and simulator use separate GPUs.  This uses all four physical GPUs
# without altering or sharding the official 100-seed protocol.
start_server baseline 0 "${BASELINE_PORT}" "${BASELINE_MODEL}"
start_server stage1 1 "${STAGE1_PORT}" "${STAGE1_MODEL}"
start_server stage2 2 "${STAGE2_PORT}" "${STAGE2_MODEL}"

wait_for_server baseline "${BASELINE_PORT}"
wait_for_server stage1 "${STAGE1_PORT}"
wait_for_server stage2 "${STAGE2_PORT}"

start_client baseline 0 "${BASELINE_PORT}"
start_client stage1 1 "${STAGE1_PORT}"
start_client stage2 3 "${STAGE2_PORT}"

failure=0
for label in baseline stage1 stage2; do
  pid="${client_pids[${label}]}"
  if wait "${pid}"; then
    log "${label} official100 client completed successfully"
  else
    status=$?
    log "ERROR: ${label} official100 client exited with status ${status}"
    failure=1
  fi
  client_pids[${label}]=""
done
((failure == 0)) || die "One or more official evaluation clients failed"

validate_result baseline
validate_result stage1
validate_result stage2
log "All three official100 evaluations finished and produced no videos"
