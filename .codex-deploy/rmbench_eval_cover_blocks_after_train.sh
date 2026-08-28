#!/usr/bin/env bash
set -euo pipefail

# Wait for the already-running Cover Blocks baseline and its checkpoint pruner,
# then gate the official 100-valid-seed RPC evaluation on a complete final
# checkpoint and a real simulator scene smoke test. This script never starts,
# restarts, or stops the training/pruning services.

readonly TRAIN_SERVICE="rmbench_train_cover_blocks_smolvla150"
readonly PRUNER_SERVICE="rmbench_prune_cover_blocks_checkpoints"

readonly RMBENCH_ROOT="/workspace/RMBench"
readonly METHOD1_ROOT="/workspace/test-TTT/policy/Method1_lerobot-pi0-ttt"
readonly METHOD1_ENV="${METHOD1_ROOT}/.venv"
readonly SIM_ENV="${RMBENCH_ROOT}/.venv-sim"

readonly DATASET_ROOT="/workspace/data_rmbench_lerobot/cover_blocks_demo_clean"
readonly DATASET_REPO_ID="rmbench/cover_blocks_demo_clean"
readonly OUTPUT_DIR="/workspace/experiments/rmbench_cover_blocks/native_smolvla_150ep"
readonly CHECKPOINT_ROOT="${OUTPUT_DIR}/checkpoints"
readonly FINAL_STEP="060000"
readonly FINAL_STEP_DIR="${CHECKPOINT_ROOT}/${FINAL_STEP}"
readonly FINAL_MODEL_DIR="${FINAL_STEP_DIR}/pretrained_model"

readonly TRAIN_LOG="/workspace/logs/rmbench_train_cover_blocks_smolvla150.log"
readonly PRUNER_LOG="/workspace/logs/rmbench_prune_cover_blocks_checkpoints.log"
readonly MODEL_SERVER_LOG="/workspace/logs/rmbench_cover_blocks_model_server.log"
readonly SMOKE_SCRIPT="${RMBENCH_ROOT}/rmbench_cover_blocks_smoke.py"
readonly MODEL_SERVER_SCRIPT="${RMBENCH_ROOT}/policy/smolvla_ttt/model_server.sh"
readonly EVAL_CLIENT_SCRIPT="${RMBENCH_ROOT}/policy/smolvla_ttt/eval_client.sh"

readonly METHOD1_PYTHON="${METHOD1_ENV}/bin/python"
readonly SIM_PYTHON="${SIM_ENV}/bin/python"
readonly SERVER_GPU="0"
readonly CLIENT_GPU="1"
readonly SERVER_HOST="127.0.0.1"
readonly SERVER_PORT="9999"
readonly SEED_SELECTOR="0"
readonly SERVICE_POLL_SECONDS="60"
readonly SERVER_POLL_SECONDS="2"
readonly SERVER_READY_TIMEOUT_SECONDS="600"
readonly INHERITED_PATH="${PATH}"

model_server_pid=""

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

stop_model_server() {
  [[ -n "${model_server_pid}" ]] || return 0

  if kill -0 "${model_server_pid}" 2>/dev/null; then
    log "Stopping model server pid=${model_server_pid}"
    kill -TERM "${model_server_pid}" 2>/dev/null || true

    local attempt
    for attempt in $(seq 1 30); do
      if ! kill -0 "${model_server_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done

    if kill -0 "${model_server_pid}" 2>/dev/null; then
      log "Model server did not stop after 30 seconds; sending SIGKILL"
      kill -KILL "${model_server_pid}" 2>/dev/null || true
    fi
  fi

  wait "${model_server_pid}" 2>/dev/null || true
  model_server_pid=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_model_server
  if [[ "${status}" -eq 0 ]]; then
    log "After-train evaluation wrapper completed successfully"
  else
    log "After-train evaluation wrapper failed with status ${status}"
  fi
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

service_status_line() {
  local service="$1"
  supervisorctl status "${service}" 2>&1
}

wait_for_service_exited() {
  local service="$1"
  local description="$2"
  local status_line state

  log "Waiting for existing ${description} service: ${service}"
  while true; do
    # supervisorctl intentionally returns 3 for an EXITED process, so the
    # textual state is authoritative here rather than the command exit code.
    status_line="$(service_status_line "${service}" || true)"
    [[ -n "${status_line}" ]] \
      || die "Supervisor returned no status for service ${service}"
    state="$(awk '{print $2}' <<<"${status_line}")"
    case "${state}" in
      RUNNING|STARTING)
        log "${service} is ${state}; waiting ${SERVICE_POLL_SECONDS}s"
        sleep "${SERVICE_POLL_SECONDS}"
        ;;
      EXITED)
        log "Confirmed ${service} is EXITED: ${status_line}"
        return 0
        ;;
      *)
        die "${service} must reach EXITED, got: ${status_line}"
        ;;
    esac
  done
}

require_nonempty_file() {
  local path="$1"
  [[ -s "${path}" ]] || die "Required non-empty file is missing: ${path}"
}

assert_no_training_processes() {
  if ! "${METHOD1_PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

output_dir = sys.argv[1]
offenders = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        raw = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    if output_dir not in command:
        continue
    if "lerobot.scripts.lerobot_train" in command or "accelerate launch" in command:
        offenders.append((int(entry.name), command))

if offenders:
    for pid, command in sorted(offenders):
        print(f"training process still running: pid={pid} command={command}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    die "Training service exited but matching training processes still exist"
  fi
  log "Confirmed no process is running the Cover Blocks training command"
}

assert_gpu_idle() {
  local gpu_id="$1"
  local purpose="$2"
  local processes

  if ! processes="$(nvidia-smi \
      --id="${gpu_id}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>&1)"; then
    die "Could not inspect GPU ${gpu_id} for ${purpose}: ${processes}"
  fi
  [[ -z "${processes}" ]] \
    || die "GPU ${gpu_id} is not idle for ${purpose}: ${processes}"
  log "Confirmed GPU ${gpu_id} is idle for ${purpose}"
}

tcp_ready() {
  "${METHOD1_PYTHON}" - "${SERVER_HOST}" "${SERVER_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
}

log "After-train Cover Blocks evaluation guard started"

for executable in \
  "${METHOD1_PYTHON}" \
  "${SIM_PYTHON}" \
  "${MODEL_SERVER_SCRIPT}" \
  "${EVAL_CLIENT_SCRIPT}"; do
  [[ -x "${executable}" ]] || die "Required executable is missing: ${executable}"
done
[[ -f "${SMOKE_SCRIPT}" ]] || die "Simulator smoke script is missing: ${SMOKE_SCRIPT}"
[[ -d "${DATASET_ROOT}" ]] || die "LeRobot dataset is missing: ${DATASET_ROOT}"
[[ -d "/workspace/logs" ]] || die "Log directory is missing: /workspace/logs"
command -v supervisorctl >/dev/null 2>&1 || die "supervisorctl is unavailable"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"

wait_for_service_exited "${TRAIN_SERVICE}" "training"
wait_for_service_exited "${PRUNER_SERVICE}" "checkpoint-pruner"

grep -Fq "End of training" "${TRAIN_LOG}" \
  || die "Training log does not contain the normal completion marker: ${TRAIN_LOG}"
grep -Fq "pruning watcher complete" "${PRUNER_LOG}" \
  || die "Pruner log does not contain its normal completion marker: ${PRUNER_LOG}"

for required_file in \
  "${FINAL_MODEL_DIR}/config.json" \
  "${FINAL_MODEL_DIR}/model.safetensors" \
  "${FINAL_MODEL_DIR}/train_config.json" \
  "${FINAL_MODEL_DIR}/policy_preprocessor.json" \
  "${FINAL_MODEL_DIR}/policy_postprocessor.json" \
  "${FINAL_STEP_DIR}/training_state/training_step.json" \
  "${FINAL_STEP_DIR}/training_state/optimizer_state.safetensors" \
  "${FINAL_STEP_DIR}/training_state/optimizer_param_groups.json" \
  "${FINAL_STEP_DIR}/training_state/rng_state.safetensors"; do
  require_nonempty_file "${required_file}"
done

[[ -L "${CHECKPOINT_ROOT}/last" ]] \
  || die "Last-checkpoint link is missing: ${CHECKPOINT_ROOT}/last"
final_resolved="$(realpath -e -- "${FINAL_STEP_DIR}")" \
  || die "Cannot resolve final checkpoint: ${FINAL_STEP_DIR}"
last_resolved="$(realpath -e -- "${CHECKPOINT_ROOT}/last")" \
  || die "Cannot resolve last-checkpoint link: ${CHECKPOINT_ROOT}/last"
[[ "${last_resolved}" == "${final_resolved}" ]] \
  || die "last points to ${last_resolved}, expected ${final_resolved}"

if ! "${METHOD1_PYTHON}" - \
    "${FINAL_MODEL_DIR}/config.json" \
    "${FINAL_STEP_DIR}/training_state/training_step.json" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
step_path = Path(sys.argv[2])
config = json.loads(config_path.read_text(encoding="utf-8"))
state = json.loads(step_path.read_text(encoding="utf-8"))
if config.get("type") != "smolvla":
    raise SystemExit(f"final policy type must be smolvla, got {config.get('type')!r}")
if int(state.get("step", -1)) != 60000:
    raise SystemExit(f"final training step must be 60000, got {state.get('step')!r}")
print("Validated final checkpoint: policy=smolvla step=60000")
PY
then
  die "Final checkpoint metadata validation failed"
fi
log "Confirmed complete final checkpoint and last -> ${FINAL_STEP}"

assert_no_training_processes
assert_gpu_idle "${SERVER_GPU}" "the model server"
assert_gpu_idle "${CLIENT_GPU}" "the simulator client"

if tcp_ready; then
  die "TCP ${SERVER_HOST}:${SERVER_PORT} is already accepting connections"
fi

if ! env \
    PATH="${SIM_ENV}/bin:${INHERITED_PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${SIM_PYTHON}" - "${RMBENCH_ROOT}/task_config/demo_clean.yml" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if config.get("data_type", {}).get("pointcloud") is not False:
    raise SystemExit("demo_clean evaluation must set data_type.pointcloud=false")
print("Validated demo_clean data_type.pointcloud=false")
PY
then
  die "RMBench demo_clean pointcloud configuration is unsafe for SmolVLA"
fi

log "Running real Cover Blocks scene smoke on physical GPU ${CLIENT_GPU}"
if ! (
  cd -- "${RMBENCH_ROOT}"
  exec env \
    PATH="${SIM_ENV}/bin:${INHERITED_PATH}" \
    CUDA_VISIBLE_DEVICES="${CLIENT_GPU}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    TORCH_EXTENSIONS_DIR="/workspace/torch_extensions" \
    "${SIM_PYTHON}" "${SMOKE_SCRIPT}"
); then
  die "Cover Blocks scene smoke failed; official evaluation will not start"
fi
log "Cover Blocks scene smoke passed"

printf '\n[%s] Starting model server for %s\n' \
  "$(date --iso-8601=seconds)" "${FINAL_MODEL_DIR}" >>"${MODEL_SERVER_LOG}"
log "Starting model server on physical GPU ${SERVER_GPU}, ${SERVER_HOST}:${SERVER_PORT}"
(
  cd -- "${RMBENCH_ROOT}"
  exec env \
    PATH="${METHOD1_ENV}/bin:${INHERITED_PATH}" \
    CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
    PYTHONDONTWRITEBYTECODE=1 \
    bash "${MODEL_SERVER_SCRIPT}" \
      "${FINAL_MODEL_DIR}" \
      "${DATASET_ROOT}" \
      "${DATASET_REPO_ID}" \
      "${SERVER_GPU}" \
      "${SERVER_PORT}" \
      "${METHOD1_ROOT}"
) >>"${MODEL_SERVER_LOG}" 2>&1 &
model_server_pid=$!
log "Model server process started with pid=${model_server_pid}; log=${MODEL_SERVER_LOG}"

ready_deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))
while ! tcp_ready; do
  if ! kill -0 "${model_server_pid}" 2>/dev/null; then
    tail -n 100 "${MODEL_SERVER_LOG}" >&2 || true
    die "Model server exited before becoming ready"
  fi
  if ((SECONDS >= ready_deadline)); then
    tail -n 100 "${MODEL_SERVER_LOG}" >&2 || true
    die "Model server did not become ready within ${SERVER_READY_TIMEOUT_SECONDS}s"
  fi
  sleep "${SERVER_POLL_SECONDS}"
done
log "Model server is ready on ${SERVER_HOST}:${SERVER_PORT}"

log "Starting official cover_blocks evaluation: GPU ${CLIENT_GPU}, seed selector ${SEED_SELECTOR}, 100 valid seeds, instruction_type=unseen"
(
  cd -- "${RMBENCH_ROOT}"
  exec env \
    PATH="${SIM_ENV}/bin:${INHERITED_PATH}" \
    CUDA_VISIBLE_DEVICES="${CLIENT_GPU}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    TORCH_EXTENSIONS_DIR="/workspace/torch_extensions" \
    bash "${EVAL_CLIENT_SCRIPT}" "${SERVER_PORT}" "${SEED_SELECTOR}"
)
log "Official 100-valid-seed Cover Blocks evaluation finished successfully"
