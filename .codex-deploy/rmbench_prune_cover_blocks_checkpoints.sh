#!/usr/bin/env bash
set -euo pipefail

# The 150-epoch Cover Blocks run writes a full ~1.5 GiB checkpoint every
# 10 epochs.  Keep the newest complete checkpoint for recovery plus the
# 50/100/150-epoch milestones so the run cannot exhaust the instance disk.

readonly TRAIN_SERVICE="rmbench_train_cover_blocks_smolvla150"
readonly OUTPUT_DIR="/workspace/experiments/rmbench_cover_blocks/native_smolvla_150ep"
readonly CHECKPOINT_ROOT="${OUTPUT_DIR}/checkpoints"
readonly FINAL_STEP="060000"

is_milestone() {
  case "$1" in
    020000|040000|060000) return 0 ;;
    *) return 1 ;;
  esac
}

is_complete_checkpoint() {
  local checkpoint_dir="$1"
  local step_name="$2"

  [[ -f "${checkpoint_dir}/pretrained_model/config.json" ]] || return 1
  [[ -f "${checkpoint_dir}/pretrained_model/model.safetensors" ]] || return 1
  [[ -f "${checkpoint_dir}/training_state/training_step.json" ]] || return 1

  /workspace/test-TTT/policy/Method1_lerobot-pi0-ttt/.venv/bin/python - \
    "${checkpoint_dir}/training_state/training_step.json" "${step_name}" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
expected = int(sys.argv[2])
state = json.loads(state_path.read_text(encoding="utf-8"))
if int(state.get("step", -1)) != expected:
    raise SystemExit(1)
PY
}

prune_after_complete_save() {
  [[ -L "${CHECKPOINT_ROOT}/last" ]] || return 0

  local latest
  latest="$(readlink -f -- "${CHECKPOINT_ROOT}/last")"
  [[ "${latest}" == "${CHECKPOINT_ROOT}/"* ]] || {
    echo "Refusing unexpected last-checkpoint target: ${latest}" >&2
    return 1
  }

  local latest_name
  latest_name="$(basename -- "${latest}")"
  [[ "${latest_name}" =~ ^[0-9]{6}$ ]] || {
    echo "Refusing malformed last-checkpoint name: ${latest_name}" >&2
    return 1
  }
  is_complete_checkpoint "${latest}" "${latest_name}" || {
    echo "Last checkpoint is not complete yet: ${latest}" >&2
    return 0
  }

  local candidate resolved step_name
  shopt -s nullglob
  for candidate in "${CHECKPOINT_ROOT}"/[0-9]*; do
    [[ -d "${candidate}" && ! -L "${candidate}" ]] || continue
    step_name="$(basename -- "${candidate}")"
    [[ "${step_name}" =~ ^[0-9]{6}$ ]] || continue
    [[ "${step_name}" != "${latest_name}" ]] || continue
    is_milestone "${step_name}" && continue
    is_complete_checkpoint "${candidate}" "${step_name}" || continue

    resolved="$(realpath -e -- "${candidate}")"
    [[ "$(dirname -- "${resolved}")" == "${CHECKPOINT_ROOT}" ]] || {
      echo "Refusing checkpoint outside exact root: ${resolved}" >&2
      return 1
    }
    echo "Removing superseded Cover Blocks checkpoint: ${resolved}"
    rm -rf --one-file-system -- "${resolved}"
  done
}

echo "Watching ${CHECKPOINT_ROOT}; preserving latest plus steps 020000/040000/060000"
while true; do
  prune_after_complete_save

  service_line="$(supervisorctl status "${TRAIN_SERVICE}" 2>&1 || true)"
  service_state="$(awk '{print $2}' <<<"${service_line}")"
  case "${service_state}" in
    RUNNING|STARTING)
      sleep 60
      ;;
    EXITED|STOPPED|FATAL|BACKOFF|UNKNOWN|"")
      prune_after_complete_save
      if [[ -f "${CHECKPOINT_ROOT}/${FINAL_STEP}/pretrained_model/model.safetensors" ]]; then
        echo "Training ended with final step ${FINAL_STEP} present; pruning watcher complete."
        exit 0
      fi
      echo "Training service ended before the final checkpoint: ${service_line}" >&2
      exit 1
      ;;
    *)
      echo "Unexpected training service state: ${service_line}" >&2
      exit 1
      ;;
  esac
done
