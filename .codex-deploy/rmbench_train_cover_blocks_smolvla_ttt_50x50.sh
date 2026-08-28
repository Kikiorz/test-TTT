#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/workspace/test-TTT/policy/Method1_lerobot-pi0-ttt"
readonly TRAIN_SCRIPT="${REPO_ROOT}/examples/rmbench/train_cover_blocks_smolvla_ttt_two_stage.sh"
readonly OUTPUT_ROOT="/workspace/experiments/rmbench_cover_blocks/smolvla_ttt_50x50"
readonly BASE_CHECKPOINT="/workspace/experiments/rmbench_cover_blocks/native_smolvla_150ep/checkpoints/060000/pretrained_model"
readonly EVAL_SERVICE="rmbench_eval_cover_blocks_after_train"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

[[ -x "${TRAIN_SCRIPT}" ]] || die "training script is missing or not executable: ${TRAIN_SCRIPT}"
[[ -s "${BASE_CHECKPOINT}/model.safetensors" ]] \
  || die "baseline checkpoint is incomplete: ${BASE_CHECKPOINT}"
[[ ! -e "${OUTPUT_ROOT}/stage1_ttt_only" ]] \
  || die "fresh Stage 1 output already exists: ${OUTPUT_ROOT}/stage1_ttt_only"
[[ ! -e "${OUTPUT_ROOT}/stage2_action_head" ]] \
  || die "fresh Stage 2 output already exists: ${OUTPUT_ROOT}/stage2_action_head"

eval_status="$(supervisorctl status "${EVAL_SERVICE}" 2>&1 || true)"
eval_state="$(awk '{print $2}' <<<"${eval_status}")"
[[ "${eval_state}" != "RUNNING" && "${eval_state}" != "STARTING" ]] \
  || die "baseline evaluation service must be stopped before Stage 1/2; state=${eval_state}"

conflicting_processes="$(
  pgrep -af 'policy_model_server.py|eval_policy_client.py|lerobot.scripts.lerobot_train|accelerate launch' \
    || true
)"
if [[ -n "${conflicting_processes}" ]]; then
  printf '%s\n' "${conflicting_processes}" >&2
  die "conflicting training/evaluation process exists"
fi

for gpu_id in 0 1 2 3; do
  gpu_processes="$(nvidia-smi --id="${gpu_id}" \
    --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
  [[ -z "${gpu_processes}" ]] \
    || die "GPU ${gpu_id} is not idle: ${gpu_processes}"
done

log "Starting Cover Blocks SmolVLA-TTT Stage 1 + Stage 2"
log "Stage 1: 50 sequence epochs, gate fixed at 0.05, TTT/register parameters only"
log "Stage 2: 50 sequence epochs, gate + TTT + SmolVLA action expert/head"
log "Four GPUs, per-device sequence batch 8, all 50 demonstrations"

cd -- "${REPO_ROOT}"
exec env \
  EXECUTE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONFAULTHANDLER=1 \
  LEROBOT_VERIFY_DDP_SYNC=1 \
  bash "${TRAIN_SCRIPT}" run
