#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-plan}"
EXECUTE="${EXECUTE:-0}"

die() {
  echo "PI0.5-TTT LIBERO-Long: $*" >&2
  exit 2
}

case "${MODE}" in
  plan|run) ;;
  *) die "mode must be plan or run" ;;
esac

REPO_ROOT="${REPO_ROOT:-/workspace/lerobot-pi0-ttt}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/venv/main/bin/accelerate}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/artifacts/models/pi05_libero_finetuned}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/artifacts/datasets/lerobot_libero}"
PHASE1_OUTPUT="${PHASE1_OUTPUT:-/workspace/outputs/train/pi05_ttt_v2_gate001_libero_long_c256_stage1_s5000_seed1000}"
PHASE2_OUTPUT="${PHASE2_OUTPUT:-/workspace/outputs/train/pi05_ttt_v2_gate001_libero_long_c256_stage2_s5000_seed1000}"
PHASE1_STEPS="${PHASE1_STEPS:-5000}"
PHASE2_STEPS="${PHASE2_STEPS:-5000}"
PHASE1_CHECKPOINT_OVERRIDE="${PHASE1_CHECKPOINT_OVERRIDE:-}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
SEED="${SEED:-1000}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PHASE1_TBPTT_SEGMENT_LENGTH="${PHASE1_TBPTT_SEGMENT_LENGTH:-8}"
PHASE2_TBPTT_SEGMENT_LENGTH="${PHASE2_TBPTT_SEGMENT_LENGTH:-4}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
EFFECTIVE_GATE_INIT="${EFFECTIVE_GATE_INIT:-0.001}"

for integer_name in \
  PHASE1_STEPS PHASE2_STEPS SAVE_FREQ SEED NUM_PROCESSES \
  PHASE1_TBPTT_SEGMENT_LENGTH PHASE2_TBPTT_SEGMENT_LENGTH N_ACTION_STEPS; do
  integer_value="${!integer_name}"
  [[ "${integer_value}" =~ ^[0-9]+$ ]] \
    || die "${integer_name} must be a non-negative integer; got '${integer_value}'"
done
(( NUM_PROCESSES == 4 )) || die "the formal recipe requires NUM_PROCESSES=4"
(( PHASE1_STEPS > 0 && PHASE2_STEPS > 0 && SAVE_FREQ > 0 )) \
  || die "step and save intervals must be positive"
(( PHASE1_TBPTT_SEGMENT_LENGTH > 0 && PHASE1_TBPTT_SEGMENT_LENGTH <= 256 )) \
  || die "PHASE1_TBPTT_SEGMENT_LENGTH must be in 1..256"
(( PHASE2_TBPTT_SEGMENT_LENGTH > 0 && PHASE2_TBPTT_SEGMENT_LENGTH <= 256 )) \
  || die "PHASE2_TBPTT_SEGMENT_LENGTH must be in 1..256"
(( N_ACTION_STEPS > 0 && N_ACTION_STEPS <= 50 )) \
  || die "N_ACTION_STEPS must be in 1..50 (the configured action chunk size)"
[[ "${EFFECTIVE_GATE_INIT}" == "0.001" ]] \
  || die "the corrected two-stage recipe requires EFFECTIVE_GATE_INIT=0.001"
[[ -x "${PYTHON_BIN}" ]] || die "Python not found: ${PYTHON_BIN}"

EPISODES="$(${PYTHON_BIN} -c 'import json; print(json.dumps(list(range(379))))')"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/artifacts/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${REPO_ROOT}/src"

if [[ "${MODE}" == run ]]; then
  [[ "${EXECUTE}" == 1 ]] || die "run mode requires EXECUTE=1"
  [[ -x "${ACCELERATE_BIN}" ]] || die "accelerate not found: ${ACCELERATE_BIN}"
  [[ -d "${REPO_ROOT}/src/lerobot" ]] || die "invalid REPO_ROOT: ${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

LAUNCH=(
  "${ACCELERATE_BIN}" launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --multi_gpu
  --mixed_precision=bf16
  --dynamo_backend=no
  -m
)

run_stage() {
  local pretrained_path="$1"
  local training_stage="$2"
  local optimizer_lr="$3"
  local scheduler_decay_lr="$4"
  local steps="$5"
  local output_dir="$6"
  local tbptt_segment_length="$7"

  local command=(
    "${LAUNCH[@]}" lerobot.scripts.lerobot_train
    --dataset.repo_id=lerobot/libero \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episodes="${EPISODES}" \
    --dataset.video_backend=pyav \
    --policy.type=pi05_ttt \
    --policy.pretrained_path="${pretrained_path}" \
    --policy.dtype=bfloat16 \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.sequence_length=256 \
    --policy.sequence_stride=256 \
    --policy.tbptt_segment_length="${tbptt_segment_length}" \
    --policy.ttt_hidden_dim=4096 \
    --policy.ttt_layer_indices='[14,15,16,17]' \
    --policy.ttt_base_inner_lr=0.1 \
    --policy.ttt_effective_gate_init="${EFFECTIVE_GATE_INIT}" \
    --policy.ttt_training_stage="${training_stage}" \
    --policy.gradient_checkpointing=false \
    --policy.compile_model=false \
    --policy.optimizer_lr="${optimizer_lr}" \
    --policy.optimizer_weight_decay=1e-5 \
    --policy.scheduler_decay_lr="${scheduler_decay_lr}" \
    --policy.push_to_hub=false \
    --batch_size=1 \
    --num_workers=2 \
    --prefetch_factor=1 \
    --persistent_workers=true \
    --steps="${steps}" \
    --log_freq=10 \
    --save_checkpoint=true \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq=0 \
    --wandb.enable=false \
    --seed="${SEED}" \
    --output_dir="${output_dir}"
  )

  if [[ "${MODE}" == plan ]]; then
    printf '  '
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

validate_phase1_checkpoint() {
  local checkpoint=$1
  "${PYTHON_BIN}" - "${checkpoint}" "${EFFECTIVE_GATE_INIT}" <<'PY'
import json
import math
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
expected_gate = float(sys.argv[2])
for filename in ("config.json", "model.safetensors"):
    path = checkpoint / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing Stage-1 checkpoint file: {path}")
config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
checks = {
    "type": config.get("type") == "pi05_ttt",
    "stage": config.get("ttt_training_stage") == "ttt_only",
    "gate initialization": math.isclose(
        float(config.get("ttt_effective_gate_init", -1.0)),
        expected_gate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ),
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"Stage-1 checkpoint is incompatible with the corrected recipe: {failed}")
PY
}

echo "PI0.5-TTT LIBERO-Long two-stage recipe"
echo "gate=learned_in_both_stages init=${EFFECTIVE_GATE_INIT}"
echo "inference_action_steps=${N_ACTION_STEPS} fast_update=once_per_action_chunk_prediction"
echo "stage1_output=${PHASE1_OUTPUT}"
echo "stage2_output=${PHASE2_OUTPUT}"

if [[ -n "${PHASE1_CHECKPOINT_OVERRIDE}" ]]; then
  PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT_OVERRIDE%/}"
else
  run_stage "${MODEL_ROOT}" ttt_only 2e-5 2e-6 "${PHASE1_STEPS}" "${PHASE1_OUTPUT}" \
    "${PHASE1_TBPTT_SEGMENT_LENGTH}"
  PHASE1_CHECKPOINT="${PHASE1_OUTPUT}/checkpoints/$(printf '%06d' "${PHASE1_STEPS}")/pretrained_model"
fi

if [[ "${MODE}" == plan ]]; then
  run_stage "${PHASE1_CHECKPOINT}" action_head 5e-5 5e-6 "${PHASE2_STEPS}" "${PHASE2_OUTPUT}" \
    "${PHASE2_TBPTT_SEGMENT_LENGTH}"
  echo "Plan only: no training process or output was created."
  exit 0
fi

validate_phase1_checkpoint "${PHASE1_CHECKPOINT}"

run_stage "${PHASE1_CHECKPOINT}" action_head 5e-5 5e-6 "${PHASE2_STEPS}" "${PHASE2_OUTPUT}" \
  "${PHASE2_TBPTT_SEGMENT_LENGTH}"
