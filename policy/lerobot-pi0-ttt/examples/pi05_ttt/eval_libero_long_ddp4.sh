#!/bin/bash

set -euo pipefail

source /venv/main/bin/activate

REPO_DIR="${PI05_TTT_REPO_DIR:-/workspace/test-TTT-eval-git/lib/lerobot-pi0-ttt}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-/workspace/outputs/train/pi05_ttt_libero_long_c256_stage2_s6000_from_stage1_004000_seed1000/checkpoints/001000/pretrained_model}"
EVAL_ROOT="${EVAL_ROOT:-/workspace/outputs/eval}"
LABEL="${EVAL_LABEL:-pi05_ttt_libero_long_stage2_001000_seed1000}"
LOG_ROOT="${LOG_ROOT:-/workspace/outputs/logs/${LABEL}}"
START_SEED="${START_SEED:-1000}"
N_EPISODES="${N_EPISODES:-10}"

cd "${REPO_DIR}"

export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/.hf_home
export HF_LEROBOT_HOME=/workspace/artifacts/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_DIR}/src"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p "${EVAL_ROOT}" "${LOG_ROOT}"

run_shard() {
  local gpu="$1"
  local task_ids="$2"
  local output_dir="${EVAL_ROOT}/${LABEL}_gpu${gpu}"
  local log_path="${LOG_ROOT}/${LABEL}_gpu${gpu}.log"

  echo "Starting PI0.5-TTT checkpoint evaluation on GPU ${gpu}, tasks ${task_ids}"
  CUDA_VISIBLE_DEVICES="${gpu}" /venv/main/bin/lerobot-eval \
    --policy.path="${POLICY_CHECKPOINT}" \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.n_action_steps=10 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=false \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids="${task_ids}" \
    --env.init_states=true \
    --env.max_parallel_tasks=1 \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.batch_size=1 \
    --eval.use_async_envs=false \
    --eval.max_episodes_rendered=0 \
    --seed="${START_SEED}" \
    --output_dir="${output_dir}" \
    >"${log_path}" 2>&1
}

status=0
run_shard 0 '[6,9]' &
pid0=$!
run_shard 1 '[2,5]' &
pid1=$!
run_shard 2 '[0,1,4]' &
pid2=$!
run_shard 3 '[3,7,8]' &
pid3=$!

for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  touch "${EVAL_ROOT}/${LABEL}_FAILED"
  echo "PI0.5-TTT evaluation failed; inspect per-GPU logs."
  exit "${status}"
fi

touch "${EVAL_ROOT}/${LABEL}_COMPLETE"
echo "PI0.5-TTT LIBERO-Long evaluation complete."
