#!/usr/bin/env bash
set -euo pipefail

METHOD1_ROOT=/workspace/test-TTT/policy/Method1_lerobot-pi0-ttt

cd "${METHOD1_ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/hf_cache
export HF_LEROBOT_HOME=/workspace/data_rmbench_lerobot
export PYTORCH_ALLOC_CONF=expandable_segments:True
export EXECUTE=1

exec bash examples/rmbench/train_cover_blocks_smolvla.sh run
