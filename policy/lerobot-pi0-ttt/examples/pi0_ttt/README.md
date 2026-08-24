# PI0-TTT

`pi0_ttt` adds RoboTTT-style fast-weight MLPs to selected PI0 action-expert layers. Training uses
contiguous episode windows, independent flow-matching noise per frame, and truncated backpropagation
through time. Base PI0 checkpoints load with only the new `model.ttt_layers.*` parameters initialized
from scratch.

## SeetaCloud RTX 5090 Start

The following command uses the existing 40-episode dataset and PI0 checkpoint. It trains the last four
TTT layers while freezing the pretrained PI0 parameters.

```bash
cd /root/lerobot-pi0-ttt
source /etc/network_turbo

PYTHONPATH=/root/lerobot-pi0-ttt/src \
/root/miniconda3/envs/lerobot/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lgy/demo_old20_new20_20260613 \
  --dataset.root=/root/.cache/huggingface/lerobot/lgy/demo_old20_new20_20260613 \
  --dataset.video_backend=pyav \
  --policy.type=pi0_ttt \
  --policy.pretrained_path=/root/autodl-tmp/outputs/train/pi0_demo/checkpoints/003000/pretrained_model \
  --policy.dtype=bfloat16 \
  --policy.sequence_length=128 \
  --policy.sequence_stride=32 \
  --policy.tbptt_segment_length=8 \
  --policy.ttt_hidden_dim=4096 \
  --policy.ttt_layer_indices='[14,15,16,17]' \
  --policy.ttt_base_inner_lr=0.1 \
  --policy.ttt_freeze_base=true \
  --policy.optimizer_lr=2e-5 \
  --policy.push_to_hub=false \
  --batch_size=1 \
  --num_workers=2 \
  --prefetch_factor=1 \
  --persistent_workers=true \
  --steps=3000 \
  --log_freq=10 \
  --save_freq=1500 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=/root/autodl-tmp/outputs/train/pi0_ttt_old_new_40_t128_v3
```

One optimizer step consumes one 128-frame window. Fast weights carry across all 16 TBPTT segments and
are detached only at segment boundaries. Start with the command above; increase sequence length before
increasing segment length, since sequence length expands memory horizon while segment length primarily
increases activation memory.

The source PI0 checkpoint's serialized processor already enables relative actions and uses its compatible
PaliGemma tokenizer. Leave `policy.use_relative_actions` unset so LeRobot reuses that processor instead of
rebuilding it with the gated default tokenizer.

## LIBERO Spatial Experiment

The repository also includes a reproducible run for the public `lerobot/libero` dataset. It selects the 432
`libero_spatial` episodes (52,970 frames), initializes from the public LIBERO-finetuned PI0 checkpoint, freezes
the PI0 backbone, and trains only the final four TTT layers. With 128-frame windows and stride 32, 177 episodes
provide 197 valid contiguous windows; shorter episodes are excluded without crossing episode boundaries.

Download `lerobot/pi0_libero_finetuned` and the selected dataset files to the paths used below, then run:

```bash
cd /root/lerobot-pi0-ttt

MODEL_ROOT=/root/autodl-tmp/hf_libero/models/pi0_libero_finetuned \
DATASET_ROOT=/root/autodl-tmp/hf_libero/lerobot/lerobot/libero \
OUTPUT_DIR=/root/autodl-tmp/outputs/train/pi0_ttt_libero_spatial_t128_v1 \
examples/pi0_ttt/train_libero_spatial.sh
```

The default run uses 3,000 optimizer steps and saves only the final checkpoint. For a no-checkpoint training
smoke test, set `STEPS=1 SAVE_CHECKPOINT=false` and use a separate `OUTPUT_DIR`.

Evaluate the source PI0 and trained PI0-TTT checkpoint with identical LIBERO task IDs, initial-state seeds,
image resolution, and `n_action_steps`. Reset TTT state at every episode boundary and carry it only within the
episode; report both overall and per-task success rates.

### Visualize Fast Weights During Inference

After training, trace the four TTT layers during one closed-loop LIBERO episode:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=/root/lerobot-pi0-ttt/src \
/root/miniconda3/envs/lerobot/bin/python \
  examples/pi0_ttt/visualize_inference_state.py \
  --checkpoint /root/autodl-tmp/outputs/train/pi0_ttt_libero_spatial_t128_v1/checkpoints/003000/pretrained_model \
  --suite libero_spatial \
  --task-id 0 \
  --seed 1000 \
  --n-action-steps 10 \
  --output-dir /root/autodl-tmp/outputs/eval/pi0_ttt_libero_spatial_trace_task0_seed1000
```

The tracer records fast-weight drift and per-observation updates for `w1`, `b1`, `w2`, and `b2`, along with
the learned inner learning rate and residual gate. It writes JSON/CSV traces, a layer timeline and heatmap,
sampled-parameter PCA trajectories, and an MP4 synchronizing the scene with fast-weight drift.

## Checkpoint Evaluation

Use fixed noise on contiguous frames to compare source PI0 and PI0-TTT action chunks. The evaluator
carries TTT state across the sequence, verifies reset replay, measures the carried-state effect against a
cold final frame, and reports both action-target error and fixed-time flow-matching loss.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=/root/lerobot-pi0-ttt/src \
/root/miniconda3/envs/lerobot/bin/python \
  examples/pi0_ttt/evaluate_checkpoint.py \
  --source-checkpoint /root/autodl-tmp/outputs/train/pi0_demo/checkpoints/003000/pretrained_model \
  --ttt-checkpoint /root/autodl-tmp/outputs/train/pi0_ttt_old_new_40_t128_v3/checkpoints/003000/pretrained_model \
  --dataset-repo-id lgy/demo_wide_20260607_20260607_180102 \
  --dataset-root /root/.cache/huggingface/lerobot/lgy/demo_wide_20260607_20260607_180102 \
  --start-index 0 \
  --num-frames 8 \
  --seed 1337 \
  --flow-time 0.5 \
  --video-backend pyav \
  --output-json /root/autodl-tmp/pi0_ttt_eval.json
```

Offline mode prevents saved tokenizer configurations from waiting on Hub availability when all required
artifacts are already cached on the training host.
