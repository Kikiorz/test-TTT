# Training SmolVLA-TTT

The policy can load a normal SmolVLA checkpoint and initialize only the new TTT tensors:

```bash
lerobot-train \
  --policy.type=smolvla_ttt \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.ttt_training_stage=ttt_only \
  --policy.ttt_num_register_tokens=16 \
  --policy.sequence_length=256 \
  --policy.sequence_stride=256 \
  --policy.tbptt_segment_length=4 \
  --dataset.repo_id=<USER>/<DATASET> \
  --batch_size=8 \
  --steps=30000
```

Each episode-local window is one independently selected training sequence of at most 256 timesteps.
It starts from the learned initial fast weights W0, carries fast state through its internal 4-timestep
TBPTT segments, detaches at segment boundaries, and produces exactly one outer optimizer step for the
window batch. Fast state never crosses loader samples. `sequence_stride` may be smaller than or equal to
`sequence_length`; overlapping windows are still independent sequences and each starts from W0. Padded
timesteps and distributed dummy lanes are excluded from both loss and recurrent updates.

The default prepends 16 learned register tokens before the 50 action tokens. Register queries
can read all register/action tokens; action queries can read all registers while preserving the
original causal action-to-action mask. SmolVLA's existing prefix state conditioning is unchanged.
Use `--policy.ttt_num_register_tokens=0` for the exact no-register ablation while keeping the
TTT-KVB update, layer placement, denoising schedule, and losses unchanged.

For the second stage, load the first-stage checkpoint and set
`--policy.ttt_training_stage=action_head`. Call `policy.reset()` between evaluation episodes so
recurrent fast weights never cross episode boundaries. `n_action_steps` may be any integer from 1 through
`chunk_size`:
one action-chunk inference performs one fast-weight update, then the policy may execute the requested
number of queued actions before the next inference.

## MIKASA two-stage run

`train_mikasa_two_stage.sh` is the reproducible launcher for the two selected MIKASA tasks:

| `TASK_ID` | Dataset directory / repo id | Environment id |
| --- | --- | --- |
| `remember_shape5` | `remember_shape_5_vla_v0` | `RememberShape5-VLA-v0` |
| `shuffle_touch` | `shell_game_shuffle_touch_vla_v0` | `ShellGameShuffleTouch-VLA-v0` |

The default protocol is deliberately fixed around the requested comparison:

- all official episode indices `0..249`, with no episode or sequence-window subsampling;
- independent tail-preserving windows with `sequence_length=sequence_stride=256`, fresh W0 per
  window, internal TBPTT carry/detach, and one outer step per window batch;
- two GPUs and per-device batch size 8;
- stage 1: `ttt_only`, learned gate initialized to `0.001`, 50 complete sequence epochs;
- stage 2: `action_head`, 50 complete sequence epochs, initialized from the final stage-1
  `pretrained_model` directory;
- a fresh optimizer, scheduler, RNG stream, global step, and output directory for stage 2;
- `save_freq=steps`, so each stage writes only its final checkpoint.

The launcher reads the real episode boundaries before training. If there are `N` independent windows,
one complete sequence epoch contains `ceil(N / global_batch_size)` slow optimizer steps. The reported
`unused_batch_slots` count is the final distributed batch's fully masked dummy lanes; no real window is
duplicated into the objective.

The input must be a local task-trained Native SmolVLA checkpoint directory containing
`config.json`, `model.safetensors`, `train_config.json`, and both processor configs. Its config must
declare `type=smolvla`. The default input location is:

```text
/workspace/experiments/native_smolvla_150/<TASK_ID>/checkpoints/last/pretrained_model
```

Inspect the fully resolved commands without creating anything:

```bash
cd /workspace/lerobot-pi0-ttt

TASK_ID=remember_shape5 MAIN_PROCESS_PORT=29501 \
  examples/smolvla_ttt/train_mikasa_two_stage.sh plan

TASK_ID=shuffle_touch MAIN_PROCESS_PORT=29502 \
  examples/smolvla_ttt/train_mikasa_two_stage.sh plan
```

Run the two tasks on separate GPU pairs, normally from two terminals:

```bash
CUDA_VISIBLE_DEVICES=0,1 TASK_ID=remember_shape5 MAIN_PROCESS_PORT=29501 EXECUTE=1 \
  examples/smolvla_ttt/train_mikasa_two_stage.sh run

CUDA_VISIBLE_DEVICES=2,3 TASK_ID=shuffle_touch MAIN_PROCESS_PORT=29502 EXECUTE=1 \
  examples/smolvla_ttt/train_mikasa_two_stage.sh run
```

Override `BASE_CHECKPOINT`, `DATASET_ROOT`, or `OUTPUT_ROOT` when the server uses different staging
paths. Both stage output directories must be absent before `run`; the launcher never treats a stage
transition as `--resume=true`. After each stage it validates the saved policy type, training stage,
processor state-file references, and exact final optimizer step before proceeding. Final checkpoints
are available at:

```text
<OUTPUT_ROOT>/stage1_ttt_only/checkpoints/last/pretrained_model
<OUTPUT_ROOT>/stage2_action_head/checkpoints/last/pretrained_model
```

## Automatic official MIKASA evaluation

`evaluate_mikasa_three_checkpoints.sh` waits for one task's complete two-stage training marker and
then evaluates these three checkpoints without manual handoff:

1. the task-trained native SmolVLA baseline;
2. the final stage-1 `ttt_only` checkpoint;
3. the final stage-2 `action_head` checkpoint.

The evaluator uses the official online simulator protocol: 50 episodes, environment seeds
`4242424242..4242424291`, `num_envs=1`, no overlays, and `success_once` as the primary success
metric. Flow-matching randomness is paired with torch seeds `7000..7049`. It calls
`policy.reset()` exactly once before every episode, so TTT state persists within an episode but can
never leak between episodes. MIKASA provides these rollouts online; there is no separate local
offline test split in the downloaded LeRobot datasets.

The native baseline keeps its intended K=50 execution cadence. Both TTT checkpoints use K=1 so the
fast state observes every physical decision. The launcher also rejects TTT checkpoints unless they
carry `ttt_sequence_state_semantics=sequence_outer_step_v1`, preventing an older episode-stream or
independent-window artifact from being evaluated under the corrected label. These models are deliberately
labeled in the output rather than presented as cadence-matched. Both selected environments are in MIKASA's official
**Short** split: `RememberShape5-VLA-v0` is Object memory and
`ShellGameShuffleTouch-VLA-v0` is Tracking memory.

Inspect the fully resolved evaluation without starting it:

```bash
TASK_ID=remember_shape5 examples/smolvla_ttt/evaluate_mikasa_three_checkpoints.sh plan
TASK_ID=shuffle_touch examples/smolvla_ttt/evaluate_mikasa_three_checkpoints.sh plan
```

Start a restart-safe waiting process for each task:

```bash
TASK_ID=remember_shape5 EXECUTE=1 \
  examples/smolvla_ttt/evaluate_mikasa_three_checkpoints.sh run

TASK_ID=shuffle_touch EXECUTE=1 \
  examples/smolvla_ttt/evaluate_mikasa_three_checkpoints.sh run
```

Each task writes per-model episode records plus a compact comparison to:

```text
/workspace/evaluations/mikasa_official50_20260827/<TASK_ID>/comparison.json
```

Completed outputs are keyed by checkpoint SHA256 and protocol identity. A restart skips a matching
completed model and resumes an interrupted model from its last atomically written episode record.

## Native 150+100 epoch duration control

`train_evaluate_native_baseline250_control.sh` tests whether the 150-epoch native baseline was simply
under-trained. For each task it loads that task's final native SmolVLA weights, resets AdamW and the
warmup/cosine scheduler, trains on all 250 demonstrations for 100 additional frame epochs with the
same per-device batch size 32 on two GPUs, and then runs the canonical K=50, 50-episode evaluation.
The original 150-epoch output is never modified.

This control is explicitly named **150+100 staged warm-start**. It has 250 total demonstration
passes, but it is not presented as a from-scratch, single-stage 250-epoch run. Native training uses
frame epochs while TTT uses sequence-window epochs, so matching the epoch count does not make their
optimizer update counts identical.

Inspect either task without creating output:

```bash
TASK_ID=remember_shape5 \
  examples/smolvla_ttt/train_evaluate_native_baseline250_control.sh plan

TASK_ID=shuffle_touch \
  examples/smolvla_ttt/train_evaluate_native_baseline250_control.sh plan
```

The default output and extended comparison are:

```text
/workspace/experiments/native_smolvla_short_memory_b32_150plus100ep_control_20260827/<TASK_ID>
/workspace/evaluations/mikasa_official50_20260827/<TASK_ID>/comparison_with_baseline250.json
```

## RememberShape5 Stage-2 extension

`train_evaluate_remember_stage3.sh` tests whether the completed Stage1+2 model benefits from more
joint optimization. It loads the concrete final RememberShape5 Stage-2 checkpoint, keeps
`ttt_training_stage=action_head` (TTT, registers, learned gate, action expert, and state/action/time
projections trainable; the vision/VLM backbone remains frozen), and runs a fresh 100-sequence-epoch
optimization stage on all 250 demonstrations. The original Stage1 and Stage2 outputs are never
modified.

The continuation requires the source checkpoint to contain
`ttt_sequence_state_semantics=sequence_outer_step_v1`. Checkpoints carrying the earlier episode-stream
semantics are retained only as ablations and are intentionally rejected as Stage3 initialization. Rerun
the corrected independent-sequence Stage1+2 protocol first.

This is a **staged warm-start**, not an in-place optimizer-state continuation. Stage2's scheduler
was constructed for exactly 800 steps and reached its LR floor; changing the total horizon while
loading that scheduler state would introduce an ambiguous LR jump. Stage3 therefore inherits all
model weights and processors while starting fresh AdamW and warmup/cosine states, matching the
existing Stage1-to-Stage2 transition semantics.

For RememberShape5, all episodes fit in one selected window, so 250 windows with B8/device on two GPUs
resolve to 16 optimizer
steps per complete sequence epoch. The extra 100 epochs are therefore exactly 1600 steps. The
launcher saves and evaluates both the +50-epoch (`000800`) and +100-epoch (`001600`) checkpoints so
an intermediate peak is not missed. Both use the same canonical K1, 50-episode seed stream as
Stage2, and the merger reports paired per-seed success flips without overwriting prior comparisons.
Because the bundled TTT evaluator emits one action per inference, this Stage3 launcher explicitly requires
`N_ACTION_STEPS=1` and rejects any override instead of silently mislabeling a K>1 checkpoint evaluation.

Inspect or run the extension:

```bash
examples/smolvla_ttt/train_evaluate_remember_stage3.sh plan

EXECUTE=1 GPU_A=2 GPU_B=3 \
  examples/smolvla_ttt/train_evaluate_remember_stage3.sh run
```

Default outputs are:

```text
/workspace/experiments/method1_smolvla_ttt_sequence_outer_v1_stage2plus100/remember_shape5/stage3_action_head
/workspace/evaluations/mikasa_official50_20260827/remember_shape5/stage2plus50_action_head/eval.json
/workspace/evaluations/mikasa_official50_20260827/remember_shape5/stage2plus100_action_head/eval.json
/workspace/evaluations/mikasa_official50_20260827/remember_shape5/comparison_with_baseline250_and_stage2_extension.json
```
