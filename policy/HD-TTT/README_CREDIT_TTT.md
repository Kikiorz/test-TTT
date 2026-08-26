# CreditTTT (V3)

**Hindsight-to-Local Query-conditioned Fast-Weight Learning for long-horizon
robot imitation**

This document is the method and experiment contract for the CreditTTT V3
implementation in `policy/HD-TTT`.  It is deliberately separate from
[`README_HD_TTT.md`](./README_HD_TTT.md), which documents the older HD/V2
protocol.  The V3 method is not obtained by renaming a V2 checkpoint: a V3
run must carry the canonical provenance object described below.

This repository contains an implementation and a reproducible benchmark
protocol.  It does **not** contain official success-rate results yet; no
number should be quoted as a result until it is produced by the MIKASA runner
and recorded in a frozen manifest.  The source tree under `lib/` and the
protected `policy/Method1_lerobot-pi0-ttt` directory are outside this method
and are not modified by the V3 recipe.

## 1. Method in one paragraph

At physical time (t), a SmolVLA action expert writes a compact, causal
interaction representation into persistent TTT fast weights.  During
training, a separate full-history causal action teacher replays successful
demonstrations and deletes the fast-weight write of one past interaction (i).
For every later query (j), the increase in the teacher's error (and the associated final
executed-action change) gives a pairwise control-credit label (u_{ij}).  The
student is then trained to reproduce that *query-conditioned* before/after
effect with its local fast-weight update, while a reader-side intervention
loss makes the correctly written memory useful to the action head.  At test
time only the local writer, fast state, and action reader remain: the teacher,
future frames, pair labels, and counterfactual branches are absent.

The three scientific claims are therefore:

1. **Full-history causal control attribution.**  Deleting one past event's
   fast-weight write identifies information that the current observation
   cannot replace but a future expert action still needs.  This is a control
   attribution signal, not an RL value or a success label.  Donor-content
   replacement is retained only as a separately named ablation.
2. **Hindsight-to-local query-conditioned distillation (QH2L).**  Long-delay
   teacher effects are distilled into a local, writer-connected fast-weight
   update objective.  The future query is used while constructing the training
   loss, not as an input available to deployment.
3. **Causal memory deployment (CMD).**  The deployed policy uses the memory in
   the same update-then-read order that was trained, persists it across
   physical decisions, and resets it at episode boundaries.  Correct/wrong/
   reset/irrelevant interventions audit this use; CMD is not an additional
   belief or progress head.

## 2. Notation and causal contract

For one episode, let

| Symbol | Meaning |
| --- | --- |
| (o_t,s_t) | current RGB observation and proprioceptive state |
| (ell) | language/task instruction |
| \(p_t=E_\theta(o_t,s_t,\ell)\) | SmolVLA observation/language/state prefix |
| (b_{t-1}) | normalized action actually executed at the preceding physical step (slot 0) |
| (x_{t,r}) | noisy 50-slot action chunk at denoising step (r) |
| (W_t) | persistent TTT fast-weight state before processing (t) |
| (a_t) | final normalized action chunk; only (a_{t,0}) is executed |

The V3 writer uses the observation-causal interaction

\[
z_t=[R,\;p_t,\;b_{t-1}],\qquad
W_t^+=U_\phi(W_t;z_t),
\]

where \(R\) denotes learned register anchors.  The current noisy action,
denoising time, and future observations are not writer inputs in the
production `prefix_only` path.  A TTT read at a query (q) is

\[
r_{t,r}=f_{W_t^+}(q_{t,r}),\qquad
v_{t,r}=A_\theta(h_{t,r},r_{t,r}),
\]

where \(A_\theta\) is the existing action output projection.  The inner update
is update-then-apply (with an optional bounded numerical stabilizer):

\[
W_t^+=W_t+g_t\bigl(U_\phi(W_t;z_t)-W_t\bigr),
\]

and (g_t=1) in the standard all-write V3 recipe.  The state is carried to
the next physical observation and is cleared by `policy.reset()`.

### Denoising order

For each observation, inference samples (x_{t,1}\sim\mathcal N(0,I)) and
runs the configured ten flow-matching steps.  The callback is called with
`update=True` only at the first step ((t=1)): it writes (W_t^+) and reads
it immediately.  The remaining nine steps call the same TTT layers with
`update=False`, so they read the already updated state without writing again.
The integrated slot 0 of the final chunk is sent to the environment and
stored as (b_t).  Thus “teacher target” below means a final executed action,
not an intermediate denoising velocity.

## 3. Network placement

The implementation is a self-contained `smolvla_ttt` policy copied from the
SmolVLA family; it does not import sibling `smolvla`, `pi0_ttt`, or
`pi05_ttt` policy code.  The default V3 recipe is:

| Component | V3 setting |
| --- | --- |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| VLM/action-expert depth | 16 aligned layers (expert width multiplier 0.75; default hidden width 720 and FFN width 2048) |
| expert attention | alternating cross-attention and causal suffix self-attention (`cross_attn`, every 2 layers) |
| action chunk | 50 slots; MIKASA normally has 7 active coordinates, internally padded to 32 |
| physical action cadence | one executed slot (`n_action_steps=1`) per observation |
| denoising | 10 steps |
| TTT layers | `[12, 13, 14, 15]`; V3 requires the numerically final layer |
| fast MLP width | 1024 in the paper launcher (the config default is a reusable lower-level default) |
| registers | 16 learned expert-width vectors |
| writer mode | `prefix_only` |

TTT is inserted after the selected action-expert attention residual and before
that layer's feed-forward MLP.  The VLM prefix is mapped to expert width by a
learned adapter.  In `prefix_only`, the writer stream is

```text
[16 learned register anchors] + [causal VLM prefix tokens] + [previous executed slot-0 action]
```

The previous action is a boundary-causal interaction feature, not a new
belief/query variable.  At an episode boundary it is the reset/zero sentinel;
it must never be borrowed from another episode.

### Register attention mask

The regular expert suffix remains

```text
[register tokens] + [50 noisy action/time tokens]
```

The asymmetric self-attention mask is intentional:

- register queries may read all valid registers and current action tokens;
- action queries cannot read register key columns and retain the original
  lower-triangular action-to-action mask.

Registers therefore enrich the TTT write/read workspace without becoming a
direct action shortcut.  `ttt_num_register_tokens=0` is a structural ablation,
not a different interpretation of the method.

## 4. Innovation 1: full-history causal teacher

`examples/mikasa/train_full_history_teacher.py` fits
`FullHistoryActionTeacher` on complete demonstration episodes.  Its input at
frame (t) is one detached frozen-prefix event token plus (b_{t-1}); a causal
recurrent state summarizes only frames up to (t).  Its prediction is the
normalized executed slot-0 action at (t).  It never receives the current
expert action as an input, a future observation, the denoising noise, or the
future query used to evaluate a label.

For an event (i) and a strictly later future (j), replay the teacher on the
same episode after deleting event (i)'s write.  This is an **event-write-only**
intervention: the demonstrated previous-executed-action sequence is held fixed
in both branches.  It is not deletion of the physical action or of the whole
interaction.  Keep the real (o_j,s_j) and the expert target (a_j^\star) fixed.
Define

\[
\ell^{\mathrm{full}}_j=\|\hat a^{\mathrm{full}}_j-a_j^\star\|^2,
\quad
\ell^{\mathrm{cf}(i)}_j=\|\hat a^{\mathrm{cf}(i)}_j-a_j^\star\|^2,
\]

\[
u_{ij}=\left[
\frac{\ell^{\mathrm{cf}(i)}_j-\ell^{\mathrm{full}}_j}
 {\tfrac12(|\ell^{\mathrm{cf}(i)}_j|+|\ell^{\mathrm{full}}_j|)+\epsilon}
\right]_+,
\qquad
\Delta a^T_{ij}=\hat a^{\mathrm{full}}_j-\hat a^{\mathrm{cf}(i)}_j.
\]

The implementation also stores the raw signed degradation and a confidence
factor for auditability.  Positive pairs are those above the frozen threshold;
low-utility pairs form the null/invariance stratum.  No physical trajectory is
edited—the intervention exists only in the teacher's input replay.

The shipped adapter is intentionally explicit about its scope:

```text
target_mode          = normalized_executed_slot0_action
teacher_adapter      = causal_action_head
flow_target_available = false
antithetic_noise     = false
```

Consequently these labels are **not** antithetic flow-velocity targets and are
not an oracle Transformer that sees the future.  A future flow-integrated
teacher can be added as a separately identified adapter; it must not be
reported as the current direct-action experiment.

## 5. Innovation 2: QH2L and CMD objectives

### 5.1 Pair representation

The artifact keeps event/future pairs instead of collapsing them to one
per-frame importance scalar.  By default it samples (K=5) pairs per event,
stratified by fixed delay bins and positive/null utility.  A pair always obeys

\[
j\ge i+\texttt{event\_block\_size},\qquad
\texttt{delay}=j-i,
\]

and both indices are episode-local.  Cross-episode pairs are impossible by
construction.

### 5.2 Query-conditioned local effect (QH2L)

At the final selected TTT layer, retain the event transition
((W_i^-,W_i^+)).  For a later query (q_j), the student replays the deployed
denoising flow (same future observation, noise, timestep schedule, and
previous executed action) in two branches and computes the final executed
slot-0 action effect

\[
\Delta a^S_{ij}
 =A(h_j,f_{W_i^+}(q_j))-A(h_j,f_{W_i^-}(q_j)).
\]

The canonical implementation uses this full-flow replay for every sampled
pair, including pairs whose future happens to lie in the current TBPTT
segment.  Pair chunks are an execution-only memory bound: they do not alter
the objective or drop pairs.  The complete reference window supplies a future
observation for cross-segment pairs, while the event snapshot is always the
only varied state.  The deployment state is never reconstructed from a future
state.  The lower-level ``v3_local_effects_from_trace`` helper is retained for
diagnostics and returns a single-phase velocity effect; it is not mixed with
the final-action labels in canonical QH2L.

The target is detached from student gradients.  With robust detached scale
(s_{ij}), a positive/null loss is

\[
\mathcal L_{\mathrm{QH2L}}
 =\operatorname{mean}_{u_{ij}>\tau}
 \operatorname{Huber}\!\left(
 \frac{\Delta a^S_{ij}-\operatorname{sg}(\Delta a^T_{ij})}{s_{ij}}
 \right)
 +\lambda_0\operatorname{mean}_{u_{ij}\le\tau}
 \operatorname{Huber}\!\left(\frac{\Delta a^S_{ij}}{s_{ij}}\right).
\]

The two strata are normalized separately, so adding null pairs or changing a
delay-bin population does not silently rescale the objective.  Utility weights
are bounded; no task-specific temperature is introduced.  QH2L is the loss
that carries the writer/meta-gradient through the local fast-weight update,
which is why V3 requires `ttt_second_order=true`.

### 5.3 Causal Memory Deployment (CMD)

CMD uses the same event snapshots but detaches them before read-only replay.
It therefore trains the query/action reader and shared action tail, while QH2L
alone trains the event writer.  The four auditable terms are:

1. distill the correct-memory action toward the full-history teacher action;
2. match the correct-minus-wrong action effect on positive pairs;
3. rank correct memory ahead of wrong memory by a fixed margin on the expert
   target;
4. enforce near-invariance on null/irrelevant pairs.

This separation prevents a reader diagnostic from becoming a hidden second
writer objective.  Correct, wrong, reset, and irrelevant memory interventions
are also evaluated after training; they are evidence that the learned memory
is actually used, not merely written.

### 5.4 Total student loss

For the canonical V3 branch (`hd_attribution_protocol=
credit_ttt_v3_query_effect`), the implementation combines

\[
\boxed{
\mathcal L
 =\mathcal L_{\mathrm{flow}}
 +\lambda_{\mathrm{local}}\mathcal L_{\mathrm{QH2L}}
 +\lambda_{\mathrm{CMD}}\mathcal L_{\mathrm{CMD}}
 +0.01\,\mathcal L_{\mathrm{anchor}}
}
\]

with launcher defaults (lambda_{\mathrm{local}}=1),
(lambda_{\mathrm{CMD}}=1), `hd_v3_null_weight=0.25`,
`hd_v3_cmd_margin=0.05`, and a fixed 0.01 K/V anchor.  The legacy V2
`hd_effect_weight` path is forced to zero; mixing it into a V3 run would change
the method under study.  The exposed numerical stabilizer (bounded RMS
inner update and finite fallback) is a fixed implementation safeguard, not a
new source of hindsight supervision.

## 6. Full-history training and sequence semantics

The complete episode is the unit of causal credit.  The canonical published
four-task recipe uses all 250 official demonstrations; no demonstration is
silently reserved for tuning:

| Purpose | Episodes |
| --- | --- |
| frozen-prefix feature extraction and teacher fitting | `[0, 250)` |
| student training and label construction | `[0, 250)` (all demos) |
| offline teacher validation | none (optional diagnostics only) |
| simulator evaluation | fixed simulator seeds, never used for selection |

The launcher defaults are `FEATURE_EPISODE_END=250`, `TRAIN_EPISODE_END=250`,
and `VALIDATION_EPISODE_START=250`; the teacher script also treats an omitted
validation threshold as full-data fitting.  A validation threshold at the dataset end
causes the teacher to report a clearly marked train-set diagnostic loss without
back-propagating it or using it for checkpoint selection.  Any reduced split
must be an explicitly named smoke/ablation override and recorded in the
manifest.

For the student, `sequence_length` is resolved to at least the longest selected
training episode, `sequence_stride == sequence_length`,
`max_windows_per_episode=1`, and `ttt_history_warmup_length=null`.  Thus one
window is one complete episode; no window crosses an episode boundary.  This
is intentionally different from a convenient bounded-window pilot.  If a
bounded-window ablation is run, its `history_mode`, context length, and offset
must be named in the manifest and it cannot be called full-history.

The dataset/collator carries `_lerobot_sequence_offset` as an **episode-local**
origin.  A TBPTT segment beginning at local position (s) is passed as
`sequence_offset = window_offset + s`; the origin resets at every episode and
is not a concatenated-dataset index.  The first frame's previous-action input
uses the episode reset sentinel.  Nonzero-offset windows must include the
causal predecessor explicitly; silently zeroing or borrowing it would violate
the teacher/student contract.

### Batch and multi-GPU contract

SmolVLA-TTT sequence windows are intentionally ragged (episode tails have
different lengths) and each window owns an independent recurrent fast-weight
state.  The default collator therefore uses `batch_size=1` **per device**;
padding unrelated episodes in time would create spurious recurrent updates and
change the causal objective.  Data parallelism remains valid: with `N` ranks,
the default effective global sequence batch is `N`, and each rank's state is
reset at the episode boundary.

An opt-in throughput path supports `BATCH_SIZE>1` only with
`EQUAL_LENGTH_BATCHING=1`.  It buckets complete trajectories by the exact pair
`(physical length, episode-local offset)`, retains a separate fast-weight state
for every batch element, and never inserts a temporal padding step or joins two
episodes.  A short bucket is completed by repeating a trajectory from the same
bucket, and complete groups are repeated only as needed to keep DDP ranks in
lockstep; no official demonstration is dropped.  The launcher computes
steps-per-sequence-epoch from this exact bucket arithmetic.  This mode is a
method-neutral batching optimization, but the batch setting and any repeated
sample count must still be recorded in the training manifest.

The sampler's global stream may assign different length buckets to different
ranks at one step (padding every bucket to a multiple of `world_size` would
otherwise duplicate many more demonstrations).  For `BATCH_SIZE>1` with the
default `tbptt_loss_weighting=valid_actions`, the trainer therefore multiplies
each rank's flow mean by `world_size * n_rank / n_global`, where `n_rank` is
its valid action-slot count.  The explicit gradient mean then equals the
global frame/slot-weighted flow objective.  Canonical V3 QH2L/CMD terms use a
separate all-rank pair denominator (described below), so they are not rescaled
by this flow factor.  Historical `BATCH_SIZE=1`, non-equal-length, and legacy
HD paths are unchanged.  This rank-weighting rule and the per-rank counts are
part of the training provenance sidecar.

For canonical V3 batches, the three QH2L/CMD stratum denominators are
all-reduced across DDP ranks before replay.  Each rank receives the global
denominator divided by the world size; this exactly cancels the trainer's
explicit gradient mean, yielding a global pair-weighted numerator/denominator
for every stratum instead of an average of rank-local ratios.  The switch is
`hd_v3_global_pair_normalization` (default `true`); single-process and B=1
paths do not issue these collectives and retain their historical values.  A
`false` value is only a named compatibility/ablation setting.

When memory limits the per-device trajectory batch, the top-level
`gradient_accumulation_steps` option (default `1`) averages gradients from that
many independent windows before one optimizer/scheduler update.  Fast-weight
and grounding states are reset at each window, while all TBPTT segments within
one window retain their existing recurrent carry.  V3 replay callbacks receive
the same `1/N` scale as the ordinary flow loss, so QH2L/CMD ratios and the
writer/reader gradient split are unchanged.  DDP reduction, clipping,
checkpoint, and evaluation cadence occur only on the final micro-window; the
sidecar records the resulting effective batch.

The four-card launcher can run two independent tasks on two cards each, or one
task on all four cards.  Native (non-TTT) SmolVLA has no recurrent-state
restriction and may use an ordinary batch of 4--8 per card.

Every executable student preflight writes
`<student_output>/training_metadata.json`.  This sidecar records the
per-device batch, Accelerate world size, exact `(length, offset)` bucket
counts, bucket-fill repeats, DDP group repeats, effective rows, and
steps-per-sequence-epoch.  Embed the reviewed sidecar when freezing the
benchmark envelope with
`benchmark_credit_ttt_v3.py manifest --training-metadata-json <path>`; the
metadata is hash-protected provenance and is never read by the optimizer.

The V3 full-flow reference replay is evaluated in a fixed, execution-only
pair micro-batch (`CREDIT_TTT_REPLAY_PAIR_CHUNK_SIZE`, default `4`).  Every
sampled event--future pair is still evaluated, and the differentiable before/
after outputs are concatenated before the same complete-window loss is
reduced; changing this bound therefore cannot change the method or its
normalizers (apart from ordinary floating-point summation order).  Chunking
prevents the checkpoint graph from scaling with `T*K` and is what makes the
larger trajectory batch practical on 32-GB cards.  A value of `0` is reserved
for an explicitly named unchunked diagnostic.

`CREDIT_TTT_REPLAY_SAVE_ON_CPU=1` is an optional host-offload escape hatch for
an independently profiled machine.  It is **off by default**: retaining the
saved activations of every pair on the host can exceed a container's RAM cap
even when the GPU has room.  With the canonical chunked path, leaving this
variable unset (or setting it to `0`) keeps each replay checkpoint on the GPU;
the denoising steps, paired noise, gradients, and loss remain unchanged.

For the published four-task benchmark, each task is a separate training run
with its own native initialization, normalization statistics, teacher, labels,
and CreditTTT student checkpoint.  The manifest accepts per-task checkpoint
maps and fails closed when any task entry is missing; no weights are shared
across tasks.

## 7. Artifacts and canonical identity

### 7.1 Pair-label fields

`examples/mikasa/build_credit_labels.py` writes a flat, frame-aligned tensor
artifact.  Each frame owns a (K)-axis of pairs.  Important fields are:

```text
hd_v3_pair_event_index       [N,K]      episode-local event i
hd_v3_pair_future_index      [N,K]      episode-local future j
hd_v3_pair_delay              [N,K]
hd_v3_pair_delay_bin          [N,K]
hd_v3_pair_utility            [N,K]
hd_v3_pair_effect             [N,K,D]   detached Delta a^T_ij
hd_v3_pair_valid/positive/null [N,K]
hd_v3_pair_teacher_full_action         [N,K,D] (CMD adapter)
hd_v3_pair_teacher_counterfactual_action[N,K,D]
hd_v3_pair_expert_action                [N,K,D]
```

Optional `hd_v3_pair_query` and `hd_v3_pair_action_tail` are accepted when a
future flow-query cache is available.  The direct action-teacher path does not
pretend to produce these fields.  Metadata records dataset ID/fps, episode
slices and lengths, feature/teacher/file hashes, pair budget, delay edges,
intervention branch/scope, the full-episode sequence contract, target mode,
and provenance.  The student trainer rejects an artifact whose declared
sequence contract is not `full_episode_replay`,
`max_windows_per_episode=1`, `sequence_stride_policy=equal_sequence_length`,
and `sequence_offset_policy=episode_local_zero`.

### 7.2 Immutable method identity

Every V3 model evaluation and mechanism artifact must carry the following
object (extra implementation fields are allowed, canonical fields are not):

```json
{
  "format": "credit_ttt_v3",
  "protocol": "creditttt_qh2l_v3",
  "version": 3,
  "pair_schema": "event_future_control_pair_v3",
  "intervention": "event_write_deletion",
  "intervention_scope": "event_write_only_previous_executed_action_held_fixed",
  "target": "final_slot0_action",
  "state": "causal_fast_weights",
  "causal": true
}
```

The concrete `intervention_mode` (`delete` for the canonical method, or
`replace` for a separately named ablation) is recorded separately.  The
canonical V3 trainer accepts only `intervention_mode=delete`, because its
student effect is the traced write-before/write-after state difference; a
replacement artifact requires a donor-state replay backend.  The benchmark envelope's
`protocol_id`/`protocol_version` is not authentication: an envelope-only JSON,
a legacy HD/V1/V2 JSON, or a result renamed into a `credit_ttt` directory is
rejected by strict validation.  The evaluator emits the object under
`model.credit_ttt_protocol`; the coordinator cross-checks it against any
envelope copy.

## 8. Reproducible MIKASA benchmark

Use the official simulator and evaluation rules from the
[MIKASA-Robo-VLA installation and benchmark documentation](https://mikasarobo.github.io/installation.html).
Run each task with its own dataset normalization statistics.

### 8.1 Two task instances

The frozen coordinator contains:

| ID | Environment | Dataset | Delay bins reported |
| --- | --- | --- | --- |
| `color` | `ShellGameColorLampTouch-VLA-v0` | `shell_game_color_lamp_touch_vla_v0` | `1-16` only |
| `shuffle_long` | `ShellGameShuffleColorLampTouch-Long-VLA-v0` | `shell_game_shuffle_color_lamp_touch_long_vla_v0` | `1-16`, `17-64`, `65-256`, `257-1024` |

Color episodes are too short to support long-delay claims; those bins must not
be extrapolated or printed as zeros.

### 8.2 Primary comparison

`examples/mikasa/benchmark_credit_ttt_v3.py` freezes this matrix:

| Method | Purpose | Evaluation cadence | V3 identity required |
| --- | --- | ---: | --- |
| Native-SmolVLA | ordinary SmolVLA baseline | (K=50) chunk | no |
| Clean-TTT | same TTT student/action cadence with HD disabled | (K=1) | no |
| CreditTTT | proposed V3 | (K=1) | yes |
| Utility-KVB | optional mechanism baseline | (K=1) | no |

Native (K=50) versus receding-horizon (K=1) changes action cadence.  Before
attributing a gain specifically to memory, add the prescribed native SmolVLA
`K=1` control or state the cadence confound explicitly.  Clean-TTT and
CreditTTT use the same backbone, optimizer budget, action cadence, episode
seeds, and reset protocol.  For a strict architectural control, launch
Clean-TTT with `hd_v3_include_previous_action=true`; this keeps the optional
previous-action projection in both students while removing only the hindsight
objective/labels.  A legacy clean checkpoint without that projection is a
separate, explicitly reported ablation rather than the primary fairness
comparison.

The default statistical contract is 50 paired simulator episodes beginning at
seed `4242424242`, torch seed `7000+i`, and student training seeds
`1000,1001,1002`.  Aggregate per-episode `success_once` with a hierarchical
paired bootstrap (10,000 replicates, 95% CI) and a two-sided exact McNemar test
on common seeds.  Native-SmolVLA is one fixed checkpoint, not three fake
replicates.  Teacher extraction/fitting cost is reported separately and is not
counted as deployed inference cost.

### 8.3 Commands (plan first)

The benchmark manifest defaults to the four published-comparable MIKASA task
profile: `shell_touch` (SGT), `intercept_medium` (IM), `remember_color3`
(RC3), and `remember_color9` (RC9).  To replay the original color/shuffle
experiment envelope, pass `--task-set legacy_two`; old manifests with the
two-task protocol ID remain readable.

The canonical `published_four` protocol is a one-model/one-task comparison:
every selected method (Native K=50, Native K=1, Clean-TTT, CreditTTT, and the
optional Utility-KVB when requested) must use a complete task-local checkpoint
map.  This is required because action-normalization statistics and the TTT
state are task-local; allowing a shared checkpoint would change the scientific
question and can silently apply the wrong task's statistics.  The coordinator
therefore fails closed if any map is absent or incomplete.  The singular
`--*-checkpoint` flags remain a compatibility fallback for the historical
`legacy_two` profile only.

For independently trained tasks, pass a JSON object (a file path or inline
object) mapping stable task IDs to checkpoints, for example:

```json
{
  "shell_touch": "/workspace/checkpoints/credit_shell_touch",
  "intercept_medium": "/workspace/checkpoints/credit_intercept_medium",
  "remember_color3": "/workspace/checkpoints/credit_remember_color3",
  "remember_color9": "/workspace/checkpoints/credit_remember_color9"
}
```

Use `--native-checkpoints-json`, `--clean-checkpoints-json`, and
`--credit-checkpoints-json` with `manifest` (and
`--utility-checkpoints-json` when `--include-optional` is enabled).  The frozen
manifest records `checkpoint_scope` and `checkpoints_by_task`; an incomplete
or shared map is rejected instead of silently falling back to a different
task's checkpoint.  The canonical guard also rejects a map that repeats one
path for multiple task IDs; the two native cadence entries may share each
task's native path, but paths must differ across tasks.  (Checkpoint hashes
remain the final provenance check once artifacts exist.)  To replay the
historical shared-checkpoint envelope, pass `--task-set legacy_two`.

All paths below are placeholders; replace them with real checkpoint and dataset
paths.  The launcher never fabricates an evaluation JSON.

```bash
cd /home/zeno-rp/2026test/test-TTT/policy/HD-TTT

# Dependency-light plan; does not touch the dataset or launch a job.
TASK_ID=shuffle_long ./examples/mikasa/train_credit_ttt.sh plan

# Stage 1: frozen-prefix features + full-history causal action teacher.
EXECUTE=1 TASK_ID=shuffle_long \
  DATASET_ROOT=/workspace/data_mikasa_robo/data_lerobot/shell_game_shuffle_color_lamp_touch_long_vla_v0 \
  BASE_CHECKPOINT=/workspace/checkpoints/clean_ttt \
  ./examples/mikasa/train_credit_ttt.sh teacher

# Stage 2: pairwise hindsight event-write-deletion labels.
EXECUTE=1 TASK_ID=shuffle_long INTERVENTION=delete \
  ./examples/mikasa/train_credit_ttt.sh labels

# Stage 3: QH2L + CMD student (same base checkpoint, named seed/output).
EXECUTE=1 TASK_ID=shuffle_long INTERVENTION=delete SEED=1000 \
  ./examples/mikasa/train_credit_ttt.sh student

# Print official Native/Clean/Credit evaluation commands; review first.
TASK_ID=shuffle_long ./examples/mikasa/train_credit_ttt.sh baselines
```

For a complete run, `EXECUTE=1 ... train_credit_ttt.sh all` performs teacher →
labels → student in that order, but it still does not evaluate.  Freeze the
benchmark envelope and commands before launching evaluation:

```bash
python examples/mikasa/benchmark_credit_ttt_v3.py self-check
python examples/mikasa/benchmark_credit_ttt_v3.py manifest \
  --output benchmark_results/credit_ttt_v3/manifest.json \
  --repo-root "$PWD" \
  --task-set legacy_two \
  --native-checkpoint /workspace/checkpoints/native_smolvla \
  --clean-checkpoint /workspace/checkpoints/clean_ttt \
  --credit-checkpoint /workspace/outputs/credit_ttt_v3/shuffle_long/student/checkpoints/last/pretrained_model
python examples/mikasa/benchmark_credit_ttt_v3.py plan \
  --manifest benchmark_results/credit_ttt_v3/manifest.json
```

After real official-runner JSON files exist:

```bash
python examples/mikasa/benchmark_credit_ttt_v3.py aggregate \
  --manifest benchmark_results/credit_ttt_v3/manifest.json \
  --results-root benchmark_results/credit_ttt_v3 \
  --output benchmark_results/credit_ttt_v3/aggregate.json
```

### 8.4 Mechanistic go/no-go audit

The benchmark is not reduced to a single SR number.  Before a paper claim is
made, supply a mechanism JSON and run `check --strict`.  Required evidence is:

- full-history teacher action loss improves over short/clean replay;
- history swap changes the teacher in the expected direction;
- QH2L teacher/student effect cosine has a positive 95% lower CI;
- local effect agrees with exact short-horizon E2E gradient credit;
- writer gradients remain nonzero in the declared long-delay bins;
- top-attributed events beat random events and recall@8 exceeds random;
- pairwise Δ-action alignment is reported separately by task and delay bin;
- deployed correct/wrong/reset memory drift exceeds irrelevant-memory drift;
- all losses/states remain finite and the fast-state RMS ratio is bounded.

Missing evidence is `UNKNOWN`, not a pass.  `check --strict` fails unknown
required fields and returns a nonzero status unless every declared gate is
resolved.  A positive benchmark SR difference is reported separately from
these mechanism checks; one cannot substitute for the other.

## 9. Structural ablations and fairness rules

Pre-register the following, keeping data, seeds, optimizer updates, and action
cadence fixed:

1. Clean-TTT (no hindsight objective) vs CreditTTT.
2. No registers (`ttt_num_register_tokens=0`) vs 16 registers.
3. `suffix` writer vs `prefix_only` writer (the former is explicitly
   action/noise-conditioned and is not the causal V3 writer).
4. QH2L only, CMD only, and QH2L+CMD.
5. Canonical event-write deletion vs phase-matched content-replacement
   intervention (offline ablation only; do not pool the two protocols).
6. Persistent state vs reset-every-step diagnostic.
7. Full-history vs a named bounded-window context, with matched seen frames.
8. Native (K=50) and native (K=1) cadence controls.

The objective-family switch is explicit in the checkpoint configuration and
launcher: `V3_ABLATION=full` (canonical, local=1/CMD=1),
`V3_ABLATION=qh2l_only` (writer/QH2L only), or
`V3_ABLATION=cmd_only` (reader/CMD only, local=0).  The model skips the
disabled replay, records a corresponding `*_disabled` metric, and rejects a
zero weight whose declared ablation does not match it.  Thus CMD-only is a
reader ablation rather than an accidentally weakened full model.

Weights such as inner learning rate, residual gate, null weight, and pair
budget are implementation parameters.  They may affect performance, but must
be fixed from the training split/recipe and not selected on simulator test
seeds.  A small predeclared parameter-neighborhood audit (for example ±20%) is
useful evidence of robustness; it is not a replacement for structural
ablations or a license to pick the best test result.

## 10. Known boundaries (state these in the paper)

- The current hindsight teacher predicts normalized executed slot-0 actions,
  not flow velocities; `flow_target_available=false` and
  `antithetic_noise=false` are intentional metadata, not missing results.
- The production writer uses projected current prefix embeddings plus the
  previous executed action.  It is not a claim that every write sees a fully
  contextualized final VLM hidden state.
- The current pair effect is defined for the final selected TTT layer and the
  executed slot-0 action.  It does not claim all 50 action slots or every
  intermediate denoising velocity has been independently attributed.
- Full-flow pair replay is computationally expensive (especially with
  second-order writer gradients); it is a training-time reference operation.
  Pair chunking bounds peak memory without changing the sampled population.
  Deployment remains one causal writer update and nine read-only denoising
  reads per observation.
- Color supports only the short delay bin in the frozen benchmark.  Long-delay
  conclusions require Shuffle-Long or another episode set with those delays.
- A passing tensor/unit smoke test proves shape, finite, or gradient contracts,
  not simulator success.  Until `aggregate.json` is generated from real
  official-runner files, report the result as **not measured**.

## 11. File map and quick checks

```text
policy/HD-TTT/
├── README_CREDIT_TTT.md                         # this V3 method/benchmark contract
├── examples/mikasa/
│   ├── train_credit_ttt.sh                       # teacher → labels → student recipe
│   ├── train_full_history_teacher.py             # causal direct-action teacher
│   ├── build_credit_labels.py                    # pairwise intervention artifact
│   ├── benchmark_credit_ttt_v3.py                # manifest, aggregation, audit gates
│   ├── evaluate_smolvla_ttt.py                   # persistent/reset MIKASA adapter
│   └── evaluate_smolvla_baseline.py              # native baseline adapter
├── src/lerobot/policies/smolvla_ttt/
│   ├── credit_ttt_v3.py                          # protocol, sampler, QH2L/CMD primitives
│   ├── history_teacher.py                        # full-history teacher/replay math
│   ├── modeling_smolvla_ttt.py                   # flow, TTT hooks, V3 losses
│   ├── ttt.py                                    # fast MLP/update/read/state
│   ├── sequence.py                               # episode windows and local offsets
│   └── hd_dataset.py                             # label/provenance loader
└── tests/                                        # protocol, gradient, offset, trace tests
```

Dependency-light checks that do not launch a GPU job:

```bash
bash -n examples/mikasa/train_credit_ttt.sh
PYTHONPATH=src python examples/mikasa/benchmark_credit_ttt_v3.py self-check
pytest -q tests/policies/smolvla_ttt/test_credit_ttt_v3.py \
  tests/policies/smolvla_ttt/test_previous_action_carry.py \
  tests/test_history_teacher.py tests/test_build_credit_labels.py
```

The last command requires the repository's test dependencies.  Always retain
the printed git commit, manifest SHA256, checkpoint/config hashes, dataset
revision, Python/PyTorch/Transformers versions, GPU, and all launcher
environment overrides with a paper run.
