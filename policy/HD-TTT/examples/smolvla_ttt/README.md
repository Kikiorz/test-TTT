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
  --batch_size=1 \
  --steps=30000
```

The default prepends 16 learned register tokens before the 50 action tokens. Register queries
can read the current register/action block; action queries retain the original causal
action-to-action mask and cannot directly read register keys. Use
`--policy.ttt_num_register_tokens=0` for the exact no-register ablation while keeping the
TTT-KVB update, layer placement, denoising schedule, and losses unchanged.

## HD-TTT training objects

The optional HD-TTT objectives are enabled with `--policy.hd_ttt_enabled=true` after a
causal teacher pass has produced attribution labels.  The paper path is the explicit v2
contract (`--policy.hd_attribution_protocol=v2_relative_antithetic_robust`,
`--policy.hd_effect_weight=1`, `--policy.hd_grounding_weight=0`, and
`--policy.ttt_writer_mode=prefix_only`).
`hd_ttt.py` contains the pure tensor implementations of the causal terms:

* `compute_hindsight_attribution` computes the leakage-safe positive credit
  (C_{i\to j}=[L^{-i}_j-L_j]_+), together with event `u` and future `rho` weights.
  `compute_robust_hindsight_attribution` is the v2 signed/antithetic-compatible
  primitive; it exposes positive and harmful credit separately.
* `local_kvb_loss` is the deployable K/V writer objective; it never requires an expert action
  at test time.
* `counterfactual_grounding_loss` matches correct-vs-wrong-memory action changes while
  enforcing invariance for low-dependency futures. Teacher tensors are always detached.
* `action_effect_distillation_loss` matches the slot-0 true-minus-wrong action effect
  with writer gradients intact; v2 uses a robust median-RMS scale and Huber penalty.

The v2 label builder uses an antithetic `(z, -z)` replay pair, symmetric relative credit,
adaptive top-`sqrt(n)` aggregation, and percentile normalization.  These are fixed
algorithmic choices, not per-task knobs.  The formal path stores one selected effect branch;
the main student consumes branch 0.  Older artifacts with K>1 branches remain readable, but
extra branches are ignored unless a separately implemented multi-event ablation is enabled.

HCA intervention labels are training-only. Deployment keeps the ordinary SmolVLA flow loss,
one update-then-apply fast-weight write per physical observation, and the recurrent state reset
at episode boundaries.

`history_teacher.py` provides an independent causal GRU-style history encoder for experiments
that need an explicit teacher. It is not imported by the default label builders, does not add an
input or parameter to a deployed student, and must be reported as a separate teacher ablation
with its format/state hash if enabled.

The frame-level hindsight builder stores one selected wrong-memory branch for grounding. To keep
that branch and `hd_rho` aligned, it requires at least 64 eligible future frames by default and
selects by mean causal credit. Short episodes/windows with no eligible event use the positive
event with the largest total credit. The rule is recorded as
`grounding_event_policy=min_future_horizon_mean_else_total_credit` and the threshold as
`grounding_min_future_frames`; override the builder CLI with
`--grounding-min-future-frames N` and pass the same value as
`--policy.hd_grounding_min_future_frames=N` during training. A value of `0` is an explicit
selection ablation, not a claim that the v2 attribution formula changed. HCA's robust v2
aggregation and the selected grounding branch must remain separately auditable.
Artifacts generated before this contract field was introduced are intentionally rejected by the
strict loader; regenerate them rather than silently interpreting their terminal-event `hd_rho`.

### Optional causal local write gate

The learned gate is an optional ablation. It is not needed by the minimal paper
objective, because H2L and action-effect distillation already train the writer:

```bash
--policy.hd_ttt_enabled=true --policy.hd_learned_write_gate=true
```

The first selected TTT layer predicts one scalar `g_t` in `(0,1)` from a masked
pool of the current observation prefix (image, language, and proprioceptive
state embeddings). It is computed before the action suffix is embedded, so it
has no access to candidate actions, flow noise, or the denoising timestep. The
same gate is used by every selected TTT layer. The offline
`hd_write_gate`/`u_t` label is only a Smooth-L1 training target; it is never
supplied at deployment. The first denoising step advances the fast state with
the predicted gate and later denoising steps only read it. A context-free
`TTTMLPLayer` can still be instantiated for an explicit action-conditioned
unit-test ablation. The minimal paper checkpoint leaves this optional head
disabled; `hd_ttt_enabled=false` also does not construct or call it when
loading a clean/base checkpoint, preserving the ordinary TTT path.

History warm-up frames carry a separate `hd_writer_valid` mask. Their action
targets remain masked, while the local K/V and gate losses still train on the
real interactions that prefill the recurrent state. If a label pass uses a
positive `--max-events` cap, `hd_write_gate_observed` masks gate distillation on
unsampled blocks; the safe default gate of 1.0 is never treated as measured
credit.

When initializing HD-TTT from an existing clean TTT checkpoint, start a new run with
`--policy.pretrained_path=<checkpoint>` and the HD flags above. Do not resume the old
optimizer state when changing the effect objective; its inner-update graph is a new
training path.

### Loading offline labels

Pass `--dataset.hd_label_path=/path/to/labels.pt` (or a directory containing
`labels.pt`, `labels.npz`, or `labels.json`) together with the SmolVLA-TTT
dataset. The loader accepts one row per dataset frame, episode/frame records,
or episode-packed columns. Canonical columns are `hd_teacher_velocity`,
`hd_teacher_true_velocity`, `hd_teacher_wrong_velocity`, `hd_attribution`,
`hd_rho`, `hd_write_gate`, `hd_write_gate_observed`,
`hd_counterfactual_write_gate`, and the optional
`hd_local_{key,value,prediction,query}` tensors. Short aliases such as `rho`,
`u`, `write_gate`, `C`, `hd_u`, and `hd_C` are accepted; a full `C[event,future]` matrix is
reduced to one future-time attribution weight before batching. Labels are
attached after episode selection and flow through the processor as
complementary `hd_*` data, so they are not normalized with observations/actions.

For example:

```bash
lerobot-train \
  --policy.type=smolvla_ttt \
  --policy.hd_ttt_enabled=true \
  --dataset.repo_id=<USER>/<DATASET> \
  --dataset.hd_label_path=/data/<DATASET>/hd_labels.pt \
  --batch_size=1
```

For the second stage, load the first-stage checkpoint and set `--policy.ttt_training_stage=action_head`. Call `policy.reset()` between evaluation episodes so recurrent fast weights never cross episode boundaries.
