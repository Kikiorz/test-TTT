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
causal teacher pass has produced attribution labels.  `hd_ttt.py` contains the pure tensor
implementations of the three paper terms:

* `compute_hindsight_attribution` computes the leakage-safe positive credit
  (C_{i\to j}=[L^{-i}_j-L_j]_+), together with event `u` and future `rho` weights.
* `local_kvb_loss` is the deployable K/V writer objective; it never requires an expert action
  at test time.
* `counterfactual_grounding_loss` matches correct-vs-wrong-memory action changes while
  enforcing invariance for low-dependency futures. Teacher tensors are always detached.

HCA intervention labels are training-only. Deployment keeps the ordinary SmolVLA flow loss,
one update-then-apply fast-weight write per physical observation, and the recurrent state reset
at episode boundaries.

For the second stage, load the first-stage checkpoint and set `--policy.ttt_training_stage=action_head`. Call `policy.reset()` between evaluation episodes so recurrent fast weights never cross episode boundaries.
