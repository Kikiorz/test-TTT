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
can read all register/action tokens; action queries can read all registers while preserving the
original causal action-to-action mask. Use
`--policy.ttt_num_register_tokens=0` for the exact no-register ablation while keeping the
TTT-KVB update, layer placement, denoising schedule, and losses unchanged.

For the second stage, load the first-stage checkpoint and set `--policy.ttt_training_stage=action_head`. Call `policy.reset()` between evaluation episodes so recurrent fast weights never cross episode boundaries.
