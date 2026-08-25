# RoboTTT policy

This directory is a standalone clean-room reconstruction of NVIDIA RoboTTT
from the public paper. You can delete every sibling directory under `policy/`
without breaking this package. It does not import `DP`, `DP_TTT`, `DiT`, or
`DiT_TTT`.

## Standalone layout

- `backbone.py`: package-local LIBERO adapter and official Isaac-GR00T DiT.
- `layer.py`: recurrent TTT-KVB fast-weight layer.
- `policy.py`: 16-layer RoboTTT action head and rollout state semantics.
- `train.py`: 30K pretrain plus 20K post-train entrypoint.
- `training_utils.py`: checkpoint and contiguous-trajectory utilities.
- `config.yaml`: complete disclosed/reconstructed configuration boundary.
- `PAPER_MAPPING.md`: paper-to-code audit and under-specified choices.
- `test_invariants.py`: causal, gate-zero, update-count, and meta-gradient tests.
- `test_isolation.py`: rejects imports from every legacy policy package.

Required external libraries are PyTorch, torchvision, NumPy, and NVIDIA
Isaac-GR00T (for `gr00t.model.modules.dit.DiT`). The GR00T source is an explicit
third-party dependency, not a sibling-policy dependency.

## Public-paper contract

- GR00T N1.7 VLM plus a 16-layer DiT action head.
- One independent TTT layer after the attention operation of every DiT block
  and before its feed-forward operation.
- Per-step TTT stream: 16 learned registers, one proprioception token, and the
  current noisy action chunk. VL tokens remain cross-attention context.
- K-to-V MSE update by standard gradient descent, followed by Q readout.
- Two-layer GeLU fast MLP, learned inner-rate multiplier over base rate 0.1,
  RoPE theta 10000, and channel-wise `tanh(alpha)` gate initialized at 0.001.
- Independent `tau = 0.999 * (1-u)`, `u ~ Beta(1.5, 1)`, and Gaussian noise for
  each action chunk in sequence training.
- Fast weights persist for the full trajectory; TBPTT detaches gradients only
  at segment boundaries.
- Sequence-only pretraining, then full-model post-training.

`strict_paper_action_head=True` is the default and rejects action heads that do
not have 16 layers, 16 registers, the reported roughly 538M-parameter DiT
scale, and roughly 10M parameters per TTT layer. The included LIBERO adapter
uses ResNet features rather than the unpublished GR00T N1.7 Eagle stack, so a
small benchmark must pass `--allow-architecture-approximation` and be described
as a scale/encoder adapter, not an exact reproduction of NVIDIA's full model.

For the reported hardware and batch schedule, run pretraining and post-training
as separate jobs: 16 GPUs for pretraining (per-device batch 4 through 4K,
then 1 above 4K), followed by 8 GPUs for post-training (per-device batch 1)
using `--stage posttrain --resume-robottt <pretrain-checkpoint>`. A single
`--stage both` job is a convenience mode and cannot reproduce that GPU split.

Setting a cache-level `outer_loss_mask[t]=false` turns timestep `t` into pure
context: it still updates fast weights but contributes no action imitation
loss. This is the paper's mechanism for in-context video and DAgger Distillation.

## Publicly under-specified choices

The paper does not disclose the fast-MLP hidden width, optimizer beta values,
QKV bias convention, learned-rate parameterization, WSD phase fractions, the
intermediate context schedule, or how one recurrent state is committed across
multiple flow-denoising evaluations at one robot timestep. We use `d -> d -> d`,
AdamW betas `(0.9, 0.95)`, bias-free QKV, an exponential positive rate
multiplier, a documented WSD split, and commit only the final denoising
candidate. These are reconstruction choices, not official source details.
