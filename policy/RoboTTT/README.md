# RoboTTT policy

This directory is the clean-room reconstruction of NVIDIA RoboTTT from the
public paper. It is deliberately separate from `policy/DiT_TTT`, which is the
legacy experimental policy that produced the archived LIBERO result.

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
not have 16 layers or policies that do not use 16 registers. The included
LIBERO adapter still uses the repository's ResNet feature encoder rather than
the unpublished GR00T N1.7 Eagle stack, so it must be described as a benchmark
adapter, not an exact reproduction of NVIDIA's full model.

## Publicly under-specified choices

The paper does not disclose the fast-MLP hidden width, optimizer beta values,
or how a single recurrent state is committed across multiple flow-denoising
evaluations at one robot timestep. We therefore use `d -> d -> d`, AdamW
betas `(0.9, 0.95)`, and start every denoising evaluation from the same
`W_(t-1)` while committing only the last candidate `W_t`. These choices are
explicitly tagged in checkpoints and are not presented as official source.
