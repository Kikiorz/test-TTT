# SmolVLA-TTT

`smolvla_ttt` is a self-contained SmolVLA variant with RoboTTT-style recurrent fast-weight MLPs in selected action-expert layers. The implementation is copied from SmolVLA and owns its configuration, model, processor, VLM/expert code, TTT layer, and sequence dataset; it does not import the sibling `smolvla`, `pi0_ttt`, or `pi05_ttt` policy implementations.

TTT is applied after each selected action-expert attention residual and before its feed-forward MLP. By default, 16 learned register tokens are prepended before the original 50 action tokens. Within self-attention, register queries read all register/action tokens, while action queries read all registers and retain the original causal action-action pattern. All tokens then participate in the same TTT-KVB update. Set `ttt_num_register_tokens=0` for the original no-register path. SmolVLA's existing prefix state conditioning is retained unchanged.

During inference, the fast state advances only on the first flow-matching denoising evaluation of an
action-chunk prediction. Later denoising evaluations reuse that candidate state. If `n_action_steps=K`
with `1 <= K <= chunk_size`, the next update occurs after those `K` queued actions have executed.
`policy.reset()` clears the live state at a rollout boundary.

Training follows RoboTTT's sequence-level outer objective. Every lane in a loader batch is one independent
training sequence: either a complete short episode or an episode-local contiguous sub-trajectory of at most
`sequence_length` timesteps. Each lane starts from the learned fast-weight initialization `W0`; fast weights
carry across its TBPTT segments, and only their autograd history is detached at each segment boundary. Exactly
one slow-weight optimizer/scheduler step occurs after the complete sequence minibatch. Fast state never crosses
into the next loader batch or across an outer optimizer update.

The default two stages are:

1. `ttt_only`: freeze the SmolVLA backbone and train the inserted TTT layers, learned registers, inner
   learning rates, and the learned near-zero (`0.001`) residual gates.
2. `action_head`: keep training TTT while also training the action expert, state projection, and action/time
   projections; the VLM remains frozen.

Start from a base SmolVLA checkpoint with `--policy.type=smolvla_ttt --policy.pretrained_path=lerobot/smolvla_base`, or initialize from the backbone configuration with `--policy.type=smolvla_ttt`.

Sequence training supports per-device batches larger than one. The sampler shuffles independent windows;
an incomplete final global batch uses fully masked dummy lanes instead of repeating real data. Variable tails,
dummy timesteps, and padded action tokens neither contribute to the outer action loss nor write padding into
the TTT-KVB update. Each active sequence is normalized by its own valid-action count before the global batch
mean.

`sequence_stride` may be smaller than `sequence_length`; overlapping windows remain independent samples and
each starts from `W0`. A final partial TBPTT segment is supported, so the two lengths need not be divisible.
Live fast state is intentionally absent from checkpoints, but checkpoints may be saved/resumed at any
completed optimizer step because no state survives a training-sequence boundary.

Changing batch size still changes the number and composition of optimizer updates; that ordinary SGD
effect cannot be made identical to `batch_size=1`. Keep batch size one when exact optimizer-trajectory
comparability is required.

Only checkpoints marked `ttt_sequence_state_semantics=sequence_outer_step_v1` implement this training
contract. Earlier `full_episode_v1`/`full_episode_outer_step_v2` checkpoints used different outer-step or
context-length semantics; they may be evaluated as labelled ablations but must not initialize a strict stage.

NVIDIA's reported GR00T model inserts TTT into all 16 action-head layers. This SmolVLA port supports the same
setting with `ttt_start_layer=0`; its default last-four-layer selection follows the smaller local PI0-TTT
engineering recipe and is an explicit compute/parameter-count choice rather than a claim about the paper.
