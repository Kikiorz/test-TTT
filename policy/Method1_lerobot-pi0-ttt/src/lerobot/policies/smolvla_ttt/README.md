# SmolVLA-TTT

`smolvla_ttt` is a self-contained SmolVLA variant with RoboTTT-style recurrent fast-weight MLPs in selected action-expert layers. The implementation is copied from SmolVLA and owns its configuration, model, processor, VLM/expert code, TTT layer, and sequence dataset; it does not import the sibling `smolvla`, `pi0_ttt`, or `pi05_ttt` policy implementations.

TTT is applied after each selected action-expert attention residual and before its feed-forward MLP. By default, 16 learned register tokens are prepended before the original 50 action tokens. Within self-attention, register queries read all register/action tokens, while action queries read all registers and retain the original causal action-action pattern. All tokens then participate in the same TTT-KVB update. Set `ttt_num_register_tokens=0` for the original no-register path.

During inference, the fast state advances only on the first flow-matching denoising evaluation for an observation. Later denoising evaluations reuse that candidate state, and `policy.reset()` clears it at an episode boundary.

Training uses episode-local sequences with truncated backpropagation through time. The default two stages are:

1. `ttt_only`: freeze the SmolVLA backbone and the residual gate; train the remaining TTT parameters.
2. `action_head`: train TTT (including the gate), the action expert, state projection, and action/time projections while keeping the VLM frozen.

Start from a base SmolVLA checkpoint with `--policy.type=smolvla_ttt --policy.pretrained_path=lerobot/smolvla_base`, or initialize from the backbone configuration with `--policy.type=smolvla_ttt`. Sequence training currently requires per-device `batch_size=1`.
