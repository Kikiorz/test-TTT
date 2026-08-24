# RoboTTT paper mapping

This repository reconstructs RoboTTT from the public paper and project page;
NVIDIA has not released the implementation. The mapping below separates direct
paper statements from implementation choices needed to make the description
executable.

## Directly specified by the paper

| Paper statement | Implementation |
|---|---|
| One TTT layer is added to each of the 16 GR00T N1.7 DiT layers (Sec. 3.4, App. A.1). | `PaperRoboTTTPolicy` creates one `RoboTTTKVBLayer` per configured DiT block. The controlled LIBERO backbone has 12 blocks, so it creates 12. |
| TTT is after self/cross attention; attention is per robot timestep and TTT operates across robot timesteps (Sec. 3.1, Fig. 2). | Each block is split into attention, TTT-gated fusion, then FFN. Every layer carries its own fast weights across decisions. |
| TTT processes register, proprioception and noised-action tokens; VL tokens bypass TTT directly (Sec. 3.1, Fig. 12). | The current token group is `[R_t, q_t, A_t^tau]`; visual tokens are only cross-attention keys/values. |
| There are 16 learned register tokens (Sec. 3.1). | `num_register_tokens=16` by default. This number is independent of the DiT layer count. |
| The fast model is a two-layer GeLU MLP and uses learned inner rate on base rate 0.1 (App. A.1). | `RoboTTTKVBLayer`; default hidden width is `4d`, and `eta=0.1*exp(multiplier)`. |
| RoPE uses theta 10000 (App. A.1). | Q and K receive recurrent token-position RoPE with `rope_theta=10000`. |
| Gate is channel-wise `tanh(alpha)`, with alpha initialized to 0.001 (Sec. 3.1). | Every TTT layer owns `gate_alpha` with shape `[d]`. |
| Sequence action forcing samples flow noise independently per action chunk (Sec. 3.2). | Every environment timestep samples its own Beta(1.5,1) flow level and noise. |
| TBPTT detaches gradients but carries numerical fast weights across segments (Sec. 3.2). | The trainer detaches every fast-state tensor only at TBPTT boundaries and does not reset it until the next episode. |

Sources: [RoboTTT paper](https://arxiv.org/html/2607.15275),
[NVIDIA project page](https://research.nvidia.com/labs/gear/robottt/).

## Explicit reconstruction choices

The paper does not specify how recurrent state is committed across multiple
flow-denoising evaluations at one environment timestep. This implementation
starts every denoising evaluation from the same `W_(t-1)` and commits the final
candidate `W_t`. Therefore one environment decision advances recurrent time
once instead of treating denoising iterations as additional robot history.

The controlled LIBERO model uses the public Isaac-GR00T 12-block DiT module,
ResNet visual features and one proprioception token rather than the unpublished
full GR00T N1.7/Eagle configuration. It is architecture-aligned but not a claim
of exact reproduction of NVIDIA's parameter count or training data.

Adding register and state tokens changes the matched backbone token layout.
Consequently `gate0` is tested against the same registered backbone with TTT
disabled, not against the earlier action-token-only checkpoint output. Existing
base weights can initialize the attention/FFN/action modules, but the new
register and TTT parameters require training.

## Result boundary

The archived LIBERO DiT-TTT result used `policy/DiT_TTT/policy.py`, the earlier
observation-summary residual implementation. It is preserved for reproducibility
and must not be reported as a result of `PaperRoboTTTPolicy`. The paper-based
implementation needs new matched gate0/frozen/online closed-loop evaluation.
