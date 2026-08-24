# Algorithm

## Fast memory

Let `z_t` be the newest observation feature. Learned projections form
`q_t`, `k_t` and `v_t`. A two-layer fast MLP with episode state `W` is read
before it is updated:

```text
r_t       = FastMLP(q_t; W_(t-1))
c'_t      = c_t + gate * Out(r_t)
L_inner   = ||FastMLP(k_t; W_(t-1)) - v_t||^2
W_t       = W_(t-1) - eta * grad_W L_inner
action_t  ~ BasePolicy(c'_t)
```

`k_t` and `v_t` depend only on `z_t`. The implementation deliberately does not
use an expert action, executed action, reward, success signal or future frame in
`L_inner`.

## DP-TTT

The DP visual encoder produces one feature per history frame. The TTT residual
is added to the flattened global conditioning vector of the conditional UNet.
One residual is reused across all DDPM denoising iterations.

Relevant files:

- `policy/DP_TTT/fast_memory.py`
- `policy/DP_TTT/policy.py`
- `policy/DP_TTT/full_episode_policy.py`
- `policy/DP_TTT/train_full_episode.py`

## GR00T-DiT baseline

`policy/DiT/model.py` uses NVIDIA's official 12-block DiT as the action head.
Two ResNet-18 encoders and the 8-D robot state create nine context tokens from
three observation steps. Eight noisy 7-D action tokens are trained using flow
matching with Beta(1.5, 1.0) time sampling and sampled using four Euler steps.

## Paper RoboTTT reconstruction for DiT

`PaperRoboTTTPolicy` follows the architecture stated in RoboTTT rather than the
earlier observation-summary residual approximation:

```text
for each DiT layer l:
    X_l = Attention_l([16 register tokens, state token, noisy action tokens], VL tokens)
    (O_ttt, W_l,t) = TTT_KVB_l(X_l, W_l,t-1)  # two-layer GeLU fast MLP
    X_l = X_l + tanh(alpha_l) * O_ttt         # channel-wise alpha_l
    X_l = X_l + FFN_l(norm(X_l))
```

TTT is inserted after attention and before the block FFN. Vision tokens stay on
the cross-attention path; TTT receives their information indirectly through the
register/state/action attention outputs. There is one independent fast state
per DiT layer. RoPE uses theta 10000, the base inner rate is 0.1, and alpha is
initialized to 0.001.

The pre-existing `DiTTTPolicy` in `policy/DiT_TTT/policy.py` is retained only to
reproduce the completed experiment. It precomputes one observation-derived
residual per layer and is not an exact reconstruction of the paper architecture.

## Required invariants

1. `gate0` equals the matched register-token backbone exactly.
2. There is one TTT layer and one fast state per DiT layer.
3. Register, proprioception and noisy-action tokens pass through TTT; VL tokens do not enter it directly.
4. Fast state persists throughout an episode and resets between episodes.
5. TBPTT detaches gradients without resetting the numerical fast state.
6. Denoising evaluations start from the same `W_(t-1)` and commit one `W_t` per environment decision.

`policy/DiT_TTT/test_robottt_invariants.py` checks these structural properties.

No clean future action is exposed at deployment. During training the current
action appears through the standard flow-matching noisy sample
`A_t^tau = tau A_t + (1-tau) epsilon`; this is sequence action forcing from the
paper, not an extra expert-action input to the deployed policy.
