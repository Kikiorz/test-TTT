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

## DiT-TTT

DiT-TTT attaches one independent fast memory after every official DiT block.
The newest observation token creates all 12 residuals once per environment
decision. Every flow-integration step reuses those residuals. Consequently the
number of writes is independent of the number of integration steps.

## Required invariants

1. `gate0` output equals the corresponding baseline exactly.
2. A decision performs exactly one observation-only write per memory.
3. Action targets influence the outer policy loss only.
4. Fast state persists throughout an episode and resets between episodes.
5. TBPTT detaches gradients without resetting the numerical fast state.

`policy/DiT_TTT/test_invariants.py` checks the first three structural
properties for the DiT family.

