# RoboTTT public-paper compliance audit

The implementation is isolated in `policy/RoboTTT`. NVIDIA has not released
the RoboTTT source, GR00T N1.7/Eagle checkpoint, or training mixture. Therefore
“compliant” below means the public algorithm is implemented literally; it does
not mean byte-identical reproduction of unpublished code and assets.

## Direct paper requirements

| Public requirement | New implementation | Status |
|---|---|---|
| GR00T N1.7 with one TTT layer in each of 16 DiT layers (Sec. 3.4, App. A.1) | Strict mode requires 16 DiT blocks and creates 16 independent fast states. | exact at action-head level |
| TTT after a block's self/cross-attention and before its FFN (Sec. 3.1) | Every official GR00T `BasicTransformerBlock` is split at that boundary. | exact |
| Per-timestep attention over registers, state and noisy actions, cross-attending to current VL tokens | Current token stream is `[R_t, q_t, A_t^tau]`; only the latest observation's VL tokens enter cross-attention. | exact |
| VL tokens do not enter TTT directly; 16 registers carry their information (Sec. 3.1) | TTT receives only attention outputs for 16 registers, one state token and noisy actions. | exact |
| K-to-V MSE gradient update, then Q readout (Eqs. 1–2) | `RoboTTTKVBLayer` performs update-then-apply with differentiable inner SGD. | exact |
| Two-layer GeLU fast MLP; standard gradient descent (App. A.1) | Functional `Linear -> GeLU -> Linear`; one SGD step per robot timestep. | exact except unpublished hidden width |
| Learned inner rate on base rate 0.1 (App. A.1) | `eta = 0.1 * exp(log_multiplier)`. | exact parameterization choice |
| RoPE with theta 10000 (App. A.1) | Applied to Q and K over the flattened recurrent token stream. | exact theta; position layout under-specified |
| Per-channel `tanh(alpha)` gate, alpha initialized 0.001 (Eq. 3) | One vector gate per DiT layer, initialized exactly to 0.001. | exact |
| Outer flow loss at every timestep, meta-gradient through inner updates (Sec. 3.2) | Outer MSE backpropagates through the inner gradient step and meta-learns Q/K/V and W0. | exact |
| `tau = 0.999(1-u)`, `u ~ Beta(1.5,1)`, independently sampled per chunk (Eq. 5) | Implemented literally with independent Gaussian noise per robot timestep. | exact |
| Full trajectory or contiguous subtrajectory; increasing pretraining context to 8K | Trainer samples contiguous windows and schedules 128→256→512→1K→2K→4K→8K. | endpoints exact; progression under-specified |
| TBPTT carries numerical fast weights and detaches their gradients at segment boundaries (Sec. 3.2) | Fast state resets only at a new trajectory window and is detached only after each TBPTT optimizer step. | exact |
| Pretrain sequence layers only; then post-train all parameters (App. A.2) | Stage names and trainability follow this rule. Encoded-LIBERO mode must freeze disconnected visual encoders and is labeled an adapter. | exact core; adapter limitation |
| 30K pretrain steps, 20K post-train steps; AdamW weight decay 1e-5 | Optimizer-step budgets are defaults; checkpoint/log counters are steps, not epochs. | exact disclosed values |
| WSD peak LR 2e-5; cosine peak LR 5e-5 | Schedulers advance every optimizer step. | exact disclosed values |
| Start each rollout at W0 and propagate fast weights across robot timesteps | `reset_ttt_state()` begins a rollout; state persists until the next reset. | exact |

Sources: [RoboTTT paper](https://arxiv.org/html/2607.15275) and
[NVIDIA project page](https://research.nvidia.com/labs/gear/robottt/).

## Corrections made in the second audit

1. The earlier reconstruction sampled `tau = 0.999u`. Equation (5) says
   `tau = 0.999(1-u)`; the new package fixes this and statistically tests it.
2. The earlier trainer expressed the paper budgets in epochs and stepped its
   scheduler once per epoch. The new trainer uses exactly 30K/20K optimizer
   steps and advances the scheduler once per update.
3. The earlier LIBERO adapter passed three explicit history frames into every
   RoboTTT timestep. The new adapter uses only the current cameras/state; past
   robot timesteps must be carried by fast weights.
4. The earlier default fast MLP expanded to `4d`, which the paper never states.
   The new least-assumptive default is `d -> d -> d`; changing it is explicitly
   an ablation.
5. The earlier code lived beside the legacy policy. It now lives solely in
   `policy/RoboTTT`; `policy/DiT_TTT` contains only the archived legacy method.

## Publicly under-specified reconstruction choices

- Fast-MLP hidden width and optimizer beta values are not reported.
- The paper gives RoPE theta but not whether token positions are shared within
  a robot timestep or assigned after flattening its token group.
- It does not state how the one recurrent `W_t` is committed across the
  multiple flow-denoising evaluations used to generate a single action chunk.
  We start each denoising evaluation from the same `W_(t-1)` and commit only
  the final candidate, so one environment decision advances history once.
- The public repository uses an encoded LIBERO adapter, not the unpublished
  Eagle VLM or NVIDIA's robot/human pretraining data mixture.

These choices are recorded in checkpoints. A paper-faithful claim must say
“public-paper reconstruction”; an exact NVIDIA reproduction is currently not
verifiable without the authors' source and model artifacts.

## Result boundary

All archived LIBERO DiT-TTT results came from `policy/DiT_TTT/policy.py`.
They are legacy evidence and are not results of `policy/RoboTTT`. The new
package requires fresh, matched `gate0`, frozen and online closed-loop runs.
