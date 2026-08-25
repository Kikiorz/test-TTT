# RoboTTT public-paper mapping

NVIDIA has not released the RoboTTT source, GR00T N1.7/Eagle checkpoint, or
training mixture. “Exact” below means a disclosed action-head requirement is
implemented literally; this package remains a public-paper reconstruction.

| Public requirement | Implementation | Status |
|---|---|---|
| One TTT layer in each of 16 DiT layers | Strict mode requires 16 DiT blocks and creates 16 fast states. | exact at action-head level |
| Original DiT 538M; each TTT layer roughly 10M; total 690M | Strict mode checks the reported DiT and per-layer TTT parameter scale. | reported-scale check |
| TTT after attention and before FFN | Each official GR00T transformer block is split at that boundary. | exact |
| Stream is registers, state, noisy action | TTT input is `[R_t, q_t, A_t^tau]`; VL tokens stay in cross-attention. | exact |
| 16 learned registers | Strict mode rejects any other register count. | exact |
| K-to-V MSE update then Q read | `RoboTTTKVBLayer` uses differentiable inner SGD before readout. | exact |
| Two-layer GeLU fast MLP | Functional `Linear -> GeLU -> Linear`. | exact except unpublished width |
| Base inner rate 0.1 with learned multiplier | Positive learned multiplier scales 0.1. | exact intent; parameterization chosen locally |
| RoPE theta 10000 | Applied to Q/K over recurrent tokens. | exact theta; layout under-specified |
| Per-channel `tanh(alpha)`, alpha 0.001 | Independent vector gate in every layer. | exact |
| Meta-gradient from outer flow loss | Outer MSE differentiates through the inner update. | exact |
| `tau=0.999(1-u)`, `u~Beta(1.5,1)` | Independently sampled per action chunk. | exact |
| Fast weights persist over trajectory | Reset is explicit; TBPTT only detaches graph history. | exact |
| 30K sequence pretrain then 20K full post-train | Optimizer-step budgets and stage trainability are explicit. | exact disclosed budgets |
| Pretrain 16 GPUs, batch/device 4 through 4K and 1 above; post-train 8 GPUs, batch/device 1 | Trainer implements the per-device schedule and supports separate resumable stages for the 16/8-GPU split. | exact disclosed schedule when launched at reported world sizes |
| WSD 2e-5 then cosine 5e-5; AdamW wd 1e-5 | Schedulers advance per optimizer update. | exact disclosed values |

The package-local `backbone.py` is a LIBERO adapter using ResNet18 plus the
public Isaac-GR00T DiT module. It is not the unpublished Eagle VLM. Its purpose
is to make this directory runnable and compatible with earlier benchmark
checkpoints without importing any legacy policy package.

Sources: [RoboTTT paper](https://arxiv.org/html/2607.15275) and
[NVIDIA project page](https://research.nvidia.com/labs/gear/robottt/).

## Explicit reconstruction choices

- Fast-MLP hidden width and AdamW beta values are not reported.
- QKV bias convention and the learned-rate multiplier parameterization are not reported.
- WSD warmup/stable/decay fractions and intermediate context-length schedule are not reported.
- RoPE token-position layout is not reported.
- The paper does not specify which candidate recurrent state is committed
  across multiple flow-denoising calls for one environment decision. This
  implementation starts all denoising evaluations at the same `W_(t-1)` and
  commits only the final candidate, producing exactly one recurrent transition
  per environment decision.
- The increasing 128-to-8192 context schedule is a documented reconstruction;
  the paper discloses the 8K endpoint, not the intermediate schedule.
