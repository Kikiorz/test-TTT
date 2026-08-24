# test-TTT

Causal test-time training (TTT) modules for robot action policies, with two
controlled policy families:

- a standard image-conditioned Diffusion Policy (DP) dependency;
- a flow-matching action policy built around NVIDIA Isaac-GR00T's official
  `gr00t.model.modules.dit.DiT`;
- DP-TTT, which adds one observation-written fast memory to DP conditioning;
- DiT-TTT, which adds one observation-written fast memory after every DiT
  block.

This repository contains **algorithm code only**. It intentionally excludes
datasets, preprocessing products, checkpoints, videos, simulator assets,
evaluation logs, credentials and machine-specific launch scripts.

## Layout

```text
policy/
├── DP/          # dependency contract for the standard DP baseline
├── DP_TTT/      # causal fast memory, DP wrapper and staged trainer
├── DiT/         # GR00T-DiT flow-matching action policy
└── DiT_TTT/     # per-layer TTT wrapper, staged trainer and invariants
configs/
└── libero_long.yaml
docs/
├── ALGORITHM.md
└── EXPERIMENT_AUDIT.md
```

The modules keep the import layout used by the validated experiment. Add
`policy/` to `PYTHONPATH` before importing `DP_TTT`, `DiT` or `DiT_TTT`.

## Causal deployment contract

At environment decision `t`:

1. read `W_(t-1)` and construct the action-policy residual;
2. sample the complete action chunk using that fixed residual;
3. write the current observation feature into `W_t` exactly once;
4. retain `W_t` for the next decision and reset it only at episode boundaries.

The write target is observation-derived. Expert actions supervise only the
outer diffusion/flow-matching objective; actions, rewards, success predicates
and future observations never enter the online write. Denoising or flow
integration reuses one residual and therefore cannot perform repeated writes.

## Training schedule

The validated schedule uses full-episode fast-state lifetime with TBPTT=64:

1. 100 epochs: freeze the base policy, force `gate=0.5`, train TTT only;
2. 20 epochs: release the gate and jointly tune TTT plus the action model at a
   lower base-policy learning rate while keeping visual encoders frozen;
3. 10 epochs: freeze the base policy again and calibrate TTT plus its gate.

Required closed-loop controls are `gate0`, `frozen` and `online`. `gate0` must
be numerically identical to the family baseline before interpreting any TTT
result.

## External dependencies

- Diffusion Policy: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- NVIDIA Isaac-GR00T commit:
  `376ba890cff8c9de64d71d982772a9c36185fdd7`
- Official GR00T DiT source SHA-256:
  `d4da27be51d9ebe1c923dedd3ae80fe2754c97f96500e82c26fe19fec0062c13`

The full GR00T model is not used: only its standard DiT action-head module is
the architectural dependency. See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for
the exact interfaces and [`docs/EXPERIMENT_AUDIT.md`](docs/EXPERIMENT_AUDIT.md)
for the current evidence and its limitations.

