# test-TTT

Causal test-time training (TTT) modules for robot action policies, with two
controlled policy families:

- a standard image-conditioned Diffusion Policy (DP) dependency;
- a flow-matching action policy built around NVIDIA Isaac-GR00T's official
  `gr00t.model.modules.dit.DiT`;
- DP-TTT, which adds one observation-written fast memory to DP conditioning;
- a separate public-paper RoboTTT reconstruction with one TTT-KVB layer after
  the attention operation of every DiT block.

This repository contains **algorithm code only**. It intentionally excludes
datasets, preprocessing products, checkpoints, videos, simulator assets,
evaluation logs, credentials and machine-specific launch scripts.

## Layout

```text
policy/
├── DP/          # dependency contract for the standard DP baseline
├── DP_TTT/      # causal fast memory, DP wrapper and staged trainer
├── DiT/         # GR00T-DiT flow-matching action policy
├── DiT_TTT/     # legacy experimental policy; archived-result boundary
└── RoboTTT/     # clean public-paper reconstruction, trainer and invariants
configs/
├── libero_long.yaml
└── robottt_paper.yaml
docs/
├── ALGORITHM.md
├── EXPERIMENT_AUDIT.md
└── PAPER_MAPPING.md
```

For standalone RoboTTT use, install the repository or add `policy/` to
`PYTHONPATH`, then import `RoboTTT`. It is the only installable package and
contains its own backbone adapter, config, trainer, paper mapping, and
isolation tests; it does not import any sibling policy.

The other policy directories are retained as source-only experiment history
and may be removed without changing RoboTTT. `policy/DiT_TTT` produced the
archived LIBERO result, which must not be attributed to RoboTTT.

## Causal deployment contract

For DP-TTT and the legacy DiT-TTT path, at environment decision `t`:

1. read `W_(t-1)` and construct the action-policy residual;
2. sample the complete action chunk using that fixed residual;
3. write the current observation feature into `W_t` exactly once;
4. retain `W_t` for the next decision and reset it only at episode boundaries.

The write target is observation-derived. Expert actions supervise only the
outer diffusion/flow-matching objective; actions, rewards, success predicates
and future observations never enter the online write. Denoising or flow
integration reuses one residual and therefore cannot perform repeated writes.

The paper RoboTTT path instead runs TTT on the current register, state and noisy
action tokens after each layer's attention operation. All denoising evaluations
start from the same `W_(t-1)`; only the final candidate `W_t` is committed, so
recurrent time advances once per environment decision.

## Training schedule

The archived legacy schedule uses full-episode fast-state lifetime with TBPTT=64:

1. 100 epochs: freeze the base policy, force `gate=0.5`, train TTT only;
2. 20 epochs: release the gate and jointly tune TTT plus the action model at a
   lower base-policy learning rate while keeping visual encoders frozen;
3. 10 epochs: freeze the base policy again and calibrate TTT plus its gate.

Required closed-loop controls are `gate0`, `frozen` and `online`. `gate0` must
be numerically identical to the family baseline before interpreting any TTT
result.

RoboTTT itself uses a different recipe: sequence-model-only pretraining followed
by all-parameter post-training, independently sampled flow noise per action
chunk, and TBPTT with numerical fast weights carried across segment boundaries.
The new trainer defaults to the paper's 30K pretraining and 20K post-training
optimizer steps and its disclosed per-device batch schedule. Strict action-head
mode also checks 16 DiT layers, 16 registers, and the reported parameter scale;
any smaller bench requires an explicit approximation flag and label.

## External dependencies

- Diffusion Policy: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- NVIDIA Isaac-GR00T commit:
  `376ba890cff8c9de64d71d982772a9c36185fdd7`
- Official GR00T DiT source SHA-256:
  `d4da27be51d9ebe1c923dedd3ae80fe2754c97f96500e82c26fe19fec0062c13`

The full GR00T model is not used: only its standard DiT action-head module is
the architectural dependency. See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for
the historical interfaces and [`docs/EXPERIMENT_AUDIT.md`](docs/EXPERIMENT_AUDIT.md)
for the current evidence and its limitations. The authoritative paper-to-code
mapping and reconstruction choices live beside the implementation in
[`policy/RoboTTT/PAPER_MAPPING.md`](policy/RoboTTT/PAPER_MAPPING.md).
