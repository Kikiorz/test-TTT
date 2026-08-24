# Experiment audit

## Evaluated task

The current closed-loop gate covers one task from the official `libero_10`
suite, which LIBERO documentation also calls LIBERO-LONG:

> put the black bowl in the bottom drawer of the cabinet and close it

The LeRobot dataset labels it with task index 8. Under LIBERO's default task
order, the simulator exposes the same language task as task ID 3. Evaluation
asserted the exact language before launching any episode.

This is **not** a complete LIBERO-LONG score. A full suite score requires all
10 tasks and the standard number of rollouts and random seeds.

## Controlled result on this task

The table below belongs to the archived implementations. In particular, its
DiT-TTT rows used `policy/DiT_TTT/policy.py`, not the later paper reconstruction
in `robottt_policy.py`. The new architecture has no closed-loop result yet.

| Family and mode | Success on official init states 0-19 |
|---|---:|
| DP baseline | 20/20 |
| DP-TTT gate0 | 20/20 |
| DP-TTT frozen | 18/20 |
| DP-TTT Stage-1 online | 20/20 |
| DP-TTT final online | 20/20 |
| GR00T-DiT baseline | 19/20 |
| DiT-TTT gate0 | 19/20 |
| DiT-TTT frozen | 17/20 |
| DiT-TTT Stage-1 online | 17/20 |
| DiT-TTT final online | 20/20 |

## Leakage audit

No direct answer leakage was found in the runtime path:

- policy input is only two camera streams plus 8-D proprioception from the
  current and two preceding frames;
- inference never reads demonstration actions, future frames or expert labels;
- `env.check_success()` is used only to stop and score a rollout;
- the archived TTT online writes are observation-only and happen once per decision;
- DP and DiT state/action normalization uses training episodes only;
- 35 demonstrations are divided by episode into 31 training and 4 validation
  episodes using seed 42.

However, the present result is not a strong generalization test:

- a separate single-task model is trained, so the task identity is implicit;
- the same BDDL task and official initial-state distribution are used for
  demonstration collection and evaluation;
- the code does not assert that the demonstration initial configurations are
  disjoint from official evaluation initial states 0-19;
- DP was evaluated at the final 100-epoch checkpoint rather than selected by a
  pre-registered validation-only rule;
- only 20 rollouts from one training seed were evaluated.

Therefore DP's 20/20 should be described as an in-distribution, single-task
closed-loop sanity result—not as 100% on LIBERO-LONG and not as evidence of
long-horizon memory.

## External reference

The original LIBERO paper did not report a Diffusion Policy baseline. A later
widely used OpenVLA evaluation reports DP-from-scratch at **50.5 +/- 1.3%** on
the complete LIBERO-LONG suite, averaged over 3 seeds and 500 rollouts per seed
(10 tasks x 50 rollouts). That number is not directly comparable to this
single-task 20-rollout experiment.

Before making a paper claim, evaluate all 10 tasks with matched seeds, select
checkpoints without closed-loop test feedback, and add an explicit held-out
initial-configuration or perturbation split.
