# PI0.5-TTT

`pi05_ttt` is a separate policy that ports the repository's PI0-TTT engineering path to the native PI0.5
action expert. It keeps PI0.5's AdaRMS conditioning, tokenizer, processors, checkpoint mapping, and action
projections. It does not import the legacy `pi0_ttt` policy or `policy/RoboTTT`.

The TTT mechanism follows the public [NVIDIA RoboTTT method](https://research.nvidia.com/labs/gear/robottt/)
where it maps cleanly to PI0.5: a two-layer GeLU fast MLP performs K-to-V gradient updates, then Q readout;
the residual is inserted after attention and before the action-expert MLP; fast weights carry over one
selected sequence while gradients are detached only at TBPTT boundaries; and every frame samples its flow
noise and time independently.

This is a RoboTTT-style PI0.5 engineering adaptation, not an architectural reproduction of NVIDIA's GR00T
model. The PI0.5 port sends only the native action-expert suffix through TTT and does **not** add the paper's
16 learned register tokens. Its default last-four-layer placement is likewise the smaller local PI0-TTT
engineering choice, not a claim that PI0.5 is identical to the NVIDIA architecture.

## LIBERO-Long two-stage run

The formal recipe uses all 379 LIBERO-Long trajectories, four data-parallel GPUs, per-device batch 1,
and maximum context 256. Windows advance by 256 frames. A short episode is kept whole, and the final
non-divisible tail is a shorter sequence, so no episode frame is discarded. Every sampled window is an
independent selected sequence that starts from the learned fast-weight initialization; state carries only
through its internal TBPTT segments, and one slow optimizer step follows the global sequence minibatch.

Both stages instantiate the large PI0.5 transformer weights in bfloat16. TTT fast weights and the
action/time projections remain float32. This is required for the second-stage action-head optimizer to
fit on 32 GiB cards. Stage 1 backpropagates through 8-frame segments; the larger stage-2 action-head
graph uses 4-frame segments. In both cases the fast state is detached only at segment boundaries and
continues through the full sequence, so this does not change the 256-frame context or four-GPU recipe.

Stage 1 runs 5,000 optimizer steps with the PI0.5 backbone frozen. All TTT slow parameters, including the
learned residual gate initialized to `0.001`, train. Stage 2 loads that checkpoint, keeps the gate and other
TTT parameters trainable, and additionally unfreezes the PI0.5 action head (Gemma action expert plus
action/time projections) for another 5,000 steps. The PaliGemma VLM remains frozen.

The default `n_action_steps=10` is an execution cadence, not ten TTT updates: one action-chunk prediction
advances the persistent fast state once, then the policy executes 10 queued actions before the next
prediction and TTT update. Set `N_ACTION_STEPS=K` with `1 <= K <= 50` to change this cadence.

```bash
cd /workspace/lerobot-pi0-ttt
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
  examples/pi05_ttt/train_libero_long_two_stage.sh plan

CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 EXECUTE=1 \
  examples/pi05_ttt/train_libero_long_two_stage.sh run
```

The default source checkpoint is `/workspace/artifacts/models/pi05_libero_finetuned`. The two output
directories are deliberately separate because the second stage starts a new optimizer and learning-rate
schedule. Their names contain `v2_gate001` so the corrected learned-`0.001` recipe cannot silently reuse
the old fixed-`0.05` output directories. A `PHASE1_CHECKPOINT_OVERRIDE` must also declare
`type=pi05_ttt`, `ttt_training_stage=ttt_only`, and `ttt_effective_gate_init=0.001`; the launcher rejects
an old incompatible Stage-1 checkpoint before Stage 2.

`eval_libero_long_ddp4.sh` defaults to the corrected Stage-2 step-5000 checkpoint and validates its policy
type, `action_head` stage, and `0.001` gate initialization before launching any GPU shard. Its
`N_ACTION_STEPS` environment variable accepts `1..50` and has the same one-update-then-K-queued-actions
meaning described above.
