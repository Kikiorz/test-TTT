# PI0.5-TTT

`pi05_ttt` is a separate policy that ports the repository's validated PI0-TTT sequence-training path to
the native PI0.5 action expert. It keeps PI0.5's AdaRMS conditioning, tokenizer, processors, checkpoint
mapping, and action projections. It does not import the legacy `pi0_ttt` policy or `policy/RoboTTT`.

The TTT mechanism follows the public NVIDIA RoboTTT method where it maps cleanly to PI0.5: a two-layer
GeLU fast MLP performs K-to-V gradient updates, then Q readout; the residual is inserted after attention
and before the action expert MLP; fast weights carry over the full sequence while gradients are detached
only at TBPTT boundaries; and every frame samples its flow noise and time independently. This is a
RoboTTT-style PI0.5 action-expert adaptation, not an architectural reproduction of GR00T N1.7: PI0.5 does
not expose GR00T's 16 learned register tokens or the same DiT action head.

## LIBERO-Long two-stage run

The formal recipe uses all 379 LIBERO-Long trajectories, four data-parallel GPUs, per-device batch 1,
and maximum context 256. Windows advance by 256 frames. A short episode is kept whole, and the final
non-divisible tail is a shorter sequence, so no episode frame is discarded.

Both stages instantiate the large PI0.5 transformer weights in bfloat16. TTT fast weights and the
action/time projections remain float32. This is required for the second-stage action-head optimizer to
fit on 32 GiB cards; it does not change the 256-frame context or the four-process data-parallel recipe.

Stage 1 runs 5,000 optimizer steps with the PI0.5 backbone frozen. The effective gate is fixed at exactly
`0.05` (`raw gate = atanh(0.05)`), while the remaining TTT parameters train. Stage 2 loads that checkpoint,
unfreezes the gate and PI0.5 action head (Gemma action expert plus action/time projections), and trains
those parameters together with TTT for another 5,000 steps. The PaliGemma VLM remains frozen.

```bash
cd /workspace/lerobot-pi0-ttt
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
  examples/pi05_ttt/train_libero_long_two_stage.sh
```

The default source checkpoint is `/workspace/artifacts/models/pi05_libero_finetuned`. The two output
directories are deliberately separate because the second stage starts a new optimizer and learning-rate
schedule, matching the pretrain/post-train boundary in RoboTTT rather than pretending it is one unchanged
optimization phase.
