# Standard Diffusion Policy baseline

DP-TTT wraps a standard image-conditioned Diffusion Policy with the following
interface:

- `obs_encoder(normalized_observations) -> frame_features`;
- `model(noisy_actions, timesteps, global_cond=condition) -> prediction`;
- `normalizer` for observations and actions;
- `horizon`, `n_obs_steps`, `n_action_steps`, `action_dim`;
- `conditional_sample(...)` and `predict_action(...)`.

The validated baseline used two independent ImageNet-pretrained ResNet-18
encoders, a conditional 1-D UNet, DDPM epsilon prediction with 100 denoising
steps, EMA weights, three observation frames, action horizon 8 and execution
chunk 6.

Upstream DP code is not vendored here. Install or mount
[`real-stanford/diffusion_policy`](https://github.com/real-stanford/diffusion_policy)
and add it to `PYTHONPATH`. This keeps this repository focused on the TTT
algorithm and avoids silently maintaining a fork of an external baseline.

