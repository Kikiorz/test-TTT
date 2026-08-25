from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from DP_TTT.policy import DPTTTPolicy


class FullEpisodeDPTTTPolicy(DPTTTPolicy):
    """DP-TTT with activation-checkpointed U-Net calls inside each TBPTT segment."""

    def _diffusion_loss(self, normalized_actions: torch.Tensor, global_condition: torch.Tensor) -> torch.Tensor:
        base = self.base_policy
        trajectory = normalized_actions
        condition_mask = base.mask_generator(trajectory.shape)
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            base.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = base.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        def run_unet(noisy: torch.Tensor, steps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            return base.model(noisy, steps, local_cond=None, global_cond=condition)

        prediction = checkpoint(
            run_unet,
            noisy_trajectory,
            timesteps,
            global_condition,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        prediction_type = base.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"unsupported prediction type: {prediction_type}")
        loss = F.mse_loss(prediction, target, reduction="none")
        loss = loss * (~condition_mask).to(loss.dtype)
        return loss.flatten(1).mean(dim=1).mean()
