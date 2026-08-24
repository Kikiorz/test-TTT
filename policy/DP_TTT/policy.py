from __future__ import annotations

from typing import Dict, Optional

import torch
from einops import reduce
from torch import nn
from torch.nn import functional as F

from diffusion_policy.common.pytorch_util import dict_apply

from .fast_memory import CausalFastMLPMemory, FastState


class DPTTTPolicy(nn.Module):
    """Checkpoint-compatible TTT wrapper around the LIBERO Diffusion Policy."""

    def __init__(
        self,
        base_policy: nn.Module,
        memory_dim: int = 256,
        fast_hidden_dim: int = 256,
        init_inner_lr: float = 0.05,
        init_gate: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        obs_dim = int(base_policy.obs_feature_dim)
        condition_dim = obs_dim * int(base_policy.n_obs_steps)
        self.ttt = CausalFastMLPMemory(
            obs_dim=obs_dim,
            condition_dim=condition_dim,
            memory_dim=memory_dim,
            fast_hidden_dim=fast_hidden_dim,
            init_inner_lr=init_inner_lr,
            init_gate=init_gate,
        )
        self._deployment_state: Optional[FastState] = None
        self._deployment_mode = "online"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def n_obs_steps(self) -> int:
        return int(self.base_policy.n_obs_steps)

    @property
    def n_action_steps(self) -> int:
        return int(self.base_policy.n_action_steps)

    def reset_ttt_state(self) -> None:
        self._deployment_state = None

    def set_deployment_mode(self, mode: str) -> None:
        if mode not in {"online", "frozen", "gate0"}:
            raise ValueError(f"unknown deployment mode: {mode}")
        self._deployment_mode = mode
        self.set_fixed_gate(0.0 if mode == "gate0" else None)
        self.reset_ttt_state()

    def set_fixed_gate(self, value: Optional[float]) -> None:
        self.ttt.set_fixed_gate(value)

    def configure_stage(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == "forced_gate_ttt":
            self.set_fixed_gate(0.1)
            for parameter in self.ttt.parameters():
                parameter.requires_grad_(True)
            self.ttt.gate_raw.requires_grad_(False)
        elif stage == "joint":
            self.set_fixed_gate(None)
            for parameter in self.ttt.parameters():
                parameter.requires_grad_(True)
            # Preserve the successful visual encoder; jointly tune the action U-Net.
            for parameter in self.base_policy.model.parameters():
                parameter.requires_grad_(True)
        elif stage == "ttt_gate_only":
            self.set_fixed_gate(None)
            for parameter in self.ttt.parameters():
                parameter.requires_grad_(True)
        else:
            raise ValueError(f"unknown stage: {stage}")
        self.base_policy.obs_encoder.requires_grad_(False)
        self.base_policy.obs_encoder.eval()

    def condition_from_frame_features(
        self,
        frame_features: torch.Tensor,
        state: Optional[FastState],
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, FastState, torch.Tensor]:
        expected = int(self.base_policy.n_obs_steps)
        if frame_features.ndim != 3 or frame_features.shape[1] != expected:
            raise ValueError(f"expected [B,{expected},Do], got {tuple(frame_features.shape)}")
        global_condition = frame_features.flatten(1)
        return self.ttt(
            frame_features[:, -1],
            global_condition,
            state,
            create_graph=create_graph,
        )

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
        prediction = base.model(
            noisy_trajectory,
            timesteps,
            local_cond=None,
            global_cond=global_condition,
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
        return reduce(loss, "b ... -> b", "mean").mean()

    def compute_loss_from_cached_features(
        self,
        frame_features: torch.Tensor,
        normalized_actions: torch.Tensor,
        state: Optional[FastState],
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, FastState, Dict[str, float]]:
        condition, next_state, inner_loss = self.condition_from_frame_features(
            frame_features,
            state,
            create_graph=create_graph,
        )
        diffusion_loss = self._diffusion_loss(normalized_actions, condition)
        metrics = {
            "diffusion_loss": float(diffusion_loss.detach()),
            "inner_loss": float(inner_loss.mean()),
            "gate": float(self.ttt.gate().detach()),
            "inner_lr": float(self.ttt.positive_inner_lr().detach()),
        }
        return diffusion_loss, next_state, metrics

    def _encode_observation_frames(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        base = self.base_policy
        normalized_obs = base.normalizer.normalize(obs_dict)
        value = next(iter(normalized_obs.values()))
        batch_size = value.shape[0]
        steps = int(base.n_obs_steps)
        this_obs = dict_apply(
            normalized_obs,
            lambda x: x[:, :steps].reshape(-1, *x.shape[2:]),
        )
        frame_features = base.obs_encoder(this_obs)
        return frame_features.reshape(batch_size, steps, -1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        base = self.base_policy
        frame_features = self._encode_observation_frames(obs_dict)
        state_in = None if self._deployment_mode == "frozen" else self._deployment_state
        condition, next_state, _ = self.condition_from_frame_features(
            frame_features,
            state_in,
            create_graph=False,
        )
        if self._deployment_mode != "frozen":
            self._deployment_state = next_state
        batch_size = frame_features.shape[0]
        horizon = int(base.horizon)
        action_dim = int(base.action_dim)
        condition_data = torch.zeros(
            (batch_size, horizon, action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        normalized_sample = base.conditional_sample(
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=condition,
            **base.kwargs,
        )
        action_prediction = base.normalizer["action"].unnormalize(normalized_sample[..., :action_dim])
        start = int(base.n_obs_steps) - 1
        end = start + int(base.n_action_steps)
        return {
            "action": action_prediction[:, start:end],
            "action_pred": action_prediction,
        }
