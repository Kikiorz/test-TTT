from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from DP_TTT.fast_memory import CausalFastMLPMemory, FastState
from DiT.model import LiberoGR00TDiT


LayerFastStates = Tuple[FastState, ...]


class DiTTTPolicy(nn.Module):
    """One causal observation-written fast memory after every official DiT block.

    A decision reads W_(t-1), constructs one residual per DiT layer, then writes
    the current observation exactly once to obtain W_t. The residuals are reused
    by all flow-integration iterations, so denoising never performs extra writes.
    """

    def __init__(
        self,
        base_policy: LiberoGR00TDiT,
        memory_dim: int = 64,
        fast_hidden_dim: int = 64,
        init_inner_lr: float = 0.05,
        init_gate: float = 0.1,
    ):
        super().__init__()
        self.base_policy = base_policy
        hidden = base_policy.config.hidden_dim
        self.memories = nn.ModuleList(
            CausalFastMLPMemory(
                obs_dim=hidden,
                condition_dim=hidden,
                memory_dim=memory_dim,
                fast_hidden_dim=fast_hidden_dim,
                init_inner_lr=init_inner_lr,
                init_gate=init_gate,
            )
            for _ in range(base_policy.config.num_layers)
        )
        self._deployment_state: Optional[LayerFastStates] = None
        self._deployment_mode = "online"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_ttt_state(self) -> None:
        self._deployment_state = None

    def set_fixed_gate(self, value: Optional[float]) -> None:
        for memory in self.memories:
            memory.set_fixed_gate(value)

    def set_deployment_mode(self, mode: str) -> None:
        if mode not in {"gate0", "frozen", "online"}:
            raise ValueError(f"unknown deployment mode: {mode}")
        self._deployment_mode = mode
        self.set_fixed_gate(0.0 if mode == "gate0" else None)
        self.reset_ttt_state()

    def gate_values(self) -> torch.Tensor:
        return torch.stack([memory.gate() for memory in self.memories])

    def inner_lr_values(self) -> torch.Tensor:
        return torch.stack([memory.positive_inner_lr() for memory in self.memories])

    def configure_stage(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.base_policy.eval()
        if stage == "fixed_gate_ttt":
            self.set_fixed_gate(0.5)
            for memory in self.memories:
                memory.requires_grad_(True)
                memory.gate_raw.requires_grad_(False)
        elif stage == "joint":
            self.set_fixed_gate(None)
            for memory in self.memories:
                memory.requires_grad_(True)
            self.base_policy.action_encoder.requires_grad_(True)
            self.base_policy.action_decoder.requires_grad_(True)
            self.base_policy.dit.requires_grad_(True)
            self.base_policy.action_encoder.train()
            self.base_policy.action_decoder.train()
            self.base_policy.dit.train()
        elif stage == "ttt_calibration":
            self.set_fixed_gate(None)
            for memory in self.memories:
                memory.requires_grad_(True)
        else:
            raise ValueError(stage)

    def prepare_layer_residuals(
        self,
        context: torch.Tensor,
        state: Optional[LayerFastStates],
        *,
        create_graph: bool,
    ) -> tuple[tuple[torch.Tensor, ...], LayerFastStates, torch.Tensor]:
        if context.ndim != 3 or context.shape[1] != self.base_policy.config.obs_steps * 3:
            raise ValueError(f"unexpected context shape {tuple(context.shape)}")
        newest_observation = context[:, -3:].mean(dim=1)
        global_summary = context.mean(dim=1)
        if state is not None and len(state) != len(self.memories):
            raise ValueError("fast-state layer count mismatch")
        residuals, next_states, inner_losses = [], [], []
        for index, memory in enumerate(self.memories):
            layer_state = None if state is None else state[index]
            adapted, next_state, inner_loss = memory(
                newest_observation,
                global_summary,
                layer_state,
                create_graph=create_graph,
            )
            residuals.append(adapted - global_summary)
            next_states.append(next_state)
            inner_losses.append(inner_loss)
        return tuple(residuals), tuple(next_states), torch.stack(inner_losses, dim=1)

    def _official_dit_with_residuals(
        self,
        action_tokens: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        layer_residuals: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        dit = self.base_policy.dit
        hidden = action_tokens.contiguous()
        context = context.contiguous()
        temb = dit.timestep_encoder(timestep)
        for index, block in enumerate(dit.transformer_blocks):
            if index % 2 == 1 and dit.config.interleave_self_attention:
                hidden = block(
                    hidden,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden = block(
                    hidden,
                    attention_mask=None,
                    encoder_hidden_states=context,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            hidden = hidden + layer_residuals[index][:, None, :]
        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden = dit.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        return dit.proj_out_2(hidden)

    def predict_velocity_with_residuals(
        self,
        normalized_actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *layer_residuals: torch.Tensor,
    ) -> torch.Tensor:
        action_tokens = self.base_policy.action_encoder(normalized_actions)
        action_tokens = action_tokens + self.base_policy.action_position_embedding.unsqueeze(0)
        hidden = self._official_dit_with_residuals(
            action_tokens, context, timestep, tuple(layer_residuals)
        )
        return self.base_policy.action_decoder(hidden)

    def flow_matching_loss_from_context(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        state: Optional[LayerFastStates],
        *,
        create_graph: bool = True,
        activation_checkpointing: bool = True,
    ) -> tuple[torch.Tensor, LayerFastStates, dict[str, float]]:
        residuals, next_state, inner_losses = self.prepare_layer_residuals(
            context, state, create_graph=create_graph
        )
        base = self.base_policy
        normalized_actions = (actions - base.action_mean) / base.action_std
        noise = torch.randn_like(normalized_actions)
        beta = torch.distributions.Beta(
            torch.tensor(1.5, device=actions.device), torch.tensor(1.0, device=actions.device)
        )
        t = beta.sample((actions.shape[0],)).to(actions.dtype).mul_(0.999)
        t_broadcast = t[:, None, None]
        noisy_actions = (1.0 - t_broadcast) * noise + t_broadcast * normalized_actions
        target_velocity = normalized_actions - noise
        timestep = (t * base.config.timestep_buckets).long()
        if activation_checkpointing and torch.is_grad_enabled():
            predicted_velocity = checkpoint(
                self.predict_velocity_with_residuals,
                noisy_actions,
                timestep,
                context,
                *residuals,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            predicted_velocity = self.predict_velocity_with_residuals(
                noisy_actions, timestep, context, *residuals
            )
        per_element = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        mask = action_mask.unsqueeze(-1).to(per_element.dtype)
        loss = (per_element * mask).sum() / (mask.sum() * actions.shape[-1]).clamp_min(1.0)
        gates = self.gate_values().detach()
        metrics = {
            "flow_loss": float(loss.detach()),
            "inner_loss": float(inner_losses.mean()),
            "gate_mean": float(gates.mean()),
            "gate_min": float(gates.min()),
            "gate_max": float(gates.max()),
            "inner_lr_mean": float(self.inner_lr_values().detach().mean()),
        }
        return loss, next_state, metrics

    @torch.no_grad()
    def sample_action(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        context = self.base_policy.encode_context(observations)
        state_in = None if self._deployment_mode == "frozen" else self._deployment_state
        residuals, next_state, _ = self.prepare_layer_residuals(
            context, state_in, create_graph=False
        )
        if self._deployment_mode != "frozen":
            self._deployment_state = next_state
        batch = context.shape[0]
        actions = torch.randn(
            batch,
            self.base_policy.config.action_horizon,
            self.base_policy.config.action_dim,
            device=context.device,
            dtype=context.dtype,
        )
        dt = 1.0 / self.base_policy.config.inference_steps
        for index in range(self.base_policy.config.inference_steps):
            continuous_t = index / float(self.base_policy.config.inference_steps)
            timestep = torch.full(
                (batch,),
                int(continuous_t * self.base_policy.config.timestep_buckets),
                device=context.device,
                dtype=torch.long,
            )
            velocity = self.predict_velocity_with_residuals(
                actions, timestep, context, *residuals
            )
            actions = actions + dt * velocity
        return actions * self.base_policy.action_std + self.base_policy.action_mean

