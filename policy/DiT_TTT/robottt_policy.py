from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from DiT.model import LiberoGR00TDiT
from DiT_TTT.robottt_layer import FastState, RoboTTTKVBLayer


LayerFastStates = Tuple[FastState, ...]


class PaperRoboTTTPolicy(nn.Module):
    """RoboTTT architecture reconstructed from Jiang et al. (2026).

    In every DiT block, current-step register, proprioception and noisy-action
    tokens first pass through the original attention operation. That layer's
    attention outputs are processed by its own recurrent TTT-KVB MLP and fused
    through a channel-wise tanh gate before the FFN. Vision tokens stay on the
    cross-attention path and never enter TTT directly.

    NVIDIA has not released RoboTTT source. This is a paper reconstruction, not
    a claim of byte-for-byte equivalence to their internal implementation.
    """

    def __init__(
        self,
        base_policy: LiberoGR00TDiT,
        *,
        num_register_tokens: int = 16,
        fast_hidden_dim: Optional[int] = None,
        base_inner_lr: float = 0.1,
        init_gate_alpha: float = 0.001,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        dim = base_policy.config.hidden_dim
        self.num_register_tokens = int(num_register_tokens)
        self.register_tokens = nn.Parameter(torch.empty(self.num_register_tokens, dim))
        nn.init.normal_(self.register_tokens, std=0.02)
        self.ttt_layers = nn.ModuleList(
            RoboTTTKVBLayer(
                dim,
                fast_hidden_dim,
                base_inner_lr=base_inner_lr,
                init_gate_alpha=init_gate_alpha,
                rope_theta=rope_theta,
            )
            for _ in range(base_policy.config.num_layers)
        )
        self._deployment_state: Optional[LayerFastStates] = None
        self._deployment_mode = "online"

    @property
    def token_prefix_length(self) -> int:
        return self.num_register_tokens + 1

    def reset_ttt_state(self) -> None:
        self._deployment_state = None

    def set_fixed_gate(self, value: Optional[float]) -> None:
        for layer in self.ttt_layers:
            layer.set_fixed_gate(value)

    def set_deployment_mode(self, mode: str) -> None:
        if mode not in {"gate0", "frozen", "online"}:
            raise ValueError(f"unknown deployment mode: {mode}")
        self._deployment_mode = mode
        self.set_fixed_gate(0.0 if mode == "gate0" else None)
        self.reset_ttt_state()

    def gate_values(self) -> torch.Tensor:
        return torch.stack([layer.gate() for layer in self.ttt_layers])

    def inner_lr_values(self) -> torch.Tensor:
        return torch.stack([layer.positive_inner_lr() for layer in self.ttt_layers])

    def configure_stage(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == "paper_sequence_pretrain":
            self.base_policy.eval()
            self.register_tokens.requires_grad_(True)
            self.ttt_layers.requires_grad_(True)
        elif stage == "paper_posttrain":
            self.train()
            for parameter in self.parameters():
                parameter.requires_grad_(True)
        else:
            raise ValueError(stage)

    def _split_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.base_policy
        expected = base.config.obs_steps * 3
        if context.ndim != 3 or context.shape[1] != expected:
            raise ValueError(f"expected [B,{expected},D], got {tuple(context.shape)}")
        frames = context.view(context.shape[0], base.config.obs_steps, 3, context.shape[-1])
        vision_tokens = frames[:, :, :2].flatten(1, 2)
        proprioception_token = frames[:, -1, 2:3]
        return vision_tokens, proprioception_token

    @staticmethod
    def _attention_then_residual(
        block: nn.Module,
        hidden: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb: torch.Tensor,
    ) -> torch.Tensor:
        if block.norm_type == "ada_norm":
            normalized = block.norm1(hidden, temb)
        else:
            normalized = block.norm1(hidden)
        if block.pos_embed is not None:
            normalized = block.pos_embed(normalized)
        attention_output = block.attn1(
            normalized,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
        )
        if block.final_dropout:
            attention_output = block.final_dropout(attention_output)
        return hidden + attention_output

    @staticmethod
    def _feed_forward(block: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + block.ff(block.norm3(hidden))

    def _dit_forward(
        self,
        normalized_actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        state: Optional[LayerFastStates],
        *,
        create_graph: bool,
        update_fast: bool,
        enable_ttt: bool = True,
    ) -> tuple[torch.Tensor, LayerFastStates, torch.Tensor]:
        base = self.base_policy
        dit = base.dit
        vision_tokens, state_token = self._split_context(context)
        action_tokens = base.action_encoder(normalized_actions)
        action_tokens = action_tokens + base.action_position_embedding.unsqueeze(0)
        registers = self.register_tokens.unsqueeze(0).expand(action_tokens.shape[0], -1, -1)
        hidden = torch.cat((registers, state_token, action_tokens), dim=1).contiguous()
        vision_tokens = vision_tokens.contiguous()
        temb = dit.timestep_encoder(timestep)

        if state is not None and len(state) != len(self.ttt_layers):
            raise ValueError("fast-state layer count mismatch")
        next_states = []
        inner_losses = []
        for index, (block, ttt_layer) in enumerate(zip(dit.transformer_blocks, self.ttt_layers)):
            use_self_attention = index % 2 == 1 and dit.config.interleave_self_attention
            attention_hidden = self._attention_then_residual(
                block,
                hidden,
                None if use_self_attention else vision_tokens,
                temb,
            )
            layer_state = None if state is None else state[index]
            if enable_ttt:
                ttt_output, layer_next_state, inner_loss = ttt_layer(
                    attention_hidden,
                    layer_state,
                    create_graph=create_graph,
                    update_fast=update_fast,
                )
                attention_hidden = attention_hidden + ttt_layer.gate()[None, None] * ttt_output
            else:
                if layer_state is None:
                    layer_state = ttt_layer.initial_state(hidden.shape[0], hidden.device)
                layer_next_state = layer_state
                inner_loss = hidden.new_zeros(hidden.shape[0])
            next_states.append(layer_next_state)
            inner_losses.append(inner_loss)
            hidden = self._feed_forward(block, attention_hidden)

        action_hidden = hidden[:, self.token_prefix_length :]
        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        action_hidden = dit.norm_out(action_hidden) * (1 + scale[:, None]) + shift[:, None]
        action_hidden = dit.proj_out_2(action_hidden)
        velocity = base.action_decoder(action_hidden)
        return velocity, tuple(next_states), torch.stack(inner_losses, dim=1)

    def flow_matching_loss_from_context(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        state: Optional[LayerFastStates],
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, LayerFastStates, dict[str, float]]:
        base = self.base_policy
        normalized_actions = (actions - base.action_mean) / base.action_std
        noise = torch.randn_like(normalized_actions)
        beta = torch.distributions.Beta(
            torch.tensor(1.5, device=actions.device),
            torch.tensor(1.0, device=actions.device),
        )
        # Sequence action forcing: each robot timestep samples its own tau/noise.
        tau = beta.sample((actions.shape[0],)).to(actions.dtype).mul_(0.999)
        noisy_actions = (1.0 - tau[:, None, None]) * noise + tau[:, None, None] * normalized_actions
        target_velocity = normalized_actions - noise
        timestep = (tau * base.config.timestep_buckets).long()
        predicted_velocity, next_state, inner_losses = self._dit_forward(
            noisy_actions,
            timestep,
            context,
            state,
            create_graph=create_graph,
            update_fast=True,
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
        base = self.base_policy
        context = base.encode_context(observations)
        batch = context.shape[0]
        actions = torch.randn(
            batch,
            base.config.action_horizon,
            base.config.action_dim,
            device=context.device,
            dtype=context.dtype,
        )
        state_in = self._deployment_state
        commit_state = state_in
        update_fast = self._deployment_mode == "online"
        dt = 1.0 / base.config.inference_steps
        for index in range(base.config.inference_steps):
            timestep = torch.full(
                (batch,),
                int(index / float(base.config.inference_steps) * base.config.timestep_buckets),
                device=context.device,
                dtype=torch.long,
            )
            # All denoising evaluations start from the same episode state. Only
            # the final candidate is committed, so one environment decision
            # advances recurrent time exactly once.
            velocity, candidate_state, _ = self._dit_forward(
                actions,
                timestep,
                context,
                state_in,
                create_graph=False,
                update_fast=update_fast,
            )
            actions = actions + dt * velocity
            commit_state = candidate_state
        if update_fast:
            self._deployment_state = commit_state
        return actions * base.action_std + base.action_mean
