#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


@dataclass(frozen=True)
class TTTFastState:
    """Per-trajectory fast weights for one TTT MLP layer."""

    w1: Tensor
    b1: Tensor
    w2: Tensor
    b2: Tensor
    position: Tensor | None = None

    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.w1, self.b1, self.w2, self.b2

    def detach(self, requires_grad: bool = True) -> "TTTFastState":
        def detach_tensor(tensor: Tensor) -> Tensor:
            return tensor.detach().requires_grad_(requires_grad)

        position = None if self.position is None else self.position.detach()
        return TTTFastState(
            *(detach_tensor(tensor) for tensor in self.tensors()),
            position=position,
        )

    @property
    def batch_size(self) -> int:
        return self.w1.shape[0]


class TTTMLPLayer(nn.Module):
    """RoboTTT update-then-apply layer with a two-layer GeLU fast MLP."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        base_inner_lr: float = 0.1,
        effective_gate_init: float = 0.001,
        gate_trainable: bool = True,
        rope_theta: float = 10_000.0,
        second_order: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")
        if base_inner_lr <= 0:
            raise ValueError("base_inner_lr must be positive")
        if not 0 <= effective_gate_init < 1:
            raise ValueError("effective_gate_init must be in [0, 1)")
        if rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.base_inner_lr = base_inner_lr
        self.rope_theta = rope_theta
        self.second_order = second_order

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.fast_w1_init = nn.Parameter(torch.empty(hidden_dim, dim))
        self.fast_b1_init = nn.Parameter(torch.empty(hidden_dim))
        self.fast_w2_init = nn.Parameter(torch.empty(dim, hidden_dim))
        self.fast_b2_init = nn.Parameter(torch.empty(dim))

        self.log_inner_lr_multiplier = nn.Parameter(torch.zeros(()))
        raw_gate = math.atanh(effective_gate_init)
        self.gate = nn.Parameter(torch.full((dim,), raw_gate), requires_grad=gate_trainable)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(projection.weight)

        nn.init.kaiming_uniform_(self.fast_w1_init, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fast_w2_init, a=math.sqrt(5))

        bound1 = 1 / math.sqrt(self.dim)
        nn.init.uniform_(self.fast_b1_init, -bound1, bound1)
        bound2 = 1 / math.sqrt(self.hidden_dim)
        nn.init.uniform_(self.fast_b2_init, -bound2, bound2)

    @property
    def inner_lr(self) -> Tensor:
        return self.base_inner_lr * self.log_inner_lr_multiplier.exp()

    @property
    def effective_gate(self) -> Tensor:
        return torch.tanh(self.gate)

    def initial_state(self, batch_size: int) -> TTTFastState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        return TTTFastState(
            w1=self.fast_w1_init.unsqueeze(0).expand(batch_size, -1, -1),
            b1=self.fast_b1_init.unsqueeze(0).expand(batch_size, -1),
            w2=self.fast_w2_init.unsqueeze(0).expand(batch_size, -1, -1),
            b2=self.fast_b2_init.unsqueeze(0).expand(batch_size, -1),
            position=torch.full(
                (batch_size,),
                -1,
                dtype=torch.int64,
                device=self.fast_w1_init.device,
            ),
        )

    @staticmethod
    def _fast_mlp(inputs: Tensor, state: TTTFastState) -> Tensor:
        hidden = torch.einsum("bsd,bhd->bsh", inputs, state.w1)
        hidden = F.gelu(hidden + state.b1[:, None, :])
        output = torch.einsum("bsh,bdh->bsd", hidden, state.w2)
        return output + state.b2[:, None, :]

    def _update(
        self,
        keys: Tensor,
        values: Tensor,
        state: TTTFastState,
        create_graph: bool,
        token_mask: Tensor | None = None,
    ) -> TTTFastState:
        prediction = self._fast_mlp(keys, state)
        per_token_loss = F.mse_loss(prediction, values, reduction="none").mean(dim=-1)
        if token_mask is None:
            per_trajectory_loss = per_token_loss.mean(dim=1)
        else:
            if token_mask.shape != per_token_loss.shape:
                raise ValueError(
                    f"Expected token_mask with shape {tuple(per_token_loss.shape)}, "
                    f"got {tuple(token_mask.shape)}"
                )
            valid_tokens = token_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
            per_trajectory_loss = (per_token_loss * valid_tokens).sum(dim=1) / valid_tokens.sum(
                dim=1
            ).clamp_min(1)
        gradients = torch.autograd.grad(
            per_trajectory_loss.sum(),
            state.tensors(),
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        return TTTFastState(
            *(
                weight - self.inner_lr * gradient
                for weight, gradient in zip(state.tensors(), gradients, strict=True)
            ),
            position=state.position,
        )

    def _apply_rope(self, inputs: Tensor, positions: Tensor) -> Tensor:
        rotary_dim = self.dim - self.dim % 2
        if rotary_dim == 0:
            return inputs

        frequencies = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, rotary_dim, 2, device=inputs.device, dtype=torch.float32) / rotary_dim)
        )
        angles = positions.to(torch.float32).unsqueeze(-1) * frequencies
        cos = angles.cos().to(dtype=inputs.dtype)
        sin = angles.sin().to(dtype=inputs.dtype)
        rotary_inputs = inputs[..., :rotary_dim].reshape(*inputs.shape[:-1], rotary_dim // 2, 2)
        first, second = rotary_inputs.unbind(dim=-1)
        rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1).flatten(
            start_dim=-2
        )
        if rotary_dim == self.dim:
            return rotated
        return torch.cat((rotated, inputs[..., rotary_dim:]), dim=-1)

    def forward(
        self,
        inputs: Tensor,
        state: TTTFastState | None = None,
        *,
        update: bool = True,
        update_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
        create_graph: bool | None = None,
    ) -> tuple[Tensor, TTTFastState]:
        """Process ``[batch, timesteps, tokens, dim]`` and return the next fast state."""
        if inputs.ndim != 4 or inputs.shape[-1] != self.dim:
            raise ValueError(
                f"Expected inputs with shape [batch, timesteps, tokens, {self.dim}], got {tuple(inputs.shape)}"
            )
        if state is not None and state.batch_size != inputs.shape[0]:
            raise ValueError(
                f"Fast-state batch size {state.batch_size} does not match input batch size {inputs.shape[0]}"
            )
        if update_mask is None:
            update_mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        elif update_mask.shape != inputs.shape[:2]:
            raise ValueError(
                f"Expected update_mask with shape {tuple(inputs.shape[:2])}, got {tuple(update_mask.shape)}"
            )
        else:
            update_mask = update_mask.to(device=inputs.device, dtype=torch.bool)
        if token_mask is None:
            token_mask = torch.ones(inputs.shape[:3], dtype=torch.bool, device=inputs.device)
        elif token_mask.shape != inputs.shape[:3]:
            raise ValueError(
                f"Expected token_mask with shape {tuple(inputs.shape[:3])}, got {tuple(token_mask.shape)}"
            )
        else:
            token_mask = token_mask.to(device=inputs.device, dtype=torch.bool)
        # A timestep with no valid token is pure padding and must neither
        # update fast weights nor advance the persistent RoPE position.
        update_mask = update_mask & token_mask.any(dim=-1)
        token_mask = token_mask & update_mask[:, :, None]

        outer_grad_enabled = torch.is_grad_enabled()
        outer_inference_enabled = torch.is_inference_mode_enabled()
        create_graph = self.training and self.second_order if create_graph is None else create_graph
        if create_graph and not outer_grad_enabled:
            create_graph = False

        input_dtype = inputs.dtype
        projection_dtype = self.q_proj.weight.dtype

        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.autocast(device_type=inputs.device.type, enabled=False),
        ):
            projected_inputs = inputs.detach().clone() if outer_inference_enabled else inputs
            projected_inputs = projected_inputs.to(dtype=projection_dtype)
            # Masks created by an outer inference_mode context are inference
            # tensors too. torch.where saves its condition for the inner
            # backward pass, so make ordinary tensors for the inner loop.
            projected_update_mask = update_mask.detach().clone() if outer_inference_enabled else update_mask
            projected_token_mask = token_mask.detach().clone() if outer_inference_enabled else token_mask
            if state is None:
                state = self.initial_state(inputs.shape[0])
            elif update and (
                outer_inference_enabled or not all(tensor.requires_grad for tensor in state.tensors())
            ):
                state = TTTFastState(
                    *(tensor.detach().clone().requires_grad_(True) for tensor in state.tensors()),
                    position=None if state.position is None else state.position.detach().clone(),
                )
            if state.position is None:
                state = TTTFastState(
                    *state.tensors(),
                    position=torch.full((state.batch_size,), -1, dtype=torch.int64, device=inputs.device),
                )

            outputs = []
            for timestep_index, timestep_inputs in enumerate(projected_inputs.unbind(dim=1)):
                timestep_update_mask = projected_update_mask[:, timestep_index]
                timestep_token_mask = projected_token_mask[:, timestep_index]
                normalized_inputs = F.layer_norm(timestep_inputs, (self.dim,))
                next_position = state.position + 1
                timestep_position = (
                    torch.where(timestep_update_mask, next_position, state.position.clamp_min(0))
                    if update
                    else state.position.clamp_min(0)
                )
                token_positions = (
                    timestep_position[:, None] * timestep_inputs.shape[1]
                    + torch.arange(timestep_inputs.shape[1], device=inputs.device)[None, :]
                )
                queries = self._apply_rope(self.q_proj(normalized_inputs), token_positions)
                if update:
                    keys = self._apply_rope(self.k_proj(normalized_inputs), token_positions)
                    values = self.v_proj(normalized_inputs)
                    candidate_state = self._update(
                        keys,
                        values,
                        state,
                        create_graph=create_graph,
                        token_mask=timestep_token_mask,
                    )
                    weight_mask = timestep_update_mask[:, None, None]
                    bias_mask = timestep_update_mask[:, None]
                    state = TTTFastState(
                        w1=torch.where(weight_mask, candidate_state.w1, state.w1),
                        b1=torch.where(bias_mask, candidate_state.b1, state.b1),
                        w2=torch.where(weight_mask, candidate_state.w2, state.w2),
                        b2=torch.where(bias_mask, candidate_state.b2, state.b2),
                        position=torch.where(timestep_update_mask, next_position, state.position),
                    )

                ttt_output = self._fast_mlp(queries, state)
                outputs.append(timestep_inputs + self.effective_gate * ttt_output)

            output = torch.stack(outputs, dim=1).to(dtype=input_dtype)
            if not outer_grad_enabled:
                output = output.detach()
                state = state.detach(requires_grad=False)

        return output, state
