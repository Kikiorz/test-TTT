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

    def clone(self, *, detach: bool = False, requires_grad: bool = True) -> "TTTFastState":
        """Clone this trajectory state for an isolated intervention branch."""

        def copy_tensor(tensor: Tensor) -> Tensor:
            copied = tensor.detach().clone() if detach else tensor.clone()
            if requires_grad and copied.is_floating_point():
                copied.requires_grad_(True)
            return copied

        position = None if self.position is None else self.position.detach().clone()
        return TTTFastState(
            *(copy_tensor(tensor) for tensor in self.tensors()),
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
        effective_gate_init: float = 0.05,
        gate_trainable: bool = False,
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

    def project_kv(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Project an interaction into the local K/V write objective."""
        normalized_inputs = F.layer_norm(inputs, (self.dim,))
        return self.k_proj(normalized_inputs), self.v_proj(normalized_inputs)

    def _update(
        self,
        keys: Tensor,
        values: Tensor,
        state: TTTFastState,
        create_graph: bool,
        write_gate: Tensor | None = None,
        detach_writer: bool = False,
    ) -> TTTFastState:
        if detach_writer:
            # Grounding is a reader-only objective.  The numerical fast-weight
            # update is still carried out so the correct/wrong branches see
            # the intended memories, but its K/V projections, inner gradient,
            # learning-rate parameter, and initial fast weights are detached
            # from the outer graph.  Query projection and the downstream
            # action head remain differentiable.
            detached_state = TTTFastState(
                *(tensor.detach().clone().requires_grad_(True) for tensor in state.tensors()),
                position=None if state.position is None else state.position.detach().clone(),
            )
            detached_keys = keys.detach()
            detached_values = values.detach()
            with torch.enable_grad():
                prediction = self._fast_mlp(detached_keys, detached_state)
                per_trajectory_loss = F.mse_loss(
                    prediction,
                    detached_values,
                    reduction="none",
                ).mean(dim=(1, 2))
                gradients = torch.autograd.grad(
                    per_trajectory_loss.sum(),
                    detached_state.tensors(),
                    create_graph=False,
                    retain_graph=False,
                )
            inner_lr = self.inner_lr.detach()
            updated_tensors = tuple(
                (weight - inner_lr * gradient).detach().requires_grad_(True)
                for weight, gradient in zip(detached_state.tensors(), gradients, strict=True)
            )
            if write_gate is not None:
                # Labels/gates are intervention controls, never writer
                # parameters in this branch.
                gate = write_gate.detach().to(
                    dtype=updated_tensors[0].dtype,
                    device=updated_tensors[0].device,
                )
                if gate.ndim == 0:
                    gate = gate.expand(detached_state.batch_size)
                updated_tensors = tuple(
                    (
                        weight
                        + gate.reshape(
                            detached_state.batch_size,
                            *([1] * (weight.ndim - 1)),
                        )
                        * (updated - weight)
                    )
                    .detach()
                    .requires_grad_(True)
                    for weight, updated in zip(
                        detached_state.tensors(), updated_tensors, strict=True
                    )
                )
            return TTTFastState(*updated_tensors, position=detached_state.position)

        prediction = self._fast_mlp(keys, state)
        per_trajectory_loss = F.mse_loss(prediction, values, reduction="none").mean(dim=(1, 2))
        gradients = torch.autograd.grad(
            per_trajectory_loss.sum(),
            state.tensors(),
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        updated_tensors = tuple(
            weight - self.inner_lr * gradient
            for weight, gradient in zip(state.tensors(), gradients, strict=True)
        )
        if write_gate is not None:
            # Interpolation keeps a learned H2L gate differentiable; a zero gate
            # is still an exact HCA event-skip intervention.
            gate = write_gate.to(dtype=updated_tensors[0].dtype, device=updated_tensors[0].device)
            if gate.ndim == 0:
                gate = gate.expand(state.batch_size)
            updated_tensors = tuple(
                weight
                + gate.reshape(state.batch_size, *([1] * (weight.ndim - 1)))
                * (updated - weight)
                for weight, updated in zip(state.tensors(), updated_tensors, strict=True)
            )
        return TTTFastState(*updated_tensors, position=state.position)

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
        create_graph: bool | None = None,
        write_gate: Tensor | None = None,
        detach_writer: bool = False,
        return_state_trace: bool = False,
    ) -> tuple[Tensor, TTTFastState] | tuple[Tensor, TTTFastState, tuple[TTTFastState, ...]]:
        """Process ``[batch, timesteps, tokens, dim]`` and return the next fast state.

        ``write_gate`` may be scalar, ``[batch]`` or ``[batch, timesteps]``. A
        zero gate skips a write exactly, while values in ``[0, 1]`` interpolate
        the local K/V update.  ``detach_writer`` keeps the numerical update but
        cuts all outer gradients through K/V/inner-update fast weights; query
        and downstream readout gradients remain active for reader-only
        counterfactual grounding. ``return_state_trace`` is used by hindsight
        replay and is off on the normal path.
        """
        if inputs.ndim != 4 or inputs.shape[-1] != self.dim:
            raise ValueError(
                f"Expected inputs with shape [batch, timesteps, tokens, {self.dim}], got {tuple(inputs.shape)}"
            )
        if state is not None and state.batch_size != inputs.shape[0]:
            raise ValueError(
                f"Fast-state batch size {state.batch_size} does not match input batch size {inputs.shape[0]}"
            )

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

            if detach_writer:
                # Even an initial state created by ``initial_state`` is an
                # expanded view of trainable fast-weight parameters.  Detach a
                # private copy before the first read/write so grounding cannot
                # update those parameters through the state path.
                state = TTTFastState(
                    *(tensor.detach().clone().requires_grad_(True) for tensor in state.tensors()),
                    position=None if state.position is None else state.position.detach().clone(),
                )

            if write_gate is not None:
                if write_gate.ndim == 0:
                    write_gate = write_gate.expand(inputs.shape[0], inputs.shape[1])
                elif write_gate.ndim == 1:
                    if write_gate.shape[0] == inputs.shape[0]:
                        write_gate = write_gate[:, None].expand(inputs.shape[0], inputs.shape[1])
                    elif inputs.shape[0] == 1 and write_gate.shape[0] == inputs.shape[1]:
                        write_gate = write_gate[None, :]
                    else:
                        raise ValueError(
                            "rank-1 write_gate must match batch (or timesteps when batch=1)"
                        )
                elif write_gate.ndim == 2 and write_gate.shape != inputs.shape[:2]:
                    raise ValueError(
                        f"write_gate must have shape {tuple(inputs.shape[:2])}, got {tuple(write_gate.shape)}"
                    )
                elif write_gate.ndim > 2:
                    raise ValueError("write_gate must be scalar, rank 1, or rank 2")

            outputs = []
            state_trace: list[TTTFastState] = []
            for timestep_index, timestep_inputs in enumerate(projected_inputs.unbind(dim=1)):
                normalized_inputs = F.layer_norm(timestep_inputs, (self.dim,))
                timestep_position = state.position + 1 if update else state.position.clamp_min(0)
                token_positions = (
                    timestep_position[:, None] * timestep_inputs.shape[1]
                    + torch.arange(timestep_inputs.shape[1], device=inputs.device)[None, :]
                )
                queries = self._apply_rope(self.q_proj(normalized_inputs), token_positions)
                if update:
                    if detach_writer:
                        # K/V are used to compute the intervention's numerical
                        # update, but their projections must not connect back to
                        # the input/action expert or writer parameters.
                        with torch.no_grad():
                            keys = self._apply_rope(
                                self.k_proj(normalized_inputs.detach()), token_positions
                            )
                            values = self.v_proj(normalized_inputs.detach())
                    else:
                        keys = self._apply_rope(self.k_proj(normalized_inputs), token_positions)
                        values = self.v_proj(normalized_inputs)
                    timestep_gate = None if write_gate is None else write_gate[:, timestep_index]
                    state = self._update(
                        keys,
                        values,
                        state,
                        create_graph=create_graph,
                        write_gate=timestep_gate,
                        detach_writer=detach_writer,
                    )
                    state = TTTFastState(*state.tensors(), position=timestep_position)

                ttt_output = self._fast_mlp(queries, state)
                residual_gate = self.effective_gate.detach() if detach_writer else self.effective_gate
                outputs.append(timestep_inputs + residual_gate * ttt_output)
                if return_state_trace:
                    state_trace.append(state.clone(detach=False, requires_grad=False))

            output = torch.stack(outputs, dim=1).to(dtype=input_dtype)
            if detach_writer:
                # Do not expose replay-only state leaves to the outer graph.
                # They were kept differentiable internally so each timestep
                # could compute its local update, but no caller should backprop
                # through the returned grounding state.
                state = state.detach(requires_grad=False)
                if return_state_trace:
                    state_trace = [trace.detach(requires_grad=False) for trace in state_trace]
            if not outer_grad_enabled:
                output = output.detach()
                state = state.detach(requires_grad=False)
                if return_state_trace:
                    state_trace = [trace.detach(requires_grad=False) for trace in state_trace]

        if return_state_trace:
            return output, state, tuple(state_trace)
        return output, state
