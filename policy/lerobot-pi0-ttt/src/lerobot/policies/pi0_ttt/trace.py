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

import torch
from torch import Tensor

from .ttt import TTTFastState, TTTMLPLayer


def clone_fast_state(state: TTTFastState) -> TTTFastState:
    """Clone an inference state so the next update can be measured against it."""
    position = None if state.position is None else state.position.detach().clone()
    return TTTFastState(*(tensor.detach().clone() for tensor in state.tensors()), position=position)


def _initial_tensors(layer: TTTMLPLayer, state: TTTFastState) -> tuple[Tensor, ...]:
    return (
        layer.fast_w1_init.unsqueeze(0).expand_as(state.w1),
        layer.fast_b1_init.unsqueeze(0).expand_as(state.b1),
        layer.fast_w2_init.unsqueeze(0).expand_as(state.w2),
        layer.fast_b2_init.unsqueeze(0).expand_as(state.b2),
    )


@torch.no_grad()
def summarize_fast_state(
    layer: TTTMLPLayer,
    state: TTTFastState,
    previous_state: TTTFastState | None = None,
) -> dict[str, float]:
    """Return compact, JSON-safe metrics for one layer's per-trajectory fast weights."""
    current_tensors = state.tensors()
    initial_tensors = _initial_tensors(layer, state)
    reference_tensors = initial_tensors if previous_state is None else previous_state.tensors()

    norm_sq = 0.0
    initial_norm_sq = 0.0
    drift_sq = 0.0
    step_delta_sq = 0.0
    dot_to_initial = 0.0
    component_metrics: dict[str, float] = {}
    for name, current, initial, reference in zip(
        ("w1", "b1", "w2", "b2"),
        current_tensors,
        initial_tensors,
        reference_tensors,
        strict=True,
    ):
        current_float = current.detach().float()
        initial_float = initial.detach().float()
        reference_float = reference.detach().float()
        component_norm_sq = float(current_float.square().sum())
        component_initial_norm_sq = float(initial_float.square().sum())
        component_drift_sq = float((current_float - initial_float).square().sum())
        component_step_delta_sq = float((current_float - reference_float).square().sum())
        norm_sq += component_norm_sq
        initial_norm_sq += component_initial_norm_sq
        drift_sq += component_drift_sq
        step_delta_sq += component_step_delta_sq
        dot_to_initial += float((current_float * initial_float).sum())
        component_metrics[f"{name}_drift_l2"] = math.sqrt(component_drift_sq)
        component_metrics[f"{name}_step_delta_l2"] = math.sqrt(component_step_delta_sq)

    norm_l2 = math.sqrt(norm_sq)
    initial_norm_l2 = math.sqrt(initial_norm_sq)
    drift_l2 = math.sqrt(drift_sq)
    step_delta_l2 = math.sqrt(step_delta_sq)
    gate = torch.tanh(layer.gate.detach().float())
    position = (
        torch.full((state.batch_size,), -1, device=state.w1.device)
        if state.position is None
        else state.position
    )
    summary = {
        "position_mean": float(position.float().mean()),
        "fast_norm_l2": norm_l2,
        "initial_norm_l2": initial_norm_l2,
        "drift_l2": drift_l2,
        "drift_relative": drift_l2 / max(initial_norm_l2, 1e-12),
        "step_delta_l2": step_delta_l2,
        "step_delta_relative": step_delta_l2 / max(norm_l2, 1e-12),
        "cosine_to_initial": dot_to_initial / max(norm_l2 * initial_norm_l2, 1e-12),
        "inner_lr": float(layer.inner_lr.detach()),
        "gate_mean": float(gate.mean()),
        "gate_abs_mean": float(gate.abs().mean()),
        "gate_rms": float(gate.square().mean().sqrt()),
        "gate_abs_max": float(gate.abs().max()),
    }
    summary.update(component_metrics)
    return summary


@torch.no_grad()
def sketch_fast_state_delta(
    layer: TTTMLPLayer,
    state: TTTFastState,
    samples_per_tensor: int = 256,
) -> Tensor:
    """Sample a deterministic low-dimensional vector from fast-weight drift for PCA plots."""
    if samples_per_tensor <= 0:
        raise ValueError("samples_per_tensor must be positive")

    samples = []
    for current, initial in zip(state.tensors(), _initial_tensors(layer, state), strict=True):
        flat_delta = (current[0] - initial[0]).detach().float().reshape(-1)
        sample_count = min(samples_per_tensor, flat_delta.numel())
        indices = (
            torch.linspace(
                0,
                flat_delta.numel() - 1,
                steps=sample_count,
                device=flat_delta.device,
            )
            .round()
            .to(torch.int64)
        )
        samples.append(flat_delta[indices].cpu())
    return torch.cat(samples)
