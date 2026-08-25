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

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config


@PreTrainedConfig.register_subclass("pi0_ttt")
@dataclass
class PI0TTTConfig(PI0Config):
    """PI0 with RoboTTT-style recurrent fast weights and TBPTT training."""

    n_action_steps: int = 1

    sequence_length: int = 128
    sequence_stride: int = 32
    tbptt_segment_length: int = 8

    ttt_hidden_dim: int = 4096
    ttt_base_inner_lr: float = 0.1
    ttt_gate_init: float = 0.001
    ttt_rope_theta: float = 10_000.0
    ttt_second_order: bool = True
    ttt_start_layer: int = 14
    ttt_layer_indices: list[int] | None = field(default=None)
    ttt_freeze_base: bool = True

    optimizer_weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive")
        if self.tbptt_segment_length <= 0:
            raise ValueError("tbptt_segment_length must be positive")
        if self.tbptt_segment_length > self.sequence_length:
            raise ValueError("tbptt_segment_length cannot exceed sequence_length")
        if self.ttt_hidden_dim <= 0:
            raise ValueError("ttt_hidden_dim must be positive")
        if self.ttt_base_inner_lr <= 0:
            raise ValueError("ttt_base_inner_lr must be positive")
        if self.ttt_rope_theta <= 0:
            raise ValueError("ttt_rope_theta must be positive")
        if self.compile_model:
            raise ValueError("pi0_ttt does not support torch.compile because its inner loop uses autograd.grad")
        if self.gradient_checkpointing:
            raise ValueError("pi0_ttt gradient checkpointing is not implemented yet; use TBPTT to bound memory")
        if self.rtc_config is not None and self.rtc_config.enabled:
            raise ValueError("pi0_ttt does not support RTC because both methods update state during denoising")

        layer_indices = self.resolved_ttt_layer_indices
        if not layer_indices:
            raise ValueError("At least one TTT layer must be selected")
        if len(layer_indices) != len(set(layer_indices)):
            raise ValueError("ttt_layer_indices must not contain duplicates")
        if any(layer_index < 0 or layer_index >= 18 for layer_index in layer_indices):
            raise ValueError("TTT layer indices must be in [0, 17] for the PI0 Gemma expert")

    @property
    def resolved_ttt_layer_indices(self) -> list[int]:
        if self.ttt_layer_indices is not None:
            return list(self.ttt_layer_indices)
        return list(range(self.ttt_start_layer, 18))
