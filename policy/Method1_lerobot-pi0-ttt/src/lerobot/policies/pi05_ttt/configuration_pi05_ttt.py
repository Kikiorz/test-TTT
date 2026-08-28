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
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_ttt")
@dataclass
class PI05TTTConfig(PI05Config):
    """PI0.5 action expert with RoboTTT-style recurrent fast weights.

    ``ttt_only`` is the sequence-pretraining stage: the PI0.5 backbone is frozen,
    and all TTT parameters, including the residual gate, train.
    ``action_head`` keeps training TTT and additionally fine-tunes the complete
    PI0.5 action expert plus its action/time projections while the VLM stays frozen.
    """

    # Fast state advances once per action-chunk prediction. The default observes
    # every environment step; larger values execute more queued actions before
    # the next prediction and therefore before the next TTT update.
    n_action_steps: int = 1

    # Every sampled episode-local window is an independent selected sequence.
    # Each lane starts from the learned W0 and carries state only through its
    # TBPTT segments; one global minibatch of such sequences is one outer step.
    sequence_length: int = 256
    sequence_stride: int = 256
    tbptt_segment_length: int = 4

    ttt_hidden_dim: int = 4096
    ttt_base_inner_lr: float = 0.1
    ttt_effective_gate_init: float = 0.001
    ttt_rope_theta: float = 10_000.0
    ttt_second_order: bool = True
    ttt_start_layer: int = 14
    ttt_layer_indices: list[int] | None = field(default=None)
    ttt_training_stage: str = "ttt_only"

    optimizer_lr: float = 2e-5
    optimizer_weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive")
        if self.sequence_stride > self.sequence_length:
            raise ValueError("sequence_stride cannot exceed sequence_length because that would drop frames")
        if self.tbptt_segment_length <= 0:
            raise ValueError("tbptt_segment_length must be positive")
        if self.tbptt_segment_length > self.sequence_length:
            raise ValueError("tbptt_segment_length cannot exceed sequence_length")
        if self.ttt_hidden_dim <= 0:
            raise ValueError("ttt_hidden_dim must be positive")
        if self.ttt_base_inner_lr <= 0:
            raise ValueError("ttt_base_inner_lr must be positive")
        if not 0 <= self.ttt_effective_gate_init < 1:
            raise ValueError("ttt_effective_gate_init must be in [0, 1)")
        if self.ttt_rope_theta <= 0:
            raise ValueError("ttt_rope_theta must be positive")
        if self.ttt_training_stage not in {"ttt_only", "action_head"}:
            raise ValueError("ttt_training_stage must be 'ttt_only' or 'action_head'")
        if self.compile_model:
            raise ValueError(
                "pi05_ttt does not support torch.compile because its inner loop uses autograd.grad"
            )
        if self.gradient_checkpointing:
            raise ValueError("pi05_ttt uses TBPTT and does not support gradient checkpointing")
        if self.rtc_config is not None and self.rtc_config.enabled:
            raise ValueError(
                "pi05_ttt does not support RTC because both methods update state during denoising"
            )

        layer_indices = self.resolved_ttt_layer_indices
        if not layer_indices:
            raise ValueError("At least one TTT layer must be selected")
        if len(layer_indices) != len(set(layer_indices)):
            raise ValueError("ttt_layer_indices must not contain duplicates")
        if any(layer_index < 0 or layer_index >= 18 for layer_index in layer_indices):
            raise ValueError("TTT layer indices must be in [0, 17] for the PI0.5 Gemma expert")

    @property
    def resolved_ttt_layer_indices(self) -> list[int]:
        if self.ttt_layer_indices is not None:
            return list(self.ttt_layer_indices)
        return list(range(self.ttt_start_layer, 18))

    @property
    def trains_action_head(self) -> bool:
        return self.ttt_training_stage == "action_head"

    @property
    def trains_gate(self) -> bool:
        # RoboTTT learns the residual gate with the other slow TTT parameters.
        return True
