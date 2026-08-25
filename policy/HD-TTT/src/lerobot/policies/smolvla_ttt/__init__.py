# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from .configuration_smolvla_ttt import SmolVLATTTConfig
from .hd_dataset import HindsightLabelDataset
from .hd_ttt import (
    HindsightAttribution,
    HindsightAttributionComputer,
    counterfactual_grounding_loss,
    compute_hindsight_attribution,
    local_kvb_loss,
)
from .modeling_smolvla_ttt import SmolVLATTTPolicy
from .processor_smolvla_ttt import make_smolvla_ttt_pre_post_processors
from .sequence import SEQUENCE_SHAPE_KEY, TailPreservingSequenceDataset, sequence_collate_fn
from .ttt import TTTFastState, TTTMLPLayer

__all__ = [
    "SEQUENCE_SHAPE_KEY",
    "SmolVLATTTConfig",
    "SmolVLATTTPolicy",
    "HindsightAttribution",
    "HindsightAttributionComputer",
    "HindsightLabelDataset",
    "TTTFastState",
    "TTTMLPLayer",
    "compute_hindsight_attribution",
    "counterfactual_grounding_loss",
    "local_kvb_loss",
    "TailPreservingSequenceDataset",
    "make_smolvla_ttt_pre_post_processors",
    "sequence_collate_fn",
]
