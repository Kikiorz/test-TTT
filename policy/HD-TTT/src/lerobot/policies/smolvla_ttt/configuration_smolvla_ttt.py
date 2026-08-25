# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla_ttt")
@dataclass
class SmolVLATTTConfig(PreTrainedConfig):
    """Independent SmolVLA policy with recurrent RoboTTT fast weights."""

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    # TTT state advances once per environment decision, so do not cache and
    # execute several actions before the next observation is processed.
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    # Episode-local sequence training and truncated backpropagation through time.
    sequence_length: int = 256
    sequence_stride: int = 256
    tbptt_segment_length: int = 4
    # Number of preceding episode frames replayed before each sampled target
    # window.  Warm-up frames advance fast weights but are masked from all
    # action/HD losses, keeping offline full-history labels state-consistent
    # even when windows are shuffled across workers.
    ttt_history_warmup_length: int | None = 0
    # Optional episode-balanced subsampling for very long demonstrations.
    # ``None`` keeps every tail-preserving window (the default, rigorous
    # setting).  A positive value selects at most that many evenly spaced
    # windows per episode, which makes a fixed 150-epoch ablation feasible on
    # long-horizon benchmarks without changing the recurrent update itself.
    max_windows_per_episode: int | None = None

    # RoboTTT fast MLPs are inserted after attention and before the expert MLP.
    ttt_hidden_dim: int = 4096
    ttt_base_inner_lr: float = 0.1
    ttt_effective_gate_init: float = 0.05
    ttt_rope_theta: float = 10_000.0
    ttt_second_order: bool = True
    ttt_start_layer: int = 12
    ttt_layer_indices: list[int] | None = field(default=None)
    # Learned expert-side tokens prepended before the action tokens. Registers
    # and actions exchange information within each timestep while the original
    # causal action-action pattern is retained. Set to 0 for the no-register path.
    ttt_num_register_tokens: int = 16
    ttt_training_stage: str = "ttt_only"

    # Hindsight-Distilled TTT (HD-TTT) auxiliary objectives.  The switches are
    # opt-in so a checkpoint loaded from ordinary SmolVLA remains a clean
    # architecture/base ablation; the HD training recipe enables them after
    # generating causal attribution labels.
    hd_ttt_enabled: bool = False
    hd_hca_weight: float = 1.0
    hd_h2l_weight: float = 1.0
    hd_grounding_weight: float = 1.0
    hd_invariance_weight: float = 0.25
    hd_event_block_size: int = 4
    hd_attribution_threshold: float = 0.0
    hd_attribution_topk: int = 8
    hd_counterfactual_margin: float = 0.05
    # Hindsight labels supervise a local, causal predictor of whether the
    # current interaction should be written to fast weights.  The predictor
    # is enabled only when ``hd_ttt_enabled`` is true; ordinary SmolVLA/TTT
    # checkpoints therefore keep their original update path exactly.
    hd_write_gate_weight: float = 1.0
    hd_write_gate_init: float = 0.95
    hd_learned_write_gate: bool = False

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_action_steps != 1:
            raise ValueError("smolvla_ttt requires n_action_steps=1 so fast state advances every decision")
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.num_vlm_layers <= 0:
            raise ValueError("smolvla_ttt requires num_vlm_layers to be positive so TTT layers are explicit")
        if self.num_expert_layers > self.num_vlm_layers:
            raise ValueError("num_expert_layers cannot exceed num_vlm_layers")
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
        if self.ttt_history_warmup_length is not None and self.ttt_history_warmup_length < 0:
            raise ValueError("ttt_history_warmup_length must be non-negative")
        if self.max_windows_per_episode is not None and self.max_windows_per_episode <= 0:
            raise ValueError("max_windows_per_episode must be positive when provided")
        if self.ttt_hidden_dim <= 0:
            raise ValueError("ttt_hidden_dim must be positive")
        if self.ttt_base_inner_lr <= 0:
            raise ValueError("ttt_base_inner_lr must be positive")
        if not 0 <= self.ttt_effective_gate_init < 1:
            raise ValueError("ttt_effective_gate_init must be in [0, 1)")
        if self.ttt_rope_theta <= 0:
            raise ValueError("ttt_rope_theta must be positive")
        if self.ttt_num_register_tokens < 0:
            raise ValueError("ttt_num_register_tokens must be non-negative")
        if self.ttt_training_stage not in {"ttt_only", "action_head"}:
            raise ValueError("ttt_training_stage must be 'ttt_only' or 'action_head'")
        for name in (
            "hd_hca_weight",
            "hd_h2l_weight",
            "hd_grounding_weight",
            "hd_invariance_weight",
            "hd_attribution_threshold",
            "hd_counterfactual_margin",
            "hd_write_gate_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 < self.hd_write_gate_init < 1:
            raise ValueError("hd_write_gate_init must be strictly between 0 and 1")
        if self.hd_event_block_size <= 0:
            raise ValueError("hd_event_block_size must be positive")
        if self.hd_attribution_topk < 0:
            raise ValueError("hd_attribution_topk must be non-negative")
        if self.compile_model:
            raise ValueError(
                "smolvla_ttt does not support torch.compile because its inner loop uses autograd.grad"
            )
        if self.rtc_config is not None and self.rtc_config.enabled:
            raise ValueError(
                "smolvla_ttt does not support RTC because both methods update state during denoising"
            )

        layer_indices = self.resolved_ttt_layer_indices
        if not layer_indices:
            raise ValueError("At least one TTT layer must be selected")
        if len(layer_indices) != len(set(layer_indices)):
            raise ValueError("ttt_layer_indices must not contain duplicates")
        if any(layer_index < 0 or layer_index >= self.num_vlm_layers for layer_index in layer_indices):
            raise ValueError(
                f"TTT layer indices must be in [0, {self.num_vlm_layers - 1}] for this SmolVLA backbone"
            )

        if 0 < self.num_expert_layers < self.num_vlm_layers:
            if self.num_vlm_layers % self.num_expert_layers != 0:
                raise ValueError("num_vlm_layers must be divisible by num_expert_layers")
            expert_stride = self.num_vlm_layers // self.num_expert_layers
            missing_expert_layers = [
                layer_index for layer_index in layer_indices if layer_index % expert_stride != 0
            ]
            if missing_expert_layers:
                raise ValueError(
                    "TTT layers must coincide with action-expert layers; no expert exists at "
                    f"{missing_expert_layers}"
                )

    @property
    def resolved_ttt_layer_indices(self) -> list[int]:
        if self.ttt_layer_indices is not None:
            return list(self.ttt_layer_indices)
        return list(range(self.ttt_start_layer, self.num_vlm_layers))

    @property
    def trains_action_head(self) -> bool:
        return self.ttt_training_stage == "action_head"

    @property
    def trains_gate(self) -> bool:
        return self.ttt_training_stage == "action_head"

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
