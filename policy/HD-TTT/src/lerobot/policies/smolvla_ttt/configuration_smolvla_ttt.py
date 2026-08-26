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


# CreditTTT stores one protocol-level intervention identity in artifacts while
# allowing the implementation to choose the concrete branch used for the
# counterfactual replay.  Keep the aliases here (rather than duplicating
# string literals across model/trainer code) so a config loaded from an older
# manifest has one unambiguous runtime representation.
_CREDIT_TTT_PROTOCOL = "credit_ttt_v3_query_effect"
_CREDIT_TTT_INTERVENTION_ALIASES = {
    "delete": "delete",
    "content_deletion": "delete",
    "replace": "replace",
    "content_replacement": "replace",
}


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
    # window.  Warm-up frames advance fast weights and are masked from action,
    # HCA, and grounding targets; labeled HD sequences retain them for the
    # separate local-writer objective through ``hd_writer_valid``.
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
    # Optional v2 numerical stabilization.  When enabled, the inner update
    # uses a bounded learned log-lr and an RMS-relative step with a finite
    # fallback.  ``False`` preserves the original RoboTTT update exactly for
    # clean/legacy checkpoints; the v2 recipe opts in explicitly.
    ttt_stable_inner_update: bool = False
    ttt_start_layer: int = 12
    ttt_layer_indices: list[int] | None = field(default=None)
    # ``suffix`` preserves the original RoboTTT writer (the current expert
    # suffix, including action/time/register tokens).  ``prefix_only`` uses
    # only the observation/language/state prefix as the K/V writer input while
    # retaining the expert suffix as the query/read path.  The latter removes
    # denoising-noise dependence from the write decision without changing the
    # fast-weight state or checkpoint tensor shapes.
    ttt_writer_mode: str = "suffix"
    # Learned expert-side tokens prepended before the action tokens. Register
    # queries may read the complete action suffix, while action queries cannot
    # directly read register columns and retain the original causal
    # action-action pattern. Set to 0 for the no-register path.
    ttt_num_register_tokens: int = 16
    ttt_training_stage: str = "ttt_only"

    # Hindsight-Distilled TTT (HD-TTT) auxiliary objectives.  The switches are
    # opt-in so a checkpoint loaded from ordinary SmolVLA remains a clean
    # architecture/base ablation; the HD training recipe enables them after
    # generating causal attribution labels.
    hd_ttt_enabled: bool = False
    hd_hca_weight: float = 1.0
    hd_h2l_weight: float = 1.0
    # v2 compact action-effect/content distillation.  Zero preserves the
    # legacy HD objective unless a v2 recipe opts in explicitly.
    # ``None`` is accepted at the decode boundary for checkpoints produced by
    # an early config writer which serialized this optional extension as JSON
    # ``null``.  ``__post_init__`` canonicalizes it back to the numeric
    # compatibility default before any model/loss code reads the field.
    hd_effect_weight: float | None = 0.0
    hd_grounding_weight: float = 1.0
    hd_invariance_weight: float = 0.25
    hd_event_block_size: int = 4
    # Maximum causal event replays used when constructing offline labels.
    # Zero means every causal block.  Keeping this in the policy config makes
    # the label/training compute contract explicit instead of silently
    # accepting an artifact generated with a different event budget.
    hd_max_events: int = 0
    # Minimum number of eligible future frames required when selecting the
    # single counterfactual branch used by grounding.  This prevents a late
    # event with only one remaining frame from winning merely because its
    # credit is averaged over a tiny horizon.  If an episode/window is too
    # short to satisfy the threshold, the offline builder falls back to the
    # highest-total-credit positive event.
    hd_grounding_min_future_frames: int = 64
    hd_attribution_threshold: float = 0.0
    hd_attribution_topk: int = 8
    # Label provenance selector.  Keep legacy as the serialization default so
    # old HD artifacts/checkpoints remain loadable; v2 experiments pass the
    # explicit protocol string.
    # As above, retain a nullable annotation solely for backwards-compatible
    # JSON decoding; runtime configs are always canonical strings.
    hd_attribution_protocol: str | None = "legacy_raw_hinge_max"
    # Flow velocities in MIKASA's normalized action space are often much
    # smaller than 0.05.  A non-zero dead-zone would therefore erase most of
    # the counterfactual grounding signal before it reaches the reader.  Keep
    # the objective faithful to the squared counterfactual error by default;
    # experiments that need a tolerance must opt into it explicitly.
    hd_counterfactual_margin: float = 0.0
    # ``random`` preserves ordinary flow-matching training. ``deployment``
    # makes the writer see the same first-denoise distribution as deployment:
    # a pure Gaussian action chunk (t=1), with no teacher-forced future action
    # chunk in the interaction that is written to fast weights.
    hd_phase_mode: str = "random"
    # Hindsight labels supervise a local, causal predictor of whether the
    # current interaction should be written to fast weights.  The predictor
    # is enabled only when ``hd_ttt_enabled`` is true; ordinary SmolVLA/TTT
    # checkpoints therefore keep their original update path exactly.
    hd_write_gate_weight: float = 1.0
    hd_write_gate_init: float = 0.95
    hd_learned_write_gate: bool = False

    # ------------------------------------------------------------------
    # CreditTTT (the paper method)
    # ------------------------------------------------------------------
    # ``hd_attribution_protocol`` remains the single protocol discriminator
    # for backwards-compatible checkpoint loading.  The value
    # ``credit_ttt_v3_query_effect`` opts into the final method: pairwise
    # full-history control credit supervises a query-conditioned local
    # fast-weight effect.  All fields below are inert for the legacy/v2
    # protocols, so an ordinary SmolVLA/TTT checkpoint is unchanged.
    hd_v3_pair_k: int = 5
    hd_v3_local_weight: float = 1.0
    # CMD is a reader/action audit loss evaluated on the same final-flow
    # replay used by QH2L.  Keeping one fixed default weight makes the
    # distinction between the writer (QH2L) and reader (CMD) explicit without
    # introducing a collection of task-specific knobs.
    hd_v3_cmd_weight: float = 1.0
    # Explicit objective-family selector for preregistered V3 ablations.
    # ``full`` is the canonical method; the two ablation names make a zero
    # loss weight auditable in a checkpoint instead of looking like an
    # accidental hyper-parameter setting.
    hd_v3_ablation: str = "full"
    hd_v3_cmd_margin: float = 0.05
    hd_v3_null_weight: float = 0.25
    hd_v3_null_threshold: float = 0.05
    # The previous *executed* slot-0 action is part of the formal CreditTTT
    # interaction token.  It is enabled by default for the V3 protocol and is
    # ``None`` means protocol default (enabled for CreditTTT, disabled for
    # legacy/clean).  An explicit boolean is retained for architecture-matched
    # clean-TTT controls: the baseline can use the same interaction projection
    # while removing only the hindsight objective.  This avoids conflating an
    # input-schema difference with the contribution of the training loss.
    hd_v3_include_previous_action: bool | None = None
    # The publication protocol deletes the event write (which is exactly the
    # student's before/after state intervention).  ``replace`` is accepted as
    # a separately named offline ablation only and is rejected by the V3
    # training contract unless a donor-state replay backend is added.
    hd_v3_intervention: str = "delete"
    # The local effect is computed at the final selected TTT layer and passed
    # through the existing action output projection.  This is a fixed
    # structural choice, not a task-specific tuning knob.
    hd_v3_effect_layer: str = "last_selected"

    def __post_init__(self):
        super().__post_init__()

        # A handful of pre-v2 checkpoints contain explicit JSON ``null`` for
        # fields that were introduced after the original SmolVLA-TTT config.
        # Normalize those values before numeric/comparison validation below so
        # both the generic draccus loader and the custom policy loader have the
        # same clean semantics.  The canonical values are serialized on the
        # next checkpoint save, so null does not propagate indefinitely.
        if self.hd_effect_weight is None:
            self.hd_effect_weight = 0.0
        if self.hd_attribution_protocol is None:
            self.hd_attribution_protocol = "legacy_raw_hinge_max"
        # A few pre-v2 config writers emitted newly added booleans as JSON
        # ``null``.  Treat that spelling as the compatibility/legacy value so
        # the generic draccus path and the custom checkpoint loader agree.
        if self.ttt_stable_inner_update is None:
            self.ttt_stable_inner_update = False

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
        if type(self.ttt_stable_inner_update) is not bool:
            raise ValueError("ttt_stable_inner_update must be a boolean")
        if not 0 <= self.ttt_effective_gate_init < 1:
            raise ValueError("ttt_effective_gate_init must be in [0, 1)")
        if self.ttt_rope_theta <= 0:
            raise ValueError("ttt_rope_theta must be positive")
        if self.ttt_writer_mode not in {"suffix", "prefix_only"}:
            raise ValueError("ttt_writer_mode must be 'suffix' or 'prefix_only'")
        if self.ttt_num_register_tokens < 0:
            raise ValueError("ttt_num_register_tokens must be non-negative")
        if self.ttt_training_stage not in {"ttt_only", "action_head"}:
            raise ValueError("ttt_training_stage must be 'ttt_only' or 'action_head'")
        if self.hd_learned_write_gate and not self.hd_ttt_enabled:
            raise ValueError("hd_learned_write_gate requires hd_ttt_enabled=True")
        if self.hd_effect_weight > 0 and not self.hd_ttt_enabled:
            raise ValueError("hd_effect_weight requires hd_ttt_enabled=True")
        if self.hd_effect_weight > 0 and self.hd_learned_write_gate:
            raise ValueError(
                "hd_effect_weight and hd_learned_write_gate cannot be enabled together: "
                "v2 action-effect replay is all-write so its writer distribution matches deployment; "
                "use the learned-gate path as a separate ablation"
            )
        if self.hd_effect_weight > 0 and not self.ttt_second_order:
            raise ValueError(
                "hd_effect_weight>0 requires ttt_second_order=True because v2 action-effect "
                "distillation differentiates through the fast-weight inner update"
            )
        for name in (
            "hd_hca_weight",
            "hd_h2l_weight",
            "hd_effect_weight",
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
        if self.hd_max_events < 0:
            raise ValueError("hd_max_events must be non-negative")
        if self.hd_grounding_min_future_frames < 0:
            raise ValueError("hd_grounding_min_future_frames must be non-negative")
        if self.hd_attribution_topk < 0:
            raise ValueError("hd_attribution_topk must be non-negative")
        # Accept the short CLI spellings used by the offline builders, then
        # serialize one canonical protocol string into checkpoints.  This
        # avoids a brittle mismatch when a recipe passes ``v2`` directly while
        # preserving the explicit legacy default for old artifacts.
        if self.hd_attribution_protocol in {"legacy", "v1"}:
            self.hd_attribution_protocol = "legacy_raw_hinge_max"
        elif self.hd_attribution_protocol == "v2":
            self.hd_attribution_protocol = "v2_relative_antithetic_robust"
        elif self.hd_attribution_protocol in {"v3", "credit_ttt_v3"}:
            self.hd_attribution_protocol = "credit_ttt_v3_query_effect"
        if self.hd_attribution_protocol not in {
            "legacy_raw_hinge_max",
            "v2_relative_antithetic_robust",
            "credit_ttt_v3_query_effect",
        }:
            raise ValueError(
                "hd_attribution_protocol must be 'legacy_raw_hinge_max', "
                "'v2_relative_antithetic_robust', or 'credit_ttt_v3_query_effect'"
            )
        if self.hd_effect_weight > 0 and self.hd_attribution_protocol not in {
            "v2_relative_antithetic_robust",
            "credit_ttt_v3_query_effect",
        }:
            raise ValueError(
                "hd_effect_weight>0 requires hd_attribution_protocol to select a v2 or "
                "CreditTTT attribution protocol"
            )
        if self.hd_v3_pair_k <= 0:
            raise ValueError("hd_v3_pair_k must be positive")
        for name in (
            "hd_v3_local_weight",
            "hd_v3_cmd_weight",
            "hd_v3_cmd_margin",
            "hd_v3_null_weight",
            "hd_v3_null_threshold",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.hd_v3_ablation, str):
            raise ValueError("hd_v3_ablation must be 'full', 'qh2l_only', or 'cmd_only'")
        ablation_aliases = {
            "canonical": "full",
            "qh2l": "qh2l_only",
            "cmd": "cmd_only",
        }
        self.hd_v3_ablation = ablation_aliases.get(
            self.hd_v3_ablation.strip().lower(), self.hd_v3_ablation.strip().lower()
        )
        if self.hd_v3_ablation not in {"full", "qh2l_only", "cmd_only"}:
            raise ValueError("hd_v3_ablation must be 'full', 'qh2l_only', or 'cmd_only'")
        if self.hd_attribution_protocol == "credit_ttt_v3_query_effect":
            expected_weights = {
                "full": (True, True),
                "qh2l_only": (True, False),
                "cmd_only": (False, True),
            }[self.hd_v3_ablation]
            if (self.hd_v3_local_weight > 0) != expected_weights[0]:
                raise ValueError(
                    f"CreditTTT {self.hd_v3_ablation} requires "
                    f"hd_v3_local_weight {'>0' if expected_weights[0] else '=0'}"
                )
            if (self.hd_v3_cmd_weight > 0) != expected_weights[1]:
                raise ValueError(
                    f"CreditTTT {self.hd_v3_ablation} requires "
                    f"hd_v3_cmd_weight {'>0' if expected_weights[1] else '=0'}"
                )
        if self.hd_v3_null_threshold > 1:
            raise ValueError("hd_v3_null_threshold must be at most 1")
        # Artifacts use ``event_write_deletion`` as the protocol-level schema
        # identity.  Command lines may still use ``delete`` or ``replace`` to
        # select a concrete offline branch; canonical V3 training accepts only
        # ``delete`` because the student trace compares write-before/write-
        # after states.  ``replace`` is retained for a separately named
        # intervention ablation.
        if not isinstance(self.hd_v3_intervention, str):
            raise ValueError(
                "hd_v3_intervention must be one of delete/replace/content_deletion/content_replacement"
            )
        intervention = _CREDIT_TTT_INTERVENTION_ALIASES.get(
            self.hd_v3_intervention.strip().lower()
        )
        if intervention is None:
            raise ValueError(
                "hd_v3_intervention must be one of delete/replace/content_deletion/content_replacement"
            )
        self.hd_v3_intervention = intervention
        if self.hd_v3_effect_layer != "last_selected":
            raise ValueError("hd_v3_effect_layer currently supports only 'last_selected'")
        if self.hd_attribution_protocol == _CREDIT_TTT_PROTOCOL:
            # The V3 objective is the pairwise query-conditioned effect.  A
            # positive ``hd_effect_weight`` would additionally activate the
            # legacy compact v2 true/wrong replay, silently changing the
            # method under study and making a baseline comparison invalid.
            if self.hd_effect_weight != 0:
                raise ValueError(
                    "CreditTTT requires hd_effect_weight=0; the legacy v2 action-effect "
                    "objective must not be mixed into the V3 protocol"
                )
            if not self.hd_ttt_enabled:
                raise ValueError(
                    "CreditTTT requires hd_ttt_enabled=True"
                )
            if self.ttt_writer_mode != "prefix_only":
                raise ValueError(
                    "CreditTTT requires ttt_writer_mode='prefix_only' so writes are observation-causal"
                )
            if not self.ttt_second_order:
                raise ValueError(
                    "CreditTTT query-effect distillation requires ttt_second_order=True"
                )
            # ``ttt_layer_indices`` need not be sorted.  The effect layer is
            # the numerically last selected layer, not the last user-provided
            # list element; using ``[-1]`` could silently select an
            # intermediate layer and apply the wrong action tail.
            resolved_v3_layers = self.resolved_ttt_layer_indices
            if not resolved_v3_layers:
                raise ValueError("CreditTTT requires at least one selected TTT layer")
            if max(resolved_v3_layers) != self.num_vlm_layers - 1:
                raise ValueError(
                    "CreditTTT requires the final VLM/action-expert layer to be selected "
                    "so the local effect can use the shared action tail"
                )
        # Resolve the protocol default only after aliases above have been
        # canonicalized.  Explicit values are meaningful for a fair,
        # architecture-matched Clean-TTT baseline and must not be overwritten.
        if self.hd_v3_include_previous_action is None:
            self.hd_v3_include_previous_action = self.hd_attribution_protocol == _CREDIT_TTT_PROTOCOL
        if type(self.hd_v3_include_previous_action) is not bool:
            raise ValueError("hd_v3_include_previous_action must be a boolean or null")
        if self.hd_phase_mode not in {"random", "deployment"}:
            raise ValueError("hd_phase_mode must be 'random' or 'deployment'")
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

    @property
    def credit_ttt_enabled(self) -> bool:
        """Whether the final CreditTTT query-effect objective is active."""

        return bool(
            self.hd_ttt_enabled
            and self.hd_attribution_protocol == "credit_ttt_v3_query_effect"
        )

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
