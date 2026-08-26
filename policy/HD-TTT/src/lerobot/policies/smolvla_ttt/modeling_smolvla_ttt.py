#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""Independent SmolVLA-TTT policy.

The SmolVLA model and processor code are duplicated in this package so the TTT
adaptation can evolve without importing the sibling ``smolvla`` policy. Selected
action-expert layers receive recurrent fast MLPs after attention and before the
feed-forward block. See this package's README for the sequence-training recipe.
"""

import json
import logging
import math
import os
from collections import deque
from dataclasses import fields
from pathlib import Path
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import (
    populate_queues,
)
from .configuration_smolvla_ttt import SmolVLATTTConfig
from .hd_ttt import (
    action_effect_distillation_loss,
    counterfactual_grounding_loss,
    local_kvb_loss,
)
from .sequence import HD_ACTION_SLOT_VALID_KEY, HD_WRITER_VALID_KEY, SEQUENCE_SHAPE_KEY
from .smolvlm_with_expert_ttt import SmolVLMWithExpertTTTModel
from .ttt import TTTFastState, TTTMLPLayer

TTTFastStates = dict[int, TTTFastState]


def _hd_loss_balance_metrics(auxiliary_loss: float, flow_loss: float) -> dict[str, float]:
    """Return scale diagnostics for the detached HD-vs-flow loss balance.

    The metric is intentionally observational: it does not rescale either
    objective or introduce another training knob.  ``hd_aux_to_flow_ratio``
    is the absolute auxiliary contribution divided by the absolute flow
    contribution.  A zero flow denominator yields ratio ``0`` (rather than
    NaN/Inf); ``hd_aux_fraction`` still reports ``1`` when a non-zero HD term
    is the only contribution.  Non-finite inputs are propagated as NaN so a
    run cannot silently hide an exploding loss.
    """

    auxiliary_value = float(auxiliary_loss)
    flow_value = float(flow_loss)
    if not math.isfinite(auxiliary_value) or not math.isfinite(flow_value):
        return {
            "hd_aux_to_flow_ratio": float("nan"),
            "hd_aux_fraction": float("nan"),
        }

    auxiliary_magnitude = abs(auxiliary_value)
    flow_magnitude = abs(flow_value)
    ratio = 0.0 if flow_magnitude <= 1e-8 else auxiliary_magnitude / flow_magnitude
    total_magnitude = auxiliary_magnitude + flow_magnitude
    fraction = 0.0 if total_magnitude <= 1e-8 else auxiliary_magnitude / total_magnitude
    return {
        "hd_aux_to_flow_ratio": ratio,
        "hd_aux_fraction": fraction,
    }


def _hd_ttt_parameter_range_metrics(ttt_layers: nn.ModuleDict | None) -> dict[str, float]:
    """Summarize selected TTT parameter ranges without touching gradients.

    ``inner_lr`` and ``effective_gate`` are the two slow controls that can
    make an otherwise identical HD recipe appear hyperparameter-sensitive.
    Logging their ranges makes that sensitivity auditable while deliberately
    leaving the update rule unchanged.  The helper tolerates lightweight fake
    layer containers used by tests and returns an empty mapping when no
    selected layers are available.
    """

    if ttt_layers is None:
        return {}
    try:
        layers = tuple(ttt_layers.values())
    except AttributeError:
        return {}

    inner_lrs: list[Tensor] = []
    effective_gates: list[Tensor] = []
    for layer in layers:
        inner_lr = getattr(layer, "inner_lr", None)
        if inner_lr is not None:
            inner_lrs.append(torch.as_tensor(inner_lr).detach().float().reshape(-1))
        effective_gate = getattr(layer, "effective_gate", None)
        if effective_gate is not None:
            effective_gates.append(torch.as_tensor(effective_gate).detach().float().reshape(-1))

    metrics: dict[str, float] = {}
    if inner_lrs:
        inner_lr_values = torch.cat(inner_lrs)
        metrics["hd_ttt_inner_lr_min"] = float(inner_lr_values.amin().item())
        metrics["hd_ttt_inner_lr_max"] = float(inner_lr_values.amax().item())
    if effective_gates:
        gate_values = torch.cat(effective_gates)
        metrics["hd_ttt_effective_gate_min"] = float(gate_values.amin().item())
        metrics["hd_ttt_effective_gate_max"] = float(gate_values.amax().item())
    return metrics


_CHECKPOINT_ARCHITECTURE_FIELDS = {
    "n_obs_steps",
    "chunk_size",
    "max_state_dim",
    "max_action_dim",
    "resize_imgs_with_padding",
    "tokenizer_max_length",
    "vlm_model_name",
    "add_image_special_tokens",
    "attention_mode",
    "prefix_length",
    "pad_language_to",
    "num_expert_layers",
    "num_vlm_layers",
    "self_attn_every_n_layers",
    "expert_width_multiplier",
    "min_period",
    "max_period",
    "ttt_hidden_dim",
    "ttt_base_inner_lr",
    "ttt_effective_gate_init",
    "ttt_rope_theta",
    "ttt_second_order",
    "ttt_start_layer",
    "ttt_layer_indices",
    "ttt_writer_mode",
    "ttt_num_register_tokens",
    "hd_ttt_enabled",
    "hd_hca_weight",
    "hd_h2l_weight",
    "hd_effect_weight",
    "hd_grounding_weight",
    "hd_invariance_weight",
    "hd_event_block_size",
    "hd_max_events",
    "hd_grounding_min_future_frames",
    "hd_attribution_threshold",
    "hd_attribution_protocol",
    "hd_attribution_topk",
    "hd_counterfactual_margin",
    "hd_phase_mode",
    "hd_write_gate_weight",
    "hd_write_gate_init",
    "hd_learned_write_gate",
}


def _restore_checkpoint_model_fields(
    config: SmolVLATTTConfig,
    source_config: SmolVLATTTConfig,
    raw_config: dict,
) -> None:
    """Restore every checkpoint-owned field that affects model structure or TTT behavior."""
    # A caller may intentionally turn HD-TTT on while initializing from an
    # ordinary TTT checkpoint (the normal baseline -> HD fine-tuning path).
    # Preserve that explicit opt-in across the structural checkpoint merge;
    # otherwise the source's ``hd_ttt_enabled=false`` silently disables the
    # new objective and learned gate.
    hd_field_names = {
        "hd_ttt_enabled",
        "hd_hca_weight",
        "hd_h2l_weight",
        "hd_effect_weight",
        "hd_grounding_weight",
        "hd_invariance_weight",
        "hd_event_block_size",
        "hd_max_events",
        "hd_grounding_min_future_frames",
        "hd_attribution_threshold",
        "hd_attribution_topk",
        "hd_attribution_protocol",
        "hd_counterfactual_margin",
        "hd_phase_mode",
        "hd_write_gate_weight",
        "hd_write_gate_init",
        "hd_learned_write_gate",
    }
    requested_hd = {
        name: getattr(config, name)
        for name in hd_field_names
        if hasattr(config, name)
    }
    requested_writer_mode = str(getattr(config, "ttt_writer_mode", None) or "suffix")
    source_writer_mode = str(getattr(source_config, "ttt_writer_mode", None) or "suffix")
    # The v2 action-effect objective is a deliberate structural conversion
    # from an ordinary first-order TTT checkpoint to a differentiable
    # (second-order) inner update.  ``ttt_second_order`` lives in the
    # checkpoint-architecture set for ordinary loads, but restoring the
    # source value here would silently turn the requested v2 path back off
    # (and fail validation) when the teacher checkpoint predates v2.
    requested_effect_weight = float(requested_hd.get("hd_effect_weight") or 0.0)
    preserve_second_order_for_effect = bool(
        requested_effect_weight > 0.0 and getattr(config, "ttt_second_order", False)
    )
    explicit_hd_opt_in = bool(
        requested_hd.get("hd_ttt_enabled", False)
        or requested_hd.get("hd_learned_write_gate", False)
    )
    # ``PreTrainedConfig.from_pretrained`` parses the checkpoint config first
    # and then applies CLI overrides.  Consequently an explicit
    # ``--policy.hd_ttt_enabled=false`` is represented here by a target value
    # that differs from the source value; a plain ``False`` value by itself is
    # not enough to distinguish the two cases.  Treat a mismatch in any HD
    # field as an explicit override as well.  This preserves both directions
    # of the intended workflow:
    #
    #   clean TTT (false) -> HD fine-tuning (true), and
    #   HD checkpoint (true) -> clean/disabled evaluation (false).
    #
    # When no CLI override is supplied, the parser copies the source values
    # into ``config`` and this comparison is false, so checkpoint-owned HD
    # settings continue to be restored normally.
    explicit_hd_override = explicit_hd_opt_in or any(
        name in raw_config
        and getattr(source_config, name, None) != value
        for name, value in requested_hd.items()
    )
    for field_name in _CHECKPOINT_ARCHITECTURE_FIELDS:
        if field_name in raw_config:
            setattr(config, field_name, getattr(source_config, field_name))
    if preserve_second_order_for_effect:
        config.ttt_second_order = True
    if explicit_hd_override:
        for field_name, value in requested_hd.items():
            setattr(config, field_name, value)
    # A writer-mode mismatch is an intentional structural conversion in the
    # v2 recipe (clean suffix teacher -> prefix-only student).  Preserve the
    # caller's requested mode in that case; otherwise inherit the checkpoint
    # mode just like the other architecture fields.
    if requested_writer_mode != source_writer_mode:
        config.ttt_writer_mode = requested_writer_mode
    config.__post_init__()


def _validate_checkpoint_keys(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    source_is_ttt: bool,
    strict: bool,
    source_has_learned_write_gate: bool = False,
    target_has_learned_write_gate: bool | None = None,
    source_writer_mode: str = "suffix",
    target_writer_mode: str = "suffix",
) -> None:
    """Allow new TTT tensors to be absent only when converting a base SmolVLA checkpoint."""
    # Callers loading pre-writer-mode configs may pass the raw JSON ``null``;
    # normalize it at this boundary as well as in ``from_pretrained`` so the
    # compatibility rule is deterministic for direct/unit-test callers.
    source_writer_mode = str(source_writer_mode or "suffix")
    target_writer_mode = str(target_writer_mode or "suffix")
    # A base SmolVLA checkpoint has no TTT tensors, so those tensors may be
    # initialized when it is converted to SmolVLA-TTT.  Once the source already
    # declares itself as a TTT checkpoint, omitting an existing TTT tensor is a
    # real incompatibility and must not be hidden by the conversion allowance.
    allowed_base_missing = (
        [
            key
            for key in missing_keys
            if key.startswith("model.ttt_layers.") or key == "model.register_tokens"
        ]
        if not source_is_ttt
        else []
    )
    # Prefix-only adds a shared projection which is absent from legacy
    # SmolVLA/TTT checkpoints.  It is safe to initialize it when the source
    # uses the suffix writer; a prefix checkpoint, however, must contain its
    # learned adapter.  Keeping this as a narrowly-scoped extension preserves
    # strict validation for every unrelated tensor.
    allowed_prefix_missing = [
        key
        for key in missing_keys
        if key.startswith("model.prefix_writer_proj.") and source_writer_mode != "prefix_only"
    ]
    # Older SmolVLA-TTT checkpoints predate the optional HD gate head.  They
    # remain loadable when the caller explicitly enables that extension, while
    # a checkpoint that declares the head is still checked strictly.
    allowed_gate_extension = [
        key
        for key in missing_keys
        if (
            (".write_gate_head." in key or ".write_gate_context_head." in key)
            and not source_has_learned_write_gate
        )
    ]
    allowed_missing = (
        set(allowed_base_missing)
        | set(allowed_prefix_missing)
        | set(allowed_gate_extension)
    )
    disallowed_missing = [key for key in missing_keys if key not in allowed_missing]
    # A short-lived prefix-gate prototype constructed both the old
    # action-token head and the new context head.  Ignore only that known
    # obsolete tensor family when loading it into the production
    # context-only architecture; all other unexpected keys remain fatal.
    allowed_legacy_unexpected = {key for key in unexpected_keys if ".write_gate_head." in key}
    if source_writer_mode == "prefix_only" and target_writer_mode == "suffix":
        # Explicitly disabling the prefix writer is a supported ablation.  Its
        # projection is unused and may safely be discarded, but no unrelated
        # checkpoint tensor is ignored.
        allowed_legacy_unexpected.update(
            key for key in unexpected_keys if key.startswith("model.prefix_writer_proj.")
        )
    # A clean/TTT ablation may intentionally disable a gate that was present
    # in the HD source checkpoint.  The optional context head is then absent
    # from the target module and appears as an unexpected tensor; it is safe
    # to discard only in this explicit source-HD -> target-clean conversion.
    if source_has_learned_write_gate and target_has_learned_write_gate is False:
        allowed_legacy_unexpected.update(
            key for key in unexpected_keys if ".write_gate_context_head." in key
        )
    disallowed_unexpected = [key for key in unexpected_keys if key not in allowed_legacy_unexpected]
    require_exact_checkpoint = source_is_ttt or strict
    # For a TTT source (or an explicitly strict load), only known optional
    # extensions may be absent.  For a non-strict base conversion the TTT
    # families above are the sole intentionally missing keys.
    required_allowed = set(allowed_prefix_missing) | set(allowed_gate_extension)
    if disallowed_unexpected or disallowed_missing or (
        require_exact_checkpoint and any(key not in required_allowed for key in missing_keys)
    ):
        raise RuntimeError(
            f"Incompatible SmolVLA checkpoint: missing={missing_keys}, unexpected={unexpected_keys}"
        )


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLATTTPolicy(PreTrainedPolicy):
    """Independent SmolVLA-TTT policy with episode-local recurrent fast weights."""

    config_class = SmolVLATTTConfig
    name = "smolvla_ttt"
    tbptt_loss_weighting = "valid_actions"

    def __init__(
        self,
        config: SmolVLATTTConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = SmolVLATTTFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    @staticmethod
    def _decode_source_config(raw_config: dict) -> SmolVLATTTConfig:
        """Decode either a base SmolVLA or a SmolVLA-TTT checkpoint config."""
        import draccus

        config_type = raw_config.get("type")
        if config_type not in {"smolvla", "smolvla_ttt"}:
            raise TypeError(f"smolvla_ttt can only load SmolVLA-family checkpoints, got {config_type!r}")
        valid_fields = {field.name for field in fields(SmolVLATTTConfig) if field.init}
        values = {key: value for key, value in raw_config.items() if key in valid_fields}
        # Early clean SmolVLA/TTT checkpoints serialized the optional HD
        # switches as JSON ``null``.  ``draccus`` cannot decode that legacy
        # spelling into a non-optional bool, even though the semantic value
        # is the disabled/clean path.  Normalize only these historical flags;
        # every other malformed field must still fail loudly during decoding.
        for flag_name in ("hd_ttt_enabled", "hd_learned_write_gate"):
            if values.get(flag_name) is None:
                values[flag_name] = False
        # The v2 fields were added after several internal checkpoints had
        # already been written.  Some of those configs contain an explicit
        # JSON ``null`` instead of omitting the new field.  Normalize the
        # nullable representation here as a second guard (the dataclass
        # annotation also accepts it for the generic draccus path).
        if values.get("hd_effect_weight") is None:
            values["hd_effect_weight"] = 0.0
        if values.get("hd_attribution_protocol") is None:
            values["hd_attribution_protocol"] = "legacy_raw_hinge_max"
        # ``ttt_writer_mode`` was added after the original suffix writer and
        # a few checkpoints serialized the optional field as JSON ``null``.
        # Treat null exactly like an absent field (the legacy suffix path),
        # while leaving any non-null invalid spelling for ``__post_init__`` to
        # reject.  This keeps old clean/TTT checkpoints loadable without
        # weakening validation of an explicitly malformed mode.
        if values.get("ttt_writer_mode") is None:
            values["ttt_writer_mode"] = "suffix"
        if config_type == "smolvla":
            # Base checkpoints often cache a full action chunk. TTT must observe
            # every environment decision and therefore executes one action at a time.
            values["n_action_steps"] = 1
            values["compile_model"] = False
            values["rtc_config"] = None
        return draccus.decode(SmolVLATTTConfig, values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """Load base SmolVLA weights while allowing only new TTT tensors to be absent."""
        import packaging
        import safetensors
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_model as load_model_as_safetensor

        model_id = str(pretrained_name_or_path)

        def resolve_file(filename: str) -> str:
            if os.path.isdir(model_id):
                resolved = os.path.join(model_id, filename)
                if not os.path.isfile(resolved):
                    raise FileNotFoundError(f"{filename} not found in {model_id}")
                return resolved
            return hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        config_file = resolve_file("config.json")
        with open(config_file) as file:
            raw_config = json.load(file)
        source_is_ttt = raw_config.get("type") == "smolvla_ttt"
        source_config = cls._decode_source_config(raw_config)

        if config is None:
            config = source_config
        elif isinstance(config, SmolVLATTTConfig):
            _restore_checkpoint_model_fields(config, source_config, raw_config)
        else:
            raise TypeError(f"Expected SmolVLATTTConfig, got {type(config).__name__}")
        config.pretrained_path = Path(pretrained_name_or_path)

        model = cls(config, **kwargs)
        load_kwargs = {"strict": False}
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            load_kwargs["device"] = config.device
        missing_keys, unexpected_keys = load_model_as_safetensor(
            model,
            resolve_file("model.safetensors"),
            **load_kwargs,
        )
        if "device" not in load_kwargs and config.device != "cpu":
            model.to(config.device)

        _validate_checkpoint_keys(
            list(missing_keys),
            list(unexpected_keys),
            source_is_ttt=source_is_ttt,
            strict=strict,
            source_has_learned_write_gate=bool(
                raw_config.get("hd_learned_write_gate", False)
            ),
            target_has_learned_write_gate=bool(getattr(config, "hd_learned_write_gate", False)),
            # Match ``_decode_source_config``: an explicit JSON null denotes
            # the pre-field suffix writer, not the literal string ``"None"``.
            source_writer_mode=str(raw_config.get("ttt_writer_mode") or "suffix"),
            target_writer_mode=str(getattr(config, "ttt_writer_mode", "suffix")),
        )
        if missing_keys:
            logging.info(
                "Loaded a SmolVLA base checkpoint; initialized %d new TTT/register tensors",
                len(missing_keys),
            )
        model.eval()
        return model

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._ttt_fast_states: TTTFastStates = {}

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions, self._ttt_fast_states = self.model.sample_actions_with_state(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            fast_states=self._ttt_fast_states,
            noise=noise,
            **kwargs,
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    @staticmethod
    def _reshape_hd_field(
        value: Tensor | None,
        sequence_shape: tuple[int, int],
        *,
        name: str,
    ) -> Tensor | None:
        """Accept flattened ``B*T`` labels or already grouped ``[B,T,...]`` labels."""

        if value is None:
            return None
        batch_size, sequence_length = sequence_shape
        if value.ndim == 0:
            return value.expand(batch_size, sequence_length)
        if value.shape[:2] == (batch_size, sequence_length):
            return value
        if value.ndim == 1 and value.shape[0] == sequence_length:
            # A shared per-time label is useful when every sample in a batch
            # comes from the same precomputed episode window.
            return value.unsqueeze(0).expand(batch_size, -1)
        if value.ndim >= 2 and value.shape[0] == sequence_length and batch_size != sequence_length:
            return value.unsqueeze(0).expand(batch_size, *value.shape)
        if value.shape[0] == batch_size * sequence_length:
            return value.reshape(batch_size, sequence_length, *value.shape[1:])
        if value.numel() == batch_size * sequence_length:
            return value.reshape(batch_size, sequence_length)
        raise ValueError(
            f"HD field {name!r} must start with [B,T]=[{batch_size},{sequence_length}] "
            f"or flattened B*T, got {tuple(value.shape)}"
        )

    def _hd_active_action_dim(self, student: Tensor, teacher: Tensor | None = None) -> int:
        """Return the action coordinates that are valid for an HD comparison.

        SmolVLA internally pads the task action (MIKASA has seven coordinates)
        to ``max_action_dim`` (normally 32).  HD labels are generated in the
        task/action space, so comparing a seven-dimensional teacher directly to
        the padded student is an error and padding the teacher with arbitrary
        values would supervise nonexistent coordinates.  We therefore compare
        only the active task coordinates, bounded by both tensors and the
        configured action feature when it is available.
        """

        student_dim = int(student.shape[-1])
        feature = getattr(self.config, "action_feature", None)
        feature_shape = getattr(feature, "shape", None)
        configured_dim = (
            int(feature_shape[0])
            if feature_shape and feature_shape[0] is not None
            else student_dim
        )
        dims = [student_dim, configured_dim]
        if teacher is not None:
            dims.append(int(teacher.shape[-1]))
        active_dim = min(dims)
        if active_dim <= 0:
            raise ValueError("HD action comparison requires a positive feature dimension")
        return active_dim

    @staticmethod
    def _hd_align_velocity_field(value: Tensor, target: Tensor, *, name: str) -> Tensor:
        """Align a teacher/intervention velocity to ``target``'s prefix shape.

        The canonical label is ``[B, T, chunk, D_task]`` while the model emits
        ``[B, T, chunk, max_action_dim]``.  A per-observation label
        ``[B, T, D_task]`` is also accepted and broadcast over the action
        chunk.  Feature dimensions are intentionally left untouched here; the
        caller slices both tensors to the active task dimension.
        """

        if value.ndim == 0 or target.ndim == 0:
            raise ValueError(f"{name} must have a feature dimension")
        aligned = value
        if aligned.ndim == target.ndim - 1 and target.ndim >= 3:
            # Per-time labels omit the chunk axis: [B,T,D] -> [B,T,1,D].
            if aligned.shape[:2] == target.shape[:2]:
                aligned = aligned.unsqueeze(-2)
        if aligned.ndim < target.ndim:
            # Permit leading singleton/batch-shared labels while preserving the
            # final feature axis.  The common case above has already inserted
            # the chunk dimension.
            while aligned.ndim < target.ndim:
                aligned = aligned.unsqueeze(0)
        if aligned.ndim != target.ndim:
            raise ValueError(
                f"{name} rank {value.ndim} cannot be aligned to student rank {target.ndim}; "
                f"got {tuple(value.shape)} vs {tuple(target.shape)}"
            )
        desired_shape = target.shape[:-1] + (aligned.shape[-1],)
        try:
            return torch.broadcast_to(aligned, desired_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} is not broadcastable to student prefix "
                f"{tuple(target.shape[:-1])}"
            ) from exc

    @staticmethod
    def _hd_step_weight(
        value: Tensor | None,
        target_shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> Tensor | None:
        """Reduce scalar, per-time, or ``[event,future]`` HD labels to ``[B,T]``.

        Offline HCA may retain a sparse ``C[i,j]`` matrix, whereas the online
        training batch normally stores its per-future aggregate.  This helper
        turns either representation into a weight for the current future time:
        matrices are summed over their event axis and any extra feature axes
        are averaged.  It never relies on accidental PyTorch broadcasting (in
        particular, ``[B,T,T]`` is not multiplied directly by ``[B,T]``).
        """

        if value is None:
            return None
        batch_size, sequence_length = target_shape
        weight = value.to(device=device, dtype=dtype)
        if weight.ndim == 0:
            return weight.expand(batch_size, sequence_length)
        if weight.shape[:2] != (batch_size, sequence_length):
            try:
                weight = torch.broadcast_to(weight, (batch_size, sequence_length, *weight.shape[2:]))
            except RuntimeError as exc:
                raise ValueError(
                    f"HD field {name!r} with shape {tuple(value.shape)} does not start with "
                    f"[B,T]=[{batch_size},{sequence_length}]"
                ) from exc
        # Remove singleton axes first, then reduce retained event/action axes.
        while weight.ndim > 2 and weight.shape[-1] == 1:
            weight = weight.squeeze(-1)
        if weight.ndim > 2:
            if weight.ndim > 3:
                # Keep the two sequence axes when this is a full C[e,j] label;
                # otherwise collapse feature/chunk axes before deciding below.
                weight = weight.mean(dim=tuple(range(3, weight.ndim)))
            if weight.ndim == 3 and weight.shape[-2] == sequence_length and weight.shape[-1] == sequence_length:
                # C[e,j] -> rho[j].  A generic per-time feature tensor
                # [B,T,K] must instead be reduced over K below.
                weight = weight.sum(dim=-2)
            elif weight.ndim > 2:
                weight = weight.mean(dim=tuple(range(2, weight.ndim)))
        if weight.shape != (batch_size, sequence_length):
            try:
                weight = torch.broadcast_to(weight, (batch_size, sequence_length))
            except RuntimeError as exc:
                raise ValueError(
                    f"HD field {name!r} reduced to shape {tuple(weight.shape)}, expected "
                    f"[{batch_size},{sequence_length}]"
                ) from exc
        return weight.clamp_min(0)

    @staticmethod
    def _hd_valid_step_weight(
        batch: dict[str, Tensor],
        sequence_shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        """Return fractional non-padding weight for each physical timestep."""

        slot_valid = SmolVLATTTPolicy._hd_action_slot_valid_weight(
            batch,
            sequence_shape,
            device=device,
            dtype=dtype,
        )
        if slot_valid is None:
            return None
        return slot_valid.mean(dim=-1)

    @staticmethod
    def _hd_action_slot_valid_weight(
        batch: dict[str, Tensor],
        sequence_shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        field_name: str = "action_is_pad",
    ) -> Tensor | None:
        """Return ``[B,T,S]`` validity for action-chunk slots.

        LeRobot normally stores ``action_is_pad`` as ``[B*T,S]``.  A few
        processors retain singleton/action-feature axes; those are reduced
        conservatively so a slot is valid only when all of its padding flags
        are false.  Keeping the slot axis lets HCA and grounding ignore
        repeated terminal actions instead of merely down-weighting the whole
        physical frame.
        """

        action_mask = batch.get(field_name)
        if action_mask is None:
            return None
        mask = SmolVLATTTPolicy._reshape_hd_field(
            action_mask,
            sequence_shape,
            name=field_name,
        ).to(device=device)
        if field_name == HD_ACTION_SLOT_VALID_KEY:
            # The sequence dataset stores validity (not padding) under this
            # key.  Reduce any retained action-feature axes conservatively so
            # a slot is valid only when all feature flags are valid.
            valid = mask.bool()
            while valid.ndim > 3:
                if valid.shape[-1] == 1:
                    valid = valid.squeeze(-1)
                else:
                    valid = valid.all(dim=-1)
            if valid.ndim == 2:
                valid = valid.unsqueeze(-1)
            return valid.to(dtype=dtype)
        pad = mask
        while pad.ndim > 3:
            if pad.shape[-1] == 1:
                pad = pad.squeeze(-1)
            else:
                pad = pad.all(dim=-1)
        if pad.ndim == 2:
            pad = pad.unsqueeze(-1)
        return (~pad.bool()).to(dtype=dtype)

    @staticmethod
    def _hd_writer_valid_step_weight(
        batch: dict[str, Tensor],
        sequence_shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        fallback: Tensor | None,
    ) -> Tensor | None:
        """Return the mask for physical interactions that may train a writer.

        ``action_is_pad`` describes whether an action *target* is present in a
        sampled window.  History warm-up frames intentionally have no such
        target, but they still represent real observations and must supervise
        the local K/V objective and gate.  ``hd_writer_valid`` is injected by
        :class:`TailPreservingSequenceDataset` only for labeled HD sequences;
        old artifacts and ordinary TTT batches fall back to the legacy action
        mask.
        """

        writer_valid = batch.get(HD_WRITER_VALID_KEY)
        if writer_valid is None:
            return fallback
        return SmolVLATTTPolicy._hd_step_weight(
            SmolVLATTTPolicy._reshape_hd_field(
                writer_valid,
                sequence_shape,
                name=HD_WRITER_VALID_KEY,
            ),
            sequence_shape,
            device=device,
            dtype=dtype,
            name=HD_WRITER_VALID_KEY,
        )

    @staticmethod
    def _hd_weighted_mean(
        values: Tensor,
        weights: Tensor | None = None,
        denominator: float | Tensor | None = None,
    ) -> Tensor:
        """Average an HD loss with optional weights, safely.

        ``denominator`` is an optional episode-level normalization constant.
        TBPTT normally computes a segment-local weighted mean, which makes a
        sparse hindsight target depend on where the segment boundary happens
        to fall.  The v2 trainer supplies the full physical-frame
        denominator so each segment contributes its numerator directly and
        the sum is invariant to the TBPTT partition.  Leaving it ``None``
        preserves the historical local-mean API for ordinary/legacy calls.
        """

        if weights is None:
            numerator = values.sum()
            local_denominator = values.new_tensor(float(values.numel()))
        else:
            weights = weights.to(device=values.device, dtype=values.dtype).clamp_min(0)
            while weights.ndim < values.ndim:
                weights = weights.unsqueeze(-1)
            weights = torch.broadcast_to(weights, values.shape)
            local_denominator = weights.sum()
            numerator = (values * weights).sum()
        if denominator is None:
            denominator_tensor = local_denominator
        else:
            denominator_tensor = torch.as_tensor(
                denominator,
                device=values.device,
                dtype=values.dtype,
            )
            if denominator_tensor.numel() != 1:
                raise ValueError("HD normalization denominator must be scalar")
            denominator_tensor = denominator_tensor.reshape(())
        safe_denominator = denominator_tensor.clamp_min(1e-8)
        return torch.where(
            denominator_tensor > 1e-8,
            numerator / safe_denominator,
            values.new_zeros(()),
        )

    @staticmethod
    def _hd_reduce_grounding_slots(
        values: Tensor,
        slot_valid: Tensor | None,
        step_weights: Tensor | None,
        denominator: float | Tensor | None = None,
    ) -> Tensor:
        """Reduce a ``[B,T,S]`` grounding field without double padding weights.

        ``slot_valid`` identifies real action-chunk slots.  When it is
        available, first average the valid slots *within each physical
        timestep* and only then apply the fractional timestep weight (for
        example ``S_valid / S``).  Multiplying the unnormalised slot field by
        that fraction would count the same padding factor twice and make a
        terminal frame with one valid slot contribute quadratically less than
        a full frame.
        """

        if slot_valid is None:
            return SmolVLATTTPolicy._hd_weighted_mean(values, step_weights, denominator)
        if values.ndim != 3 or slot_valid.ndim != 3:
            raise ValueError(
                "grounding slot reduction expects values and slot_valid with shape [B,T,S]"
            )
        slot_valid = torch.broadcast_to(slot_valid, values.shape).to(
            device=values.device, dtype=values.dtype
        )
        valid_count = slot_valid.sum(dim=-1)
        per_step = (values * slot_valid).sum(dim=-1) / valid_count.clamp_min(1.0)
        if step_weights is None:
            step_weights = (valid_count > 0).to(dtype=values.dtype)
        return SmolVLATTTPolicy._hd_weighted_mean(per_step, step_weights, denominator)

    @staticmethod
    def _hd_grounding_rho_weight(
        rho: Tensor | None,
        sequence_shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return episode-normalized grounding weights without segment renormalization.

        ``hd_rho`` is normalized once by the offline episode label builder.
        TBPTT segments must preserve those absolute values; normalizing by a
        segment-local maximum would make the same interaction receive a
        different grounding target solely because its segment was truncated.
        """

        weight = SmolVLATTTPolicy._hd_step_weight(
            rho,
            sequence_shape,
            device=device,
            dtype=dtype,
            name="hd_rho",
        )
        if weight is None:
            return torch.ones(sequence_shape, device=device, dtype=dtype)
        return weight.clamp(0, 1)

    @staticmethod
    def _clone_fast_states(
        fast_states: TTTFastStates | None,
        *,
        detach: bool = False,
        requires_grad: bool = True,
    ) -> TTTFastStates | None:
        if fast_states is None:
            return None
        return {
            layer_index: state.clone(detach=detach, requires_grad=requires_grad)
            for layer_index, state in fast_states.items()
        }

    def _hd_auxiliary_losses(
        self,
        batch: dict[str, Tensor],
        sequence_shape: tuple[int, int],
        *,
        student_velocity: Tensor,
        wrong_student_velocity: Tensor | None = None,
        grounding_student_velocity: Tensor | None = None,
        effect_student_true: Tensor | None = None,
        effect_student_wrong: Tensor | None = None,
        local_ttt_loss: Tensor | None = None,
        predicted_write_gate: Tensor | None = None,
        normalization_denominator: float | Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute optional HD terms from training-only teacher/intervention labels.

        The ordinary LeRobot batch contains none of these keys, so the function
        is a strict no-op for base SmolVLA/TTT training.  A hindsight data pass
        may attach flattened labels under the documented ``hd_*`` names; all
        teacher tensors are detached here before they can influence gradients.
        Teacher/intervention velocities may use the task dimension (for example
        ``[B*T, 50, 7]`` for MIKASA) while the model internally emits padded
        ``[B*T, 50, 32]`` tensors.  Only the active task coordinates are compared;
        per-future C/rho matrices are reduced to safe ``[B,T]`` weights.  A v2
        artifact may additionally carry a compact slot-0 ``hd_teacher_effect``
        target; when differentiable true/wrong branches are supplied, the
        action-effect term trains writer content as well as the reader.  Its
        absence leaves the legacy path unchanged.
        """

        if not getattr(self.config, "hd_ttt_enabled", False):
            return student_velocity.new_zeros(()), {}

        B, T = sequence_shape
        student_velocity = self._reshape_hd_field(
            student_velocity,
            sequence_shape,
            name="student_velocity",
        )
        total = student_velocity.new_zeros(())
        metrics: dict[str, float] = {}
        hca_weight = float(getattr(self.config, "hd_hca_weight", 1.0))
        h2l_weight = float(getattr(self.config, "hd_h2l_weight", 1.0))
        grounding_weight = float(getattr(self.config, "hd_grounding_weight", 1.0))
        invariance_weight = float(getattr(self.config, "hd_invariance_weight", 1.0))
        write_gate_weight = float(getattr(self.config, "hd_write_gate_weight", 1.0))
        counterfactual_margin = float(getattr(self.config, "hd_counterfactual_margin", 0.0))
        valid_steps = self._hd_valid_step_weight(
            batch,
            sequence_shape,
            device=student_velocity.device,
            dtype=student_velocity.dtype,
        )
        writer_valid_steps = self._hd_writer_valid_step_weight(
            batch,
            sequence_shape,
            device=student_velocity.device,
            dtype=student_velocity.dtype,
            fallback=valid_steps,
        )
        # ``TailPreservingSequenceDataset`` preserves the pre-warm-up action
        # slot mask under this auxiliary key.  Use its fractional validity for
        # the local writer objective so terminal rows with repeated/padded
        # future actions do not dominate H2L, while history warm-up rows stay
        # trainable through ``writer_valid_steps``.
        writer_slot_valid_steps: Tensor | None = None
        if HD_ACTION_SLOT_VALID_KEY in batch:
            preserved_slots = self._hd_action_slot_valid_weight(
                batch,
                sequence_shape,
                device=student_velocity.device,
                dtype=student_velocity.dtype,
                field_name=HD_ACTION_SLOT_VALID_KEY,
            )
            if preserved_slots is not None:
                writer_slot_valid_steps = preserved_slots.mean(dim=-1)

        teacher_velocity = self._reshape_hd_field(
            batch.get("hd_teacher_velocity"), sequence_shape, name="hd_teacher_velocity"
        )
        attribution = self._reshape_hd_field(
            batch.get("hd_attribution"), sequence_shape, name="hd_attribution"
        )
        if teacher_velocity is not None:
            teacher_velocity = teacher_velocity.to(
                device=student_velocity.device,
                dtype=student_velocity.dtype,
            )
            teacher_velocity = self._hd_align_velocity_field(
                teacher_velocity,
                student_velocity,
                name="hd_teacher_velocity",
            )
            active_dim = self._hd_active_action_dim(student_velocity, teacher_velocity)
            student_active = student_velocity[..., :active_dim]
            teacher_active = teacher_velocity[..., :active_dim].detach()
            hca_error = (student_active - teacher_active).square()
            if hca_error.ndim >= 4:
                # [B,T,chunk,action_dim]: ignore repeated/padded future
                # actions before reducing to one HCA value per physical frame.
                per_slot = hca_error.mean(dim=-1)
                slot_valid = self._hd_action_slot_valid_weight(
                    batch,
                    sequence_shape,
                    device=per_slot.device,
                    dtype=per_slot.dtype,
                )
                if slot_valid is not None:
                    slot_valid = torch.broadcast_to(slot_valid, per_slot.shape)
                    per_step = (per_slot * slot_valid).sum(dim=-1) / slot_valid.sum(
                        dim=-1
                    ).clamp_min(1.0)
                else:
                    per_step = per_slot.mean(dim=-1)
            else:
                hca_reduce_dims = tuple(range(2, hca_error.ndim))
                per_step = hca_error.mean(dim=hca_reduce_dims) if hca_reduce_dims else hca_error
            attribution_weight = self._hd_step_weight(
                attribution,
                (B, T),
                device=per_step.device,
                dtype=per_step.dtype,
                name="hd_attribution",
            )
            if valid_steps is not None:
                attribution_weight = (
                    valid_steps
                    if attribution_weight is None
                    else attribution_weight * valid_steps
                )
            hca = self._hd_weighted_mean(
                per_step,
                attribution_weight,
                normalization_denominator,
            )
            total = total + hca_weight * hca
            metrics["hd_hca"] = float(hca.detach().item())

        # The deployable writer objective is computed directly inside each
        # TTT layer.  It is the joint local K/V reconstruction exposed by the
        # existing TTT API (the value projection remains differentiable) and
        # is intentionally returned as an un-gated [B,T] loss
        # and weighted here by hindsight ``hd_write_gate`` plus the physical
        # writer-valid mask (which includes labeled history warm-up frames).
        #
        # ``hd_write_gate_observed`` is intentionally *not* multiplied into
        # this local objective.  With ``max_events > 0`` the collector only
        # replays a sampled subset of causal blocks and assigns unobserved
        # blocks the safe default gate=1.0.  Keeping those rows in H2L is the
        # deliberate all-write fallback: the local writer still learns its
        # ordinary deployment objective on every valid interaction, while the
        # observed mask gates only hindsight gate distillation below.  Masking
        # H2L by observed would be a different compute-budget ablation and
        # would turn unsampled rows into no-supervision rows.
        # This removes the need for unavailable/offline ``hd_local_*`` labels.
        if local_ttt_loss is not None:
            local_loss = self._reshape_hd_field(
                local_ttt_loss,
                sequence_shape,
                name="local_ttt_loss",
            ).to(device=student_velocity.device, dtype=student_velocity.dtype)
            local_gate = self._hd_step_weight(
                self._reshape_hd_field(batch.get("hd_write_gate"), sequence_shape, name="hd_write_gate"),
                (B, T),
                device=student_velocity.device,
                dtype=student_velocity.dtype,
                name="hd_write_gate",
            )
            if writer_valid_steps is not None:
                local_gate = (
                    writer_valid_steps
                    if local_gate is None
                    else local_gate * writer_valid_steps
                )
            if writer_slot_valid_steps is not None:
                local_gate = (
                    writer_slot_valid_steps
                    if local_gate is None
                    else local_gate * writer_slot_valid_steps
                )
            kvb = self._hd_weighted_mean(
                local_loss,
                local_gate,
                normalization_denominator,
            )
            total = total + h2l_weight * kvb
            metrics["hd_h2l"] = float(kvb.detach().item())
            if writer_slot_valid_steps is not None:
                metrics["hd_h2l_slot_valid_fraction"] = float(
                    writer_slot_valid_steps.detach().mean().item()
                )
        else:
            # Backward-compatible fallback for old checkpoints/collectors that
            # explicitly stored projected local K/V tensors.  New HD training
            # never depends on these fields because ``forward_with_state``
            # supplies ``local_ttt_loss`` above.
            local_key = self._reshape_hd_field(batch.get("hd_local_key"), sequence_shape, name="hd_local_key")
            local_value = self._reshape_hd_field(batch.get("hd_local_value"), sequence_shape, name="hd_local_value")
            local_prediction = self._reshape_hd_field(
                batch.get("hd_local_prediction"), sequence_shape, name="hd_local_prediction"
            )
            local_query = self._reshape_hd_field(batch.get("hd_local_query"), sequence_shape, name="hd_local_query")
            if local_key is not None and local_value is not None and local_prediction is not None:
                local_key = local_key.to(device=student_velocity.device, dtype=student_velocity.dtype)
                local_value = local_value.to(device=student_velocity.device, dtype=student_velocity.dtype)
                local_prediction = local_prediction.to(
                    device=student_velocity.device,
                    dtype=student_velocity.dtype,
                )
                if local_query is not None:
                    local_query = local_query.to(
                        device=student_velocity.device,
                        dtype=student_velocity.dtype,
                    )
                local_gate = self._reshape_hd_field(batch.get("hd_write_gate"), sequence_shape, name="hd_write_gate")
                if local_gate is not None:
                    local_gate = local_gate.to(
                        device=student_velocity.device,
                        dtype=student_velocity.dtype,
                    )
                # Keep the same all-write fallback as the differentiable
                # ``local_ttt_loss`` path above; observed replay coverage is
                # only a gate-target validity signal.
                if writer_valid_steps is not None:
                    local_gate = (
                        writer_valid_steps
                        if local_gate is None
                        else local_gate * writer_valid_steps
                    )
                if writer_slot_valid_steps is not None:
                    local_gate = (
                        writer_slot_valid_steps
                        if local_gate is None
                        else local_gate * writer_slot_valid_steps
                    )
                kvb = local_kvb_loss(local_query, local_key, local_value, local_prediction, local_gate)
                total = total + h2l_weight * kvb
                metrics["hd_h2l"] = float(kvb.detach().item())
                if writer_slot_valid_steps is not None:
                    metrics["hd_h2l_slot_valid_fraction"] = float(
                        writer_slot_valid_steps.detach().mean().item()
                    )

        # Hindsight ``u_i`` is available only offline.  Distill it into the
        # causal gate predicted from the current interaction so deployment
        # does not need labels or a teacher.  The target is detached and the
        # The writer-valid/observed masks keep history warm-up trainable while
        # excluding padded or unsampled interactions.
        if predicted_write_gate is not None and batch.get("hd_write_gate") is not None:
            predicted_gate = self._reshape_hd_field(
                predicted_write_gate,
                sequence_shape,
                name="predicted_write_gate",
            ).to(device=student_velocity.device, dtype=student_velocity.dtype).clamp(0, 1)
            target_gate = self._hd_step_weight(
                self._reshape_hd_field(batch.get("hd_write_gate"), sequence_shape, name="hd_write_gate"),
                (B, T),
                device=student_velocity.device,
                dtype=student_velocity.dtype,
                name="hd_write_gate",
            )
            if target_gate is not None:
                gate_error = F.smooth_l1_loss(
                    predicted_gate,
                    target_gate.detach().clamp(0, 1),
                    reduction="none",
                )
                gate_observed = self._hd_step_weight(
                    self._reshape_hd_field(
                        batch.get("hd_write_gate_observed"),
                        sequence_shape,
                        name="hd_write_gate_observed",
                    ),
                    (B, T),
                    device=student_velocity.device,
                    dtype=student_velocity.dtype,
                    name="hd_write_gate_observed",
                )
                gate_weights = writer_valid_steps
                if gate_observed is not None:
                    gate_weights = (
                        gate_observed
                        if gate_weights is None
                        else gate_weights * gate_observed
                    )
                gate_loss = self._hd_weighted_mean(
                    gate_error,
                    gate_weights,
                    normalization_denominator,
                )
                total = total + write_gate_weight * gate_loss
                metrics["hd_gate"] = float(gate_loss.detach().item())
                metrics["hd_gate_pred_mean"] = float(
                    self._hd_weighted_mean(predicted_gate.detach(), gate_weights).item()
                )
                metrics["hd_gate_target_mean"] = float(
                    self._hd_weighted_mean(target_gate.detach(), gate_weights).item()
                )
                if gate_weights is None:
                    metrics["hd_gate_observed_fraction"] = 1.0
                else:
                    metrics["hd_gate_observed_fraction"] = float(
                        (gate_weights > 0).to(dtype=student_velocity.dtype).mean().item()
                    )

                # A matching mean is not enough to show that the local gate
                # uses the current interaction: a constant predictor can
                # achieve the same value whenever the target distribution is
                # imbalanced.  Keep these detached diagnostics out of the
                # optimized objective so they do not alter the HD recipe.
                diagnostic_pred = predicted_gate.detach().float()
                diagnostic_target = target_gate.detach().float().clamp(0, 1)
                if gate_weights is None:
                    diagnostic_weights = torch.ones_like(diagnostic_pred)
                else:
                    diagnostic_weights = gate_weights.detach().float().clamp_min(0)
                diagnostic_denominator = diagnostic_weights.sum().clamp_min(1e-8)

                def _gate_mean(value: Tensor) -> Tensor:
                    return (value * diagnostic_weights).sum() / diagnostic_denominator

                pred_mean = _gate_mean(diagnostic_pred)
                target_mean = _gate_mean(diagnostic_target)
                pred_variance = _gate_mean((diagnostic_pred - pred_mean).square())
                target_variance = _gate_mean((diagnostic_target - target_mean).square())
                covariance = _gate_mean(
                    (diagnostic_pred - pred_mean) * (diagnostic_target - target_mean)
                )
                correlation_denominator = (pred_variance * target_variance).sqrt()
                correlation = torch.where(
                    correlation_denominator > 1e-8,
                    covariance / correlation_denominator.clamp_min(1e-8),
                    torch.zeros_like(correlation_denominator),
                )
                constant_error = F.smooth_l1_loss(
                    torch.full_like(diagnostic_target, target_mean),
                    diagnostic_target,
                    reduction="none",
                )
                constant_loss = _gate_mean(constant_error)
                gain_vs_constant = torch.where(
                    constant_loss > 1e-8,
                    (constant_loss - gate_loss.detach().float()) / constant_loss,
                    torch.zeros_like(constant_loss),
                )
                metrics["hd_gate_pred_std"] = float(pred_variance.sqrt().item())
                metrics["hd_gate_target_std"] = float(target_variance.sqrt().item())
                metrics["hd_gate_corr"] = float(correlation.clamp(-1, 1).item())
                metrics["hd_gate_constant"] = float(constant_loss.item())
                metrics["hd_gate_gain_vs_constant"] = float(gain_vs_constant.item())
                metrics["hd_gate_weight_mass"] = float(diagnostic_weights.sum().item())

        # ------------------------------------------------------------------
        # v2 action-effect/content distillation
        # ------------------------------------------------------------------
        # The offline builder stores a selected-event axis (new v2 artifacts
        # have K=1), and the first branch is always the selected grounding
        # event.  Keep the online computation compatible with older K>1
        # artifacts by consuming branch zero only; extra branches are ignored
        # unless a separately implemented multi-event ablation is used.
        # Crucially,
        # this branch is only entered when the caller supplied *differentiable*
        # true/wrong replays (see ``forward_sequence_segment`` below).  Legacy
        # labels/checkpoints never pay the extra forward passes or receive a
        # changed loss.
        if (
            effect_student_true is not None
            and effect_student_wrong is not None
            and batch.get("hd_teacher_effect") is not None
        ):
            effect_true = self._reshape_hd_field(
                effect_student_true,
                sequence_shape,
                name="effect_student_true",
            )
            effect_wrong = self._reshape_hd_field(
                effect_student_wrong,
                sequence_shape,
                name="effect_student_wrong",
            )
            if effect_true is None or effect_wrong is None:
                raise RuntimeError("effect student branches unexpectedly resolved to None")
            # ``forward_with_state`` emits [B,T,chunk,D].  Compact v2 labels
            # target only the executed slot, so select slot 0 before comparing.
            if effect_true.ndim < 4 or effect_wrong.ndim < 4:
                raise ValueError(
                    "v2 action-effect branches must include [B,T,chunk,D] dimensions"
                )
            effect_true = effect_true[:, :, 0, :]
            effect_wrong = effect_wrong[:, :, 0, :]
            teacher_effect = self._reshape_hd_field(
                batch.get("hd_teacher_effect"),
                sequence_shape,
                name="hd_teacher_effect",
            )
            if teacher_effect is None:
                raise RuntimeError("hd_teacher_effect unexpectedly resolved to None")
            if teacher_effect.ndim >= 4:
                # [B,T,K,D] -> selected branch [B,T,D].
                teacher_effect = teacher_effect[:, :, 0, :]
            elif teacher_effect.ndim != 3:
                raise ValueError(
                    "hd_teacher_effect must have [B,T,D] or [B,T,K,D] shape"
                )
            active_dim = self._hd_active_action_dim(effect_true, teacher_effect)
            effect_true = effect_true[..., :active_dim]
            effect_wrong = effect_wrong[..., :active_dim]
            teacher_effect = teacher_effect.to(
                device=effect_true.device,
                dtype=effect_true.dtype,
            )[..., :active_dim].detach()
            effect_rho = self._reshape_hd_field(
                batch.get("hd_effect_rho"), sequence_shape, name="hd_effect_rho"
            )
            if effect_rho is None:
                effect_rho = self._hd_grounding_rho_weight(
                    self._reshape_hd_field(batch.get("hd_rho"), sequence_shape, name="hd_rho"),
                    (B, T),
                    device=effect_true.device,
                    dtype=effect_true.dtype,
                )
            elif effect_rho.ndim >= 3:
                effect_rho = effect_rho[..., 0]
            effect_rho = effect_rho.to(
                device=effect_true.device,
                dtype=effect_true.dtype,
            ).clamp(0, 1)
            effect_valid = self._reshape_hd_field(
                batch.get("hd_effect_valid"), sequence_shape, name="hd_effect_valid"
            )
            if effect_valid is not None and effect_valid.ndim >= 3:
                effect_valid = effect_valid[..., 0]
            if effect_valid is None:
                effect_valid = torch.ones_like(effect_rho)
            else:
                effect_valid = effect_valid.to(
                    device=effect_true.device,
                    dtype=effect_true.dtype,
                ).clamp_min(0)
            # Padded/warm-up interactions do not provide a valid action-effect
            # target.  The effect compares *executed slot 0*, so use that
            # exact slot validity rather than the all-slot mean used by HCA.
            # ``hd_action_slot_valid`` preserves the physical mask through
            # warm-up rows; fall back to ``action_is_pad`` for older batches.
            slot_valid_field = (
                HD_ACTION_SLOT_VALID_KEY
                if batch.get(HD_ACTION_SLOT_VALID_KEY) is not None
                else "action_is_pad"
            )
            effect_slot_valid = self._hd_action_slot_valid_weight(
                batch,
                sequence_shape,
                device=effect_valid.device,
                dtype=effect_valid.dtype,
                field_name=slot_valid_field,
            )
            if effect_slot_valid is not None:
                if effect_slot_valid.shape[-1] < 1:
                    raise ValueError("action-effect validity requires at least one action slot")
                effect_valid = effect_valid * effect_slot_valid[..., 0]
            effect_parts = action_effect_distillation_loss(
                effect_true,
                effect_wrong,
                teacher_effect=teacher_effect,
                importance=effect_rho,
                valid_mask=effect_valid > 0,
                reduction="none",
                return_components=True,
            )
            effect_loss = effect_parts.total
            # ``effect_parts.total`` is [B,T] under reduction='none'.  Apply
            # fractional writer validity without renormalizing per segment;
            # this preserves the episode-level attribution scale across TBPTT.
            effect_weight = effect_valid
            local_effect_mass = effect_weight.sum().clamp_min(1e-8)
            effect_denominator = (
                local_effect_mass
                if normalization_denominator is None
                else torch.as_tensor(
                    normalization_denominator,
                    device=effect_loss.device,
                    dtype=effect_loss.dtype,
                ).clamp_min(1e-8)
            )
            effect_scalar = (effect_loss * effect_weight).sum() / effect_denominator
            # Missing fields are possible when loading a pre-v2 config.  Keep
            # that legacy path a strict no-op; v2 recipes opt in explicitly
            # through the config field (whose default is zero).
            hd_effect_weight = float(getattr(self.config, "hd_effect_weight", 0.0))
            total = total + hd_effect_weight * effect_scalar
            metrics["hd_effect"] = float(effect_scalar.detach().item())
            metrics["hd_effect_direction"] = float(effect_parts.effect.detach().mean().item())
            metrics["hd_effect_invariance"] = float(effect_parts.invariance.detach().mean().item())
            metrics["hd_effect_weight_mass"] = float(local_effect_mass.detach().item())

        if wrong_student_velocity is not None:
            grounding_student = (
                student_velocity
                if grounding_student_velocity is None
                else self._reshape_hd_field(
                    grounding_student_velocity,
                    sequence_shape,
                    name="grounding_student_velocity",
                )
            )
            teacher_true = self._reshape_hd_field(
                batch.get("hd_teacher_true_velocity"), sequence_shape, name="hd_teacher_true_velocity"
            )
            teacher_wrong = self._reshape_hd_field(
                batch.get("hd_teacher_wrong_velocity"), sequence_shape, name="hd_teacher_wrong_velocity"
            )
            rho = self._reshape_hd_field(batch.get("hd_rho"), sequence_shape, name="hd_rho")
            if teacher_true is not None and teacher_wrong is not None:
                grounding_student = grounding_student.to(
                    device=student_velocity.device,
                    dtype=student_velocity.dtype,
                )
                wrong_student_velocity = self._reshape_hd_field(
                    wrong_student_velocity,
                    sequence_shape,
                    name="wrong_student_velocity",
                )
                wrong_student_velocity = self._hd_align_velocity_field(
                    wrong_student_velocity,
                    grounding_student,
                    name="wrong_student_velocity",
                )
                teacher_true = teacher_true.to(
                    device=student_velocity.device,
                    dtype=student_velocity.dtype,
                )
                teacher_wrong = teacher_wrong.to(
                    device=student_velocity.device,
                    dtype=student_velocity.dtype,
                )
                teacher_true = self._hd_align_velocity_field(
                    teacher_true,
                    grounding_student,
                    name="hd_teacher_true_velocity",
                )
                teacher_wrong = self._hd_align_velocity_field(
                    teacher_wrong,
                    grounding_student,
                    name="hd_teacher_wrong_velocity",
                )
                active_dim = self._hd_active_action_dim(grounding_student, teacher_true)
                student_true_active = grounding_student[..., :active_dim]
                student_wrong_active = wrong_student_velocity[..., :active_dim]
                teacher_true_active = teacher_true[..., :active_dim]
                teacher_wrong_active = teacher_wrong[..., :active_dim]
                rho_weight = self._hd_grounding_rho_weight(
                    rho,
                    (B, T),
                    device=student_velocity.device,
                    dtype=student_velocity.dtype,
                )
                grounding_parts = counterfactual_grounding_loss(
                    student_true_active,
                    student_wrong_active,
                    teacher_true_active,
                    teacher_wrong_active,
                    rho_weight,
                    margin=counterfactual_margin,
                    reduction="none",
                    return_components=True,
                )
                grounding_per_token = grounding_parts.direction + (
                    invariance_weight * grounding_parts.invariance
                )
                slot_valid = self._hd_action_slot_valid_weight(
                    batch,
                    sequence_shape,
                    device=grounding_per_token.device,
                    dtype=grounding_per_token.dtype,
                )
                if slot_valid is not None:
                    slot_valid = torch.broadcast_to(slot_valid, grounding_per_token.shape)
                grounding_step_weights = valid_steps
                if slot_valid is not None and grounding_step_weights is None:
                    grounding_step_weights = (slot_valid.sum(dim=-1) > 0).to(
                        dtype=grounding_per_token.dtype
                    )
                # Reduce valid action slots within a physical timestep before
                # applying the fractional timestep mask.  This avoids the
                # previous K^2/S weighting at episode boundaries (K valid
                # slots out of S total slots).
                grounding = self._hd_reduce_grounding_slots(
                    grounding_per_token,
                    slot_valid,
                    grounding_step_weights,
                    normalization_denominator,
                )
                total = total + grounding_weight * grounding
                metrics["hd_grounding"] = float(grounding.detach().item())

                # Detached support/scale diagnostics distinguish an actually
                # absent counterfactual signal from a small loss rounded to
                # zero in the terminal log.  They never enter ``total``.
                teacher_delta = (teacher_true_active - teacher_wrong_active).detach().float()
                student_delta = (student_true_active - student_wrong_active).detach().float()
                teacher_delta_rms = self._hd_reduce_grounding_slots(
                    teacher_delta.square().mean(dim=-1).sqrt(),
                    slot_valid,
                    grounding_step_weights,
                )
                student_delta_rms = self._hd_reduce_grounding_slots(
                    student_delta.square().mean(dim=-1).sqrt(),
                    slot_valid,
                    grounding_step_weights,
                )
                direction_metric = self._hd_reduce_grounding_slots(
                    grounding_parts.direction.detach().float(),
                    slot_valid,
                    grounding_step_weights,
                )
                invariance_metric = self._hd_reduce_grounding_slots(
                    grounding_parts.invariance.detach().float(),
                    slot_valid,
                    grounding_step_weights,
                )
                # Match the exact element-wise dead-zone used by
                # ``counterfactual_grounding_loss``: high-rho rows compare
                # student and teacher counterfactual deltas, while low-rho
                # rows enforce student invariance.  Measuring only the raw
                # teacher delta would report nearly 100% activity at
                # margin=0 even when the optimized hinge is inactive.
                direction_active = (
                    (student_delta - teacher_delta).abs() > counterfactual_margin
                ).to(dtype=teacher_delta.dtype).mean(dim=-1)
                invariance_active = (student_delta.abs() > counterfactual_margin).to(
                    dtype=teacher_delta.dtype
                ).mean(dim=-1)
                margin_active_field = (
                    rho_weight.detach().float().unsqueeze(-1) * direction_active
                    + (1.0 - rho_weight.detach().float().unsqueeze(-1)) * invariance_active
                )
                margin_active = self._hd_reduce_grounding_slots(
                    margin_active_field,
                    slot_valid,
                    grounding_step_weights,
                )
                if grounding_step_weights is None:
                    grounding_support = teacher_delta.new_tensor(float(B * T))
                else:
                    grounding_support = grounding_step_weights.detach().float().sum()
                rho_support = self._hd_weighted_mean(
                    (rho_weight.detach().float() > 1e-6).to(dtype=teacher_delta.dtype),
                    grounding_step_weights,
                )
                counterfactual_gate = self._hd_step_weight(
                    self._reshape_hd_field(
                        batch.get("hd_counterfactual_write_gate"),
                        sequence_shape,
                        name="hd_counterfactual_write_gate",
                    ),
                    (B, T),
                    device=teacher_delta.device,
                    dtype=teacher_delta.dtype,
                    name="hd_counterfactual_write_gate",
                )
                if counterfactual_gate is None:
                    wrong_gate_zero_fraction = teacher_delta.new_zeros(())
                else:
                    wrong_gate_zero_fraction = self._hd_weighted_mean(
                        (counterfactual_gate.detach() <= 1e-6).to(dtype=teacher_delta.dtype),
                        grounding_step_weights,
                    )
                delta_ratio = torch.where(
                    teacher_delta_rms > 1e-8,
                    student_delta_rms / teacher_delta_rms,
                    torch.zeros_like(teacher_delta_rms),
                )
                metrics["hd_grounding_direction"] = float(direction_metric.item())
                metrics["hd_grounding_invariance"] = float(invariance_metric.item())
                metrics["hd_grounding_weight_mass"] = float(grounding_support.item())
                metrics["hd_grounding_rho_nonzero_fraction"] = float(rho_support.item())
                metrics["hd_grounding_wrong_gate_zero_fraction"] = float(
                    wrong_gate_zero_fraction.item()
                )
                metrics["hd_grounding_teacher_delta_rms"] = float(teacher_delta_rms.item())
                metrics["hd_grounding_student_delta_rms"] = float(student_delta_rms.item())
                metrics["hd_grounding_delta_ratio"] = float(delta_ratio.item())
                metrics["hd_grounding_margin_active_fraction"] = float(margin_active.item())

        return total, metrics

    def forward_sequence_segment(
        self,
        batch: dict[str, Tensor],
        sequence_shape: tuple[int, int],
        fast_states: TTTFastStates | None = None,
        reduction: str = "mean",
        noise: Tensor | None = None,
        time: Tensor | None = None,
        *,
        grounding_states: dict[str, TTTFastStates | None] | None = None,
        flow_loss_weight: float | Tensor | None = None,
        hd_normalization_denominator: float | Tensor | None = None,
    ) -> tuple[Tensor, dict, TTTFastStates]:
        """Train one contiguous TBPTT segment and return its numerical fast state.

        ``grounding_states`` is an optional mutable pair of detached replay
        states (``"true"``/``"wrong"``).  When provided, the two
        counterfactual branches continue from their own previous segment
        states.  This preserves full-episode causal interventions across
        TBPTT boundaries while keeping the historical three-item return API.
        The container is created and discarded by the sequence-level trainer;
        it must never be reused across episodes/windows.  ``flow_loss_weight``
        and ``hd_normalization_denominator`` are optional sequence-level
        controls used by the v2 trainer: the former weights only this segment's
        valid-action flow numerator, while the latter normalizes every HD
        auxiliary numerator by the full physical-frame count.  Keeping these
        factors separate makes the objective independent of the chosen TBPTT
        segment length; omitted values preserve the historical local-mean
        behavior.
        """
        batch_size, sequence_length = sequence_shape
        expected_flat_batch = batch_size * sequence_length
        if batch[ACTION].shape[0] != expected_flat_batch:
            raise ValueError(
                f"Sequence shape {sequence_shape} requires {expected_flat_batch} flattened samples, "
                f"but the action batch has {batch[ACTION].shape[0]}"
            )

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        # ``hd_ttt_enabled`` is an architecture/deployment switch, while
        # ``hd_labels_present`` only indicates that this training batch carries
        # hindsight teacher fields.  Keeping them separate is essential:
        # an HD checkpoint must use its learned local gate at deployment even
        # though no offline labels are available then.
        hd_enabled = bool(getattr(self.config, "hd_ttt_enabled", False))
        # Keep compatibility with older artifacts that only contain projected
        # local K/V or counterfactual columns; phase/teacher fields are checked
        # independently below when they are actually consumed.
        hd_labels_present = hd_enabled and any(key.startswith("hd_") for key in batch)
        # v2 compact action-effect labels are optional.  Their presence, not
        # the HD switch alone, enables the extra differentiable true/wrong
        # writer replays below; legacy HD artifacts therefore retain exactly
        # the historical compute/loss path.
        effect_labels_present = bool(
            hd_enabled
            and float(getattr(self.config, "hd_effect_weight", 0.0)) > 0.0
            and batch.get("hd_teacher_effect") is not None
            and (
                batch.get("hd_effect_write_gate") is not None
                or batch.get("hd_counterfactual_write_gate") is not None
            )
        )
        if effect_labels_present and not bool(getattr(self.config, "ttt_second_order", True)):
            raise ValueError(
                "HD v2 action-effect distillation requires ttt_second_order=true: "
                "the writer-connected effect gradient is a meta-gradient through the inner update. "
                "Set hd_effect_weight=0 for a first-order/no-effect ablation."
            )
        # A hindsight collector may store the exact flow phase/noise used by
        # its causal teacher.  Reusing them makes HCA distillation phase
        # matched; ordinary TTT batches continue to sample fresh values.
        if noise is None and hd_labels_present and batch.get("hd_noise") is not None:
            labeled_noise = self._reshape_hd_field(
                batch["hd_noise"], sequence_shape, name="hd_noise"
            )
            labeled_noise = labeled_noise.to(device=actions.device, dtype=actions.dtype)
            # HD fields are grouped as ``[B,T,chunk,D]`` after sequence
            # collation, while the action expert consumes flattened
            # ``[B*T,chunk,D]`` tensors.  Flatten the sequence axes before the
            # rank/prefix checks below; otherwise a perfectly valid per-frame
            # phase label is rejected as rank four versus action rank three.
            if labeled_noise.ndim >= 2 and labeled_noise.shape[:2] == (batch_size, sequence_length):
                labeled_noise = labeled_noise.reshape(
                    expected_flat_batch, *labeled_noise.shape[2:]
                )
            if labeled_noise.ndim == actions.ndim - 1:
                labeled_noise = labeled_noise.unsqueeze(-2)
            if labeled_noise.ndim != actions.ndim:
                raise ValueError(
                    f"hd_noise shape {tuple(labeled_noise.shape)} must match action rank {actions.ndim}"
                )
            if labeled_noise.shape[:-1] != actions.shape[:-1]:
                try:
                    labeled_noise = torch.broadcast_to(
                        labeled_noise, actions.shape[:-1] + (labeled_noise.shape[-1],)
                    )
                except RuntimeError as exc:
                    raise ValueError(
                        f"hd_noise prefix {tuple(labeled_noise.shape[:-1])} does not match "
                        f"actions {tuple(actions.shape[:-1])}"
                    ) from exc
            if labeled_noise.shape[-1] < actions.shape[-1]:
                labeled_noise = F.pad(labeled_noise, (0, actions.shape[-1] - labeled_noise.shape[-1]))
            elif labeled_noise.shape[-1] > actions.shape[-1]:
                labeled_noise = labeled_noise[..., : actions.shape[-1]]
            noise = labeled_noise.reshape_as(actions)
        if noise is None:
            noise = self.model.sample_noise(actions.shape, actions.device)
        if time is None and hd_labels_present and batch.get("hd_time") is not None:
            labeled_time = self._reshape_hd_field(
                batch["hd_time"], sequence_shape, name="hd_time"
            ).to(device=actions.device, dtype=torch.float32)
            labeled_time = labeled_time.reshape(-1)
            if labeled_time.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"hd_time has {labeled_time.shape[0]} values but flattened action batch has {actions.shape[0]}"
                )
            if hd_enabled and getattr(self.config, "hd_phase_mode", "random") == "deployment":
                # Do not silently trust a random-phase artifact under a
                # deployment-causal configuration.  The explicit check makes
                # phase matching a reproducible contract rather than a comment
                # in the label-generation command.
                if not torch.allclose(labeled_time, torch.ones_like(labeled_time), atol=1e-6, rtol=0):
                    raise ValueError(
                        "hd_phase_mode='deployment' requires hd_time=1 labels; "
                        "regenerate the artifact with --phase-mode deployment"
                    )
                time = torch.ones_like(labeled_time)
            else:
                time = labeled_time
        if time is None:
            if hd_enabled and getattr(self.config, "hd_phase_mode", "random") == "deployment":
                # Match the first deployment denoise exactly: the interaction
                # written to fast weights is projected from pure Gaussian
                # action noise, not from the future expert action chunk.
                time = torch.ones(actions.shape[0], device=actions.device, dtype=torch.float32)
            else:
                time = self.model.sample_time(actions.shape[0], actions.device)

        hd_write_gate = self._reshape_hd_field(
            batch.get("hd_write_gate"), sequence_shape, name="hd_write_gate"
        ) if hd_labels_present else None
        if hd_write_gate is not None:
            hd_write_gate = hd_write_gate.clamp(0, 1)
        learned_write_gate = bool(
            hd_enabled and getattr(self.config, "hd_learned_write_gate", False)
        )
        if learned_write_gate and hd_labels_present and hd_write_gate is None:
            raise ValueError(
                "HD learned-gate training requires the frame-aligned 'hd_write_gate' label"
            )
        initial_fast_states = self._clone_fast_states(fast_states) if hd_labels_present else None

        # The ordinary main state is carried by the TBPTT trainer.  Grounding
        # needs two additional *independent* numerical trajectories so a
        # zero-write intervention in an early segment remains in effect for
        # every later segment.  The trainer owns the container lifetime; this
        # function only updates its detached values in place.
        if grounding_states is not None:
            grounding_states.setdefault("true", None)
            grounding_states.setdefault("wrong", None)
            if effect_labels_present:
                grounding_states.setdefault("effect_true", None)
                grounding_states.setdefault("effect_wrong", None)
        if hd_enabled:
            # In the learned-gate variant hindsight ``u_i`` is a target, not
            # an online input.  The same is true for the v2 no-gate path: its
            # main writer is deliberately all-write so training and
            # deployment have identical update semantics.  Only the legacy
            # no-effect path retains the direct label override for backwards
            # compatibility.
            # v2's minimal paper path has no online gate head: all writes are
            # used during training exactly as they are at deployment, while
            # hindsight ``hd_write_gate`` only reweights the local H2L loss.
            # The direct label override remains solely for legacy/no-effect
            # compatibility, where it is an explicit training ablation.
            writer_gate_override = (
                None
                if learned_write_gate or effect_labels_present
                else hd_write_gate
            )
            forward_result = self.model.forward_with_state(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                actions,
                noise,
                time,
                sequence_shape=sequence_shape,
                fast_states=fast_states,
                write_gate=writer_gate_override,
                return_velocity=True,
                return_local_loss=hd_labels_present,
                use_learned_write_gate=learned_write_gate,
                return_write_gate=learned_write_gate,
            )
            if hd_labels_present and learned_write_gate:
                student_velocity, fast_states, local_ttt_loss, predicted_write_gate = forward_result
            elif hd_labels_present:
                student_velocity, fast_states, local_ttt_loss = forward_result
                predicted_write_gate = None
            elif learned_write_gate:
                student_velocity, fast_states, predicted_write_gate = forward_result
                local_ttt_loss = None
            else:
                student_velocity, fast_states = forward_result
                local_ttt_loss = None
                predicted_write_gate = None
            flow_target = noise - actions
            losses = F.mse_loss(flow_target, student_velocity, reduction="none")
            wrong_student_velocity = None
            grounding_student_velocity = None
            effect_student_true = None
            effect_student_wrong = None

            if effect_labels_present:
                # H2L content/effect replay.  The existing ``grounding``
                # branches below deliberately use ``detach_writer=True`` and
                # train only the reader.  These two branches are separate:
                # their fast-weight updates keep the writer graph connected
                # and use a second-order inner update so the action-effect
                # loss can change what is written.  Numerical states are
                # carried across TBPTT segments through the same mutable
                # container as grounding, but detached at the boundary.
                effect_state_store = grounding_states if grounding_states is not None else {}
                # The main v2 branch is already the exact all-write replay
                # used as the ``true`` intervention (same phase, prefix
                # writer, and second-order inner update).  Reusing it avoids a
                # third full VLM forward, removes stochastic disagreement
                # between two nominally identical true branches, and leaves
                # the writer-connected graph intact for the effect loss.
                # ``fast_states`` is copied into the replay store only as a
                # detached numerical carry; the main branch itself remains
                # available to the flow/HCA/H2L terms below.
                effect_student_true = student_velocity
                effect_true_next_states = fast_states
                effect_state_store["effect_true"] = {
                    layer_index: effect_state.detach(requires_grad=False)
                    for layer_index, effect_state in effect_true_next_states.items()
                }
                effect_gate = self._reshape_hd_field(
                    batch.get("hd_effect_write_gate"),
                    sequence_shape,
                    name="hd_effect_write_gate",
                )
                if effect_gate is None:
                    effect_gate = self._reshape_hd_field(
                        batch.get("hd_counterfactual_write_gate"),
                        sequence_shape,
                        name="hd_counterfactual_write_gate",
                    )
                if effect_gate is None:
                    effect_gate = torch.ones(
                        batch_size,
                        sequence_length,
                        device=actions.device,
                        dtype=actions.dtype,
                    )
                elif effect_gate.ndim >= 3:
                    # Compact v2 labels use [B,T,K]; branch zero is the
                    # selected event.  Keep this explicit rather than reducing
                    # over K, which would define a different intervention.
                    effect_gate = effect_gate[..., 0]
                effect_gate = effect_gate.to(
                    device=actions.device,
                    dtype=actions.dtype,
                ).clamp(0, 1)
                effect_wrong_source = effect_state_store.get("effect_wrong")
                if effect_wrong_source is None:
                    effect_wrong_source = initial_fast_states
                effect_wrong_initial = self._clone_fast_states(
                    effect_wrong_source,
                    detach=True,
                    requires_grad=False,
                )
                effect_student_wrong, effect_wrong_next_states = self.model.forward_with_state(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    actions,
                    noise,
                    time,
                    sequence_shape=sequence_shape,
                    fast_states=effect_wrong_initial,
                    create_graph=True,
                    write_gate=effect_gate,
                    detach_writer=False,
                    return_velocity=True,
                    use_learned_write_gate=False,
                )
                effect_state_store["effect_wrong"] = {
                    layer_index: effect_state.detach(requires_grad=False)
                    for layer_index, effect_state in effect_wrong_next_states.items()
                }
            wrong_gate = self._reshape_hd_field(
                batch.get("hd_counterfactual_write_gate"),
                sequence_shape,
                name="hd_counterfactual_write_gate",
            )
            if wrong_gate is not None:
                wrong_gate = wrong_gate.clamp(0, 1)
            # v2 action-effect replay already supplies the true/wrong
            # intervention with a writer-connected graph.  Running the old
            # detached grounding pair on top would duplicate the same
            # counterfactual, add two full forwards, and introduce a second
            # reader-only objective whose relative weight is a tuning knob.
            # Keep detached grounding for legacy/no-effect artifacts and as a
            # clearly isolated ablation.
            has_grounding_labels = (
                not effect_labels_present
                and float(getattr(self.config, "hd_grounding_weight", 1.0)) > 0.0
                and wrong_gate is not None
                and batch.get("hd_teacher_true_velocity") is not None
                and batch.get("hd_teacher_wrong_velocity") is not None
            )
            if has_grounding_labels:
                # Re-run both memory branches with writer/inner-update graphs
                # detached.  The numerical updates are identical, but the
                # resulting grounding loss can only train query/readout/action
                # pathways.  Each branch receives an independent copy of its
                # own prior replay state (or the common pre-segment state on
                # the first segment), and neither replay mutates the
                # persistent main ``fast_states`` returned above.
                true_branch_source = (
                    grounding_states.get("true")
                    if grounding_states is not None and grounding_states.get("true") is not None
                    else initial_fast_states
                )
                grounding_initial_states = self._clone_fast_states(
                    true_branch_source,
                    detach=True,
                    requires_grad=False,
                )
                grounding_student_velocity, true_branch_next_states = self.model.forward_with_state(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    actions,
                    noise,
                    time,
                    sequence_shape=sequence_shape,
                    fast_states=grounding_initial_states,
                    create_graph=False,
                    # The hindsight teacher's true branch is the ordinary
                    # all-write replay.  Do not accidentally apply the
                    # learned/label gate here: grounding should train the
                    # reader/action path, not compare a gated student branch
                    # against an all-write teacher target.
                    write_gate=None,
                    detach_writer=True,
                    return_velocity=True,
                    use_learned_write_gate=False,
                )
                if grounding_states is not None:
                    # ``detach_writer=True`` already cuts the outer graph;
                    # clone the returned state once more to avoid aliasing
                    # temporary replay tensors across TBPTT segments.
                    grounding_states["true"] = {
                        layer_index: state.detach(requires_grad=False)
                        for layer_index, state in true_branch_next_states.items()
                    }
                wrong_branch_source = (
                    grounding_states.get("wrong")
                    if grounding_states is not None and grounding_states.get("wrong") is not None
                    else initial_fast_states
                )
                wrong_initial_states = self._clone_fast_states(
                    wrong_branch_source,
                    detach=True,
                    requires_grad=False,
                )
                wrong_student_velocity, wrong_branch_next_states = self.model.forward_with_state(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    actions,
                    noise,
                    time,
                    sequence_shape=sequence_shape,
                    fast_states=wrong_initial_states,
                    create_graph=False,
                    write_gate=wrong_gate,
                    detach_writer=True,
                    return_velocity=True,
                    # ``wrong_gate`` is the explicit offline intervention;
                    # multiplying it by the learned online gate would create
                    # a different counterfactual from the teacher artifact.
                    use_learned_write_gate=False,
                )
                if grounding_states is not None:
                    grounding_states["wrong"] = {
                        layer_index: state.detach(requires_grad=False)
                        for layer_index, state in wrong_branch_next_states.items()
                    }
            hd_aux_loss, hd_metrics = self._hd_auxiliary_losses(
                batch,
                sequence_shape,
                student_velocity=student_velocity,
                wrong_student_velocity=wrong_student_velocity,
                grounding_student_velocity=grounding_student_velocity,
                effect_student_true=effect_student_true,
                effect_student_wrong=effect_student_wrong,
                local_ttt_loss=local_ttt_loss,
                predicted_write_gate=predicted_write_gate,
                normalization_denominator=hd_normalization_denominator,
            )
        else:
            losses, fast_states = self.model.forward_with_state(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                actions,
                noise,
                time,
                sequence_shape=sequence_shape,
                fast_states=fast_states,
            )
            hd_aux_loss = losses.new_zeros(())
            hd_metrics = {}
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            valid = (~actions_is_pad).unsqueeze(-1)
            losses = losses * valid
            valid_steps = valid.sum().clamp_min(1)
            loss_per_dim = losses.sum(dim=(0, 1)) / valid_steps
        else:
            loss_per_dim = losses.mean(dim=(0, 1))

        loss_dict = {"loss_per_dim": loss_per_dim.detach().cpu().tolist()}
        loss_dict.update({key: value for key, value in hd_metrics.items()})
        # Compute the flow term before adding HD auxiliaries.  The sequence
        # trainer may pass a global action-valid segment weight for TBPTT; HD
        # terms already use their own episode-level denominator and must not
        # be multiplied by that action weight (warm-up/effect rows would then
        # disappear whenever their action target is padded).
        if actions_is_pad is None:
            flow_loss = losses.mean()
        else:
            num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
            flow_loss = losses.sum() / num_valid
        if flow_loss_weight is not None:
            flow_weight = torch.as_tensor(
                flow_loss_weight,
                device=flow_loss.device,
                dtype=flow_loss.dtype,
            )
            if flow_weight.numel() != 1:
                raise ValueError("flow_loss_weight must be scalar")
            flow_loss = flow_loss * flow_weight.reshape(())

        if hd_enabled:
            # These values are diagnostics only.  In the v2 TBPTT path the
            # trainer recomputes the two balance metrics after summing all
            # segment numerators, so the reported ratio is invariant to the
            # chosen segment length.  A direct one-segment call (including
            # legacy/non-TBPTT training) receives the local balance instead.
            loss_dict["hd_auxiliary_loss"] = float(hd_aux_loss.detach().item())
            loss_dict["hd_flow_loss"] = float(flow_loss.detach().item())
            loss_dict.update(
                _hd_loss_balance_metrics(
                    hd_aux_loss.detach().item(),
                    flow_loss.detach().item(),
                )
            )
            loss_dict.update(
                _hd_ttt_parameter_range_metrics(
                    getattr(self.model, "ttt_layers", None),
                )
            )

        if reduction == "none":
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            per_sample_loss = per_sample_loss + hd_aux_loss
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict, fast_states
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")

        loss = flow_loss + hd_aux_loss
        loss_dict["loss"] = loss.item()
        return loss, loss_dict, fast_states

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        if SEQUENCE_SHAPE_KEY not in batch:
            raise ValueError(
                f"smolvla_ttt training batches must contain {SEQUENCE_SHAPE_KEY!r}; "
                "use TailPreservingSequenceDataset and sequence_collate_fn"
            )
        sequence_shape = tuple(int(value) for value in batch[SEQUENCE_SHAPE_KEY])
        loss, loss_dict, _ = self.forward_sequence_segment(
            batch,
            sequence_shape=sequence_shape,
            reduction=reduction,
            noise=noise,
            time=time,
        )
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            # LeRobot's offline loader can intentionally return uint8 images
            # (the training loop normally converts them before preprocessing),
            # while direct evaluation/label collection may call the policy
            # without that loop.  Normalize here as a safe boundary so byte
            # tensors never reach bilinear interpolation or the [-1, 1] cast.
            if img.dtype == torch.uint8:
                img = img.to(dtype=torch.float32) / 255.0
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class SmolVLATTTFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLATTTConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertTTTModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        # Prefix-only HD writes cross the VLM/action-expert representation
        # boundary.  A shared learned adapter is preferable to treating
        # feature channels as a spatial axis (which would make the mapping
        # depend on arbitrary hidden-dimension ordering).  Register the
        # adapter only for that structural mode: suffix checkpoints then keep
        # the exact original state-dict schema and do not acquire an unused
        # random parameter family.
        if getattr(config, "ttt_writer_mode", "suffix") == "prefix_only":
            self.prefix_writer_proj: nn.Linear | None = nn.Linear(
                self.vlm_with_expert.config.text_config.hidden_size,
                self.vlm_with_expert.expert_hidden_size,
                bias=False,
            )
            nn.init.xavier_uniform_(self.prefix_writer_proj.weight)
        else:
            self.prefix_writer_proj = None
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )
        if config.ttt_num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.empty(config.ttt_num_register_tokens, self.vlm_with_expert.expert_hidden_size)
            )
            nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        else:
            self.register_tokens = None

        model_layers = self.vlm_with_expert.get_model_layers(
            [self.vlm_with_expert.get_vlm_model().text_model, self.vlm_with_expert.lm_expert]
        )
        invalid_layers = [
            layer_index
            for layer_index in config.resolved_ttt_layer_indices
            if layer_index >= len(model_layers[1]) or model_layers[1][layer_index] is None
        ]
        if invalid_layers:
            raise ValueError(f"No SmolVLA action-expert layer exists at TTT indices {invalid_layers}")
        self.write_gate_layer_index = min(config.resolved_ttt_layer_indices)
        self.ttt_layers = nn.ModuleDict(
            {
                str(layer_index): TTTMLPLayer(
                    dim=self.vlm_with_expert.expert_hidden_size,
                    hidden_dim=config.ttt_hidden_dim,
                    base_inner_lr=config.ttt_base_inner_lr,
                    effective_gate_init=config.ttt_effective_gate_init,
                    gate_trainable=config.trains_gate,
                    rope_theta=config.ttt_rope_theta,
                    second_order=config.ttt_second_order,
                    # One scalar gate is shared by all selected TTT layers for
                    # each physical interaction.  Predict it at the first
                    # selected layer so later layers cannot disagree about
                    # what was written to the recurrent state.
                    learned_write_gate=(
                        config.hd_ttt_enabled
                        and config.hd_learned_write_gate
                        and layer_index == self.write_gate_layer_index
                    ),
                    write_gate_init=config.hd_write_gate_init,
                    write_gate_token_index=config.ttt_num_register_tokens,
                    write_gate_context_dim=(
                        self.vlm_with_expert.config.text_config.hidden_size
                        if (
                            config.hd_ttt_enabled
                            and config.hd_learned_write_gate
                            and layer_index == self.write_gate_layer_index
                        )
                        else None
                    ),
                )
                for layer_index in config.resolved_ttt_layer_indices
            }
        )
        self._configure_trainable_parameters()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _configure_trainable_parameters(self) -> None:
        self.requires_grad_(False)
        for parameter in self.ttt_layers.parameters():
            parameter.requires_grad_(True)
        if self.register_tokens is not None:
            self.register_tokens.requires_grad_(True)
        if (
            getattr(self.config, "ttt_writer_mode", "suffix") == "prefix_only"
            and self.prefix_writer_proj is not None
        ):
            self.prefix_writer_proj.requires_grad_(True)
        for layer in self.ttt_layers.values():
            layer.gate.requires_grad_(self.config.trains_gate)

        if self.config.trains_action_head:
            for module in (
                self.vlm_with_expert.lm_expert,
                self.state_proj,
                self.action_in_proj,
                self.action_out_proj,
                self.action_time_mlp_in,
                self.action_time_mlp_out,
            ):
                module.requires_grad_(True)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.vlm_with_expert.vlm.eval()
        self.vlm_with_expert.lm_expert.train(mode and self.config.trains_action_head)
        for module in (
            self.state_proj,
            self.action_in_proj,
            self.action_out_proj,
            self.action_time_mlp_in,
            self.action_time_mlp_out,
        ):
            module.train(mode and self.config.trains_action_head)
        if self.prefix_writer_proj is not None:
            self.prefix_writer_proj.train(
                mode and getattr(self.config, "ttt_writer_mode", "suffix") == "prefix_only"
            )
        self.ttt_layers.train(mode)
        return self

    def _make_expert_layer_callback(
        self,
        sequence_shape: tuple[int, int],
        fast_states: TTTFastStates,
        *,
        update: bool,
        create_graph: bool | None,
        write_gate: Tensor | None = None,
        detach_writer: bool = False,
        return_local_loss: bool = False,
        local_loss_accumulator: list[Tensor] | None = None,
        use_learned_write_gate: bool = False,
        write_gate_accumulator: list[Tensor] | None = None,
    ):
        """Build an expert callback, optionally collecting H2L writer losses.

        ``return_local_loss`` is opt-in so every existing callback invocation
        keeps its original output/state API.  When enabled, each selected TTT
        layer appends a ``[B,T]`` raw inner K/V loss to
        ``local_loss_accumulator``; :meth:`forward_with_state` averages these
        layer-wise losses before returning them.
        """
        batch_size, sequence_length = sequence_shape
        writer_mode = getattr(getattr(self, "config", None), "ttt_writer_mode", "suffix")

        # The learned gate is deliberately shared by all selected layers.  A
        # closure-local cache ensures the first selected layer computes one
        # scalar per physical interaction and every later layer reuses it.
        predicted_write_gate: Tensor | None = None
        gate_context: Tensor | None = None
        writer_inputs: Tensor | None = None
        writer_mask: Tensor | None = None

        def set_gate_context(context: Tensor) -> None:
            nonlocal gate_context
            gate_context = context

        def set_writer_inputs(inputs: Tensor, mask: Tensor | None = None) -> None:
            nonlocal writer_inputs, writer_mask
            if inputs.ndim != 3:
                raise ValueError(
                    "prefix writer inputs must be flattened [B*T,N,D], "
                    f"got {tuple(inputs.shape)}"
                )
            if inputs.shape[0] != batch_size * sequence_length:
                raise ValueError(
                    "prefix writer inputs must have the flattened sequence batch size "
                    f"{batch_size * sequence_length}, got {inputs.shape[0]}"
                )
            if mask is not None:
                if mask.ndim != 2 or mask.shape != inputs.shape[:2]:
                    raise ValueError(
                        "prefix writer mask must have shape [B*T,N] matching writer inputs, "
                        f"got {tuple(mask.shape)} for {tuple(inputs.shape)}"
                    )
                writer_mask = mask
            else:
                writer_mask = None
            writer_inputs = inputs

        def apply_ttt(layer_index: int, hidden_states: Tensor) -> Tensor:
            nonlocal predicted_write_gate
            layer_key = str(layer_index)
            if layer_key not in self.ttt_layers:
                return hidden_states
            if hidden_states.shape[0] != batch_size * sequence_length:
                raise ValueError(
                    f"TTT expected a flattened batch of {batch_size}*{sequence_length}, "
                    f"got {hidden_states.shape[0]}"
                )
            sequence = hidden_states.reshape(
                batch_size, sequence_length, hidden_states.shape[1], hidden_states.shape[2]
            )
            layer_write_gate = write_gate
            if use_learned_write_gate and update:
                if predicted_write_gate is None:
                    if layer_index != self.write_gate_layer_index:
                        raise RuntimeError(
                            "The shared HD write gate must be predicted at the first selected TTT layer"
                        )
                    context_sequence = None
                    if gate_context is not None:
                        if gate_context.shape[0] != batch_size * sequence_length:
                            raise ValueError(
                                "prefix gate context must have the flattened sequence batch size "
                                f"{batch_size * sequence_length}, got {gate_context.shape[0]}"
                            )
                        context_sequence = gate_context.reshape(
                            batch_size,
                            sequence_length,
                            gate_context.shape[-1],
                        )
                    predicted_write_gate = self.ttt_layers[layer_key].predict_write_gate(
                        sequence,
                        context=context_sequence,
                    )
                    if write_gate_accumulator is not None:
                        write_gate_accumulator.append(predicted_write_gate)
                learned_gate = predicted_write_gate.detach() if detach_writer else predicted_write_gate
                layer_write_gate = learned_gate if write_gate is None else learned_gate * write_gate
            if layer_write_gate is not None:
                if layer_write_gate.shape != (batch_size, sequence_length):
                    raise ValueError(
                        "write_gate must match the callback sequence shape "
                        f"{(batch_size, sequence_length)}, got {tuple(layer_write_gate.shape)}"
                    )
            if update and writer_mode == "prefix_only" and writer_inputs is None:
                raise RuntimeError(
                    "ttt_writer_mode='prefix_only' requires set_writer_inputs() before the expert callback"
                )
            layer_output = self.ttt_layers[layer_key](
                sequence,
                fast_states.get(layer_index),
                writer_inputs=(
                    None
                    if writer_mode == "suffix"
                    else (
                        writer_inputs.reshape(
                            batch_size,
                            sequence_length,
                            writer_inputs.shape[1],
                            writer_inputs.shape[2],
                        )
                        if writer_inputs is not None
                        else None
                    )
                ),
                writer_mask=(
                    None
                    if writer_mask is None
                    else writer_mask.reshape(batch_size, sequence_length, writer_mask.shape[1])
                ),
                update=update,
                create_graph=create_graph,
                write_gate=layer_write_gate,
                detach_writer=detach_writer,
                return_local_loss=return_local_loss,
            )
            if return_local_loss:
                sequence, next_state, local_loss = layer_output
                if local_loss_accumulator is not None:
                    local_loss_accumulator.append(local_loss)
            else:
                sequence, next_state = layer_output
            fast_states[layer_index] = next_state
            return sequence.reshape_as(hidden_states)

        # ``FlowMatching.forward``/``sample_actions`` install the current
        # observation-only prefix summary through this setter before the first
        # expert layer executes.  Keeping it as an attribute preserves the
        # existing two-argument callback API used by the sibling model/tests.
        apply_ttt.set_gate_context = set_gate_context
        apply_ttt.set_writer_inputs = set_writer_inputs
        return apply_ttt

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def _make_prefix_writer_inputs(
        self, prefix_embs: Tensor, prefix_pad_masks: Tensor
    ) -> Tensor:
        """Build deterministic observation-prefix inputs for the K/V writer.

        The VLM prefix and action expert have different widths.  The shared
        learned adapter maps the semantic VLM embedding into the expert TTT
        space; padding tokens are returned separately to the callback so they
        cannot update fast weights.  The returned tensor is flattened like
        expert hidden states and is installed once per physical observation.
        """
        if prefix_embs.ndim != 3 or prefix_pad_masks.ndim != 2:
            raise ValueError(
                "prefix writer inputs expect [B,P,D] embeddings and [B,P] padding mask, "
                f"got {tuple(prefix_embs.shape)} and {tuple(prefix_pad_masks.shape)}"
            )
        if prefix_embs.shape[:2] != prefix_pad_masks.shape:
            raise ValueError(
                "prefix embeddings and padding mask must share [B,P], got "
                f"{tuple(prefix_embs.shape)} and {tuple(prefix_pad_masks.shape)}"
            )
        projection = getattr(self, "prefix_writer_proj", None)
        if projection is None:
            # Low-level callers that construct a partially initialized flow
            # object (and old suffix-only ablations) have no learned adapter.
            # Keep a deterministic shape-compatible fallback for that test
            # boundary; production prefix-only models always take the learned
            # semantic projection above.
            expert_dim = int(self.vlm_with_expert.expert_hidden_size)
            writer = F.adaptive_avg_pool1d(
                prefix_embs.reshape(-1, 1, prefix_embs.shape[-1]), expert_dim
            ).reshape(prefix_embs.shape[0], prefix_embs.shape[1], expert_dim)
        else:
            writer = projection(prefix_embs.to(projection.weight.dtype))
        # Zeroing is redundant with ``writer_mask`` in the normal callback,
        # but makes this helper safe for standalone callers and guarantees a
        # padded prefix can never inject a bias-free K/V signal by accident.
        return writer * prefix_pad_masks.to(device=writer.device, dtype=writer.dtype).unsqueeze(-1)

    def _prefix_writer_inputs_with_registers(
        self, prefix_embs: Tensor, prefix_pad_masks: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Add static register anchors to the observation-only writer stream.

        Prefix-only HD writing intentionally excludes the current noisy action
        and timestep.  Without this explicit path, the prepended registers
        would be invisible to the prefix writer (and action queries are
        forbidden from reading register columns by the asymmetric suffix
        mask), making them dead parameters in the paper configuration.  The
        anchors are learned expert-width vectors, valid for every observation,
        and are concatenated after the projected causal prefix.  They do not
        reintroduce action/noise dependence; suffix mode keeps its original
        register-as-writer behavior unchanged.
        """

        writer = self._make_prefix_writer_inputs(prefix_embs, prefix_pad_masks)
        register_tokens = getattr(self, "register_tokens", None)
        if register_tokens is None:
            return writer, prefix_pad_masks
        register = register_tokens.to(device=writer.device, dtype=writer.dtype)
        register = register.unsqueeze(0).expand(writer.shape[0], -1, -1)
        register_mask = torch.ones(
            writer.shape[0],
            register.shape[1],
            dtype=prefix_pad_masks.dtype,
            device=prefix_pad_masks.device,
        )
        return torch.cat([register, writer], dim=1), torch.cat(
            [register_mask, prefix_pad_masks], dim=1
        )

    def embed_suffix(self, noisy_actions, timestep):
        """Embed action/time tokens and optional learned registers for the expert."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        if self.register_tokens is not None:
            register_embs = self.register_tokens.to(device=device, dtype=dtype)
            register_embs = register_embs.unsqueeze(0).expand(bsize, -1, -1)
            embs.append(register_embs)
            register_mask = torch.ones(
                bsize,
                self.config.ttt_num_register_tokens,
                dtype=torch.bool,
                device=device,
            )
            pad_masks.append(register_mask)
            # Registers are prepended in one shared segment. The 2-D suffix-mask
            # helper lets register queries read every action token, without
            # exposing registers as direct keys to action queries.
            att_masks += [1] + [0] * (self.config.ttt_num_register_tokens - 1)

        # Keep the original causal action-action attention pattern. The
        # register-aware 2-D mask is deliberately asymmetric (registers read
        # actions; actions do not read registers).
        embs.append(action_time_emb)
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def _make_suffix_att_2d_masks(self, pad_masks: Tensor, att_masks: Tensor) -> Tensor:
        """Build one-way register attention for the expert suffix.

        Registers are a readout workspace: they may read the complete current
        action block (and one another), while action queries retain the original
        causal action-to-action mask and cannot directly read registers. This is
        intentionally asymmetric; otherwise a register would become an
        unidentifiable second action pathway.
        """
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        num_register_tokens = self.config.ttt_num_register_tokens
        if num_register_tokens == 0:
            return att_2d_masks

        expected_suffix_length = num_register_tokens + self.config.chunk_size
        if pad_masks.shape[1] != expected_suffix_length:
            raise ValueError(
                f"Expected {expected_suffix_length} suffix tokens, got {pad_masks.shape[1]}"
            )
        register_queries_are_valid = pad_masks[:, :num_register_tokens, None]
        all_suffix_keys_are_valid = pad_masks[:, None, :]
        att_2d_masks[:, :num_register_tokens, :] = register_queries_are_valid & all_suffix_keys_are_valid

        # Remove the register columns from action queries, then restore the
        # causal action-only triangle. Padding remains respected for both axes.
        action_pad = pad_masks[:, num_register_tokens:]
        action_causal = torch.tril(
            torch.ones(
                self.config.chunk_size,
                self.config.chunk_size,
                dtype=torch.bool,
                device=pad_masks.device,
            )
        )[None, :, :]
        att_2d_masks[:, num_register_tokens:, :] = False
        att_2d_masks[:, num_register_tokens:, num_register_tokens:] = (
            action_pad[:, :, None] & action_pad[:, None, :] & action_causal
        )
        return att_2d_masks

    def _select_action_tokens(self, suffix_output: Tensor) -> Tensor:
        """Select action tokens after the prepended register-token block."""
        action_start = self.config.ttt_num_register_tokens
        action_end = action_start + self.config.chunk_size
        if suffix_output.shape[1] < action_end:
            raise ValueError(
                f"Expected at least {action_end} expert tokens, got {suffix_output.shape[1]}"
            )
        return suffix_output[:, action_start:action_end]

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        *,
        expert_layer_callback=None,
        return_velocity: bool = False,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        need_gate_context = bool(
            expert_layer_callback is not None
            and getattr(self.config, "hd_ttt_enabled", False)
            and getattr(self.config, "hd_learned_write_gate", False)
        )
        if need_gate_context:
            prefix_weights = prefix_pad_masks.to(dtype=prefix_embs.dtype).unsqueeze(-1)
            prefix_context = (prefix_embs * prefix_weights).sum(dim=1) / prefix_weights.sum(
                dim=1
            ).clamp_min(1.0)
        else:
            prefix_context = None
        if expert_layer_callback is not None:
            set_gate_context = getattr(expert_layer_callback, "set_gate_context", None)
            if set_gate_context is not None and prefix_context is not None:
                # Prefix tokens contain only the current observation, language
                # instruction, and proprioceptive state.  Pooling them before
                # the suffix is a strict causal gate context: no action chunk
                # or denoising timestep can leak into the write decision.
                set_gate_context(prefix_context)
            if getattr(self.config, "ttt_writer_mode", "suffix") == "prefix_only":
                set_writer_inputs = getattr(expert_layer_callback, "set_writer_inputs", None)
                if set_writer_inputs is None:
                    raise RuntimeError("prefix_only writer requires a callback writer-input setter")
                writer_inputs, writer_mask = self._prefix_writer_inputs_with_registers(
                    prefix_embs, prefix_pad_masks
                )
                set_writer_inputs(writer_inputs, writer_mask)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        suffix_length = suffix_pad_masks.shape[1]
        att_2d_masks[:, -suffix_length:, -suffix_length:] = self._make_suffix_att_2d_masks(
            suffix_pad_masks, suffix_att_masks
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            expert_layer_callback=expert_layer_callback,
        )
        suffix_out = self._select_action_tokens(suffix_out)
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if return_velocity:
            return v_t
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def forward_with_state(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise,
        time,
        *,
        sequence_shape: tuple[int, int],
        fast_states: TTTFastStates | None = None,
        create_graph: bool | None = None,
        write_gate: Tensor | None = None,
        detach_writer: bool = False,
        return_velocity: bool = False,
        return_local_loss: bool = False,
        use_learned_write_gate: bool = False,
        return_write_gate: bool = False,
    ) -> (
        tuple[Tensor, TTTFastStates]
        | tuple[Tensor, TTTFastStates, Tensor]
        | tuple[Tensor, TTTFastStates, Tensor]
        | tuple[Tensor, TTTFastStates, Tensor, Tensor]
    ):
        """Run flow matching while optionally exposing the local H2L loss.

        The default two-item return is unchanged.  With
        ``return_local_loss=True``, a third ``[B,T]`` tensor is returned.  It
        is the mean raw inner K/V prediction loss over selected TTT layers,
        before any hindsight ``write_gate`` weighting.  This keeps the local
        objective available without requiring precomputed ``hd_local_*``
        tensors in the dataset.
        """
        fast_states = {} if fast_states is None else dict(fast_states)
        local_loss_parts: list[Tensor] | None = [] if return_local_loss else None
        write_gate_parts: list[Tensor] | None = [] if return_write_gate else None
        callback = self._make_expert_layer_callback(
            sequence_shape,
            fast_states,
            update=True,
            create_graph=create_graph,
            write_gate=write_gate,
            detach_writer=detach_writer,
            return_local_loss=return_local_loss,
            local_loss_accumulator=local_loss_parts,
            use_learned_write_gate=use_learned_write_gate,
            write_gate_accumulator=write_gate_parts,
        )
        losses = self.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            expert_layer_callback=callback,
            return_velocity=return_velocity,
        )
        local_loss = None
        if return_local_loss:
            if local_loss_parts:
                local_loss = torch.stack(local_loss_parts, dim=0).mean(dim=0)
            else:
                local_loss = losses.new_zeros(sequence_shape)
        predicted_gate = None
        if return_write_gate:
            if write_gate_parts:
                predicted_gate = torch.stack(write_gate_parts, dim=0).mean(dim=0)
            else:
                predicted_gate = losses.new_ones(sequence_shape)
        if return_local_loss and return_write_gate:
            return losses, fast_states, local_loss, predicted_gate
        if return_local_loss:
            return losses, fast_states, local_loss
        if return_write_gate:
            return losses, fast_states, predicted_gate
        return losses, fast_states

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        _expert_layer_callback_factory=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        need_gate_context = bool(
            _expert_layer_callback_factory is not None
            and getattr(self.config, "hd_ttt_enabled", False)
            and getattr(self.config, "hd_learned_write_gate", False)
        )
        if need_gate_context:
            prefix_weights = prefix_pad_masks.to(dtype=prefix_embs.dtype).unsqueeze(-1)
            prefix_context = (prefix_embs * prefix_weights).sum(dim=1) / prefix_weights.sum(
                dim=1
            ).clamp_min(1.0)
        else:
            prefix_context = None
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            expert_layer_callback = (
                _expert_layer_callback_factory(step == 0)
                if _expert_layer_callback_factory is not None
                else None
            )
            if expert_layer_callback is not None:
                set_gate_context = getattr(expert_layer_callback, "set_gate_context", None)
                if set_gate_context is not None and prefix_context is not None:
                    set_gate_context(prefix_context)
                if getattr(self.config, "ttt_writer_mode", "suffix") == "prefix_only":
                    set_writer_inputs = getattr(expert_layer_callback, "set_writer_inputs", None)
                    if set_writer_inputs is None:
                        raise RuntimeError("prefix_only writer requires a callback writer-input setter")
                    writer_inputs, writer_mask = self._prefix_writer_inputs_with_registers(
                        prefix_embs, prefix_pad_masks
                    )
                    set_writer_inputs(writer_inputs, writer_mask)

            def denoise_step_partial_call(
                input_x_t,
                current_timestep=time_tensor,
                current_expert_layer_callback=expert_layer_callback,
            ):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                    expert_layer_callback=current_expert_layer_callback,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    @torch.no_grad()
    def sample_actions_with_state(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        fast_states: TTTFastStates | None = None,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> tuple[Tensor, TTTFastStates]:
        """Denoise an action chunk while advancing TTT memory exactly once."""
        fast_states = {} if fast_states is None else dict(fast_states)
        sequence_shape = (state.shape[0], 1)

        def callback_factory(update: bool):
            return self._make_expert_layer_callback(
                sequence_shape,
                fast_states,
                update=update,
                create_graph=False,
                use_learned_write_gate=(
                    getattr(self.config, "hd_ttt_enabled", False)
                    and getattr(self.config, "hd_learned_write_gate", False)
                ),
            )

        actions = self.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            _expert_layer_callback_factory=callback_factory,
            **kwargs,
        )
        return actions, fast_states

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        expert_layer_callback=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = self._make_suffix_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            expert_layer_callback=expert_layer_callback,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = self._select_action_tokens(suffix_out)
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def predict_velocity(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: Tensor,
        timestep: Tensor,
        expert_layer_callback=None,
    ) -> Tensor:
        """Public raw flow-velocity interface used by HCA and interventions.

        Unlike :meth:`forward`, this returns the model velocity ``v_t`` before
        comparing it with an expert target.  It is therefore usable by a
        teacher replay when the target action is held fixed and only the
        historical fast-weight branch changes.
        """

        return self.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
            expert_layer_callback=expert_layer_callback,
        )
