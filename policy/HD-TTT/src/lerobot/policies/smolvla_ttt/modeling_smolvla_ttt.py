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
from .hd_ttt import counterfactual_grounding_loss, local_kvb_loss
from .sequence import SEQUENCE_SHAPE_KEY
from .smolvlm_with_expert_ttt import SmolVLMWithExpertTTTModel
from .ttt import TTTFastState, TTTMLPLayer

TTTFastStates = dict[int, TTTFastState]

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
    "ttt_num_register_tokens",
    "hd_ttt_enabled",
    "hd_hca_weight",
    "hd_h2l_weight",
    "hd_grounding_weight",
    "hd_invariance_weight",
    "hd_event_block_size",
    "hd_attribution_threshold",
    "hd_attribution_topk",
    "hd_counterfactual_margin",
}


def _restore_checkpoint_model_fields(
    config: SmolVLATTTConfig,
    source_config: SmolVLATTTConfig,
    raw_config: dict,
) -> None:
    """Restore every checkpoint-owned field that affects model structure or TTT behavior."""
    for field_name in _CHECKPOINT_ARCHITECTURE_FIELDS:
        if field_name in raw_config:
            setattr(config, field_name, getattr(source_config, field_name))
    config.__post_init__()


def _validate_checkpoint_keys(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    source_is_ttt: bool,
    strict: bool,
) -> None:
    """Allow new TTT tensors to be absent only when converting a base SmolVLA checkpoint."""
    allowed_base_missing = [
        key
        for key in missing_keys
        if key.startswith("model.ttt_layers.") or key == "model.register_tokens"
    ]
    disallowed_missing = [key for key in missing_keys if key not in allowed_base_missing]
    require_exact_checkpoint = source_is_ttt or strict
    if unexpected_keys or disallowed_missing or (require_exact_checkpoint and missing_keys):
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

        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is None:
            return None
        pad = SmolVLATTTPolicy._reshape_hd_field(
            action_is_pad, sequence_shape, name="action_is_pad"
        ).to(device=device)
        if pad.ndim <= 2:
            valid = (~pad.bool()).to(dtype=dtype)
        else:
            valid = (~pad.bool()).to(dtype=dtype).mean(dim=tuple(range(2, pad.ndim)))
        return valid

    @staticmethod
    def _hd_weighted_mean(
        values: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Average an HD loss with optional non-negative weights, safely."""

        if weights is None:
            return values.mean()
        weights = weights.to(device=values.device, dtype=values.dtype).clamp_min(0)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        weights = torch.broadcast_to(weights, values.shape)
        denominator = weights.sum()
        numerator = (values * weights).sum()
        safe_denominator = denominator.clamp_min(1e-8)
        return torch.where(
            denominator > 1e-8,
            numerator / safe_denominator,
            values.new_zeros(()),
        )

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
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute optional HD terms from training-only teacher/intervention labels.

        The ordinary LeRobot batch contains none of these keys, so the function
        is a strict no-op for base SmolVLA/TTT training.  A hindsight data pass
        may attach flattened labels under the documented ``hd_*`` names; all
        teacher tensors are detached here before they can influence gradients.
        Teacher/intervention velocities may use the task dimension (for example
        ``[B*T, 50, 7]`` for MIKASA) while the model internally emits padded
        ``[B*T, 50, 32]`` tensors.  Only the active task coordinates are compared;
        per-future C/rho matrices are reduced to safe ``[B,T]`` weights.
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
        counterfactual_margin = float(getattr(self.config, "hd_counterfactual_margin", 0.0))
        valid_steps = self._hd_valid_step_weight(
            batch,
            sequence_shape,
            device=student_velocity.device,
            dtype=student_velocity.dtype,
        )

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
            hca = self._hd_weighted_mean(per_step, attribution_weight)
            total = total + hca_weight * hca
            metrics["hd_hca"] = float(hca.detach().item())

        # The writer gate is supplied by the hindsight teacher.  The local K/V
        # objective itself can optionally be logged/optimized when a collector
        # stores the projected key/value/prediction tensors.
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
            kvb = local_kvb_loss(local_query, local_key, local_value, local_prediction, local_gate)
            total = total + h2l_weight * kvb
            metrics["hd_h2l"] = float(kvb.detach().item())

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
                rho_weight = self._hd_step_weight(
                    rho,
                    (B, T),
                    device=student_velocity.device,
                    dtype=student_velocity.dtype,
                    name="hd_rho",
                )
                if rho_weight is None:
                    rho_weight = student_velocity.new_ones((B, T))
                else:
                    # ``rho`` may be a raw column sum of C rather than a
                    # pre-normalized dependency label.  Grounding interprets
                    # it as a mixture coefficient in [0,1], so normalize per
                    # sequence before the tensor utility clamps/broadcasts it.
                    rho_max = rho_weight.amax(dim=-1, keepdim=True)
                    rho_weight = rho_weight / rho_max.clamp_min(1e-8)
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
                grounding_weights = valid_steps
                grounding = self._hd_weighted_mean(grounding_per_token, grounding_weights)
                total = total + grounding_weight * grounding
                metrics["hd_grounding"] = float(grounding.detach().item())

        return total, metrics

    def forward_sequence_segment(
        self,
        batch: dict[str, Tensor],
        sequence_shape: tuple[int, int],
        fast_states: TTTFastStates | None = None,
        reduction: str = "mean",
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict, TTTFastStates]:
        """Train one contiguous TBPTT segment and return its numerical fast state."""
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
        hd_fields_present = self.config.hd_ttt_enabled and any(
            key.startswith("hd_") for key in batch
        )
        # A hindsight collector may store the exact flow phase/noise used by
        # its causal teacher.  Reusing them makes HCA distillation phase
        # matched; ordinary TTT batches continue to sample fresh values.
        if noise is None and hd_fields_present and batch.get("hd_noise") is not None:
            labeled_noise = self._reshape_hd_field(
                batch["hd_noise"], sequence_shape, name="hd_noise"
            )
            labeled_noise = labeled_noise.to(device=actions.device, dtype=actions.dtype)
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
        if time is None and hd_fields_present and batch.get("hd_time") is not None:
            labeled_time = self._reshape_hd_field(
                batch["hd_time"], sequence_shape, name="hd_time"
            ).to(device=actions.device, dtype=torch.float32)
            time = labeled_time.reshape(-1)
            if time.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"hd_time has {time.shape[0]} values but flattened action batch has {actions.shape[0]}"
                )
        if time is None:
            time = self.model.sample_time(actions.shape[0], actions.device)

        hd_write_gate = self._reshape_hd_field(
            batch.get("hd_write_gate"), sequence_shape, name="hd_write_gate"
        ) if hd_fields_present else None
        if hd_write_gate is not None:
            hd_write_gate = hd_write_gate.clamp(0, 1)
        initial_fast_states = self._clone_fast_states(fast_states) if hd_fields_present else None
        if hd_fields_present:
            student_velocity, fast_states = self.model.forward_with_state(
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
                write_gate=hd_write_gate,
                return_velocity=True,
            )
            flow_target = noise - actions
            losses = F.mse_loss(flow_target, student_velocity, reduction="none")
            wrong_student_velocity = None
            grounding_student_velocity = None
            wrong_gate = self._reshape_hd_field(
                batch.get("hd_counterfactual_write_gate"),
                sequence_shape,
                name="hd_counterfactual_write_gate",
            )
            if wrong_gate is not None:
                wrong_gate = wrong_gate.clamp(0, 1)
            has_grounding_labels = (
                wrong_gate is not None
                and batch.get("hd_teacher_true_velocity") is not None
                and batch.get("hd_teacher_wrong_velocity") is not None
            )
            if has_grounding_labels:
                # Re-run both memory branches with writer/inner-update graphs
                # detached.  The numerical updates are identical, but the
                # resulting grounding loss can only train query/readout/action
                # pathways.  Each branch receives an independent copy of the
                # same pre-segment state, and neither replay mutates the
                # persistent ``fast_states`` returned above.
                grounding_initial_states = self._clone_fast_states(
                    initial_fast_states,
                    detach=True,
                    requires_grad=False,
                )
                grounding_student_velocity, _ = self.model.forward_with_state(
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
                    write_gate=hd_write_gate,
                    detach_writer=True,
                    return_velocity=True,
                )
                wrong_initial_states = self._clone_fast_states(
                    initial_fast_states,
                    detach=True,
                    requires_grad=False,
                )
                wrong_student_velocity, _ = self.model.forward_with_state(
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
                )
            hd_aux_loss, hd_metrics = self._hd_auxiliary_losses(
                batch,
                sequence_shape,
                student_velocity=student_velocity,
                wrong_student_velocity=wrong_student_velocity,
                grounding_student_velocity=grounding_student_velocity,
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

        if actions_is_pad is None:
            loss = losses.mean()
        else:
            num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
            loss = losses.sum() / num_valid
        loss = loss + hd_aux_loss
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
    ):
        """Build an expert callback, optionally in reader-only replay mode."""
        batch_size, sequence_length = sequence_shape

        def apply_ttt(layer_index: int, hidden_states: Tensor) -> Tensor:
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
            if layer_write_gate is not None:
                if layer_write_gate.shape != (batch_size, sequence_length):
                    raise ValueError(
                        "write_gate must match the callback sequence shape "
                        f"{(batch_size, sequence_length)}, got {tuple(layer_write_gate.shape)}"
                    )
            sequence, next_state = self.ttt_layers[layer_key](
                sequence,
                fast_states.get(layer_index),
                update=update,
                create_graph=create_graph,
                write_gate=layer_write_gate,
                detach_writer=detach_writer,
            )
            fast_states[layer_index] = next_state
            return sequence.reshape_as(hidden_states)

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
    ) -> tuple[Tensor, TTTFastStates]:
        """Run flow matching while optionally detaching TTT writer updates."""
        fast_states = {} if fast_states is None else dict(fast_states)
        callback = self._make_expert_layer_callback(
            sequence_shape,
            fast_states,
            update=True,
            create_graph=create_graph,
            write_gate=write_gate,
            detach_writer=detach_writer,
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
