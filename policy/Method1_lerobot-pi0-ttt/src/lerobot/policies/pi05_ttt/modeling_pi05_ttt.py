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

import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers.models.gemma import modeling_gemma

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PaliGemmaWithExpertModel,
    PI05Policy,
    PI05Pytorch,
    clone_past_key_values,
    get_gemma_config,
    make_att_2d_masks,
)
from lerobot.policies.pi_gemma import _gated_residual, layernorm_forward
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import require_package

from .configuration_pi05_ttt import PI05TTTConfig
from .sequence import SEQUENCE_SHAPE_KEY
from .ttt import TTTFastState, TTTMLPLayer

TTTFastStates = dict[int, TTTFastState]


def _compute_layer_with_ttt(
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    layers,
    rotary_emb,
    *,
    layer_index: int,
    expert_layer_callback,
):
    """PI0.5 joint VLM/expert layer with TTT after expert attention and before its MLP."""
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for stream_index, hidden_states in enumerate(inputs_embeds):
        layer = layers[stream_index]
        hidden_states, gate = layernorm_forward(
            layer.input_layernorm, hidden_states, adarms_cond[stream_index]
        )
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    paligemma_layer = layers[0]
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        paligemma_layer.self_attn.scaling,
    )
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(query_states.shape[0], -1, 8 * head_dim)

    outputs_embeds = []
    start_pos = 0
    for stream_index, hidden_states in enumerate(inputs_embeds):
        layer = layers[stream_index]
        end_pos = start_pos + hidden_states.shape[1]
        stream_att_output = att_output[:, start_pos:end_pos]
        if stream_att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            stream_att_output = stream_att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(stream_att_output)
        out_emb = _gated_residual(hidden_states, out_emb, gates[stream_index])
        if stream_index == 1:
            out_emb = expert_layer_callback(layer_index, out_emb)
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[stream_index])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        outputs_embeds.append(_gated_residual(after_first_residual, out_emb, gate))
        start_pos = end_pos
    return outputs_embeds


class PaliGemmaWithExpertTTTModel(PaliGemmaWithExpertModel):
    """PI0.5 PaliGemma/action expert with an action-stream post-attention callback."""

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
        expert_layer_callback=None,
    ):
        if expert_layer_callback is None or inputs_embeds[1] is None:
            return super().forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
            )

        if adarms_cond is None:
            adarms_cond = [None, None]

        if inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
                layer_callback=expert_layer_callback,
            )
            return [None, suffix_output.last_hidden_state], None

        paligemma_layers = self.paligemma.model.language_model.layers
        gemma_expert_layers = self.gemma_expert.model.layers
        rotary_emb = self.paligemma.model.language_model.rotary_emb
        use_gradient_checkpointing = (
            hasattr(self.gemma_expert.model, "gradient_checkpointing")
            and self.gemma_expert.model.gradient_checkpointing
            and self.training
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)
        if use_gradient_checkpointing:
            raise ValueError("pi05_ttt callbacks are not compatible with gradient checkpointing")

        for layer_index, layers in enumerate(zip(paligemma_layers, gemma_expert_layers, strict=True)):
            inputs_embeds = _compute_layer_with_ttt(
                inputs_embeds,
                attention_mask,
                position_ids,
                adarms_cond,
                layers,
                rotary_emb,
                layer_index=layer_index,
                expert_layer_callback=expert_layer_callback,
            )

        final_norms = (
            self.paligemma.model.language_model.norm,
            self.gemma_expert.model.norm,
        )
        outputs_embeds = []
        for stream_index, hidden_states in enumerate(inputs_embeds):
            out_emb, _ = layernorm_forward(
                final_norms[stream_index], hidden_states, adarms_cond[stream_index]
            )
            outputs_embeds.append(out_emb)
        return outputs_embeds, None


class PI05TTTPytorch(PI05Pytorch):
    """PI0.5 core with independent TTT memories in selected action-expert layers."""

    config: PI05TTTConfig

    def __init__(self, config: PI05TTTConfig, rtc_processor=None) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)
        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertTTTModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=False,
            train_expert_only=False,
        )
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.gradient_checkpointing_enabled = False

        self.ttt_layers = nn.ModuleDict(
            {
                str(layer_index): TTTMLPLayer(
                    dim=action_expert_config.width,
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

    def _configure_trainable_parameters(self) -> None:
        self.requires_grad_(False)
        for parameter in self.ttt_layers.parameters():
            parameter.requires_grad_(True)
        for layer in self.ttt_layers.values():
            layer.gate.requires_grad_(self.config.trains_gate)

        if self.config.trains_action_head:
            for module in (
                self.paligemma_with_expert.gemma_expert,
                self.action_in_proj,
                self.action_out_proj,
                self.time_mlp_in,
                self.time_mlp_out,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.paligemma_with_expert.paligemma.eval()
        self.paligemma_with_expert.gemma_expert.train(mode and self.config.trains_action_head)
        for module in (self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out):
            module.train(mode and self.config.trains_action_head)
        self.ttt_layers.train(mode)
        return self

    def _make_expert_layer_callback(
        self,
        sequence_shape: tuple[int, int],
        fast_states: TTTFastStates,
        *,
        update: bool,
        update_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
        create_graph: bool | None,
    ):
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
            sequence, next_state = self.ttt_layers[layer_key](
                sequence,
                fast_states.get(layer_index),
                update=update,
                update_mask=update_mask,
                token_mask=token_mask,
                create_graph=create_graph,
            )
            fast_states[layer_index] = next_state
            return sequence.reshape_as(hidden_states)

        return apply_ttt

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        *,
        action_token_mask: Tensor | None = None,
        expert_layer_callback=None,
    ) -> Tensor:
        if expert_layer_callback is None:
            return super().forward(images, img_masks, tokens, masks, actions, noise, time)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)
        suffix_key_mask = None
        if action_token_mask is not None:
            if action_token_mask.shape != suffix_pad_masks.shape:
                raise ValueError(
                    f"Expected action_token_mask with shape {tuple(suffix_pad_masks.shape)}, "
                    f"got {tuple(action_token_mask.shape)}"
                )
            suffix_key_mask = action_token_mask.to(
                device=suffix_pad_masks.device,
                dtype=torch.bool,
            )

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        if suffix_key_mask is not None:
            # Padded action slots must not become keys for valid action tokens.
            # Keep their query rows readable, however: fully masked query rows
            # are unsafe for attention implementations that use -inf masks.
            key_mask = torch.cat([prefix_pad_masks, suffix_key_mask], dim=1)
            att_2d_masks = att_2d_masks & key_mask[:, None, :]
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            expert_layer_callback=expert_layer_callback,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    def forward_with_state(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        *,
        sequence_shape: tuple[int, int],
        update_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
        fast_states: TTTFastStates | None = None,
        create_graph: bool | None = None,
    ) -> tuple[Tensor, TTTFastStates]:
        fast_states = {} if fast_states is None else dict(fast_states)
        callback = self._make_expert_layer_callback(
            sequence_shape,
            fast_states,
            update=True,
            update_mask=update_mask,
            token_mask=token_mask,
            create_graph=create_graph,
        )
        losses = self.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            action_token_mask=(
                None if token_mask is None else token_mask.reshape(sequence_shape[0] * sequence_shape[1], -1)
            ),
            expert_layer_callback=callback,
        )
        return losses, fast_states

    def _denoise_step_with_callback(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        expert_layer_callback,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=clone_past_key_values(past_key_values),
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            expert_layer_callback=expert_layer_callback,
        )
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions_with_state(
        self,
        images,
        img_masks,
        tokens,
        masks,
        fast_states: TTTFastStates | None = None,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> tuple[Tensor, TTTFastStates]:
        del kwargs
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        fast_states = {} if fast_states is None else dict(fast_states)
        batch_size = tokens.shape[0]
        device = tokens.device
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), device
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        sequence_shape = (batch_size, 1)
        for step in range(num_steps):
            time_tensor = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=device).expand(batch_size)
            callback = self._make_expert_layer_callback(
                sequence_shape,
                fast_states,
                update=step == 0,
                create_graph=False,
            )
            v_t = self._denoise_step_with_callback(
                prefix_pad_masks,
                past_key_values,
                x_t,
                time_tensor,
                callback,
            )
            x_t = x_t + dt * v_t
        return x_t, fast_states


_CHECKPOINT_BASE_FIELDS = {
    "paligemma_variant",
    "action_expert_variant",
    "n_obs_steps",
    "chunk_size",
    "max_state_dim",
    "max_action_dim",
    "num_inference_steps",
    "time_sampling_beta_alpha",
    "time_sampling_beta_beta",
    "time_sampling_scale",
    "time_sampling_offset",
    "min_period",
    "max_period",
    "use_relative_actions",
    "relative_exclude_joints",
    "image_resolution",
    "empty_cameras",
    "tokenizer_max_length",
    "normalization_mapping",
}

_CHECKPOINT_TTT_FIELDS = {
    "ttt_hidden_dim",
    "ttt_base_inner_lr",
    "ttt_effective_gate_init",
    "ttt_rope_theta",
    "ttt_second_order",
    "ttt_start_layer",
    "ttt_layer_indices",
}


class PI05TTTPolicy(PI05Policy):
    """Independent PI0.5-TTT policy; it does not import the legacy PI0-TTT policy."""

    config_class = PI05TTTConfig
    name = "pi05_ttt"
    tbptt_loss_weighting = "valid_actions"

    def __init__(self, config: PI05TTTConfig, **kwargs) -> None:
        del kwargs
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05TTTPytorch(config, rtc_processor=self.rtc_processor)
        self.model.to(config.device)
        self.reset()

    @staticmethod
    def _decode_source_config(raw_config: dict) -> PI05Config:
        import draccus

        config_type = raw_config.get("type")
        if config_type == "pi05_ttt":
            config_class = PI05TTTConfig
        elif config_type == "pi05":
            config_class = PI05Config
        else:
            raise TypeError(f"pi05_ttt can only load PI0.5-family checkpoints, got {config_type!r}")
        valid_fields = {field.name for field in fields(config_class) if field.init}
        return draccus.decode(
            config_class,
            {key: value for key, value in raw_config.items() if key in valid_fields},
        )

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
        from safetensors.torch import load_file
        from transformers.utils import cached_file

        config_file = (
            Path(pretrained_name_or_path) / "config.json"
            if Path(pretrained_name_or_path).is_dir()
            else cached_file(
                pretrained_name_or_path,
                "config.json",
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
        if config_file is None or not Path(config_file).exists():
            raise FileNotFoundError(f"config.json not found in {pretrained_name_or_path}")
        with open(config_file) as file:
            raw_config = json.load(file)
        source_config = cls._decode_source_config(raw_config)

        if config is None:
            if isinstance(source_config, PI05TTTConfig):
                config = source_config
                config.pretrained_path = Path(pretrained_name_or_path)
            else:
                base_values = {
                    field.name: getattr(source_config, field.name)
                    for field in fields(PI05Config)
                    if field.init
                }
                base_values.update(
                    pretrained_path=Path(pretrained_name_or_path),
                    n_action_steps=1,
                    gradient_checkpointing=False,
                    compile_model=False,
                )
                config = PI05TTTConfig(**base_values)
        elif isinstance(config, PI05TTTConfig):
            for field_name in _CHECKPOINT_BASE_FIELDS:
                setattr(config, field_name, getattr(source_config, field_name))
            if isinstance(source_config, PI05TTTConfig):
                # A TTT checkpoint owns the architecture and numerical
                # behavior of its serialized TTT tensors. Runtime cadence,
                # selected-sequence/TBPTT lengths, and training stage remain
                # explicit choices of the caller-provided config.
                for field_name in _CHECKPOINT_TTT_FIELDS:
                    setattr(config, field_name, getattr(source_config, field_name))
            config.gradient_checkpointing = False
            config.compile_model = False
        else:
            raise TypeError(f"Expected PI05TTTConfig, got {type(config).__name__}")
        config.pretrained_path = Path(pretrained_name_or_path)
        config.__post_init__()

        model = cls(config, **kwargs)
        resolved_file = cached_file(
            pretrained_name_or_path,
            "model.safetensors",
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
        )
        if resolved_file is None:
            raise FileNotFoundError(f"model.safetensors not found in {pretrained_name_or_path}")

        source_state_dict = load_file(resolved_file)
        fixed_state_dict = model._fix_pytorch_state_dict_keys(source_state_dict, model.config)
        remapped_state_dict = {
            key if key.startswith("model.") else f"model.{key}": value
            for key, value in fixed_state_dict.items()
        }
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        source_is_ttt = isinstance(source_config, PI05TTTConfig)
        disallowed_missing_keys = (
            list(missing_keys)
            if source_is_ttt
            else [key for key in missing_keys if not key.startswith("model.ttt_layers.")]
        )
        if unexpected_keys or disallowed_missing_keys or (strict and missing_keys):
            raise RuntimeError(
                f"Incompatible PI0.5 checkpoint: missing={missing_keys}, unexpected={unexpected_keys}"
            )
        if missing_keys:
            logging.info(
                "Loaded PI0.5 base checkpoint; initialized %d new TTT parameters",
                len(missing_keys),
            )
        model.eval()
        return model

    def reset(self) -> None:
        super().reset()
        self._ttt_fast_states: TTTFastStates = {}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions, self._ttt_fast_states = self.model.sample_actions_with_state(
            images,
            img_masks,
            tokens,
            masks,
            fast_states=self._ttt_fast_states,
            **kwargs,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    def forward_sequence_segment(
        self,
        batch: dict[str, Tensor],
        sequence_shape: tuple[int, int],
        fast_states: TTTFastStates | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict, TTTFastStates]:
        batch_size, sequence_length = sequence_shape
        expected_flat_batch = batch_size * sequence_length
        if batch[ACTION].shape[0] != expected_flat_batch:
            raise ValueError(
                f"Sequence shape {sequence_shape} requires {expected_flat_batch} flattened samples, "
                f"but the action batch has {batch[ACTION].shape[0]}"
            )

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        update_mask = None
        token_mask = None
        if actions_is_pad is not None:
            expected_mask_shape = (expected_flat_batch, self.config.chunk_size)
            if not isinstance(actions_is_pad, Tensor) or actions_is_pad.shape != expected_mask_shape:
                actual_shape = (
                    tuple(actions_is_pad.shape)
                    if isinstance(actions_is_pad, Tensor)
                    else type(actions_is_pad).__name__
                )
                raise ValueError(f"action_is_pad must have shape {expected_mask_shape}, got {actual_shape}")
            actions_is_pad = actions_is_pad.to(device=actions.device, dtype=torch.bool)
            token_mask = (~actions_is_pad).reshape(
                batch_size,
                sequence_length,
                self.config.chunk_size,
            )
            update_mask = token_mask.any(dim=-1)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        losses, fast_states = self.model.forward_with_state(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            sequence_shape=sequence_shape,
            update_mask=update_mask,
            token_mask=token_mask,
            fast_states=fast_states,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        if actions_is_pad is None:
            valid_actions = torch.ones(
                losses.shape[:2],
                dtype=torch.bool,
                device=losses.device,
            )
        else:
            valid_actions = ~actions_is_pad
        losses = losses * valid_actions.unsqueeze(-1)
        valid_steps = valid_actions.sum().clamp_min(1)
        loss_per_dim = losses.sum(dim=(0, 1)) / valid_steps
        loss_dict = {"loss_per_dim": loss_per_dim.detach().cpu().tolist()}
        if reduction == "none":
            num_valid = (valid_actions.sum(dim=1) * losses.shape[-1]).clamp_min(1)
            per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict, fast_states
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        num_valid = (valid_actions.sum() * losses.shape[-1]).clamp_min(1)
        loss = losses.sum() / num_valid
        loss_dict["loss"] = loss.item()
        return loss, loss_dict, fast_states

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if SEQUENCE_SHAPE_KEY not in batch:
            raise ValueError(
                f"pi05_ttt training batches must contain {SEQUENCE_SHAPE_KEY!r}; "
                "use TailPreservingSequenceDataset and sequence_collate_fn"
            )
        sequence_shape = tuple(int(value) for value in batch[SEQUENCE_SHAPE_KEY])
        loss, loss_dict, _ = self.forward_sequence_segment(
            batch, sequence_shape=sequence_shape, reduction=reduction
        )
        return loss, loss_dict
