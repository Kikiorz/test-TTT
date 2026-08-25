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
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import (
    ActionSelectKwargs,
    PI0Policy,
    PI0Pytorch,
    get_gemma_config,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_pi0_ttt import PI0TTTConfig
from .sequence import SEQUENCE_SHAPE_KEY
from .ttt import TTTFastState, TTTMLPLayer

TTTFastStates = dict[int, TTTFastState]


class PI0TTTPytorch(PI0Pytorch):
    """PI0 core with TTT hooks in selected action-expert layers."""

    config: PI0TTTConfig

    def __init__(self, config: PI0TTTConfig, rtc_processor=None) -> None:
        super().__init__(config, rtc_processor=rtc_processor)
        expert_width = get_gemma_config(config.action_expert_variant).width
        self.ttt_layers = nn.ModuleDict(
            {
                str(layer_index): TTTMLPLayer(
                    dim=expert_width,
                    hidden_dim=config.ttt_hidden_dim,
                    base_inner_lr=config.ttt_base_inner_lr,
                    gate_init=config.ttt_gate_init,
                    rope_theta=config.ttt_rope_theta,
                    second_order=config.ttt_second_order,
                )
                for layer_index in config.resolved_ttt_layer_indices
            }
        )

        if config.ttt_freeze_base:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in self.ttt_layers.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.ttt_freeze_base:
            self.paligemma_with_expert.eval()
            self.action_in_proj.eval()
            self.action_out_proj.eval()
            self.state_proj.eval()
            self.action_time_mlp_in.eval()
            self.action_time_mlp_out.eval()
            self.ttt_layers.train(mode)
        return self

    def _make_expert_layer_callback(
        self,
        sequence_shape: tuple[int, int],
        fast_states: TTTFastStates,
        *,
        update: bool,
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
                create_graph=create_graph,
            )
            fast_states[layer_index] = next_state
            return sequence.reshape_as(hidden_states)

        return apply_ttt

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
    ) -> tuple[Tensor, TTTFastStates]:
        fast_states = {} if fast_states is None else dict(fast_states)
        callback = self._make_expert_layer_callback(
            sequence_shape,
            fast_states,
            update=True,
            create_graph=create_graph,
        )
        losses = super().forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            expert_layer_callback=callback,
        )
        return losses, fast_states

    def sample_actions_with_state(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        fast_states: TTTFastStates | None = None,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> tuple[Tensor, TTTFastStates]:
        fast_states = {} if fast_states is None else dict(fast_states)
        sequence_shape = (state.shape[0], 1)

        def callback_factory(update: bool):
            return self._make_expert_layer_callback(
                sequence_shape,
                fast_states,
                update=update,
                create_graph=False,
            )

        actions = super().sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=num_steps,
            _expert_layer_callback_factory=callback_factory,
            **kwargs,
        )
        return actions, fast_states


class PI0TTTPolicy(PI0Policy):
    """PI0 policy extended with RoboTTT-style sequence memory."""

    config_class = PI0TTTConfig
    name = "pi0_ttt"
    model_class = PI0TTTPytorch

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
        """Load PI0 or PI0-TTT weights, allowing only new TTT keys to be absent."""
        if config is None:
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
            if raw_config.get("type") not in {"pi0", "pi0_ttt"}:
                raise TypeError(
                    f"pi0_ttt can only load PI0-family checkpoints, got {raw_config.get('type')!r}"
                )

            import draccus

            if raw_config.get("type") == "pi0_ttt":
                valid_fields = {field.name for field in fields(PI0TTTConfig) if field.init}
                source_config = draccus.decode(
                    PI0TTTConfig,
                    {key: value for key, value in raw_config.items() if key in valid_fields},
                )
            else:
                valid_fields = {field.name for field in fields(PI0Config) if field.init}
                source_config = draccus.decode(
                    PI0Config,
                    {key: value for key, value in raw_config.items() if key in valid_fields},
                )
            if isinstance(source_config, PI0TTTConfig):
                config = source_config
            elif isinstance(source_config, PI0Config):
                ttt_owned_fields = {
                    "n_action_steps",
                    "gradient_checkpointing",
                    "compile_model",
                    "optimizer_weight_decay",
                    "pretrained_path",
                }
                config_values = {
                    field.name: getattr(source_config, field.name)
                    for field in fields(PI0Config)
                    if field.init and field.name not in ttt_owned_fields
                }
                config_values["pretrained_path"] = Path(pretrained_name_or_path)
                config = PI0TTTConfig(**config_values)
            else:
                raise TypeError(
                    f"pi0_ttt can only load PI0-family checkpoints, got {type(source_config).__name__}"
                )
        if not isinstance(config, PI0TTTConfig):
            raise TypeError(f"Expected PI0TTTConfig, got {type(config).__name__}")

        model = cls(config, **kwargs)
        from safetensors.torch import load_file
        from transformers.utils import cached_file

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
        disallowed_missing_keys = [
            key for key in missing_keys if not key.startswith("model.ttt_layers.")
        ]
        if unexpected_keys or disallowed_missing_keys or (strict and missing_keys):
            raise RuntimeError(
                "Incompatible PI0 checkpoint: "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )
        if missing_keys:
            logging.info(
                "Loaded PI0 base checkpoint; initialized %d new TTT parameters from PI0TTTConfig",
                len(missing_keys),
            )
        model.eval()
        return model

    def reset(self) -> None:
        super().reset()
        self._ttt_fast_states: TTTFastStates = {}

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)

        actions, self._ttt_fast_states = self.model.sample_actions_with_state(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
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
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
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

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }
        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict, fast_states
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")

        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict, fast_states

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if SEQUENCE_SHAPE_KEY not in batch:
            raise ValueError(
                f"pi0_ttt training batches must contain {SEQUENCE_SHAPE_KEY!r}; "
                "use ContiguousSequenceDataset and sequence_collate_fn"
            )
        sequence_shape = tuple(int(value) for value in batch[SEQUENCE_SHAPE_KEY])
        loss, loss_dict, _ = self.forward_sequence_segment(
            batch, sequence_shape=sequence_shape, reduction=reduction
        )
        return loss, loss_dict
