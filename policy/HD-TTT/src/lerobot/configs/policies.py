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
import abc
import builtins
import json
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.utils.hub import HubMixin

from .types import FeatureType, PolicyFeature

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):  # type: ignore[misc,name-defined] #TODO: draccus issue
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
    """

    n_obs_steps: int = 1

    # `input_features` can be set to None/null in order to infer those values from the dataset.
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    device: str | None = None  # e.g. "cuda", "cuda:0", "cpu", or "mps"
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    # Whether the policy employed PEFT for training.
    use_peft: bool = False

    push_to_hub: bool = True  # type: ignore[assignment] # TODO: use a different name to avoid override
    repo_id: str | None = None

    # Upload on private repository on the Hugging Face hub.
    private: bool | None = None
    # Add tags to your policy on the hub.
    tags: list[str] | None = None
    # Add tags to your policy on the hub.
    license: str | None = None
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch.
    pretrained_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:  # type: ignore[type-arg] #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        # Encode the registered choice so draccus includes the concrete
        # ``type`` discriminator.  ``from_pretrained`` needs that key to
        # resolve the policy class; relying on ``dump(self)`` is
        # version-dependent and can silently omit it.  draccus 0.8 has two
        # incompatible ``encode`` signatures in the environments supported by
        # this project, so use the typed form when available and fall back to
        # the one-argument form.  Finally fill the discriminator explicitly as
        # a last guard against a registry encoder that omits it.
        with open(save_directory / CONFIG_NAME, "w") as f:
            try:
                encoded = draccus.encode(self, PreTrainedConfig)
            except TypeError as error:
                if "positional argument" not in str(error) and "takes 1" not in str(error):
                    raise
                encoded = draccus.encode(self)
            if isinstance(encoded, dict) and not encoded.get("type"):
                encoded = dict(encoded)
                encoded["type"] = self.type
            json.dump(encoded, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with open(config_file) as f:
            config = json.load(f)

        # HACK: Parse the original config to get the config subclass, so that
        # we can apply cli overrides.  Older SmolVLA-TTT checkpoints may carry
        # JSON null for fields that are non-nullable in the older draccus
        # runtime (notably the two HD booleans).  Sanitize a private temporary
        # copy before this *first* parse; normalizing only after parsing is too
        # late on the benchmark's Python 3.11 environment, where draccus
        # rejects ``null`` immediately.
        # This is very ugly, ideally we'd like to be able to do that natively
        # with draccus, something like --policy.path (in addition to
        # --policy.type)
        config_type = config.get("type")
        orig_config_file = config_file
        if config_type == "smolvla_ttt":
            config_for_probe = dict(config)
            for flag_name in ("hd_ttt_enabled", "hd_learned_write_gate"):
                if config_for_probe.get(flag_name) is None:
                    config_for_probe[flag_name] = False
            if config_for_probe.get("hd_effect_weight") is None:
                config_for_probe["hd_effect_weight"] = 0.0
            if config_for_probe.get("hd_attribution_protocol") is None:
                config_for_probe["hd_attribution_protocol"] = "legacy_raw_hinge_max"
            if config_for_probe != config:
                config = config_for_probe
                with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as probe:
                    json.dump(config_for_probe, probe)
                    orig_config_file = probe.name
        with draccus.config_type("json"):
            orig_config = draccus.parse(cls, orig_config_file, args=[])

        config.pop("type")
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        # A clean evaluation/fine-tuning opt-out of an HD SmolVLA-TTT
        # checkpoint should not fail while draccus reconstructs the source
        # config.  The source may carry ``hd_effect_weight>0``; when the user
        # explicitly sets ``hd_ttt_enabled=false`` and does not provide an
        # effect override, inject the semantically implied zero before parsing
        # so ``SmolVLATTTConfig.__post_init__`` can validate a coherent clean
        # config.  Keep this narrowly scoped to the registered SmolVLA-TTT
        # config so unrelated policy classes never receive an unknown option,
        # and preserve an explicit positive effect value as a deliberate
        # (and consequently rejected) configuration error.
        cli_overrides = list(cli_overrides or [])
        if getattr(orig_config, "type", None) == "smolvla_ttt":

            def _override_value(field_name: str) -> str | None:
                prefix = f"--{field_name}="
                # YAML-derived overrides precede command-line overrides; use
                # the last occurrence to mirror argparse/draccus precedence
                # when callers intentionally provide the same field twice.
                for argument in reversed(cli_overrides):
                    if argument.startswith(prefix):
                        return argument[len(prefix) :].strip().lower()
                return None

            hd_enabled_override = _override_value("hd_ttt_enabled")
            effect_override = _override_value("hd_effect_weight")
            gate_override = _override_value("hd_learned_write_gate")
            if (
                hd_enabled_override in {"false", "0", "no", "off"}
                and effect_override is None
            ):
                cli_overrides.append("--hd_effect_weight=0.0")
            if (
                hd_enabled_override in {"false", "0", "no", "off"}
                and gate_override is None
            ):
                cli_overrides.append("--hd_learned_write_gate=false")
        with draccus.config_type("json"):
            parsed_config = draccus.parse(orig_config.__class__, config_file, args=cli_overrides)
        # Keep the generic parser and the custom SmolVLA-TTT checkpoint
        # decoder on the same canonical representation.  Early checkpoints
        # occasionally serialized the two boolean HD switches as JSON null;
        # allowing ``None`` to escape here makes metadata/contract checks
        # nondeterministic even though downstream ``bool(None)`` happens to
        # disable the feature.  Normalize only this policy family.
        if getattr(orig_config, "type", None) == "smolvla_ttt":
            for field_name in ("hd_ttt_enabled", "hd_learned_write_gate"):
                if getattr(parsed_config, field_name, None) is None:
                    setattr(parsed_config, field_name, False)
        return parsed_config
