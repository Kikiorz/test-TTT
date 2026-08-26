#!/usr/bin/env python

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
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, make_dataset
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.policies.pi0_ttt.configuration_pi0_ttt import PI0TTTConfig
from lerobot.policies.pi0_ttt.sequence import (
    SEQUENCE_SHAPE_KEY,
    ContiguousSequenceDataset,
    sequence_collate_fn,
)
from lerobot.policies.pi05_ttt.configuration_pi05_ttt import PI05TTTConfig
from lerobot.policies.pi05_ttt.sequence import (
    TailPreservingSequenceDataset,
    sequence_collate_fn as pi05_ttt_sequence_collate_fn,
)
from lerobot.policies.smolvla_ttt.configuration_smolvla_ttt import SmolVLATTTConfig
from lerobot.policies.smolvla_ttt.hd_dataset import HindsightLabelDataset
from lerobot.policies.smolvla_ttt.sequence import (
    SEQUENCE_OFFSET_KEY,
    EqualLengthBatchSampler,
    TailPreservingSequenceDataset as SmolVLATTTSequenceDataset,
    batched_sequence_collate_fn,
    sequence_collate_fn as smolvla_ttt_sequence_collate_fn,
)
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import eval_policy_all


# Keep the offline selected-event rule explicit in the training contract.  A
# frame/window label artifact generated with a different rule can have a
# single-branch ``hd_rho`` that no longer matches its stored wrong velocity.
_HD_GROUNDING_EVENT_POLICY = "min_future_horizon_mean_else_total_credit"
_HD_ATTRIBUTION_PROTOCOL_LEGACY = "legacy_raw_hinge_max"
_HD_ATTRIBUTION_PROTOCOL_V2 = "v2_relative_antithetic_robust"
_HD_ATTRIBUTION_PROTOCOL_V3 = "credit_ttt_v3_query_effect"


def _normalize_hd_attribution_protocol(value: Any, *, default: str) -> str:
    """Canonicalize an HD label protocol, including legacy JSON ``null``.

    Early artifacts omitted this optional field (and some serializers emitted
    it as explicit ``null``).  Treat both forms as the legacy protocol rather
    than converting ``None`` to the literal string ``"None"`` and rejecting a
    otherwise valid artifact.  Unknown values are left as strings so the
    caller can raise the existing contract error with a useful value.
    """

    if value is None:
        value = default
    if not isinstance(value, str):
        value = str(value)
    if value in {"legacy", "v1"}:
        return _HD_ATTRIBUTION_PROTOCOL_LEGACY
    if value == "v2":
        return _HD_ATTRIBUTION_PROTOCOL_V2
    return value


def _validate_hd_v2_label_contract(
    metadata: Mapping[str, Any],
    *,
    label_keys: set[str] | None = None,
) -> str:
    """Validate the immutable protocol fields required by an HD-v2 artifact.

    The causal attribution/effect labels are a coupled offline protocol, not
    independently swappable columns.  Checking these fields at trainer
    startup prevents a legacy all-slot artifact (or a partially regenerated
    effect target) from silently being optimized as the paper method.  The
    effect branch count intentionally remains ``> 0`` rather than hard-coding
    two, so an artifact with a different fixed branch budget remains readable
    by the selected-branch student implementation.

    Returns the canonical protocol string.  Legacy artifacts are accepted and
    returned as ``legacy_raw_hinge_max``; callers can then skip v2-only checks.
    """

    protocol = _normalize_hd_attribution_protocol(
        metadata.get("attribution_protocol"),
        default=_HD_ATTRIBUTION_PROTOCOL_LEGACY,
    )
    if protocol != _HD_ATTRIBUTION_PROTOCOL_V2:
        if protocol != _HD_ATTRIBUTION_PROTOCOL_LEGACY:
            raise ValueError(f"HD labels have unsupported attribution_protocol={protocol!r}")
        return protocol

    required_fields = {
        "attribution_slot_mode": "slot0",
        "attribution_replays": 2,
        "effect_target": "plus_noise_full_minus_wrong",
    }
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field_name, expected in required_fields.items():
        actual = metadata.get(field_name)
        # ``bool`` is an ``int`` subclass; reject it explicitly for the replay
        # count so malformed JSON cannot pass as a valid protocol declaration.
        if field_name == "attribution_replays":
            valid = type(actual) is int and actual == expected
        else:
            valid = actual == expected
        if not valid:
            mismatches[field_name] = (actual, expected)
    effect_branches = metadata.get("effect_branches")
    if type(effect_branches) is not int or effect_branches <= 0:
        mismatches["effect_branches"] = (effect_branches, "> 0")
    if mismatches:
        raise ValueError(
            "HD v2 label protocol contract mismatch: "
            f"{mismatches} (expected slot0/2/plus_noise_full_minus_wrong and positive effect_branches)"
        )
    if label_keys is not None:
        required_labels = {
            "hd_teacher_effect",
            "hd_effect_rho",
            "hd_effect_write_gate",
            "hd_effect_valid",
        }
        missing = sorted(required_labels - set(label_keys))
        if missing:
            raise ValueError(f"HD v2 label artifact is missing action-effect columns: {missing}")
    return protocol


def _hd_ttt_finite_guard(
    *,
    policy: torch.nn.Module | None = None,
    loss: Any = None,
    grad_norm: Any = None,
    fast_states: Any = None,
    observations: tuple[tuple[str, Any], ...] = (),
    accelerator: "Accelerator | None" = None,
    stage: str,
    segment_index: int | None = None,
    check_gradients: bool = True,
    check_parameters: bool = True,
) -> None:
    """Fail a HD-TTT update before ``optimizer.step`` can poison a checkpoint.

    The sequence trainer owns the only custom optimizer path for HD-TTT.  A
    non-finite inner-loop state can otherwise reach the outer optimizer after
    gradient clipping (``clip_grad_norm_`` is intentionally configured not to
    raise).  This guard is deliberately a no-op unless its caller is on the
    HD path; ordinary LeRobot and clean TTT updates never call it.

    In distributed training every rank participates in one small reduction of
    a bad-value flag.  Thus a rank that observes a bad local batch causes all
    ranks to raise together instead of leaving a collective or dataloader
    deadlocked.  The check happens before the optimizer step, so the in-memory
    parameters and the last checkpoint written by the training loop remain
    finite and untouched.
    """

    # Keep the values around for a detailed message only if the global flag is
    # bad.  The finite reductions themselves stay on-device and incur one host
    # synchronization, rather than one synchronization per parameter tensor.
    tensor_values: list[tuple[str, torch.Tensor]] = []
    python_bad_names: list[str] = []

    def collect(name: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            tensor_values.append((name, value))
            return
        # ``TTTFastState`` is intentionally a small dataclass and importing it
        # here would pull the policy implementation into ordinary LeRobot
        # training.  Duck-typing its public ``tensors`` method keeps this
        # guard local to the trainer while still checking every recurrent
        # weight in an HD replay.
        tensors_method = getattr(value, "tensors", None)
        if callable(tensors_method):
            try:
                for index, nested in enumerate(tensors_method()):
                    collect(f"{name}.tensors[{index}]", nested)
            except (TypeError, RuntimeError):
                python_bad_names.append(f"{name}(uncheckable)")
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                collect(f"{name}.{key}", nested)
            return
        if isinstance(value, (tuple, list)):
            for index, nested in enumerate(value):
                collect(f"{name}[{index}]", nested)
            return
        try:
            if name.endswith("nonfinite_seen") and bool(value):
                python_bad_names.append(name)
                return
            if not math.isfinite(float(value)):
                python_bad_names.append(name)
        except (TypeError, ValueError):
            # Non-numeric diagnostics are not part of the finite contract.
            return

    collect("loss", loss)
    collect("grad_norm", grad_norm)
    collect("fast_state", fast_states)
    for name, value in observations:
        collect(name, value)

    if policy is not None:
        for parameter_name, parameter in policy.named_parameters():
            if not parameter.requires_grad:
                continue
            if check_parameters:
                collect(f"parameter:{parameter_name}", parameter)
            if check_gradients and parameter.grad is not None:
                gradient = parameter.grad
                # ``isfinite`` does not operate on sparse layouts on all
                # supported torch versions; checking sparse values is enough.
                if gradient.is_sparse:
                    gradient = gradient.coalesce().values()
                collect(f"gradient:{parameter_name}", gradient)

    # Select a device without importing/constructing an Accelerator on the
    # ordinary path.  All model tensors normally share the accelerator device;
    # moving only the scalar flags also handles CPU-only unit tests.
    if accelerator is not None and hasattr(accelerator, "device"):
        flag_device = accelerator.device
    elif tensor_values:
        flag_device = tensor_values[0][1].device
    else:
        flag_device = torch.device("cpu")

    finite_flags: list[torch.Tensor] = []
    for name, value in tensor_values:
        if value.numel() == 0 or not (value.is_floating_point() or value.is_complex()):
            if name.endswith("nonfinite_seen") and value.numel() > 0:
                finite_flags.append(
                    value.detach().bool().any().to(device=flag_device, dtype=torch.int32)
                )
            continue
        if name.endswith("nonfinite_seen"):
            finite_flags.append(
                value.detach().bool().any().to(device=flag_device, dtype=torch.int32)
            )
        finite_flags.append(
            (~torch.isfinite(value.detach()).all()).to(device=flag_device, dtype=torch.int32)
        )
    if finite_flags:
        local_bad = torch.stack(finite_flags).amax()
    else:
        local_bad = torch.zeros((), device=flag_device, dtype=torch.int32)
    if python_bad_names:
        local_bad = torch.maximum(local_bad, torch.ones_like(local_bad))

    # Distributed failures otherwise report only ``another distributed rank``
    # on the main process.  Keep an opt-in, rank-local report for numerical
    # bring-up; the normal path does not format tensors or synchronize any
    # additional values.  This is intentionally an execution diagnostic, not
    # part of the training objective.
    if os.environ.get("HD_TTT_DEBUG_FINITE", "0") == "1":
        local_bad_names = list(python_bad_names)
        for name, value in tensor_values:
            if value.numel() == 0 or not (value.is_floating_point() or value.is_complex()):
                if name.endswith("nonfinite_seen") and value.numel() > 0 and bool(value.detach().bool().any().item()):
                    local_bad_names.append(name)
                continue
            try:
                if not bool(torch.isfinite(value.detach()).all().item()):
                    local_bad_names.append(name)
                elif name.endswith("nonfinite_seen") and bool(value.detach().bool().any().item()):
                    local_bad_names.append(name)
            except RuntimeError:
                local_bad_names.append(f"{name}(uncheckable)")
        process_index = getattr(accelerator, "process_index", 0) if accelerator is not None else 0
        print(
            "[HD-TTT finite debug] "
            f"rank={process_index} stage={stage!r} segment={segment_index} "
            f"local_bad={bool(local_bad.detach().item())} "
            f"names={local_bad_names[:16]}",
            flush=True,
        )

    if accelerator is not None and getattr(accelerator, "num_processes", 1) > 1:
        global_bad = accelerator.reduce(local_bad, reduction="sum")
        has_bad_value = bool(global_bad.detach().item() > 0)
    else:
        has_bad_value = bool(local_bad.detach().item() != 0)
    if not has_bad_value:
        return

    # Only inspect individual tensors after the aggregate flag is bad.  This
    # keeps the normal step cheap while still making the failure actionable.
    bad_names = list(python_bad_names)
    for name, value in tensor_values:
        if value.numel() == 0 or not (value.is_floating_point() or value.is_complex()):
            continue
        try:
            if name.endswith("nonfinite_seen") and bool(value.detach().bool().any().item()):
                bad_names.append(name)
            elif not bool(torch.isfinite(value.detach()).all().item()):
                bad_names.append(name)
        except RuntimeError:
            bad_names.append(f"{name}(uncheckable)")
        if len(bad_names) >= 8:
            break
    if not bad_names:
        bad_names.append("another distributed rank")
    location = stage
    if segment_index is not None:
        location = f"{location}, segment={segment_index}"
    raise RuntimeError(
        "HD-TTT finite guard failed before optimizer.step "
        f"({location}): non-finite {', '.join(bad_names[:8])}. "
        "The optimizer step was not executed; in-memory parameters and the "
        "last finite checkpoint remain untouched."
    )


def _ttt_finite_guard_enabled(policy_config: Any) -> bool:
    """Enable the pre-step finite guard for robust HD or stable TTT runs."""

    return bool(
        getattr(policy_config, "hd_ttt_enabled", False)
        or getattr(policy_config, "ttt_stable_inner_update", False)
    )


def _teacher_config_sha256(checkpoint: str | Path | None) -> str:
    """Resolve and hash the exact teacher ``config.json`` used for labels.

    HD labels are counterfactual outputs of one specific clean teacher.  A
    path string alone is not a sufficient provenance key because the same
    path can be replaced or a Hub alias can resolve to a different revision.
    The label builders record the raw config-file SHA; training therefore
    hashes the current ``pretrained_path`` before attaching labels and fails
    on any mismatch.  Hub resolution is deliberately lazy so ordinary
    non-HD training does not import or contact the Hub.
    """

    if checkpoint is None:
        raise ValueError(
            "HD labels require policy.pretrained_path so the teacher config can be verified"
        )
    checkpoint_text = str(checkpoint)
    checkpoint_path = Path(checkpoint_text).expanduser()
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / "config.json"
    elif checkpoint_path.is_file() and checkpoint_path.name == "config.json":
        config_path = checkpoint_path
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(repo_id=checkpoint_text, filename="config.json")
            )
        except Exception as error:
            raise ValueError(
                "Could not resolve config.json for the current HD teacher "
                f"{checkpoint_text!r}: {error}"
            ) from error
    if not config_path.is_file():
        raise ValueError(
            "Current HD teacher is missing config.json: "
            f"{config_path} (from {checkpoint_text!r})"
        )
    try:
        config_bytes = config_path.read_bytes()
    except OSError as error:
        raise ValueError(f"Could not read current HD teacher config {config_path}: {error}") from error
    return hashlib.sha256(config_bytes).hexdigest()


def _configured_hd_teacher_checkpoint(cfg: TrainPipelineConfig) -> str | Path | None:
    """Return the clean teacher path recorded by an initial run.

    On resume, LeRobot intentionally changes ``policy.pretrained_path`` to the
    HD checkpoint being restored.  That checkpoint's config contains the HD
    switches and therefore cannot have the clean-teacher SHA stored in a
    hindsight artifact.  The checkpoint also preserves the original
    ``train_config.json``; recover its ``policy.pretrained_path`` so resuming a
    valid run keeps the same provenance check instead of spuriously failing.
    """

    if not getattr(cfg, "resume", False):
        return getattr(cfg.policy, "pretrained_path", None)
    checkpoint_dir = getattr(cfg, "checkpoint_path", None)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        candidates = (
            checkpoint_dir / "pretrained_model" / "train_config.json",
            checkpoint_dir / "train_config.json",
        )
        for train_config_path in candidates:
            if not train_config_path.is_file():
                continue
            try:
                raw = json.loads(train_config_path.read_text(encoding="utf-8"))
                original = raw.get("policy", {}).get("pretrained_path")
            except (OSError, json.JSONDecodeError, AttributeError) as error:
                raise ValueError(
                    f"Could not read resumed HD train config {train_config_path}: {error}"
                ) from error
            if original:
                return original
    # A hand-built resume config may not retain the original train config.  In
    # that case use the active path and let the normal SHA check produce a
    # precise mismatch rather than silently disabling provenance validation.
    return getattr(cfg.policy, "pretrained_path", None)


def _attach_hd_labels(dataset, cfg: TrainPipelineConfig, *, is_smolvla_ttt: bool):
    """Attach offline HD-TTT labels while preserving the LeRobot dataset API.

    Labels are loaded after ``make_dataset`` has resolved episode subsets and
    delta-timestamp action chunks.  This keeps the artifact indexed by the
    exact frame order consumed by the trainer and lets the normal processor
    pipeline carry ``hd_*`` fields as complementary data.  The wrapper is
    deliberately restricted to SmolVLA-TTT so a typo cannot silently alter a
    baseline or a different policy's input schema.
    """

    label_path = getattr(cfg.dataset, "hd_label_path", None)
    if not label_path:
        return dataset
    if not is_smolvla_ttt:
        raise ValueError(
            "dataset.hd_label_path is only supported with policy.type=smolvla_ttt; "
            "remove it for baseline/PI0 training"
        )
    logging.info("Loading HD-TTT hindsight labels from %s", label_path)
    labeled_dataset = HindsightLabelDataset(dataset, label_path, strict=True)
    policy_cfg = cfg.policy
    if not bool(getattr(policy_cfg, "hd_ttt_enabled", False)):
        raise ValueError(
            "dataset.hd_label_path requires policy.hd_ttt_enabled=true; "
            "remove the label path for a clean TTT/base run"
        )

    # A hindsight artifact is tied to the exact recurrent/window protocol
    # used to generate it.  Fail at startup rather than silently training a
    # different state distribution (especially important for long episodes).
    metadata = labeled_dataset.label_metadata
    if not metadata:
        raise ValueError(
            "HD label artifacts must include guarded provenance metadata; "
            "regenerate labels with build_hd_labels.py"
        )
    # The bounded inner-update mode was added after the first HD artifacts.
    # Missing/explicit-null has the legacy clean semantics; materialize that
    # default locally so the contract below can compare a real boolean without
    # mutating the serialized dataset object.
    metadata = dict(metadata)
    if metadata.get("teacher_ttt_stable_inner_update") is None:
        metadata["teacher_ttt_stable_inner_update"] = False
    # v2 hindsight artifacts carry an explicit causal-credit protocol.  Legacy
    # files predate this field and are interpreted as the raw-hinge protocol;
    # this keeps old checkpoints loadable while preventing a silent mix of
    # slot-0/antithetic labels with the legacy all-slot objective.
    artifact_attribution_protocol = _normalize_hd_attribution_protocol(
        metadata.get("attribution_protocol"),
        default=_HD_ATTRIBUTION_PROTOCOL_LEGACY,
    )
    if artifact_attribution_protocol not in {
        _HD_ATTRIBUTION_PROTOCOL_LEGACY,
        _HD_ATTRIBUTION_PROTOCOL_V2,
        _HD_ATTRIBUTION_PROTOCOL_V3,
    }:
        raise ValueError(
            "HD labels have unsupported attribution_protocol="
            f"{artifact_attribution_protocol!r}"
        )
    expected_attribution_protocol = _normalize_hd_attribution_protocol(
        getattr(policy_cfg, "hd_attribution_protocol", artifact_attribution_protocol),
        default=_HD_ATTRIBUTION_PROTOCOL_LEGACY,
    )
    if expected_attribution_protocol != artifact_attribution_protocol:
        raise ValueError(
            "HD attribution protocol mismatch: artifact uses "
            f"{artifact_attribution_protocol!r}, policy.hd_attribution_protocol="
            f"{expected_attribution_protocol!r}"
        )
    if artifact_attribution_protocol == _HD_ATTRIBUTION_PROTOCOL_V3:
        # CreditTTT labels come from the independent explicit action teacher,
        # not from the legacy clean SmolVLA replay.  Validate their compact
        # pair schema here and return before the v1/v2 provenance checks below
        # (which intentionally require velocity/replay fields that V3 does
        # not contain).
        required_v3_metadata = {
            "format",
            "protocol",
            "version",
            "pair_schema",
            "dataset_repo_id",
            "fps",
            "event_dim",
            "action_dim",
            "pair_k",
            "event_block_size",
            "delay_edges",
            "intervention",
            "intervention_scope",
            "intervention_type",
            "protocol_variant",
            "canonical_delay_edges",
            "target_mode",
            # Immutable protocol identity/provenance.  These fields make a
            # direct-action teacher impossible to misreported as an
            # antithetic-flow teacher and make causal state semantics explicit.
            "state",
            "causal",
            "denoise_steps",
            "antithetic_noise",
            "includes_previous_executed_action",
            "teacher_adapter",
            "flow_target_available",
            "teacher_checkpoint_sha256",
            "feature_artifact_sha256",
            # Full-history is a data/sequence contract, not merely a README
            # convention.  Keep its structural fields in the artifact so a
            # bounded-window run cannot be relabelled as the canonical method.
            "history_mode",
            "min_sequence_length",
            "sequence_stride_policy",
            "max_windows_per_episode",
            "ttt_history_warmup_length",
            "sequence_offset_policy",
            "episode_slices",
            "episode_lengths",
        }
        missing_v3 = sorted(required_v3_metadata - set(metadata))
        if missing_v3:
            raise ValueError(
                "CreditTTT V3 label artifact is missing required provenance fields: "
                f"{missing_v3}"
            )
        if metadata.get("format") != "credit_ttt_v3" or metadata.get("protocol") != "creditttt_qh2l_v3":
            raise ValueError("CreditTTT V3 label format/protocol declaration is invalid")
        if type(metadata.get("version")) is not int or metadata.get("version") != 3 or metadata.get("pair_schema") != "event_future_control_pair_v3":
            raise ValueError("CreditTTT V3 pair schema/version mismatch")
        if type(metadata.get("event_block_size")) is not int or metadata.get("event_block_size") != 1:
            raise ValueError(
                "CreditTTT V3 canonical training requires event_block_size=1; "
                "multi-frame event blocks need a dedicated block-state replay"
            )
        try:
            delay_edges = tuple(int(edge) for edge in metadata["delay_edges"])
        except (TypeError, ValueError) as error:
            raise ValueError("CreditTTT V3 delay_edges must be an integer sequence") from error
        if delay_edges != (1, 17, 65, 257, 1025, 2**31 - 1):
            raise ValueError(
                "CreditTTT V3 delay_edges do not match the frozen publication bins "
                "(1-16, 17-64, 65-256, 257-1024, 1025+)"
            )
        denoise_steps = metadata.get("denoise_steps")
        if type(denoise_steps) is not int or denoise_steps <= 0:
            raise ValueError("CreditTTT V3 denoise_steps must be a positive integer")
        if str(metadata.get("dataset_repo_id")) != str(getattr(cfg.dataset, "repo_id", None)):
            raise ValueError(
                "CreditTTT V3 label dataset mismatch: artifact was generated for "
                f"{metadata.get('dataset_repo_id')!r}, training dataset is "
                f"{getattr(cfg.dataset, 'repo_id', None)!r}"
            )
        dataset_fps = getattr(getattr(dataset, "meta", None), "fps", None)
        try:
            if int(metadata["fps"]) != int(dataset_fps):
                raise ValueError(
                    f"CreditTTT V3 fps mismatch: artifact={metadata['fps']!r}, dataset={dataset_fps!r}"
                )
        except (TypeError, ValueError) as error:
            raise ValueError("CreditTTT V3 artifact/dataset fps is malformed or mismatched") from error
        expected_k = int(getattr(policy_cfg, "hd_v3_pair_k", metadata["pair_k"]))
        if int(metadata["pair_k"]) != expected_k:
            raise ValueError(
                f"CreditTTT V3 pair_k mismatch: artifact={metadata['pair_k']}, policy={expected_k}"
            )
        expected_dim = int(getattr(policy_cfg, "max_action_dim", metadata["action_dim"]))
        if int(metadata["action_dim"]) > expected_dim:
            raise ValueError("CreditTTT V3 action_dim exceeds policy.max_action_dim")
        required_v3_labels = {
            "hd_v3_pair_event_index",
            "hd_v3_pair_future_index",
            "hd_v3_pair_utility",
            "hd_v3_pair_effect",
            "hd_v3_pair_valid",
            "hd_v3_pair_positive",
            "hd_v3_pair_null",
            "hd_v3_pair_delay",
            "hd_v3_pair_delay_bin",
            "hd_v3_pair_event_end",
        }
        missing_labels = sorted(required_v3_labels - set(labeled_dataset.label_keys))
        if missing_labels:
            raise ValueError(
                "CreditTTT V3 label artifact is missing pair columns: " f"{missing_labels}"
            )
        if metadata.get("state") != "causal_fast_weights" or metadata.get("causal") is not True:
            raise ValueError(
                "CreditTTT V3 requires causal_fast_weights state with causal=true"
            )
        if metadata.get("intervention") != "event_write_deletion":
            raise ValueError(
                "CreditTTT V3 intervention identity must be event_write_deletion; "
                "use intervention_type for explicitly named ablations"
            )
        if metadata.get("intervention_scope") != (
            "event_write_only_previous_executed_action_held_fixed"
        ):
            raise ValueError(
                "CreditTTT V3 canonical training requires intervention_scope="
                "'event_write_only_previous_executed_action_held_fixed'; "
                "a whole-interaction or content-replacement intervention is a "
                "separate ablation"
            )
        # The student effect is defined by the traced state transition
        # ``W_i^- -> W_i^+``.  A donor-content replacement would require a
        # second donor-state trace, which is intentionally not part of the
        # canonical implementation.  Rejecting it here prevents an artifact
        # with a superficially valid protocol header from training against a
        # mismatched counterfactual.
        if metadata.get("intervention_type") != "delete":
            raise ValueError(
                "CreditTTT V3 canonical training requires intervention_type='delete'; "
                "content replacement is an offline ablation and needs a donor-state replay backend"
            )
        if metadata.get("canonical_delay_edges") is not True:
            raise ValueError(
                "CreditTTT V3 canonical training requires canonical_delay_edges=true; "
                "custom delay schedules are separate ablations"
            )
        if metadata.get("protocol_variant") != "canonical_event_write_deletion":
            raise ValueError(
                "CreditTTT V3 artifact is not the canonical event-write-deletion variant; "
                "use a separately named ablation output"
            )
        if metadata.get("history_mode") != "full_episode_replay":
            raise ValueError(
                "CreditTTT V3 canonical training requires history_mode='full_episode_replay'"
            )
        try:
            min_sequence_length = int(metadata["min_sequence_length"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "CreditTTT V3 min_sequence_length must be a positive integer"
            ) from error
        if min_sequence_length <= 0:
            raise ValueError("CreditTTT V3 min_sequence_length must be positive")
        if metadata.get("sequence_stride_policy") != "equal_sequence_length":
            raise ValueError(
                "CreditTTT V3 canonical training requires sequence_stride_policy="
                "'equal_sequence_length'"
            )
        if type(metadata.get("max_windows_per_episode")) is not int or metadata.get(
            "max_windows_per_episode"
        ) != 1:
            raise ValueError(
                "CreditTTT V3 canonical training requires max_windows_per_episode=1"
            )
        if metadata.get("ttt_history_warmup_length") not in {None, 0}:
            raise ValueError(
                "CreditTTT V3 canonical training requires no history warm-up"
            )
        if metadata.get("sequence_offset_policy") != "episode_local_zero":
            raise ValueError(
                "CreditTTT V3 canonical training requires episode-local zero offsets"
            )
        # Labels are generated on a (possibly larger) feature split, whereas
        # training may select a strict episode subset.  Validate the mapping
        # explicitly instead of comparing only the global maximum length: a
        # long held-out episode must not force the train sequence length, while
        # a missing/short selected episode must fail before workers start.
        slices = metadata.get("episode_slices")
        lengths_declared = metadata.get("episode_lengths")
        if not isinstance(slices, list) or not slices:
            raise ValueError("CreditTTT V3 episode_slices must be a non-empty list")
        if not isinstance(lengths_declared, list) or len(lengths_declared) != len(slices):
            raise ValueError("CreditTTT V3 episode_lengths must align with episode_slices")
        declared_lengths: dict[int, int] = {}
        for item, length_raw in zip(slices, lengths_declared, strict=True):
            if not isinstance(item, Mapping):
                raise ValueError("CreditTTT V3 episode_slices entries must be objects")
            try:
                episode_id = int(item["episode_index"])
                item_length = int(item["length"])
                declared_length = int(length_raw)
                row_start = int(item["row_start"])
                row_end = int(item["row_end"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("CreditTTT V3 episode_slices contain malformed indices/lengths") from error
            if episode_id in declared_lengths or item_length != declared_length or item_length <= 0:
                raise ValueError("CreditTTT V3 episode_slices contain duplicate or inconsistent lengths")
            if row_start < 0 or row_end - row_start != item_length:
                raise ValueError("CreditTTT V3 episode_slices row bounds are inconsistent")
            declared_lengths[episode_id] = item_length
        if int(metadata["min_sequence_length"]) != max(declared_lengths.values()):
            raise ValueError(
                "CreditTTT V3 min_sequence_length does not match episode_lengths"
            )
        meta_episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
        selected_ids: list[int] | None = None
        actual_lengths: dict[int, int] = {}
        # ``LeRobotDatasetMetadata.episodes`` is a HuggingFace Dataset in the
        # normal loader, but lightweight tests/adapters may expose a mapping
        # or a list of row mappings.  Normalize all three representations so
        # a selected short episode is not accidentally treated as an
        # artifact-wide max-length request.
        starts = ends = None
        if isinstance(meta_episodes, Mapping):
            starts = meta_episodes.get("dataset_from_index")
            ends = meta_episodes.get("dataset_to_index")
        elif meta_episodes is not None:
            column_names = getattr(meta_episodes, "column_names", None)
            if column_names and {
                "dataset_from_index",
                "dataset_to_index",
            }.issubset(set(column_names)):
                starts = meta_episodes["dataset_from_index"]
                ends = meta_episodes["dataset_to_index"]
            else:
                try:
                    rows = list(meta_episodes)
                except TypeError:
                    rows = []
                if rows and isinstance(rows[0], Mapping):
                    starts = [row.get("dataset_from_index") for row in rows]
                    ends = [row.get("dataset_to_index") for row in rows]
        if starts is not None and ends is not None:
            total = len(starts)
            selected_raw = getattr(dataset, "episodes", None)
            selected_ids = (
                list(range(total))
                if selected_raw is None
                else [int(v) for v in selected_raw]
            )
            if any(
                ep < 0 or ep >= total or starts[ep] is None or ends[ep] is None
                for ep in selected_ids
            ):
                raise ValueError("CreditTTT V3 selected dataset episodes have invalid metadata bounds")
            actual_lengths = {ep: int(ends[ep]) - int(starts[ep]) for ep in selected_ids}
        if selected_ids is not None:
            unknown = [ep for ep in selected_ids if ep not in declared_lengths]
            if unknown:
                raise ValueError(
                    "CreditTTT V3 labels do not cover selected dataset episodes: "
                    f"{unknown[:8]}"
                )
            mismatched = [
                ep for ep in selected_ids if declared_lengths[ep] != actual_lengths[ep]
            ]
            if mismatched:
                raise ValueError(
                    "CreditTTT V3 episode length mismatch for selected episodes: "
                    f"{mismatched[:8]}"
                )
            selected_max_length = max(actual_lengths.values()) if actual_lengths else 0
        else:
            # Lightweight datasets without episode metadata can only be
            # checked against the artifact-wide contract.
            selected_max_length = max(declared_lengths.values())
        if type(metadata.get("antithetic_noise")) is not bool:
            raise ValueError("CreditTTT V3 antithetic_noise must be a boolean")
        if type(metadata.get("includes_previous_executed_action")) is not bool:
            raise ValueError(
                "CreditTTT V3 includes_previous_executed_action must be a boolean"
            )
        if metadata.get("includes_previous_executed_action") is not True:
            raise ValueError(
                "CreditTTT V3 canonical labels must include the preceding executed action; "
                "omit it only in a separately named ablation"
            )
        if not isinstance(metadata.get("teacher_adapter"), str) or not metadata.get("teacher_adapter"):
            raise ValueError("CreditTTT V3 teacher_adapter must be a non-empty string")
        if type(metadata.get("flow_target_available")) is not bool:
            raise ValueError("CreditTTT V3 flow_target_available must be a boolean")
        if metadata.get("teacher_adapter") == "causal_action_head" and metadata.get("antithetic_noise"):
            raise ValueError(
                "CreditTTT direct causal_action_head artifacts cannot claim antithetic_noise=true"
            )
        if metadata.get("flow_target_available") and not metadata.get("antithetic_noise"):
            raise ValueError(
                "CreditTTT flow_target_available requires antithetic_noise=true"
            )
        if metadata.get("target_mode") != "normalized_executed_slot0_action":
            raise ValueError(
                "CreditTTT V3 target_mode must be normalized_executed_slot0_action; "
                "velocity labels cannot be mixed into QH2L"
            )
        # CMD is part of the canonical method, not an optional label column.
        # A QH2L-only run may set cmd weight to zero as an explicit ablation;
        # otherwise fail closed when the teacher action triplet is absent.
        if float(getattr(policy_cfg, "hd_v3_cmd_weight", 1.0)) > 0:
            required_cmd_labels = {
                "hd_v3_pair_teacher_full_action",
                "hd_v3_pair_teacher_counterfactual_action",
                "hd_v3_pair_expert_action",
            }
            missing_cmd = sorted(required_cmd_labels - set(labeled_dataset.label_keys))
            if missing_cmd:
                raise ValueError(
                    "CreditTTT V3 CMD is enabled but label artifact is missing action columns: "
                    f"{missing_cmd}"
                )
        # The current public V3 student path requires one complete episode
        # sequence for its sparse event/query trace.  A future query-replay
        # backend may relax this, but silently training truncated local pairs
        # would invalidate the delay claim.
        if getattr(policy_cfg, "ttt_history_warmup_length", 0) not in {None, 0}:
            raise ValueError(
                "CreditTTT V3 frame labels require ttt_history_warmup_length=None or 0"
            )
        if int(getattr(policy_cfg, "sequence_length", 0)) < selected_max_length:
            raise ValueError(
                "CreditTTT V3 sequence_length is shorter than the longest selected episode "
                f"to build labels ({getattr(policy_cfg, 'sequence_length', None)} < "
                f"{selected_max_length})"
            )
        if int(getattr(policy_cfg, "sequence_stride", 0)) != int(
            getattr(policy_cfg, "sequence_length", 0)
        ):
            raise ValueError(
                "CreditTTT V3 canonical training requires sequence_stride == sequence_length"
            )
        if getattr(policy_cfg, "max_windows_per_episode", None) != 1:
            raise ValueError(
                "CreditTTT V3 canonical training requires max_windows_per_episode=1"
            )
        labeled_dataset.hd_attribution_protocol = _HD_ATTRIBUTION_PROTOCOL_V3
        logging.info(
            "CreditTTT V3 label contract: pair_k=%s, delay_edges=%s, teacher=%s",
            metadata["pair_k"],
            metadata["delay_edges"],
            metadata["teacher_checkpoint_sha256"],
        )
        return labeled_dataset
    # A few early artifacts serialized the optional writer field as JSON null;
    # normalize it exactly like checkpoint configs before comparing protocol
    # strings.  Never turn null into the literal string ``"None"``.
    artifact_writer_mode = str(metadata.get("teacher_ttt_writer_mode") or "suffix")
    expected_writer_mode = str(getattr(policy_cfg, "ttt_writer_mode", None) or "suffix")
    if artifact_attribution_protocol == "v2_relative_antithetic_robust" and artifact_writer_mode != expected_writer_mode:
        raise ValueError(
            "HD v2 writer-mode mismatch: artifact teacher uses "
            f"{artifact_writer_mode!r}, policy.ttt_writer_mode={expected_writer_mode!r}; "
            "regenerate labels with the same prefix/suffix writer protocol"
        )
    if artifact_attribution_protocol == "v2_relative_antithetic_robust":
        if artifact_writer_mode not in {"suffix", "prefix_only"}:
            raise ValueError(
                "HD v2 labels must declare teacher_ttt_writer_mode='suffix' or 'prefix_only'"
            )
        # Validate the immutable v2 replay contract at the trainer boundary.
        # In particular, do not allow an artifact with the right protocol name
        # but legacy all-slot attribution, a single replay, or a mismatched
        # effect target to silently train.  ``effect_branches`` remains a
        # positive-count compatibility field (the student consumes slot 0).
        _validate_hd_v2_label_contract(
            metadata,
            label_keys=set(labeled_dataset.label_keys),
        )
    common_required_metadata = {
        "phase_mode",
        "history_mode",
        "event_block_size",
        "max_events",
        "grounding_event_policy",
        "grounding_min_future_frames",
        "attribution_threshold",
        "dataset_repo_id",
        "action_chunk_size",
        "max_action_dim",
        "fps",
        "frame_batch_size",
        "checkpoint",
        "teacher_checkpoint",
        "teacher_policy_type",
        "teacher_config_sha256",
        "teacher_ttt_layer_indices",
        "teacher_ttt_num_register_tokens",
        "teacher_ttt_stable_inner_update",
        "teacher_hd_ttt_enabled",
        "teacher_hd_learned_write_gate",
        "seed",
    }
    missing_common = sorted(common_required_metadata - set(metadata))
    if missing_common:
        raise ValueError(
            "HD label artifact is missing required guarded provenance fields: "
            f"{missing_common}"
        )
    if labeled_dataset.hd_window_local:
        required_metadata = {
            "sequence_length",
            "sequence_stride",
            "context_length",
            "max_windows_per_episode",
        }
        missing_metadata = sorted(required_metadata - set(metadata))
        if missing_metadata:
            raise ValueError(
                "Window-local HD artifact is missing required provenance fields: "
                f"{missing_metadata}"
            )

    expected_phase = getattr(policy_cfg, "hd_phase_mode", "random")
    artifact_phase = metadata.get("phase_mode")
    if artifact_phase not in {"random", "deployment"}:
        raise ValueError(f"HD label artifact has invalid phase_mode={artifact_phase!r}")
    if str(artifact_phase) != str(expected_phase):
        raise ValueError(
            "HD label phase mismatch: artifact uses "
            f"{artifact_phase!r}, policy.hd_phase_mode={expected_phase!r}"
        )
    checks = {
        "event_block_size": getattr(policy_cfg, "hd_event_block_size", None),
        "max_events": getattr(policy_cfg, "hd_max_events", None),
        "grounding_min_future_frames": getattr(
            policy_cfg, "hd_grounding_min_future_frames", 64
        ),
        "attribution_threshold": getattr(policy_cfg, "hd_attribution_threshold", None),
        "action_chunk_size": getattr(policy_cfg, "chunk_size", None),
        "max_action_dim": getattr(policy_cfg, "max_action_dim", None),
    }
    if labeled_dataset.hd_window_local:
        checks.update(
            {
                "sequence_length": getattr(policy_cfg, "sequence_length", None),
                "sequence_stride": getattr(policy_cfg, "sequence_stride", None),
                # ``None`` is meaningful here (full-history replay or no cap),
                # so it is an exact value rather than a wildcard.
                "context_length": getattr(policy_cfg, "ttt_history_warmup_length", None),
                "max_windows_per_episode": getattr(
                    policy_cfg, "max_windows_per_episode", None
                ),
            }
        )
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in checks.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"HD label/window contract mismatch: {mismatches}")
    artifact_grounding_policy = metadata.get("grounding_event_policy")
    if artifact_grounding_policy != _HD_GROUNDING_EVENT_POLICY:
        raise ValueError(
            "HD label grounding event policy mismatch: artifact uses "
            f"{artifact_grounding_policy!r}, expected {_HD_GROUNDING_EVENT_POLICY!r}"
        )
    artifact_grounding_horizon = metadata.get("grounding_min_future_frames")
    if type(artifact_grounding_horizon) is not int or artifact_grounding_horizon < 0:
        raise ValueError(
            "HD labels have malformed grounding_min_future_frames; "
            "expected a non-negative integer"
        )

    if metadata.get("teacher_policy_type") != "smolvla_ttt":
        raise ValueError(
            "HD labels must be generated by a SmolVLA-TTT teacher; "
            f"got {metadata.get('teacher_policy_type')!r}"
        )
    # The current offline replay explicitly controls the ordinary TTT write
    # gate and does not execute the learned HD gate.  Accepting an HD teacher
    # here would silently train from clean/all-write labels while claiming HD
    # teacher provenance, so require the explicit clean-teacher contract emitted
    # by both hindsight builders.
    for flag_name in ("teacher_hd_ttt_enabled", "teacher_hd_learned_write_gate"):
        flag_value = metadata.get(flag_name)
        if type(flag_value) is not bool:
            raise ValueError(
                f"HD labels have malformed {flag_name}; expected a JSON boolean"
            )
        if flag_value:
            raise ValueError(
                "HD labels require a clean SmolVLA-TTT teacher with "
                "teacher_hd_ttt_enabled=false and "
                "teacher_hd_learned_write_gate=false; regenerate labels with "
                "the clean/all-write replay teacher"
            )
    artifact_stable_inner_update = metadata.get("teacher_ttt_stable_inner_update")
    if type(artifact_stable_inner_update) is not bool:
        raise ValueError(
            "HD labels have malformed teacher_ttt_stable_inner_update; "
            "expected a JSON boolean"
        )
    expected_stable_inner_update = bool(
        getattr(policy_cfg, "ttt_stable_inner_update", False)
    )
    if artifact_stable_inner_update != expected_stable_inner_update:
        raise ValueError(
            "HD teacher/student stable-inner-update mismatch: artifact teacher="
            f"{artifact_stable_inner_update}, student={expected_stable_inner_update}; "
            "regenerate labels with the same ttt_stable_inner_update setting"
        )
    if (
        float(getattr(policy_cfg, "hd_effect_weight", 0.0) or 0.0) > 0.0
        and not expected_stable_inner_update
    ):
        raise ValueError(
            "HD v2 action-effect training requires policy.ttt_stable_inner_update=true; "
            "set the robust recurrence explicitly in the v2 recipe"
        )
    if str(metadata.get("teacher_checkpoint")) != str(metadata.get("checkpoint")):
        raise ValueError(
            "HD label teacher_checkpoint and checkpoint provenance disagree; regenerate the artifact"
        )
    teacher_hash = str(metadata.get("teacher_config_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", teacher_hash) is None:
        raise ValueError(
            "HD labels have an invalid teacher_config_sha256; "
            "regenerate labels with the guarded builder"
        )
    teacher_layers = metadata.get("teacher_ttt_layer_indices")
    expected_layers = list(getattr(policy_cfg, "resolved_ttt_layer_indices", ()))
    if not isinstance(teacher_layers, (list, tuple)):
        raise ValueError("HD labels have malformed teacher_ttt_layer_indices")
    try:
        teacher_layers = [int(value) for value in teacher_layers]
    except (TypeError, ValueError) as error:
        raise ValueError("HD labels have malformed teacher_ttt_layer_indices") from error
    if teacher_layers != expected_layers:
        raise ValueError(
            "HD teacher/student TTT layer mismatch: "
            f"teacher={teacher_layers}, student={expected_layers}"
        )
    try:
        teacher_registers = int(metadata.get("teacher_ttt_num_register_tokens"))
    except (TypeError, ValueError) as error:
        raise ValueError("HD labels have malformed teacher_ttt_num_register_tokens") from error
    expected_registers = int(getattr(policy_cfg, "ttt_num_register_tokens", -1))
    if teacher_registers != expected_registers:
        raise ValueError(
            "HD teacher/student register-token mismatch: "
            f"teacher={teacher_registers}, student={expected_registers}"
        )
    if labeled_dataset.hd_window_local:
        if metadata.get("history_mode") != "bounded_window_replay":
            raise ValueError(
                "Window-local HD labels must declare history_mode='bounded_window_replay'"
            )
    elif metadata.get("history_mode") != "full_episode_replay":
        raise ValueError(
            "Frame-level HD labels must declare history_mode='full_episode_replay'; "
            "regenerate them with the guarded hindsight builder"
        )
    elif getattr(policy_cfg, "ttt_history_warmup_length", 0) is not None:
        raise ValueError(
            "Frame-level full-episode HD labels require "
            "policy.ttt_history_warmup_length=None; use window-keyed labels for bounded history"
        )
    artifact_repo = metadata.get("dataset_repo_id")
    configured_repo = getattr(cfg.dataset, "repo_id", None)
    if artifact_repo is None or configured_repo is None or str(artifact_repo) != str(configured_repo):
        raise ValueError(
            "HD label dataset mismatch: artifact was generated for "
            f"{artifact_repo!r}, training dataset is {configured_repo!r}"
        )
    # The dataset root is intentionally not part of the identity check: a
    # byte-identical LeRobot tree may be staged at a different local path on a
    # worker.  Dataset identity and sampling rate are stable, however, and a
    # mismatch would invalidate frame-indexed hindsight credits.
    artifact_fps = metadata.get("fps")
    dataset_meta = getattr(dataset, "meta", None)
    dataset_fps = getattr(dataset_meta, "fps", None)
    if artifact_fps is None or dataset_fps is None:
        raise ValueError("HD label/dataset provenance is missing fps")
    try:
        artifact_fps_int = int(artifact_fps)
        dataset_fps_int = int(dataset_fps)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"HD label/dataset fps must be integer-like, got artifact={artifact_fps!r}, "
            f"dataset={dataset_fps!r}"
        ) from error
    if type(artifact_fps) is bool or artifact_fps_int != dataset_fps_int:
        raise ValueError(
            "HD label fps mismatch: artifact was generated at "
            f"{artifact_fps!r} Hz, training dataset is {dataset_fps!r} Hz"
        )

    # ``teacher_config_sha256`` is the strongest portable identity for the
    # clean replay teacher.  Verify it against the exact config currently
    # requested by the training run, not merely against the path string stored
    # in the label artifact.
    artifact_teacher_hash = str(metadata.get("teacher_config_sha256", ""))
    teacher_checkpoint = _configured_hd_teacher_checkpoint(cfg)
    current_teacher_hash = _teacher_config_sha256(teacher_checkpoint)
    if artifact_teacher_hash != current_teacher_hash:
        raise ValueError(
            "HD teacher config mismatch: label artifact SHA256="
            f"{artifact_teacher_hash}, current pretrained_path SHA256={current_teacher_hash}; "
            "regenerate labels with the exact clean teacher used for training"
        )
    if (
        labeled_dataset.hd_window_local
        and not labeled_dataset.hd_window_keyed
        and getattr(policy_cfg, "sequence_stride", None)
        != getattr(policy_cfg, "sequence_length", None)
    ):
        raise ValueError(
            "window-local HD labels require sequence_stride == sequence_length; "
            "use a window-keyed artifact for overlapping windows"
        )
    logging.info(
        "HD label contract: phase=%s, attribution=%s, writer=%s, window_local=%s, teacher=%s (config_sha256=%s)",
        artifact_phase,
        artifact_attribution_protocol,
        expected_writer_mode,
        labeled_dataset.hd_window_local,
        metadata.get("checkpoint", "unknown"),
        current_teacher_hash,
    )
    logging.info("HD teacher provenance path used for hash: %s", teacher_checkpoint)
    return labeled_dataset


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        sample_weighter: Optional SampleWeighter instance for per-sample loss weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if sample_weights is not None:
            # Use per-sample loss for weighted training
            # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Weighted loss: each sample's contribution is scaled by its weight.
            # We divide by weight sum (not batch size) so that if some weights are zero,
            # the remaining samples contribute proportionally more, preserving gradient scale.
            # Weights are pre-normalized to sum to batch_size for stable training dynamics.
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

            # Log weighting statistics
            if output_dict is None:
                output_dict = {}
            for key, value in weight_stats.items():
                output_dict[f"sample_weight_{key}"] = value
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def _slice_flattened_sequence_batch(
    batch: Any,
    batch_size: int,
    sequence_length: int,
    start: int,
    end: int,
) -> Any:
    """Select a time segment from data flattened in batch-major ``B*T`` order."""
    flat_batch_size = batch_size * sequence_length
    flat_indices = [
        batch_index * sequence_length + timestep
        for batch_index in range(batch_size)
        for timestep in range(start, end)
    ]

    def select(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == flat_batch_size:
            indices = torch.tensor(flat_indices, dtype=torch.long, device=value.device)
            return value.index_select(0, indices)
        if isinstance(value, list) and len(value) == flat_batch_size:
            return [value[index] for index in flat_indices]
        if isinstance(value, dict):
            return {key: select(nested_value) for key, nested_value in value.items()}
        return value

    return select(batch)


def _sequence_offset_from_batch(batch: Mapping[str, Any] | dict[str, Any]) -> int:
    """Read the episode-local origin preserved by the SmolVLA-TTT collator.

    ``sequence_collate_fn`` emits one scalar, but accepting an older repeated
    ``[T]`` representation keeps checkpoints/data-loader workers compatible.
    A malformed or mixed origin is rejected because falling back to zero would
    silently misalign event/future CreditTTT labels.
    """

    value = batch.get(SEQUENCE_OFFSET_KEY)
    if value is None:
        return 0
    try:
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except Exception as exc:
        raise ValueError(f"{SEQUENCE_OFFSET_KEY!r} must be an integer scalar") from exc
    if tensor.numel() == 0:
        raise ValueError(f"{SEQUENCE_OFFSET_KEY!r} cannot be empty")
    flattened = tensor.reshape(-1)
    if tensor.is_floating_point():
        if not bool(torch.isfinite(flattened).all().item()):
            raise ValueError(f"{SEQUENCE_OFFSET_KEY!r} must be finite")
        if not bool((flattened == flattened.round()).all().item()):
            raise ValueError(f"{SEQUENCE_OFFSET_KEY!r} must contain integral values")
    if not bool((flattened == flattened[0]).all().item()):
        raise ValueError(
            f"All sequence rows must share {SEQUENCE_OFFSET_KEY!r}; "
            f"got {flattened.detach().cpu().tolist()}"
        )
    offset = int(flattened[0].item())
    if offset < 0:
        raise ValueError(f"{SEQUENCE_OFFSET_KEY!r} must be non-negative, got {offset}")
    return offset


def _compute_hd_effect_normalization_floor(
    batch: dict[str, Any],
    sequence_shape: tuple[int, int],
    policy_config: Any,
) -> torch.Tensor | None:
    """Compute one robust action-effect floor for a complete TBPTT window.

    ``action_effect_distillation_loss`` normally estimates its floor from the
    tensors passed to one call.  That is undesirable for TBPTT because each
    segment would then have a different median.  This helper mirrors both
    legacy/v2 per-frame labels (flattened ``B*T`` rows, optional event axis,
    selected slot 0) and canonical V3 pair labels (``[B,T,K,D]``): the
    complete-window effect population is assembled *before* the sequence is
    sliced, and the detached scalar is reused by every segment.  The active
    action coordinates are limited to the configured task dimension so padded
    model coordinates do not alter the statistic.
    """

    teacher_effect = batch.get("hd_teacher_effect")
    pair_effect_field = batch.get("hd_v3_pair_effect")
    # A correctly generated V3 artifact only needs the pair field, but a
    # migration/merge step may temporarily carry both legacy and V3 columns.
    # Let the declared protocol select the canonical population in that case:
    # V3 must never silently fall back to the legacy slot-0 statistic, while
    # callers without an explicit V3 protocol retain the historical field
    # precedence and direct-helper behavior.
    declared_protocol = str(getattr(policy_config, "hd_attribution_protocol", ""))
    # ``SmolVLATTTConfig`` canonicalizes these aliases at construction time;
    # accepting them here as well keeps the standalone trainer helper
    # deterministic for lightweight callers/tests that use a namespace.
    declared_v3 = declared_protocol in {
        _HD_ATTRIBUTION_PROTOCOL_V3,
        "credit_ttt_v3",
        "v3",
    }
    pair_effect = bool(
        pair_effect_field is not None
        and (teacher_effect is None or declared_v3)
    )
    if pair_effect:
        teacher_effect = pair_effect_field
    if teacher_effect is None:
        return None
    if not isinstance(teacher_effect, torch.Tensor) or teacher_effect.ndim == 0:
        field_name = "hd_v3_pair_effect" if pair_effect else "hd_teacher_effect"
        raise ValueError(f"{field_name} must be a non-scalar tensor")

    batch_size, sequence_length = sequence_shape
    if (
        teacher_effect.ndim >= 2
        and teacher_effect.shape[0] == batch_size
        and teacher_effect.shape[1] == sequence_length
    ):
        effect = teacher_effect
    elif teacher_effect.shape[0] == batch_size * sequence_length:
        effect = teacher_effect.reshape(batch_size, sequence_length, *teacher_effect.shape[1:])
    elif (
        teacher_effect.ndim >= 2
        and teacher_effect.shape[0] == sequence_length
        and batch_size != sequence_length
    ):
        # Match ``_reshape_hd_field`` for a shared [T,...] label field.
        effect = teacher_effect.unsqueeze(0).expand(batch_size, *teacher_effect.shape)
    else:
        field_name = "hd_v3_pair_effect" if pair_effect else "hd_teacher_effect"
        raise ValueError(
            f"{field_name} must start with [B,T] or flattened B*T dimensions, "
            f"got {tuple(teacher_effect.shape)} for sequence {sequence_shape}"
        )

    if pair_effect:
        # V3 supervises sampled event--future pairs.  Keep every pair in the
        # floor population (rather than selecting slot 0 as in v2), while
        # excluding invalid padded rows when the mask is available.  Invalid
        # rows are normally zero, but filtering them makes the statistic
        # robust to malformed/legacy artifacts without changing valid labels.
        if effect.ndim == 3:
            # Accept a compact K=1 spelling ``[B,T,D]`` for old collators.
            effect = effect.unsqueeze(2)
        elif effect.ndim != 4:
            raise ValueError(
                "hd_v3_pair_effect must have [B,T,K,D] (or compact [B,T,D]) shape, "
                f"got {tuple(effect.shape)}"
            )
        valid_field = batch.get("hd_v3_pair_valid")
        if valid_field is not None:
            if not isinstance(valid_field, torch.Tensor) or valid_field.ndim == 0:
                raise ValueError("hd_v3_pair_valid must be a non-scalar tensor")
            if (
                valid_field.ndim >= 2
                and valid_field.shape[0] == batch_size
                and valid_field.shape[1] == sequence_length
            ):
                valid = valid_field
            elif valid_field.shape[0] == batch_size * sequence_length:
                valid = valid_field.reshape(batch_size, sequence_length, *valid_field.shape[1:])
            elif (
                valid_field.ndim >= 1
                and valid_field.shape[0] == sequence_length
                and batch_size != sequence_length
            ):
                valid = valid_field.unsqueeze(0).expand(batch_size, *valid_field.shape)
            else:
                raise ValueError(
                    "hd_v3_pair_valid must start with [B,T] or flattened B*T dimensions, "
                    f"got {tuple(valid_field.shape)} for sequence {sequence_shape}"
                )
            if valid.ndim == 2:
                valid = valid.unsqueeze(-1)
            if valid.shape[:3] != effect.shape[:3]:
                try:
                    valid = torch.broadcast_to(valid, effect.shape[:3])
                except RuntimeError as exc:
                    raise ValueError(
                        "hd_v3_pair_valid leading shape must align with hd_v3_pair_effect; "
                        f"got {tuple(valid.shape)} vs {tuple(effect.shape[:3])}"
                    ) from exc
            effect = effect[valid.bool()]
        else:
            effect = effect.reshape(-1, effect.shape[-1])
    else:
        if effect.ndim == 4:
            if effect.shape[2] < 1:
                raise ValueError("hd_teacher_effect must contain at least one event slot")
            effect = effect[:, :, 0, :]
        elif effect.ndim != 3:
            raise ValueError(
                "hd_teacher_effect must have [B,T,D] or [B,T,K,D] shape, "
                f"got {tuple(effect.shape)}"
            )

    feature = getattr(policy_config, "action_feature", None)
    feature_shape = getattr(feature, "shape", None)
    configured_dim = (
        int(feature_shape[0])
        if feature_shape and feature_shape[0] is not None
        else int(effect.shape[-1])
    )
    active_dim = min(int(effect.shape[-1]), configured_dim)
    if active_dim <= 0:
        raise ValueError("hd_teacher_effect requires a positive active action dimension")

    # Import lazily so ordinary policies do not import HD-TTT tensor helpers.
    from lerobot.policies.smolvla_ttt.hd_ttt import compute_action_effect_normalization_floor

    return compute_action_effect_normalization_floor(effect[..., :active_dim])


def _tbptt_segment_loss_weights(
    batch: dict[str, Any],
    sequence_shape: tuple[int, int],
    segment_length: int,
    *,
    weight_by_valid_actions: bool,
    include_writer_valid: bool = False,
) -> list[float]:
    """Return segment weights for action and optional HD-writer supervision.

    Ordinary TTT uses valid action counts so padded action chunks do not bias
    the sequence mean.  HD windows can additionally contain replayed history
    frames whose ``action_is_pad`` flag is intentionally true; when requested,
    those physical interactions are counted through ``hd_writer_valid`` so a
    warm-up-only segment still contributes its gate/H2L gradients.
    """
    batch_size, sequence_length = sequence_shape
    actions_is_pad = batch.get("action_is_pad") if weight_by_valid_actions else None
    segment_valid_counts: list[int] = []

    if actions_is_pad is not None:
        if not isinstance(actions_is_pad, torch.Tensor):
            raise TypeError("action_is_pad must be a tensor for TTT sequence training")
        expected_flat_batch = batch_size * sequence_length
        if actions_is_pad.ndim < 2 or actions_is_pad.shape[0] != expected_flat_batch:
            raise ValueError(
                "action_is_pad must have flattened batch-major shape "
                f"[{expected_flat_batch}, ...], got {tuple(actions_is_pad.shape)}"
            )
        action_padding = actions_is_pad.reshape(batch_size, sequence_length, -1)

    writer_valid = batch.get("hd_writer_valid") if include_writer_valid else None
    if writer_valid is not None:
        if not isinstance(writer_valid, torch.Tensor):
            raise TypeError("hd_writer_valid must be a tensor for TTT sequence training")
        writer_valid = writer_valid.reshape(-1)
        if writer_valid.numel() != batch_size * sequence_length:
            raise ValueError(
                "hd_writer_valid must contain one value per flattened sequence frame "
                f"[{batch_size * sequence_length}], got {tuple(writer_valid.shape)}"
            )
        writer_valid = writer_valid.reshape(batch_size, sequence_length).bool()

    for segment_start in range(0, sequence_length, segment_length):
        segment_end = min(segment_start + segment_length, sequence_length)
        if actions_is_pad is None:
            action_valid_count = batch_size * (segment_end - segment_start)
        elif writer_valid is not None:
            # HD auxiliary losses are normalized per physical interaction,
            # not per action-chunk slot.  Count a target frame once here;
            # otherwise a 50-slot action chunk would down-weight a warm-up
            # writer segment by roughly 50x even though its gate/H2L loss is
            # itself a per-frame mean.
            action_valid_count = int(
                (~action_padding[:, segment_start:segment_end]).any(dim=-1).sum().item()
            )
        else:
            action_valid_count = int((~action_padding[:, segment_start:segment_end]).sum().item())
        if writer_valid is not None:
            writer_valid_count = int(writer_valid[:, segment_start:segment_end].sum().item())
            # The segment contains two internally normalized objectives.  A
            # union/max count gives zero-action warm-up segments non-zero
            # weight while retaining the usual action weighting elsewhere.
            valid_count = max(action_valid_count, writer_valid_count)
        else:
            valid_count = action_valid_count
        segment_valid_counts.append(valid_count)

    total_valid_count = sum(segment_valid_counts)
    if total_valid_count == 0:
        raise ValueError("A TTT training sequence must contain at least one supervised action")
    return [valid_count / total_valid_count for valid_count in segment_valid_counts]


def _sequence_valid_action_slots(
    batch: Mapping[str, Any] | dict[str, Any], sequence_shape: tuple[int, int]
) -> int:
    """Count valid action slots in one flattened sequence batch.

    ``TailPreservingSequenceDataset`` keeps the time axis flattened as
    ``B*T``.  The flow loss averages over valid action *slots* (the chunk axis)
    and then over feature dimensions, so this is the measure that must be used
    when combining unequal-length ranks.  Keeping the count in one helper also
    makes the distributed weighting contract match
    :func:`_tbptt_segment_loss_weights` exactly.
    """

    batch_size, sequence_length = sequence_shape
    expected_flat_batch = batch_size * sequence_length
    action_is_pad = batch.get("action_is_pad")
    if action_is_pad is None:
        return expected_flat_batch
    if not isinstance(action_is_pad, torch.Tensor):
        raise TypeError("action_is_pad must be a tensor for TTT sequence training")
    if action_is_pad.ndim < 2 or action_is_pad.shape[0] != expected_flat_batch:
        raise ValueError(
            "action_is_pad must have flattened batch-major shape "
            f"[{expected_flat_batch}, ...], got {tuple(action_is_pad.shape)}"
        )
    return int((~action_is_pad.bool()).reshape(expected_flat_batch, -1).sum().item())


def _ddp_frame_weighted_flow_scale(
    batch: Mapping[str, Any] | dict[str, Any],
    sequence_shape: tuple[int, int],
    accelerator: "Accelerator",
    *,
    enabled: bool,
) -> tuple[float, int, int]:
    """Return a rank scale that makes a variable-length DDP flow mean exact.

    ``EqualLengthBatchSampler`` guarantees equal ``T`` *within* each rank,
    but its global batch stream can assign different length buckets to ranks
    at the same optimizer step.  A plain mean of rank-local losses would then
    give a short trajectory the same weight as a long one.  If enabled, scale
    rank ``r``'s local flow mean by ``P*n_r/N`` before the trainer's explicit
    DDP gradient mean, where ``n_r`` is its valid action-slot count and ``N``
    is the all-rank count.  The resulting gradient is exactly the global
    frame/slot-weighted mean.  B=1 and non-equal-length paths never call this
    helper, preserving their historical semantics.

    The returned tuple is ``(rank_scale, local_count, global_count)``.  Counts
    are integers for provenance/diagnostics; the scale is a Python float so it
    can be multiplied into the existing per-segment weights without adding a
    tensor to the autograd graph.
    """

    local_count = _sequence_valid_action_slots(batch, sequence_shape)
    if not enabled or accelerator.num_processes <= 1:
        return 1.0, local_count, local_count

    count_tensor = torch.tensor(
        float(local_count), dtype=torch.float32, device=accelerator.device
    )
    global_tensor = accelerator.reduce(count_tensor, reduction="sum")
    if not isinstance(global_tensor, torch.Tensor):
        raise RuntimeError("Accelerator.reduce must return a tensor for flow-slot weighting")
    global_count = int(round(float(global_tensor.detach().item())))
    if global_count <= 0:
        raise ValueError("Distributed TTT sequence batches contain no valid action slots")
    rank_scale = float(accelerator.num_processes * local_count / global_count)
    return rank_scale, local_count, global_count


def _v3_ddp_pair_normalizers(
    policy: torch.nn.Module,
    reference_batch: Mapping[str, Any] | None,
    sequence_shape: tuple[int, int],
    accelerator: "Accelerator",
    *,
    enabled: bool,
) -> dict[str, torch.Tensor] | None:
    """Return all-rank V3 stratum denominators for explicit DDP training.

    The V3 primitives normalize each stratum by a detached complete-window
    denominator.  With a trajectory batch, computing that scalar independently
    on every rank makes the final explicit gradient mean an *average of local
    ratios*.  When enabled, this helper sums the three local denominators over
    ranks and returns ``global_denominator / world_size``.  Dividing by the
    world size is deliberate: the sequence trainer reduces gradients with a
    mean after the forward pass, so the two factors cancel and the resulting
    gradient is the global pair-weighted numerator divided by the global
    denominator.

    The helper is deliberately a no-op for single-process/B=1 compatibility,
    for missing reference windows, or when the policy switch is false.  It
    invokes the model's existing private label-preparation routine instead of
    reimplementing pair validity/utility rules, keeping the protocol identical
    to the direct V3 path.  A missing pair artifact contributes zero
    denominators but still participates in all three collectives, preventing a
    rank-order mismatch if a malformed/empty batch reaches this boundary.
    """

    if not enabled or accelerator.num_processes <= 1 or reference_batch is None:
        return None
    prepare_labels = getattr(policy, "_prepare_v3_pair_labels", None)
    normalizer_fn = getattr(policy, "_v3_pair_normalizers", None)
    if not callable(prepare_labels) or not callable(normalizer_fn):
        raise TypeError(
            "Canonical V3 DDP normalization requires the SmolVLA policy pair-label helpers"
        )
    reference_shape_fn = getattr(policy, "_v3_reference_sequence_shape", None)
    if callable(reference_shape_fn):
        reference_shape = reference_shape_fn(reference_batch)
    else:
        reference_shape = sequence_shape
    reference_offset = _sequence_offset_from_batch(reference_batch)
    pair_labels = prepare_labels(
        reference_batch,
        reference_shape,
        sequence_offset=reference_offset,
        allow_cross_segment=True,
    )
    local_normalizers = normalizer_fn(pair_labels)
    zero = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    global_normalizers: dict[str, torch.Tensor] = {}
    for name in ("full", "positive", "null"):
        local_value = zero
        if local_normalizers is not None and name in local_normalizers:
            local_value = torch.as_tensor(
                local_normalizers[name], device=accelerator.device, dtype=torch.float32
            ).detach()
            if local_value.numel() != 1:
                raise ValueError(f"V3 pair normalizer {name!r} must be scalar")
            local_value = local_value.reshape(())
        reduced = accelerator.reduce(local_value, reduction="sum")
        if not isinstance(reduced, torch.Tensor) or reduced.numel() != 1:
            raise RuntimeError("Accelerator.reduce returned an invalid V3 pair normalizer")
        global_normalizers[name] = (
            reduced.detach() / float(accelerator.num_processes)
        ).reshape(())
    return global_normalizers


def _ddp_reduce_gradients(
    policy: torch.nn.Module,
    accelerator: "Accelerator",
) -> None:
    """Synchronize sequence-policy gradients with a small number of collectives.

    Sequence TTT deliberately calls the *unwrapped* policy so its recurrent
    fast state can be carried across TBPTT segments.  Consequently Accelerate
    cannot install DDP gradient hooks and the trainer has to reduce gradients
    explicitly.  Reducing one tensor per parameter is prohibitively expensive
    for SmolVLA (and becomes especially visible on four PCIe-connected GPUs),
    so dense gradients are flattened per ``(device, dtype)`` group, reduced
    once, and copied back to their individual ``.grad`` buffers.

    The first collective preserves the conditional-branch contract used by the
    old implementation: a parameter that is unused on one rank receives an
    explicit zero there whenever another rank used it.  A second, tiny
    presence vector identifies unusual sparse/layout-mismatched gradients; the
    corresponding parameters take the compatibility per-parameter path on all
    ranks, preventing collective-order divergence.  Ordinary dense models use
    one all-reduce per dtype/device group instead of one per parameter.  The
    function is only called for multi-process sequence training; the B=1 /
    single-process path is therefore untouched.
    """

    if accelerator.num_processes <= 1:
        return

    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        return

    # Determine which parameters are used anywhere in the data-parallel
    # group before constructing the flattened buffers.  ``accelerator.reduce``
    # clones its input, so this remains a tiny O(number-of-parameters) tensor.
    local_presence = torch.tensor(
        [parameter.grad is not None for parameter in trainable_parameters],
        dtype=torch.int32,
        device=accelerator.device,
    )
    global_presence = accelerator.reduce(local_presence, reduction="sum")
    if not isinstance(global_presence, torch.Tensor):
        raise RuntimeError("Accelerator.reduce must return a tensor for gradient presence")
    global_presence = global_presence.reshape(-1)
    if global_presence.numel() != len(trainable_parameters):
        raise RuntimeError(
            "Gradient-presence reduction returned an unexpected shape: "
            f"got {tuple(global_presence.shape)}, expected ({len(trainable_parameters)},)"
        )
    # Materialize the tiny control vectors once. Calling ``.item()`` inside
    # the parameter loop would re-synchronize the CUDA stream for every
    # parameter and erase the benefit of flattened gradient reduction.
    global_presence_values = global_presence.detach().cpu().tolist()

    # Dense flattening requires the same layout/device/dtype on every rank.
    # Usually this is guaranteed by a deterministic model and autocast, but
    # discover anomalous local gradients collectively so a sparse or cast
    # branch cannot make ranks enter different numbers of all-reduces.
    local_unsupported = torch.tensor(
        [
            int(
                int(global_presence_values[index]) > 0
                and (
                    parameter.layout != torch.strided
                    or (
                        parameter.grad is not None
                        and (
                            parameter.grad.layout != torch.strided
                            or parameter.grad.device != parameter.device
                            or parameter.grad.dtype != parameter.dtype
                        )
                    )
                )
            )
            for index, parameter in enumerate(trainable_parameters)
        ],
        dtype=torch.int32,
        device=accelerator.device,
    )
    global_unsupported = accelerator.reduce(local_unsupported, reduction="sum")
    if not isinstance(global_unsupported, torch.Tensor):
        raise RuntimeError("Accelerator.reduce must return a tensor for gradient layouts")
    global_unsupported = global_unsupported.reshape(-1)
    if global_unsupported.numel() != len(trainable_parameters):
        raise RuntimeError(
            "Gradient-layout reduction returned an unexpected shape: "
            f"got {tuple(global_unsupported.shape)}, expected ({len(trainable_parameters)},)"
        )
    global_unsupported_values = global_unsupported.detach().cpu().tolist()

    dense_groups: dict[tuple[torch.device, torch.dtype], list[tuple[torch.nn.Parameter, torch.Tensor]]] = {}
    fallback_parameters: list[torch.nn.Parameter] = []

    for index, parameter in enumerate(trainable_parameters):
        if int(global_presence_values[index]) <= 0:
            # No rank produced a gradient, matching the historical behavior
            # of leaving ``parameter.grad`` as ``None``.
            continue
        if int(global_unsupported_values[index]) > 0:
            fallback_parameters.append(parameter)
            continue
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        # The collective layout check above should make this branch true on
        # every rank. Keep a defensive fallback for unusual custom modules or
        # test doubles that mutate a gradient between the two collectives.
        if (
            gradient.layout != torch.strided
            or gradient.device != parameter.device
            or gradient.dtype != parameter.dtype
            or parameter.layout != torch.strided
            or parameter.numel() == 0
        ):
            fallback_parameters.append(parameter)
            continue
        key = (parameter.device, parameter.dtype)
        dense_groups.setdefault(key, []).append((parameter, gradient))

    for entries in dense_groups.values():
        parameters, gradients = zip(*entries, strict=True)
        flat = torch._utils._flatten_dense_tensors(list(gradients))
        reduced_flat = accelerator.reduce(flat, reduction="mean")
        if not isinstance(reduced_flat, torch.Tensor) or reduced_flat.shape != flat.shape:
            raise RuntimeError("Accelerator.reduce returned an invalid flattened gradient")
        reduced_gradients = torch._utils._unflatten_dense_tensors(
            reduced_flat, list(gradients)
        )
        for parameter, reduced_gradient in zip(parameters, reduced_gradients, strict=True):
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.copy_(reduced_gradient)
        # ``reduced_gradients`` are views into ``reduced_flat``; deleting both
        # here avoids retaining a model-sized flat buffer until the optimizer
        # step while leaving each parameter's own gradient in place.
        del reduced_gradients, reduced_flat, flat

    for parameter in fallback_parameters:
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        if gradient.layout != torch.strided:
            # Sparse gradients are uncommon for this policy, but densifying
            # them keeps all ranks on the same dense collective contract when
            # only one rank took a sparse branch.
            gradient = gradient.to_dense()
        if gradient.device != parameter.device or gradient.dtype != parameter.dtype:
            gradient = gradient.to(device=parameter.device, dtype=parameter.dtype)
        reduced_gradient = accelerator.reduce(gradient, reduction="mean")
        if not isinstance(reduced_gradient, torch.Tensor):
            raise RuntimeError("Accelerator.reduce returned an invalid fallback gradient")
        if parameter.layout == torch.strided:
            if parameter.grad is None or parameter.grad.layout != torch.strided:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.copy_(reduced_gradient)
        else:
            # Preserve the only non-dense parameter layout supported by this
            # compatibility branch; optimizer implementations that support it
            # can consume the reduced sparse tensor directly.
            parameter.grad = reduced_gradient


def update_policy_tbptt(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: dict[str, Any],
    sequence_shape: tuple[int, int],
    segment_length: int,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    *,
    optimizer_step: bool = True,
    zero_grad_before: bool = True,
    gradient_scale: float = 1.0,
) -> tuple[MetricsTracker, dict]:
    """Update a TTT policy over one sequence while truncating gradients between segments.

    Each accelerator process owns an independent trajectory window and its local fast state. The
    persistent PI0-TTT parameters are synchronized once, after all TBPTT segments have contributed
    gradients and before clipping/stepping the optimizer. This preserves trajectory-local memory while
    implementing data parallel training for the outer parameters.

    ``optimizer_step=False`` is an explicit gradient-accumulation mode used by
    the sequence trainer.  In that mode this function still executes exactly one
    complete sequence window (including its own fast-state reset), but leaves its
    outer gradients in ``optimizer`` for a later call.  The final call performs
    the one DDP reduction, clipping, optimizer/scheduler step, and buffer update.
    ``gradient_scale`` scales every backward contribution, including streamed
    CreditTTT V3 replay callbacks, while leaving reported losses unscaled.
    The defaults preserve the historical one-window/one-step behavior exactly.
    """

    if not math.isfinite(float(gradient_scale)) or gradient_scale <= 0:
        raise ValueError(f"gradient_scale must be finite and positive, got {gradient_scale!r}")

    start_time = time.perf_counter()
    policy.train()
    if zero_grad_before:
        optimizer.zero_grad()
    unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    batch_size, sequence_length = sequence_shape
    # The sampler's origin is episode-local and may be non-zero when a window
    # contains only a bounded warm-up prefix.  TBPTT segment offsets are
    # relative to that window, so retain the physical origin before slicing.
    window_sequence_offset = _sequence_offset_from_batch(batch)
    fast_states = None
    policy_config = getattr(unwrapped_policy, "config", None)
    finite_guard_enabled = _ttt_finite_guard_enabled(policy_config)
    # v2 HD batches contain per-frame action-effect targets.  Their auxiliary
    # terms are normalized by the complete physical window inside the policy,
    # so the trainer must weight only the flow numerator per segment.  This is
    # deliberately inferred from the artifact/protocol rather than exposed as
    # another user knob: changing TBPTT length cannot silently change the
    # hindsight objective.
    use_global_hd_normalization = bool(
        getattr(policy_config, "hd_ttt_enabled", False)
        and str(getattr(policy_config, "hd_attribution_protocol", ""))
        in {"v2", "v2_relative_antithetic_robust", _HD_ATTRIBUTION_PROTOCOL_V3}
        and (
            batch.get("hd_teacher_effect") is not None
            or batch.get("hd_v3_pair_effect") is not None
        )
    )
    # Imported lazily so ordinary/non-SmolVLA training does not pull the
    # heavyweight SmolVLA model module merely for a logging helper.
    hd_balance_metrics = None
    if bool(getattr(policy_config, "hd_ttt_enabled", False)):
        from lerobot.policies.smolvla_ttt.modeling_smolvla_ttt import _hd_loss_balance_metrics

        hd_balance_metrics = _hd_loss_balance_metrics
    # Hindsight grounding has two detached counterfactual trajectories.  Keep
    # them across TBPTT segments of this one sequence so an early zero-write
    # intervention remains present in later reads.  Ordinary TTT/clean batches
    # do not carry these fields and retain the original call path.
    grounding_states = None
    if bool(getattr(getattr(unwrapped_policy, "config", None), "hd_ttt_enabled", False)):
        has_reader_grounding = all(
            batch.get(key) is not None
            for key in (
                "hd_counterfactual_write_gate",
                "hd_teacher_true_velocity",
                "hd_teacher_wrong_velocity",
            )
        )
        has_effect_grounding = all(
            batch.get(key) is not None
            for key in ("hd_teacher_effect", "hd_effect_write_gate")
        )
        if has_reader_grounding or has_effect_grounding:
            # The model adds ``effect_true/effect_wrong`` lazily when v2
            # labels are present.  Keeping one container preserves both the
            # historical detached reader branches and the new
            # writer-gradient effect branches across TBPTT segments.
            grounding_states = {"true": None, "wrong": None}
    total_loss = torch.zeros((), device=accelerator.device)
    loss_per_dim = None
    num_segments = 0
    auxiliary_metric_sums: dict[str, float] = {}
    segment_loss_weights = _tbptt_segment_loss_weights(
        batch,
        sequence_shape,
        segment_length,
        weight_by_valid_actions=(
            getattr(unwrapped_policy, "tbptt_loss_weighting", None) == "valid_actions"
        ),
        # With v2's separate flow/HD paths, flow is weighted only by valid
        # action slots.  Legacy/ordinary calls retain the prior union with
        # writer-valid warm-up frames.
        include_writer_valid=(
            False
            if use_global_hd_normalization
            else bool(
                getattr(policy_config, "hd_ttt_enabled", False)
                and "hd_writer_valid" in batch
            )
        ),
    )
    # Exact-length batching keeps every local batch rectangular, but the
    # sampler may still hand different physical ``T`` buckets to different
    # DDP ranks at one optimizer step.  For the ordinary flow objective (and
    # V3, whose auxiliary terms are already normalized per complete window),
    # combine those rank-local means by valid action-slot count.  The trainer
    # performs an explicit mean reduction of gradients below, so multiplying
    # rank ``r`` by ``P*n_r/N`` yields the global frame-weighted mean.  Keep
    # this opt-in to ``B>1`` equal-length runs: historical B=1 and legacy HD
    # paths retain their exact weighting and loss decomposition.
    frame_weighted_ddp = bool(
        batch_size > 1
        and accelerator.num_processes > 1
        and getattr(policy_config, "equal_length_batching", False)
        and getattr(unwrapped_policy, "tbptt_loss_weighting", None) == "valid_actions"
        and (
            not bool(getattr(policy_config, "hd_ttt_enabled", False))
            or use_global_hd_normalization
        )
    )
    if frame_weighted_ddp:
        flow_rank_scale, _, _ = _ddp_frame_weighted_flow_scale(
            batch,
            sequence_shape,
            accelerator,
            enabled=True,
        )
    else:
        # Do not even inspect ``action_is_pad`` on historical paths.  Apart
        # from avoiding an unnecessary host sync, this keeps malformed or
        # legacy auxiliary metadata from changing B=1 behavior.
        flow_rank_scale = 1.0
    hd_normalization_denominator = (
        float(batch_size * sequence_length) if use_global_hd_normalization else None
    )
    # ``SEQUENCE_SHAPE_KEY`` is consumed (popped) by the outer training loop
    # before preprocessing.  CreditTTT replay nevertheless needs the complete
    # reference-window shape to gather future observations and to compute
    # episode-level pair denominators.  Keep a shallow metadata-enriched view;
    # tensors remain shared and no second copy of the sequence is allocated.
    v3_reference_batch = None
    if str(getattr(policy_config, "hd_attribution_protocol", "")) == _HD_ATTRIBUTION_PROTOCOL_V3:
        v3_reference_batch = dict(batch)
        v3_reference_batch[SEQUENCE_SHAPE_KEY] = torch.tensor(
            [batch_size, sequence_length], dtype=torch.int64, device=accelerator.device
        )
    v3_pair_normalizers = _v3_ddp_pair_normalizers(
        unwrapped_policy,
        v3_reference_batch,
        sequence_shape,
        accelerator,
        enabled=bool(
            str(getattr(policy_config, "hd_attribution_protocol", ""))
            == _HD_ATTRIBUTION_PROTOCOL_V3
            and getattr(policy_config, "hd_v3_global_pair_normalization", True)
        ),
    )
    # Estimate the robust action-effect floor from the complete physical
    # window, before TBPTT slicing.  Reusing this detached scalar in every
    # segment keeps action-effect normalization invariant to segment length while
    # retaining the per-timestep RMS normalization inside the loss.
    effect_normalization_floor = (
        _compute_hd_effect_normalization_floor(batch, sequence_shape, policy_config)
        if use_global_hd_normalization
        else None
    )

    # Canonical CreditTTT V3 replay can be substantially larger than the
    # action-flow batch (one event may supervise several future queries).  The
    # SmolVLA policy accepts this callback to backward each CMD/QH2L pair
    # chunk synchronously and release its checkpoint graph before constructing
    # the next one.  ``accelerator.backward`` is intentionally kept here,
    # outside the model, so the policy remains framework-agnostic and the
    # historical non-V3/B=1 call path is unchanged.
    stream_v3_replay = bool(
        str(getattr(policy_config, "hd_attribution_protocol", ""))
        == _HD_ATTRIBUTION_PROTOCOL_V3
        and use_global_hd_normalization
    )

    def _stream_v3_backward(loss: torch.Tensor, retain_graph: bool) -> None:
        # Streaming replay invokes backward inside the policy.  Apply the same
        # accumulation scale as the ordinary segment loss so QH2L/CMD retain
        # their objective ratio when several independent windows are averaged.
        accelerator.backward(loss * gradient_scale, retain_graph=retain_graph)

    local_num_segments = len(segment_loss_weights)
    # Tail-preserving windows intentionally have variable physical lengths.
    # In DDP, ranks can therefore reach the end of their local sequence at
    # different TBPTT segment indices.  Every rank must nevertheless execute
    # the same sequence of finite-guard and reduction collectives.  Determine
    # a common loop bound once; shorter ranks contribute a differentiable zero
    # loss for the missing suffix and keep their local fast state unchanged.
    if accelerator.num_processes > 1:
        segment_count = torch.tensor(
            local_num_segments, dtype=torch.int32, device=accelerator.device
        )
        # Accelerate's public reduction contract is sum/mean; passing "max"
        # is backend/version dependent and has silently behaved like a sum in
        # supported MIKASA environments.  Gather one scalar per rank and take
        # the maximum explicitly so short trajectories do not create spurious
        # extra dummy segments.
        gathered_segment_counts = accelerator.gather(segment_count.reshape(1))
        global_num_segments = int(gathered_segment_counts.amax().item())
    else:
        global_num_segments = local_num_segments
    last_segment_output: dict[str, Any] = {}

    for segment_index in range(global_num_segments):
        has_local_segment = segment_index < local_num_segments
        if has_local_segment:
            segment_start = segment_index * segment_length
            segment_end = min(segment_start + segment_length, sequence_length)
            current_segment_length = segment_end - segment_start
            segment_batch = _slice_flattened_sequence_batch(
                batch,
                batch_size,
                sequence_length,
                segment_start,
                segment_end,
            )

            with accelerator.autocast():
                segment_kwargs = {
                    "sequence_shape": (batch_size, current_segment_length),
                    "fast_states": fast_states,
                }
                if grounding_states is not None:
                    # Only SmolVLA-TTT exposes the optional grounding container;
                    # PI0/PI05 sequence policies keep their historical signature.
                    segment_kwargs["grounding_states"] = grounding_states
                if use_global_hd_normalization or frame_weighted_ddp:
                    segment_kwargs["flow_loss_weight"] = (
                        segment_loss_weights[segment_index] * flow_rank_scale
                    )
                    segment_kwargs["hd_normalization_denominator"] = hd_normalization_denominator
                    segment_kwargs["effect_normalization_floor"] = effect_normalization_floor
                if _HD_ATTRIBUTION_PROTOCOL_V3 == str(
                    getattr(policy_config, "hd_attribution_protocol", "")
                ):
                    # Pair indices are episode-local.  The model uses this
                    # offset to map them into the current TBPTT segment; the
                    # complete batch remains available for cross-segment
                    # query replay and its global pair normalizers.
                    segment_kwargs["sequence_offset"] = window_sequence_offset + segment_start
                    segment_kwargs["v3_reference_batch"] = v3_reference_batch
                    if v3_pair_normalizers is not None:
                        segment_kwargs["v3_pair_normalizers"] = v3_pair_normalizers
                    if stream_v3_replay:
                        segment_kwargs["v3_streaming_backward"] = _stream_v3_backward
                segment_loss, segment_output, fast_states = unwrapped_policy.forward_sequence_segment(
                    segment_batch,
                    **segment_kwargs,
                )
                segment_weight = segment_loss_weights[segment_index]
                metric_segment_weight = segment_weight * flow_rank_scale
                # In the v2 path the policy has already separated and normalized
                # the two objectives.  Multiplying the combined scalar here would
                # attenuate warm-up/effect supervision by the action-valid mask.
                weighted_segment_loss = (
                    segment_loss
                    if use_global_hd_normalization or frame_weighted_ddp
                    else segment_loss * segment_weight
                )
            last_segment_output = segment_output
        else:
            segment_weight = 0.0
            # ``requires_grad`` lets Accelerator.backward follow the same
            # control path on every rank without fabricating gradients for any
            # policy parameter.  The subsequent presence-aware reduction fills
            # zeros only where another rank used that parameter.
            weighted_segment_loss = torch.zeros(
                (), device=accelerator.device, requires_grad=True
            )
            segment_output = {
                "ttt_nonfinite_seen": torch.zeros(
                    (), device=accelerator.device, dtype=torch.bool
                )
            }

        if finite_guard_enabled:
            # Check before autograd can propagate a malformed segment.  The
            # distributed reduction inside the guard makes this collective
            # safe even when only one rank receives a bad window.
            _hd_ttt_finite_guard(
                loss=weighted_segment_loss,
                fast_states=fast_states,
                observations=(
                    ("segment_loss_per_dim", segment_output.get("loss_per_dim")),
                    ("ttt_nonfinite_seen", segment_output.get("ttt_nonfinite_seen")),
                ),
                accelerator=accelerator,
                stage="segment loss before backward",
                segment_index=segment_index,
                check_gradients=False,
                check_parameters=False,
            )

        accelerator.backward(weighted_segment_loss * gradient_scale)
        total_loss += weighted_segment_loss.detach()
        if has_local_segment:
            segment_loss_per_dim = torch.tensor(
                segment_output["loss_per_dim"], device=accelerator.device
            )
            if loss_per_dim is None:
                loss_per_dim = segment_loss_per_dim * metric_segment_weight
            else:
                loss_per_dim += segment_loss_per_dim * metric_segment_weight
            for metric_name, metric_value in segment_output.items():
                if metric_name.startswith("hd_"):
                    # Ratios are recomputed from the sequence-level sums below;
                    # averaging per-segment ratios would make them depend on the
                    # TBPTT partition, especially under v2 global normalization.
                    if metric_name in {"hd_aux_to_flow_ratio", "hd_aux_fraction"}:
                        continue
                    additive_metric = metric_name in {
                        "hd_hca",
                        "hd_h2l",
                        "hd_effect",
                        "hd_grounding",
                        "hd_gate",
                        "hd_auxiliary_loss",
                        "hd_flow_loss",
                        # CreditTTT primitives return sequence-level losses
                        # and pair counts normalized by the complete window.
                        # Aggregate these as numerators, never as a mean of
                        # TBPTT-segment means (which would depend on the
                        # arbitrary truncation length).
                        "hd_v3_qh2l",
                        "hd_v3_qh2l_positive",
                        "hd_v3_qh2l_null",
                        "hd_v3_qh2l_streamed_loss",
                        "hd_v3_cmd",
                        "hd_v3_cmd_full",
                        "hd_v3_cmd_effect",
                        "hd_v3_cmd_rank",
                        "hd_v3_cmd_null",
                        "hd_v3_cmd_streamed_loss",
                        "hd_v3_streamed_loss",
                        "hd_v3_pairs",
                        "hd_v3_positive_pairs",
                        "hd_v3_null_pairs",
                        "hd_v3_pairs_skipped",
                        "hd_v3_cmd_pairs",
                        "hd_v3_cmd_pairs_skipped",
                        "hd_v3_cross_segment_pairs",
                        "hd_v3_cmd_cross_segment_pairs",
                    }
                    metric_weight = (
                        1.0 if use_global_hd_normalization and additive_metric else segment_weight
                    )
                    auxiliary_metric_sums[metric_name] = auxiliary_metric_sums.get(
                        metric_name, 0.0
                    ) + float(metric_value) * metric_weight
                elif metric_name.startswith("ttt_"):
                    value = float(metric_value)
                    if metric_name.endswith("_max") or metric_name == "ttt_nonfinite_seen":
                        auxiliary_metric_sums[metric_name] = max(
                            auxiliary_metric_sums.get(metric_name, value), value
                        )
                    elif metric_name.endswith("_min"):
                        auxiliary_metric_sums[metric_name] = min(
                            auxiliary_metric_sums.get(metric_name, value), value
                        )
                    else:
                        auxiliary_metric_sums[metric_name] = auxiliary_metric_sums.get(
                            metric_name, 0.0
                        ) + value * segment_weight
            fast_states = {
                layer_index: fast_state.detach() for layer_index, fast_state in fast_states.items()
            }
        num_segments += 1

    if optimizer_step and accelerator.num_processes > 1:
        # Sequence TTT calls the unwrapped policy, so synchronize all outer
        # gradients explicitly.  The helper preserves conditional unused-
        # parameter handling while flattening dense tensors to avoid one
        # all-reduce per model parameter.  In accumulation mode this reduction
        # intentionally happens only on the final window: reducing each partial
        # sum in place would average already-synchronized gradients repeatedly.
        _ddp_reduce_gradients(policy, accelerator)

        total_loss = accelerator.reduce(total_loss, reduction="mean")
        loss_per_dim = accelerator.reduce(loss_per_dim, reduction="mean")
        for metric_name, metric_value in list(auxiliary_metric_sums.items()):
            metric_tensor = torch.tensor(metric_value, device=accelerator.device)
            if metric_name.endswith("_max") or metric_name == "ttt_nonfinite_seen":
                gathered_metric = accelerator.gather(metric_tensor.reshape(1))
                auxiliary_metric_sums[metric_name] = float(gathered_metric.amax().item())
            elif metric_name.endswith("_min"):
                gathered_metric = accelerator.gather(metric_tensor.reshape(1))
                auxiliary_metric_sums[metric_name] = float(gathered_metric.amin().item())
            else:
                auxiliary_metric_sums[metric_name] = float(
                    accelerator.reduce(metric_tensor, reduction="mean").item()
                )

    if optimizer_step and finite_guard_enabled:
        # This is the last point before gradient clipping and the optimizer
        # step.  A bad loss, gradient, recurrent fast state, or parameter is
        # therefore reported without allowing any mutation by the optimizer.
        _hd_ttt_finite_guard(
            policy=policy,
            loss=total_loss,
            fast_states=fast_states,
            observations=(
                ("loss_per_dim", loss_per_dim),
                (
                    "ttt_nonfinite_seen",
                    last_segment_output.get("ttt_nonfinite_seen"),
                ),
            ),
            accelerator=accelerator,
            stage="before gradient clipping",
        )

    # Report a single sequence-level balance value.  For v2, each HD scalar is
    # already normalized by the complete physical-frame denominator and each
    # flow scalar carries its segment-valid fraction, so summing the two
    # metrics above yields a TBPTT-partition-invariant ratio.  Legacy paths use
    # the historical segment weights for both terms.
    if (
        hd_balance_metrics is not None
        and "hd_auxiliary_loss" in auxiliary_metric_sums
        and "hd_flow_loss" in auxiliary_metric_sums
    ):
        auxiliary_metric_sums.update(
            hd_balance_metrics(
                auxiliary_metric_sums["hd_auxiliary_loss"],
                auxiliary_metric_sums["hd_flow_loss"],
            )
        )

    if optimizer_step:
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )

        if finite_guard_enabled:
            # Clipping can itself produce an infinite norm for otherwise finite
            # but overflowing gradients.  Check again after clipping and still
            # abort before entering the lock/optimizer section.
            _hd_ttt_finite_guard(
                policy=policy,
                loss=total_loss,
                grad_norm=grad_norm,
                fast_states=fast_states,
                accelerator=accelerator,
                stage="after gradient clipping",
            )

        with lock if lock is not None else nullcontext():
            optimizer.step()

        if accelerator.num_processes > 1 and os.environ.get("LEROBOT_VERIFY_DDP_SYNC") == "1":
            max_parameter_difference = torch.zeros((), device=accelerator.device)
            for parameter in policy.parameters():
                if not parameter.requires_grad:
                    continue
                rank_zero_parameter = parameter.detach().clone()
                torch.distributed.broadcast(rank_zero_parameter, src=0)
                max_parameter_difference = torch.maximum(
                    max_parameter_difference,
                    (parameter.detach() - rank_zero_parameter).abs().max(),
                )
            torch.distributed.all_reduce(
                max_parameter_difference,
                op=torch.distributed.ReduceOp.MAX,
            )
            if max_parameter_difference.item() != 0:
                raise RuntimeError(
                    "TTT policy data-parallel replicas diverged after optimizer step: "
                    f"max parameter difference={max_parameter_difference.item()}"
                )
            if accelerator.is_main_process:
                logging.info("Verified TTT data-parallel replicas are identical after optimizer step")

        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()
        if has_method(unwrapped_policy, "update"):
            unwrapped_policy.update()
    else:
        # Gradients are intentionally left in place for the next independent
        # window.  No clipping/reduction/scheduler step is allowed here: doing
        # either would change the sum that the final accumulation call sees.
        grad_norm = torch.zeros((), device=accelerator.device)

    train_metrics.loss = total_loss.item()
    # A deferred micro-window has no post-accumulation norm yet.  Recording a
    # synthetic zero would bias the running diagnostic toward zero whenever
    # accumulation is enabled, so publish the clipped norm only on the final
    # optimizer call (the historical path always takes this branch).
    if optimizer_step:
        train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    output_dict = {
        "loss": total_loss.item(),
        "loss_per_dim": loss_per_dim.detach().cpu().tolist(),
        "tbptt_segments": num_segments,
    }
    if "ttt_nonfinite_seen" in last_segment_output:
        output_dict["ttt_nonfinite_seen"] = last_segment_output["ttt_nonfinite_seen"]
    output_dict.update(auxiliary_metric_sums)
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator

    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    is_pi0_ttt = isinstance(cfg.policy, PI0TTTConfig)
    is_pi05_ttt = isinstance(cfg.policy, PI05TTTConfig)
    is_smolvla_ttt = isinstance(cfg.policy, SmolVLATTTConfig)
    is_sequence_ttt = is_pi0_ttt or is_pi05_ttt or is_smolvla_ttt
    raw_gradient_accumulation_steps = getattr(cfg, "gradient_accumulation_steps", 1)
    if (
        type(raw_gradient_accumulation_steps) is not int
        or raw_gradient_accumulation_steps < 1
    ):
        # ``TrainPipelineConfig.validate`` catches malformed CLI values in
        # normal runs.  Keep this local guard for callers that construct a
        # config object programmatically and invoke ``train`` directly.
        raise ValueError(
            "gradient_accumulation_steps must be a positive integer, got "
            f"{raw_gradient_accumulation_steps!r}"
        )
    gradient_accumulation_steps = raw_gradient_accumulation_steps
    if gradient_accumulation_steps > 1 and not is_sequence_ttt:
        raise ValueError(
            "gradient_accumulation_steps>1 is currently supported only for "
            "episode-local sequence-TTT policies"
        )
    if is_sequence_ttt:
        if cfg.sample_weighting is not None:
            raise ValueError("TTT sequence training does not support sample weighting yet")
        if cfg.peft is not None:
            raise ValueError("TTT sequence training does not support PEFT yet")
        if cfg.dataset.streaming:
            raise ValueError("TTT sequence training requires a map-style dataset; streaming is not supported")
    smolvla_equal_length_batching = bool(
        is_smolvla_ttt and getattr(cfg.policy, "equal_length_batching", False)
    )
    if smolvla_equal_length_batching:
        # Accelerate's split-batch mode slices one trajectory group across
        # ranks.  That would make the collator's ``[B,T]`` state layout and
        # per-trajectory fast weights disagree with the sampler contract.
        # Fail closed instead of silently changing the effective batch.
        split_batches = getattr(accelerator, "split_batches", None)
        if split_batches is None:
            split_batches = getattr(
                getattr(accelerator, "dataloader_config", None), "split_batches", False
            )
        if bool(split_batches):
            raise ValueError(
                "SmolVLA-TTT equal_length_batching requires Accelerate split_batches=false"
            )
    if (is_pi05_ttt or is_smolvla_ttt) and cfg.batch_size != 1 and not smolvla_equal_length_batching:
        raise ValueError("Tail-preserving TTT sequences require per-device batch_size=1")

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        dataset = _attach_hd_labels(dataset, cfg, is_smolvla_ttt=is_smolvla_ttt)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)
        dataset = _attach_hd_labels(dataset, cfg, is_smolvla_ttt=is_smolvla_ttt)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path
    if (
        getattr(active_cfg, "use_relative_actions", False)
        and processor_pretrained_path is not None
        and not cfg.resume
    ):
        logging.warning(
            "use_relative_actions=true with pretrained processors can skip relative transforms if "
            "the checkpoint processors do not define them. Building processors from current policy config."
        )
        processor_pretrained_path = None

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            **processor_kwargs,
            **postprocessor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes * gradient_accumulation_steps
        logging.info(
            "Effective optimizer batch size: %s x %s x %s accumulation = %s "
            "(per-device window x processes x windows/update)",
            cfg.batch_size,
            num_processes,
            gradient_accumulation_steps,
            effective_bs,
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if is_smolvla_ttt:
        dataloader_dataset = SmolVLATTTSequenceDataset(
            dataset,
            sequence_length=active_cfg.sequence_length,
            sequence_stride=active_cfg.sequence_stride,
            max_windows_per_episode=getattr(active_cfg, "max_windows_per_episode", None),
            history_warmup_length=getattr(active_cfg, "ttt_history_warmup_length", 0),
        )
        if smolvla_equal_length_batching:
            # Bucket trajectory windows by exact physical T.  The sampler
            # balances short buckets and globally DDP-divides batch count by
            # repeating complete trajectories within the same (T, offset)
            # bucket; no temporal padding or cross-episode concatenation
            # occurs.
            shuffle = False
            sampler = None
            batch_sampler = EqualLengthBatchSampler(
                dataloader_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_replicas=accelerator.num_processes,
                seed=int(cfg.seed or 0),
            )
            collate_fn = functools.partial(
                batched_sequence_collate_fn,
                require_zero_offsets=(
                    str(getattr(active_cfg, "hd_attribution_protocol", ""))
                    == _HD_ATTRIBUTION_PROTOCOL_V3
                ),
            )
        else:
            shuffle = True
            sampler = None
            batch_sampler = None
            collate_fn = smolvla_ttt_sequence_collate_fn
    elif is_pi05_ttt:
        dataloader_dataset = TailPreservingSequenceDataset(
            dataset,
            sequence_length=active_cfg.sequence_length,
            sequence_stride=active_cfg.sequence_stride,
        )
        shuffle = True
        sampler = None
        collate_fn = pi05_ttt_sequence_collate_fn
    elif is_pi0_ttt:
        dataloader_dataset = ContiguousSequenceDataset(
            dataset,
            sequence_length=active_cfg.sequence_length,
            sequence_stride=active_cfg.sequence_stride,
        )
        shuffle = True
        sampler = None
        collate_fn = sequence_collate_fn
    elif hasattr(active_cfg, "drop_n_last_frames"):
        dataloader_dataset = dataset
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=active_cfg.drop_n_last_frames,
            shuffle=True,
        )
        collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    else:
        dataloader_dataset = dataset
        shuffle = True
        sampler = None
        # Only swap in the language-aware collate when the dataset actually
        # declares language columns; otherwise stay on PyTorch's default
        # collate so non-language training runs are unaffected.
        collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader_kwargs = dict(
        dataset=dataloader_dataset,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    if smolvla_equal_length_batching:
        dataloader_kwargs["batch_sampler"] = batch_sampler
    else:
        dataloader_kwargs.update(
            batch_size=cfg.batch_size,
            shuffle=shuffle and not cfg.dataset.streaming,
            sampler=sampler,
            drop_last=False,
        )
    dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)

    # TTT policies call a custom sequence-segment method on the unwrapped policy so fast state can be
    # carried across TBPTT segments. In multi-process mode, wrapping that policy in DDP would bypass
    # DDP's forward bookkeeping. Prepare the optimizer/dataloader/scheduler only and synchronize the
    # outer gradients explicitly in update_policy_tbptt instead.
    accelerator.wait_for_everyone()
    if is_sequence_ttt and accelerator.num_processes > 1:
        optimizer, dataloader, lr_scheduler = accelerator.prepare(optimizer, dataloader, lr_scheduler)
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    if is_smolvla_ttt and getattr(cfg.policy, "hd_ttt_enabled", False):
        for metric_name in (
            "hd_hca",
            "hd_h2l",
            "hd_effect",
            "hd_gate",
            "hd_grounding",
            "hd_gate_pred_mean",
            "hd_gate_target_mean",
            "hd_gate_observed_fraction",
            "hd_v3_qh2l",
            "hd_v3_qh2l_positive",
            "hd_v3_qh2l_null",
            "hd_v3_cmd",
            "hd_v3_cmd_full",
            "hd_v3_cmd_effect",
            "hd_v3_cmd_rank",
            "hd_v3_cmd_null",
            "hd_v3_pairs",
            "hd_v3_positive_pairs",
            "hd_v3_null_pairs",
            "hd_v3_pairs_skipped",
            "hd_v3_cmd_pairs",
            "hd_v3_cmd_pairs_skipped",
            "hd_v3_cross_segment_pairs",
            "hd_v3_cmd_cross_segment_pairs",
            "hd_v3_cmd_disabled",
            "hd_v3_cmd_missing_targets",
            "hd_v3_cmd_no_reference",
        ):
            train_metrics[metric_name] = AverageMeter(metric_name, ":.3f")
        # These diagnostics are deliberately separate from the optimized
        # objectives.  Their finer display precision is needed to distinguish
        # a genuinely inactive grounding loss from a small positive value that
        # would otherwise be rounded to ``0.000``.
        for metric_name in (
            "hd_gate_pred_std",
            "hd_gate_target_std",
            "hd_gate_corr",
            "hd_gate_constant",
            "hd_gate_gain_vs_constant",
            "hd_gate_weight_mass",
            "hd_grounding_direction",
            "hd_grounding_invariance",
            "hd_grounding_weight_mass",
            "hd_grounding_rho_nonzero_fraction",
            "hd_grounding_wrong_gate_zero_fraction",
            "hd_grounding_teacher_delta_rms",
            "hd_grounding_student_delta_rms",
            "hd_grounding_delta_ratio",
            "hd_grounding_margin_active_fraction",
            "hd_effect_direction",
            "hd_effect_invariance",
            "hd_effect_weight_mass",
            "hd_aux_to_flow_ratio",
            "hd_aux_fraction",
            "hd_ttt_inner_lr_min",
            "hd_ttt_inner_lr_max",
            "hd_ttt_effective_gate_min",
            "hd_ttt_effective_gate_max",
            "hd_v3_delay_mean",
            "hd_v3_teacher_effect_rms",
            "hd_v3_student_effect_rms",
            "hd_v3_kvb_anchor",
        ):
            train_metrics[metric_name] = AverageMeter(metric_name, ":.5f")
    if is_smolvla_ttt and getattr(cfg.policy, "ttt_stable_inner_update", False):
        for metric_name in (
            "ttt_nonfinite_seen",
            "ttt_state_rms_ratio_min",
            "ttt_state_rms_ratio_mean",
            "ttt_state_rms_ratio_max",
        ):
            train_metrics[metric_name] = AverageMeter(metric_name, ":.5f")

    def _record_sequence_output(output_dict: dict[str, Any]) -> None:
        """Add one sequence-window's diagnostics to the running meters.

        Gradient accumulation executes several *independent* windows before a
        single optimizer update.  Keeping this bookkeeping in a helper lets
        each micro-window contribute its diagnostics without moving the
        optimizer/checkpoint cadence or accidentally sharing its fast state.
        """

        if is_smolvla_ttt and cfg.policy.hd_ttt_enabled:
            for metric_name in (
                "hd_hca",
                "hd_h2l",
                "hd_effect",
                "hd_gate",
                "hd_grounding",
                "hd_gate_pred_mean",
                "hd_gate_target_mean",
                "hd_gate_observed_fraction",
                "hd_v3_qh2l",
                "hd_v3_qh2l_positive",
                "hd_v3_qh2l_null",
                "hd_v3_cmd",
                "hd_v3_cmd_full",
                "hd_v3_cmd_effect",
                "hd_v3_cmd_rank",
                "hd_v3_cmd_null",
                "hd_v3_pairs",
                "hd_v3_positive_pairs",
                "hd_v3_null_pairs",
                "hd_v3_pairs_skipped",
                "hd_v3_cmd_pairs",
                "hd_v3_cmd_pairs_skipped",
                "hd_v3_cross_segment_pairs",
                "hd_v3_cmd_cross_segment_pairs",
                "hd_v3_cmd_disabled",
                "hd_v3_cmd_missing_targets",
                "hd_v3_cmd_no_reference",
                "hd_gate_pred_std",
                "hd_gate_target_std",
                "hd_gate_corr",
                "hd_gate_constant",
                "hd_gate_gain_vs_constant",
                "hd_gate_weight_mass",
                "hd_grounding_direction",
                "hd_grounding_invariance",
                "hd_grounding_weight_mass",
                "hd_grounding_rho_nonzero_fraction",
                "hd_grounding_wrong_gate_zero_fraction",
                "hd_grounding_teacher_delta_rms",
                "hd_grounding_student_delta_rms",
                "hd_grounding_delta_ratio",
                "hd_grounding_margin_active_fraction",
                "hd_effect_direction",
                "hd_effect_invariance",
                "hd_effect_weight_mass",
                "hd_aux_to_flow_ratio",
                "hd_aux_fraction",
                "hd_ttt_inner_lr_min",
                "hd_ttt_inner_lr_max",
                "hd_ttt_effective_gate_min",
                "hd_ttt_effective_gate_max",
                "hd_v3_delay_mean",
                "hd_v3_teacher_effect_rms",
                "hd_v3_student_effect_rms",
                "hd_v3_kvb_anchor",
            ):
                if metric_name in output_dict:
                    setattr(train_tracker, metric_name, output_dict[metric_name])
        if is_smolvla_ttt and getattr(cfg.policy, "ttt_stable_inner_update", False):
            for metric_name in (
                "ttt_nonfinite_seen",
                "ttt_state_rms_ratio_min",
                "ttt_state_rms_ratio_mean",
                "ttt_state_rms_ratio_max",
            ):
                if metric_name in output_dict:
                    setattr(train_tracker, metric_name, output_dict[metric_name])

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = (
        cfg.batch_size * accelerator.num_processes * gradient_accumulation_steps
    )
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
        sample_multiplier=gradient_accumulation_steps,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        # Each micro-window is an independent trajectory sample.  Only the
        # persistent outer gradients span accumulation windows; fast states,
        # grounding branches, and TBPTT graphs are created/reset inside
        # ``update_policy_tbptt`` for every call.
        output_dict: dict[str, Any] = {}
        for accumulation_index in range(gradient_accumulation_steps):
            start_time = time.perf_counter()
            batch = next(dl_iter)
            sequence_shape = None
            if is_sequence_ttt:
                sequence_shape = tuple(int(value) for value in batch.pop(SEQUENCE_SHAPE_KEY))
            for cam_key in dataset.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)
            train_tracker.dataloading_s = time.perf_counter() - start_time

            if is_sequence_ttt:
                train_tracker, output_dict = update_policy_tbptt(
                    train_tracker,
                    policy,
                    batch,
                    sequence_shape,
                    active_cfg.tbptt_segment_length,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    accelerator=accelerator,
                    lr_scheduler=lr_scheduler,
                    # Synchronize/step only once after all independent windows
                    # have contributed their scaled gradients.
                    optimizer_step=(accumulation_index == gradient_accumulation_steps - 1),
                    zero_grad_before=(accumulation_index == 0),
                    gradient_scale=1.0 / gradient_accumulation_steps,
                )
            else:
                # Non-sequence policies are guarded above to keep the scope of
                # explicit accumulation unambiguous and preserve their native
                # update path.
                train_tracker, output_dict = update_policy(
                    train_tracker,
                    policy,
                    batch,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    accelerator=accelerator,
                    lr_scheduler=lr_scheduler,
                    sample_weighter=sample_weighter,
                )
            _record_sequence_output(output_dict)

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        # One logical step may consume several independent sequence windows;
        # account for all of them in sample/epoch diagnostics while retaining
        # one scheduler/checkpoint/evaluation step.
        train_tracker.step(sample_multiplier=gradient_accumulation_steps)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log sample weighting statistics if enabled
                if sample_weighter is not None:
                    weighter_stats = sample_weighter.get_stats()
                    wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model)
            else:
                unwrapped_model.push_model_to_hub(cfg)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
