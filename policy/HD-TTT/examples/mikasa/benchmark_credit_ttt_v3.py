#!/usr/bin/env python
"""Reproducible benchmark protocol for CreditTTT on MIKASA-Robo-VLA.

This file is an *experiment coordinator*, not a replacement for either of
the existing MIKASA adapters.  It has three deliberately separate jobs:

``manifest``
    Freeze the task, seed, checkpoint, and fairness contract and emit the
    commands which can later be run by a human or a job scheduler.

``aggregate``
    Read the per-episode JSON files emitted by the official adapters, verify
    their provenance, align identical episode seeds, and calculate paired
    bootstrap confidence intervals and McNemar tests.

``check``
    Apply pre-declared mechanistic go/no-go checks to a JSON artifact (for
    example the full-history-teacher audit or exact-gradient audit).

No subcommand launches a job unless ``run --execute`` is explicitly supplied.
The default command is therefore safe to use while planning an experiment.

The protocol intentionally uses functional method names rather than version
labels.  ``CreditTTT`` is the proposed method; ``Clean-TTT`` and a native
SmolVLA **K=1 receding-horizon control** are the cadence-matched baselines.
The original Native-SmolVLA K=50 behavior is retained as a descriptive
reference, but is never placed in the primary memory comparison.  A legacy
result whose metadata says ``HD-TTT`` (or an old v1/v2 protocol) is rejected
when it is supplied as a CreditTTT result.  This prevents both an old
checkpoint and a cadence confound from silently becoming a reported V3 gain.

The official MIKASA metric and seed convention are documented at
https://mikasarobo.github.io/evaluation_protocol.html.  The script keeps the
benchmark's ``success_once`` latch as the sole primary metric and treats
returns as diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - project runtime always has numpy
    raise RuntimeError("This benchmark coordinator requires numpy") from exc


# The benchmark envelope is versioned independently from the model's canonical
# CreditTTT identity below.  The published-four profile is the default paper
# profile; the original two-task profile remains loadable for old runs.
PUBLISHED_FOUR_TASK_PROTOCOL_ID = "credit_ttt_v3_mikasa_published_four_task"
LEGACY_TWO_TASK_PROTOCOL_ID = "credit_ttt_v3_mikasa_two_task"
SUPPORTED_PROTOCOL_IDS = {
    PUBLISHED_FOUR_TASK_PROTOCOL_ID,
    LEGACY_TWO_TASK_PROTOCOL_ID,
}
PROTOCOL_ID = PUBLISHED_FOUR_TASK_PROTOCOL_ID
PROTOCOL_VERSION = "credit_ttt_v3_baseline_protocol_1"
# Checkpoint/label implementations may serialize the same method under one
# of these names.  Keep the benchmark manifest version out of this set: it is
# an experiment-envelope identifier, not proof that a model used the V3
# policy.  Otherwise an arbitrary result could inherit
# ``protocol_version=credit_ttt_v3_baseline_protocol_1`` from an envelope and
# be misclassified as CreditTTT without canonical model metadata.
V3_PROTOCOL_MARKERS = {
    "credit_ttt_v3",
    "credit_ttt_v3_query_effect",
    "creditttt_qh2l_v3",
}
# ``PROTOCOL_ID``/``PROTOCOL_VERSION`` identify this *benchmark envelope*;
# they do not authenticate a checkpoint.  A CreditTTT result must carry this
# independent, exact identity object.  Keeping the fields here (rather than
# accepting a free-form ``protocol_version`` string) prevents an old HD-TTT
# or a hand-edited result from being promoted to the paper method merely by
# changing its directory name.
CANONICAL_V3_PROTOCOL_IDENTITY: dict[str, Any] = {
    "format": "credit_ttt_v3",
    "protocol": "creditttt_qh2l_v3",
    "version": 3,
    "pair_schema": "event_future_control_pair_v3",
    "intervention": "event_write_deletion",
    "intervention_scope": "event_write_only_previous_executed_action_held_fixed",
    "target": "final_slot0_action",
    "state": "causal_fast_weights",
    "causal": True,
}
CANONICAL_V3_PROTOCOL_FIELDS = tuple(CANONICAL_V3_PROTOCOL_IDENTITY)
OFFICIAL_PROTOCOL = "MIKASA-Robo-VLA official runner"
# The native SmolVLA model always predicts a 50-slot flow chunk.  The runner
# may consume all 50 slots (the canonical native contract) or only slot 0 and
# re-query at the next physical step (the cadence-matched control).  Keeping
# these values explicit prevents a result's runner chunk from being confused
# with the model's action horizon.
NATIVE_MODEL_ACTION_HORIZON = 50
NATIVE_CADENCE_CHUNK = "native_chunk"
NATIVE_CADENCE_RECEDING = "receding_horizon"
NATIVE_VARIANT_CHUNK = "native_smolvla"
NATIVE_VARIANT_K1 = "native_smolvla_k1"
DEFAULT_START_SEED = 4_242_424_242
DEFAULT_EPISODES = 50
DEFAULT_TORCH_SEED = 7_000
DEFAULT_TRAIN_SEEDS = (1000, 1001, 1002)
DEFAULT_RESULTS_ROOT = "benchmark_results/credit_ttt_v3"

LEGACY_TWO_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "color",
        "env_id": "ShellGameColorLampTouch-VLA-v0",
        "dataset_repo_id": "shell_game_color_lamp_touch_vla_v0",
        "dataset_root": "/workspace/data_mikasa_robo/data_lerobot/shell_game_color_lamp_touch_vla_v0",
        "horizon_split": "Short",
        "memory_type": "Spatial",
        "max_episode_steps": 30,
        "demo_count": 250,
        "train_demo_indices": [0, 199],
        "validation_demo_indices": [200, 249],
        "delay_bins_present": ["1-16"],
    },
    {
        "id": "shuffle_long",
        "env_id": "ShellGameShuffleColorLampTouch-Long-VLA-v0",
        "dataset_repo_id": "shell_game_shuffle_color_lamp_touch_long_vla_v0",
        "dataset_root": "/workspace/data_mikasa_robo/data_lerobot/shell_game_shuffle_color_lamp_touch_long_vla_v0",
        "horizon_split": "Medium",
        "memory_type": "Tracking",
        "max_episode_steps": 600,
        "demo_count": 250,
        "train_demo_indices": [0, 199],
        "validation_demo_indices": [200, 249],
        "delay_bins_present": ["1-16", "17-64", "65-256", "257-1024"],
    },
)

# Four tasks selected because their environment IDs and abbreviations are
# present in the published MemoryVLA MIKASA-Robo table (SGT, IM, RC3, RC9).
# They deliberately span a simple spatial interaction, a dynamic interception
# task, and low/high-capacity object-memory tasks while retaining one common
# short-horizon evaluation cadence.  Dataset roots are explicit so each task's
# normalization statistics remain isolated.
PUBLISHED_COMPARABLE_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "shell_touch",
        "env_id": "ShellGameTouch-VLA-v0",
        "dataset_repo_id": "shell_game_touch_vla_v0",
        "dataset_root": "/workspace/data_mikasa_robo/data_lerobot/shell_game_touch_vla_v0",
        "horizon_split": "Short",
        "memory_type": "Spatial",
        "max_episode_steps": 30,
        "demo_count": 250,
        # All 250 official demonstrations are used for fitting/credit labels.
        # Simulator episodes are the held-out evaluation; no demo is reserved
        # for model selection in the canonical recipe.
        "train_demo_indices": [0, 249],
        "validation_demo_indices": [],
        "delay_bins_present": ["1-16"],
        "published_abbreviation": "SGT",
    },
    {
        "id": "intercept_medium",
        "env_id": "InterceptMedium-VLA-v0",
        "dataset_repo_id": "intercept_medium_vla_v0",
        "dataset_root": "/workspace/data_mikasa_robo/data_lerobot/intercept_medium_vla_v0",
        "horizon_split": "Short",
        "memory_type": "Spatial",
        "max_episode_steps": 60,
        "demo_count": 250,
        "train_demo_indices": [0, 249],
        "validation_demo_indices": [],
        "delay_bins_present": ["1-16"],
        "published_abbreviation": "IM",
    },
    {
        "id": "remember_color3",
        "env_id": "RememberColor3-VLA-v0",
        "dataset_repo_id": "remember_color_3_vla_v0",
        "dataset_root": "/workspace/data_mikasa_robo/data_lerobot/remember_color_3_vla_v0",
        "horizon_split": "Short",
        "memory_type": "Object",
        "max_episode_steps": 25,
        "demo_count": 250,
        "train_demo_indices": [0, 249],
        "validation_demo_indices": [],
        "delay_bins_present": ["1-16"],
        "published_abbreviation": "RC3",
    },
    {
        "id": "remember_color9",
        "env_id": "RememberColor9-VLA-v0",
        "dataset_repo_id": "remember_color_9_vla_v0",
        "dataset_root": "/workspace/data_mikasa_robo/data_lerobot/remember_color_9_vla_v0",
        "horizon_split": "Short",
        "memory_type": "Object",
        "max_episode_steps": 25,
        "demo_count": 250,
        "train_demo_indices": [0, 249],
        "validation_demo_indices": [],
        "delay_bins_present": ["1-16"],
        "published_abbreviation": "RC9",
    },
)

# Keep the historical public name as an alias for callers that imported it.
# New manifests select a profile explicitly (published_four by default).
DEFAULT_TASKS = LEGACY_TWO_TASKS
TASK_SET_ALIASES = {
    "published_four": "published_four",
    "published": "published_four",
    "four": "published_four",
    "legacy_two": "legacy_two",
    "legacy": "legacy_two",
    "two": "legacy_two",
}
TASK_SETS: dict[str, tuple[dict[str, Any], ...]] = {
    "published_four": PUBLISHED_COMPARABLE_TASKS,
    "legacy_two": LEGACY_TWO_TASKS,
}
TASK_SET_PROTOCOL_IDS = {
    "published_four": PUBLISHED_FOUR_TASK_PROTOCOL_ID,
    "legacy_two": LEGACY_TWO_TASK_PROTOCOL_ID,
}
DEFAULT_TASK_SET = "published_four"

_DELAY_BIN_ORDER = ("1-16", "17-64", "65-256", "257-1024", "1025+")

# These names are intentionally stable and human-readable in tables.  The
# optional Utility-KVB entry is a mechanism baseline, not a version of our
# method; it is omitted from the primary comparison unless result files exist.
METHODS: tuple[dict[str, Any], ...] = (
    {
        "id": "native_smolvla",
        "label": "Native-SmolVLA",
        "role": "cadence_reference_baseline",
        "evaluator": "examples/mikasa/evaluate_smolvla_baseline.py",
        "expected_action_chunk_size": NATIVE_MODEL_ACTION_HORIZON,
        "expected_model_action_horizon": NATIVE_MODEL_ACTION_HORIZON,
        "expected_execution_action_steps": NATIVE_MODEL_ACTION_HORIZON,
        "expected_execution_cadence": NATIVE_CADENCE_CHUNK,
        "benchmark_variant": NATIVE_VARIANT_CHUNK,
        "comparison_scope": "cadence_mismatched_reference",
        "deployable": True,
        "requires_v3_metadata": False,
        "optional": False,
        "replicate_policy": "fixed_checkpoint",
    },
    {
        "id": "native_smolvla_k1",
        "label": "Native-SmolVLA-K1",
        "role": "matched_cadence_baseline",
        "evaluator": "examples/mikasa/evaluate_smolvla_baseline.py",
        "expected_action_chunk_size": 1,
        "expected_model_action_horizon": NATIVE_MODEL_ACTION_HORIZON,
        "expected_execution_action_steps": 1,
        "expected_execution_cadence": NATIVE_CADENCE_RECEDING,
        "benchmark_variant": NATIVE_VARIANT_K1,
        "comparison_scope": "matched_cadence",
        "deployable": True,
        "requires_v3_metadata": False,
        "optional": False,
        "replicate_policy": "fixed_checkpoint",
    },
    {
        "id": "clean_ttt",
        "label": "Clean-TTT",
        "role": "primary_baseline",
        "evaluator": "examples/mikasa/evaluate_smolvla_ttt.py",
        "expected_action_chunk_size": 1,
        "expected_model_action_horizon": NATIVE_MODEL_ACTION_HORIZON,
        "expected_execution_action_steps": 1,
        "expected_execution_cadence": NATIVE_CADENCE_RECEDING,
        "comparison_scope": "matched_cadence",
        "deployable": True,
        "requires_v3_metadata": False,
        "optional": False,
        "replicate_policy": "student_seed",
    },
    {
        "id": "credit_ttt",
        "label": "CreditTTT",
        "role": "ours",
        "evaluator": "examples/mikasa/evaluate_smolvla_ttt.py",
        "expected_action_chunk_size": 1,
        "expected_model_action_horizon": NATIVE_MODEL_ACTION_HORIZON,
        "expected_execution_action_steps": 1,
        "expected_execution_cadence": NATIVE_CADENCE_RECEDING,
        "comparison_scope": "matched_cadence",
        "deployable": True,
        "requires_v3_metadata": True,
        "optional": False,
        "replicate_policy": "student_seed",
    },
    {
        "id": "utility_kvb",
        "label": "Utility-KVB (optional mechanism baseline)",
        "role": "mechanism_baseline",
        "evaluator": "examples/mikasa/evaluate_smolvla_ttt.py",
        "expected_action_chunk_size": 1,
        "expected_model_action_horizon": NATIVE_MODEL_ACTION_HORIZON,
        "expected_execution_action_steps": 1,
        "expected_execution_cadence": NATIVE_CADENCE_RECEDING,
        "comparison_scope": "matched_cadence",
        "deployable": True,
        "requires_v3_metadata": False,
        "optional": True,
        "replicate_policy": "student_seed",
    },
)

# The checks are intentionally conservative.  They are evidence gates, not
# hyperparameter-selection objectives.  Missing values are reported as
# ``unknown`` by default; ``--strict`` turns unknown required checks into a
# failed gate so a paper run cannot accidentally omit a mechanistic audit.
DEFAULT_GO_NO_GO: tuple[dict[str, Any], ...] = (
    {
        "id": "full_teacher_action_improvement",
        "aliases": (
            "teacher_full_vs_short_action_loss_relative_improvement",
            "full_teacher_vs_short_relative_loss_improvement",
            "full_history_teacher_relative_action_loss_improvement",
        ),
        "operator": ">=",
        "threshold": 0.10,
        "required": True,
        "reason": "The hindsight source must contain information beyond a short/clean replay.",
    },
    {
        "id": "history_swap_changes_control",
        "aliases": (
            "teacher_history_swap_action_delta_cosine",
            "history_swap_action_effect_cosine",
            "teacher_history_swap_cosine",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": "A content intervention must change the teacher in a measurable direction.",
    },
    {
        "id": "local_effect_alignment",
        "aliases": (
            "q_h2l_effect_cosine",
            "qh2l_teacher_effect_cosine",
            "local_effect_teacher_cosine",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": "The local writer update must reproduce the teacher's control effect.",
    },
    {
        "id": "local_effect_alignment_ci",
        "aliases": (
            "q_h2l_effect_cosine_ci95_low",
            "q_h2l_effect_cosine_lower_ci",
            "qh2l_teacher_effect_cosine_ci95_low",
            "local_effect_teacher_cosine_ci95_low",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": (
            "The query-conditioned local effect must align with the teacher with a "
            "positive lower confidence bound, rather than only a positive point estimate."
        ),
    },
    {
        "id": "exact_gradient_alignment",
        "aliases": (
            "exact_e2e_gradient_alignment_cosine",
            "short_task_exact_gradient_cosine",
            "e2e_local_update_cosine",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": "Local credit should agree with exact short-horizon E2E credit.",
    },
    {
        "id": "long_delay_gradient_retention",
        "aliases": (
            "long_delay_gradient_nonzero_fraction",
            "delay_over_tbptt_nonzero_gradient_fraction",
            "writer_gradient_nonzero_fraction_long_delay",
        ),
        "operator": ">=",
        "threshold": 0.80,
        "required": True,
        "reason": "The proposed local objective must not lose writer credit after TBPTT.",
    },
    {
        "id": "top_event_intervention",
        "aliases": (
            "top_vs_random_intervention_delta",
            "top_minus_random_action_loss_degradation",
            "top_event_vs_random_delta",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": "Attribution should rank control-relevant events above random events.",
    },
    {
        "id": "pairwise_effect_alignment",
        "aliases": (
            "pairwise_high_utility_delta_a_cosine",
            "high_utility_action_effect_cosine",
            "pairwise_delta_a_cosine",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": (
            "High-utility event/future pairs must carry a directional final-action effect; "
            "report this separately by delay bin in the audit artifact."
        ),
    },
    {
        "id": "history_intervention_recall_excess_random",
        "aliases": (
            "history_intervention_recall_at_k_minus_random",
            "history_intervention_recall_excess_random",
            "intervention_recall_at_k_excess",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": (
            "Attribution should retrieve control-relevant history above a random-event "
            "top-K reference, with K frozen before evaluation."
        ),
    },
    {
        "id": "deployment_memory_selectivity",
        "aliases": (
            "deployment_correct_minus_irrelevant_action_drift",
            "correct_memory_drift_minus_irrelevant_drift",
            "correct_vs_irrelevant_action_drift_delta",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": (
            "The deployed reader must respond more to a correct memory intervention "
            "than to an irrelevant-memory control."
        ),
    },
    {
        "id": "deployment_counterfactual_sensitivity",
        "aliases": (
            "deployment_wrong_reset_action_drift",
            "wrong_or_reset_memory_action_drift",
            "deployment_counterfactual_action_drift",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": (
            "Replacing or resetting a control-relevant memory must measurably change "
            "the deployed action; otherwise the writer is not causally used."
        ),
    },
    {
        "id": "deployment_wrong_selectivity",
        "aliases": (
            "deployment_wrong_minus_irrelevant_action_drift",
            "wrong_memory_drift_minus_irrelevant_drift",
            "wrong_vs_irrelevant_action_drift_delta",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": (
            "A wrong-memory intervention should be distinguishable from an irrelevant "
            "memory replacement in the deployed reader."
        ),
    },
    {
        "id": "deployment_reset_selectivity",
        "aliases": (
            "deployment_reset_minus_irrelevant_action_drift",
            "reset_memory_drift_minus_irrelevant_drift",
            "reset_vs_irrelevant_action_drift_delta",
        ),
        "operator": ">",
        "threshold": 0.0,
        "required": True,
        "reason": (
            "Resetting a control-relevant memory should be distinguishable from an "
            "irrelevant-memory control in the deployed reader."
        ),
    },
    {
        "id": "finite_training",
        "aliases": (
            "nonfinite_steps",
            "nonfinite_count",
            "ttt_nonfinite_seen",
        ),
        "operator": "==",
        "threshold": 0.0,
        "required": True,
        "reason": "A method with non-finite updates is not a valid benchmark result.",
    },
    {
        "id": "state_stability",
        "aliases": (
            "state_rms_ratio_p99",
            "ttt_state_rms_ratio_p99",
            "fast_state_rms_ratio_p99",
        ),
        "operator": "between",
        "threshold": [0.5, 2.0],
        "required": True,
        "reason": "The recurrent state should remain bounded without task-specific tuning.",
    },
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return a deterministic SHA256 for a JSON-compatible object."""

    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _stable_seed(*parts: Any) -> int:
    """Derive a process-independent RNG seed from labels.

    Python's built-in ``hash`` is salted per process, so using it for a
    bootstrap stream would make two aggregation invocations disagree.  A
    short SHA256 prefix gives deterministic, platform-independent offsets
    while retaining the manifest's explicit base seed.
    """

    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100_000


def _is_v3_marker(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        # A nested identity is handled by ``_validate_canonical_v3_identity``;
        # only its canonical protocol string is a marker here.  Converting a
        # mapping to ``str`` would make malformed metadata look like a valid
        # marker and would also produce process-dependent key ordering.
        value = value.get("protocol")
        if value is None:
            return False
    text = str(value).strip().lower()
    return text in V3_PROTOCOL_MARKERS or text.startswith("creditttt_qh2l_v3")


def _validate_canonical_v3_identity(
    value: Any,
    *,
    path: Path | str = "result",
) -> dict[str, Any]:
    """Validate the non-negotiable identity of a deployed CreditTTT model.

    The benchmark envelope has its own version and is intentionally *not*
    accepted as a substitute for this object.  Extra implementation fields
    (for example ``denoise_steps`` or ``intervention_mode``) are allowed, but
    every canonical field is required and type-checked.  In particular,
    Python's ``bool`` is an ``int`` subclass, so ``causal=1`` is rejected.
    """

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{path}: canonical CreditTTT protocol identity must be a JSON object, "
            f"got {type(value).__name__}"
        )
    missing = [key for key in CANONICAL_V3_PROTOCOL_FIELDS if key not in value]
    if missing:
        raise ValueError(
            f"{path}: canonical CreditTTT protocol identity is missing {missing}"
        )
    for key, expected in CANONICAL_V3_PROTOCOL_IDENTITY.items():
        actual = value[key]
        if key == "causal":
            if type(actual) is not bool:
                raise ValueError(
                    f"{path}: canonical field causal must be a JSON boolean, got {actual!r}"
                )
        elif key == "version":
            if type(actual) is not int:
                raise ValueError(
                    f"{path}: canonical field version must be an integer, got {actual!r}"
                )
        elif not isinstance(actual, str):
            raise ValueError(
                f"{path}: canonical field {key!r} must be a string, got {actual!r}"
            )
        if actual != expected:
            raise ValueError(
                f"{path}: canonical field {key!r}={actual!r} does not match "
                f"{expected!r}"
            )
    return {key: value[key] for key in CANONICAL_V3_PROTOCOL_FIELDS}


def _canonical_identity_from_result(
    result: Mapping[str, Any],
    *,
    path: Path | str,
    required: bool,
) -> dict[str, Any] | None:
    """Read and cross-check nested identity objects on a result/envelope.

    Evaluators may put provenance on ``model`` or on the result envelope.  If
    both are present they must be byte-for-byte equivalent on canonical
    fields; accepting one over a contradictory other would make provenance
    order-dependent.
    """

    model = result.get("model")
    model = model if isinstance(model, Mapping) else {}
    candidates: list[tuple[str, Any]] = []
    for owner, container in (("result", result), ("model", model)):
        if "credit_ttt_protocol" in container:
            candidates.append((f"{path}:{owner}.credit_ttt_protocol", container["credit_ttt_protocol"]))
    if not candidates:
        if required:
            raise ValueError(
                f"{path}: CreditTTT requires nested credit_ttt_protocol canonical identity; "
                "benchmark envelope protocol_version alone is not sufficient"
            )
        return None
    identities = [
        _validate_canonical_v3_identity(candidate, path=location)
        for location, candidate in candidates
    ]
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise ValueError(f"{path}: contradictory CreditTTT canonical protocol identities")
    return first


def _canonical_identity_from_artifact(
    payload: Mapping[str, Any],
    *,
    path: Path | str,
    required: bool,
) -> dict[str, Any] | None:
    """Validate canonical identity carried by a mechanism/label artifact.

    Offline artifacts historically store protocol fields flat in
    ``metadata`` whereas evaluators use ``credit_ttt_protocol``.  Both are
    accepted only when the complete immutable field set is present; a lone
    ``protocol=creditttt_qh2l_v3`` marker never authenticates an audit.
    """

    candidates: list[tuple[str, Any]] = []
    containers: list[tuple[str, Mapping[str, Any]]] = [("artifact", payload)]
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        containers.append(("metadata", metadata))
    for owner, container in containers:
        if "credit_ttt_protocol" in container:
            candidates.append(
                (f"{path}:{owner}.credit_ttt_protocol", container["credit_ttt_protocol"])
            )
        # Canonical pair-label artifacts use a flat metadata schema.  Treat a
        # partial set as an error rather than silently falling back to a
        # scalar marker.
        declares_v3 = (
            container.get("format") == CANONICAL_V3_PROTOCOL_IDENTITY["format"]
            or _is_v3_marker(container.get("protocol"))
        )
        # A label payload itself has a convenience top-level ``format`` and a
        # nested ``metadata`` object.  Do not mistake that wrapper for a flat
        # identity unless at least one additional canonical field is present.
        flat_identity_fields = set(CANONICAL_V3_PROTOCOL_FIELDS) & set(container)
        if declares_v3 and (len(flat_identity_fields) > 1 or "protocol" in container):
            candidates.append((f"{path}:{owner}", container))
    if not candidates:
        if required:
            raise ValueError(
                f"{path}: strict CreditTTT mechanism audit is missing canonical protocol identity"
            )
        return None
    identities = [
        _validate_canonical_v3_identity(candidate, path=location)
        for location, candidate in candidates
    ]
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise ValueError(f"{path}: contradictory canonical identities in mechanism artifact")
    return first


def _manifest_with_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON {path}: {exc}") from exc


def _verify_manifest(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    if not expected:
        raise ValueError("Manifest has no manifest_sha256; regenerate with the manifest subcommand")
    actual_payload = dict(manifest)
    actual_payload.pop("manifest_sha256", None)
    actual = sha256_json(actual_payload)
    if str(expected) != actual:
        raise ValueError(
            "Manifest hash mismatch: file may have been edited after freezing "
            f"(expected {expected}, computed {actual})"
        )


def _parse_int_list(value: str) -> list[int]:
    if not value.strip():
        raise ValueError("Expected a non-empty comma-separated integer list")
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"Invalid integer list {value!r}") from exc
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate values in integer list {value!r}")
    return result


def _normalize_task_set(value: str | None) -> str:
    """Return a canonical task-profile name.

    Aliases are accepted on the command line to make old experiment scripts
    easy to replay, while the manifest always records one of the two stable
    profile identifiers.
    """

    raw = DEFAULT_TASK_SET if value is None else str(value).strip().lower()
    try:
        return TASK_SET_ALIASES[raw]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_SETS))
        raise ValueError(f"Unknown task_set={value!r}; expected one of {choices}") from exc


def _task_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    task_set = _normalize_task_set(getattr(args, "task_set", None))
    tasks = [dict(task) for task in TASK_SETS[task_set]]

    # Root overrides are deliberately keyed by stable task IDs rather than by
    # positional indices.  This keeps the legacy color/shuffle flags working
    # and prevents a four-task profile from silently receiving the wrong
    # normalization statistics when task order changes.
    root_flags = {
        "color": "color_dataset_root",
        "shuffle_long": "shuffle_dataset_root",
        "shell_touch": "shell_touch_dataset_root",
        "intercept_medium": "intercept_medium_dataset_root",
        "remember_color3": "remember_color3_dataset_root",
        "remember_color9": "remember_color9_dataset_root",
    }
    for task in tasks:
        flag_name = root_flags.get(str(task["id"]))
        override = getattr(args, flag_name, None) if flag_name else None
        if override:
            task["dataset_root"] = str(Path(override))
    return tasks


def _union_delay_bins(tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the declared delay bins in the protocol's canonical order."""

    present = {
        str(item)
        for task in tasks
        for item in task.get("delay_bins_present", [])
    }
    unknown = present.difference(_DELAY_BIN_ORDER)
    if unknown:
        raise ValueError(f"Unknown delay bin(s) in task profile: {sorted(unknown)}")
    return [item for item in _DELAY_BIN_ORDER if item in present]


def _load_task_checkpoint_map(
    raw: Any,
    tasks: Sequence[Mapping[str, Any]],
    *,
    option_name: str,
) -> dict[str, str] | None:
    """Parse and validate a task-id → checkpoint JSON mapping.

    ``raw`` may be a path to a JSON file or an inline JSON object.  Every task
    in the selected profile must be present when a map is supplied; silently
    falling back to a common checkpoint for one task would make a benchmark
    comparison ambiguous.  Environment IDs are accepted as keys as a
    convenience, but the manifest always stores canonical task IDs.
    """

    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    source = Path(text).expanduser()
    if source.is_file():
        payload = _read_json(source)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{option_name} must be a JSON object or path to a JSON file; "
                f"could not parse {text!r}"
            ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{option_name} must contain a JSON object")

    task_ids = {str(task["id"]) for task in tasks}
    env_to_id = {str(task["env_id"]): str(task["id"]) for task in tasks}
    result: dict[str, str] = {}
    unknown: list[str] = []
    for raw_key, value in payload.items():
        key = str(raw_key)
        task_id = key if key in task_ids else env_to_id.get(key)
        if task_id is None:
            unknown.append(key)
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{option_name}[{key!r}] must be a non-empty checkpoint path/string"
            )
        if task_id in result:
            raise ValueError(f"{option_name} contains duplicate task key for {task_id!r}")
        result[task_id] = value
    if unknown:
        raise ValueError(
            f"{option_name} contains unknown task key(s): {sorted(unknown)}; "
            f"expected task IDs {sorted(task_ids)}"
        )
    missing = sorted(task_ids.difference(result))
    if missing:
        raise ValueError(
            f"{option_name} is incomplete; missing checkpoint(s) for task(s): {missing}"
        )
    return result


def _method_specs(include_optional: bool) -> list[dict[str, Any]]:
    return [dict(method) for method in METHODS if include_optional or not method["optional"]]


def _eval_command(
    *,
    repo_root: str,
    method: Mapping[str, Any],
    task: Mapping[str, Any],
    checkpoint: str,
    results_root: str,
    train_seed: str | int,
    n_episodes: int,
    start_seed: int,
    torch_seed: int,
    sim_backend: str,
    python_bin: str,
) -> list[str]:
    output = (
        Path(results_root)
        / str(method["id"])
        / f"train_seed_{train_seed}"
        / str(task["id"])
        / "eval.json"
    )
    evaluator = Path(repo_root) / str(method["evaluator"])
    command = [
        python_bin,
        str(evaluator),
        "--checkpoint",
        checkpoint,
        "--dataset-repo-id",
        str(task["dataset_repo_id"]),
        "--dataset-root",
        str(task["dataset_root"]),
        "--task",
        str(task["env_id"]),
        "--num-episodes",
        str(n_episodes),
        "--start-seed",
        str(start_seed),
        "--torch-seed",
        str(torch_seed),
        "--sim-backend",
        sim_backend,
        "--device",
        "cuda",
        "--output",
        str(output),
    ]
    if method["id"] == "clean_ttt":
        # This is an explicit structural clean baseline.  It prevents a
        # checkpoint carrying HD fields from being evaluated as a hidden HD
        # model while retaining the same TTT student/action cadence.  The
        # previous-action projection is also forced on: it is part of the
        # architecture-matched input schema, not part of the hindsight loss.
        command.extend(
            [
                "--no-hd-ttt-enabled",
                "--no-hd-learned-write-gate",
                "--hd-v3-include-previous-action",
            ]
        )
    elif method["id"] == NATIVE_VARIANT_K1:
        # The native checkpoint still predicts its complete 50-slot chunk;
        # only the runner consumption horizon changes.  This is the explicit
        # cadence-matched control for Clean/Credit K=1 and must not be
        # approximated by evaluating the canonical K=50 result and relabeling
        # its metadata.
        command.extend(["--execution-action-steps", "1"])
    return command


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.n_episodes) <= 0:
        raise ValueError("--n-episodes must be positive")
    if int(args.start_seed) < 0 or int(args.torch_seed) < 0:
        raise ValueError("seeds must be non-negative")
    train_seeds = _parse_int_list(args.train_seeds)
    task_set = _normalize_task_set(getattr(args, "task_set", None))
    tasks = _task_specs(args)
    methods = _method_specs(args.include_optional)
    protocol_id = TASK_SET_PROTOCOL_IDS[task_set]
    # Native SmolVLA is normally one frozen base checkpoint.  Student methods
    # are often trained independently for each task because LeRobot action
    # normalization statistics are task-local.  Optional maps make that
    # distinction explicit while retaining the historical common-checkpoint
    # command-line arguments for shared/multitask runs.
    checkpoint_map_inputs = {
        NATIVE_VARIANT_CHUNK: getattr(args, "native_checkpoints_json", None),
        NATIVE_VARIANT_K1: getattr(args, "native_checkpoints_json", None),
        "clean_ttt": getattr(args, "clean_checkpoints_json", None),
        "credit_ttt": getattr(args, "credit_checkpoints_json", None),
        "utility_kvb": getattr(args, "utility_checkpoints_json", None),
    }
    task_checkpoint_maps: dict[str, dict[str, str]] = {}
    checkpoint_scope: dict[str, str] = {}
    common_checkpoints = {
        NATIVE_VARIANT_CHUNK: str(args.native_checkpoint),
        NATIVE_VARIANT_K1: str(args.native_checkpoint),
        "clean_ttt": str(args.clean_checkpoint),
        "credit_ttt": str(args.credit_checkpoint),
    }
    if args.include_optional:
        common_checkpoints["utility_kvb"] = str(args.utility_checkpoint)
    for method_id in common_checkpoints:
        mapped = _load_task_checkpoint_map(
            checkpoint_map_inputs.get(method_id),
            tasks,
            option_name=f"{method_id}_checkpoints_json",
        )
        if mapped is None:
            task_checkpoint_maps[method_id] = {
                str(task["id"]): common_checkpoints[method_id] for task in tasks
            }
            checkpoint_scope[method_id] = "shared"
        else:
            task_checkpoint_maps[method_id] = mapped
            checkpoint_scope[method_id] = "per_task"
    checkpoint_map = {
        NATIVE_VARIANT_CHUNK: str(args.native_checkpoint),
        # Same frozen native checkpoint as K=50; this map entry changes only
        # runner cadence, not model weights or architecture.
        NATIVE_VARIANT_K1: str(args.native_checkpoint),
        "clean_ttt": str(args.clean_checkpoint),
        "credit_ttt": str(args.credit_checkpoint),
    }
    if args.include_optional:
        checkpoint_map["utility_kvb"] = str(args.utility_checkpoint)

    full_demo_recipe = all(
        list(task.get("train_demo_indices", [])) == [0, int(task.get("demo_count", 0)) - 1]
        and not task.get("validation_demo_indices")
        for task in tasks
    )
    manifest: dict[str, Any] = {
        "protocol_id": protocol_id,
        "protocol_version": PROTOCOL_VERSION,
        "task_set": task_set,
        # This object authenticates the *model method* and is deliberately
        # independent from the benchmark-envelope version above.  It is
        # copied into every frozen manifest and required on CreditTTT eval
        # records by ``_validate_eval_record``.
        "credit_ttt_protocol": dict(CANONICAL_V3_PROTOCOL_IDENTITY),
        "benchmark": "MIKASA-Robo-VLA",
        "created_by": "benchmark_credit_ttt_v3.py",
        "tasks": tasks,
        "methods": methods,
        "checkpoints": checkpoint_map,
        # Commands consume this task-specific map.  The legacy ``checkpoints``
        # field remains as the common-checkpoint fallback for old tooling.
        "checkpoints_by_task": task_checkpoint_maps,
        "checkpoint_scope": checkpoint_scope,
        "evaluation": {
            "n_episodes": int(args.n_episodes),
            "start_seed": int(args.start_seed),
            "episode_seeds": [int(args.start_seed) + i for i in range(int(args.n_episodes))],
            "torch_seed_base": int(args.torch_seed),
            "torch_seed_rule": "episode_i = torch_seed_base + i",
            "sim_backend": str(args.sim_backend),
            "obs_mode": "rgb",
            "control_mode": "pd_ee_delta_pose",
            "reward_mode": "normalized_dense",
            "wrapper_chain": "apply_mikasa_vla_wrappers(include_overlays=False)",
            "primary_metric": "success_once",
            "debug_metric": "mean_return",
            "confidence_interval": "hierarchical paired bootstrap, 95%, 10000 replicates",
            "paired_test": "two-sided exact McNemar on common episode seeds",
        },
        "training": {
            "student_train_seeds": train_seeds,
            "student_budget_policy": "same optimizer updates and seen frames for every student baseline",
            "teacher_cost_accounting": "reported separately; teacher is not deployed",
            "demo_split": {
                "train_and_label": (
                    "all official demonstrations [0, 249]"
                    if full_demo_recipe
                    else "task-profile declared training range"
                ),
                "offline_validation": (
                    "none (diagnostic-only metrics may reuse training demos)"
                    if full_demo_recipe
                    else "task-profile declared validation range"
                ),
                "test": "simulator seeds only; no test-seed tuning",
            },
            "official_demo_count": 250 if full_demo_recipe else None,
            "all_official_demos_used": full_demo_recipe,
            "validation_affects_training": False,
            "task_specific_statistics": True,
            "label_reuse_across_tasks": False,
        },
        "fairness_controls": {
            "match_trainable_student_parameters": True,
            "clean_ttt_architecture_control_required": True,
            "clean_ttt_architecture_control_note": (
                "The primary Clean-TTT checkpoint must be trained with the same "
                "SmolVLA-TTT config and hd_v3_include_previous_action=true, while "
                "hd_ttt_enabled/HD losses are disabled. A legacy checkpoint without "
                "that projection is reported only as an explicitly named ablation."
            ),
            "match_action_tail_unfreezing": True,
            # K=1 is the only native comparison with the same observation /
            # action cadence as Clean-TTT and CreditTTT.  K=50 remains useful
            # as the original-policy reference, but it is not a fair primary
            # memory comparison and is excluded from the primary pairwise
            # confidence gates below.
            "native_k1_control_required": True,
            "native_k1_control": NATIVE_VARIANT_K1,
            "native_k1_control_note": (
                "Native-SmolVLA-K1 uses the identical frozen native checkpoint and "
                "50-slot model horizon while the runner consumes one slot and "
                "re-queries every physical step."
            ),
            # Backward-compatible spelling retained for consumers of earlier
            # manifests.  Its value now means that the explicit K=1 control
            # is required; the K=50 reference itself is not a matched control.
            "native_chunk_control_required": True,
            "native_chunk_control_note": (
                "Deprecated alias: satisfy this requirement with the native_k1_control; "
                "Native K=50 is cadence-mismatched and exploratory only."
            ),
            "native_k50_reference": NATIVE_VARIANT_CHUNK,
            "native_k50_reference_note": (
                "Native-SmolVLA K=50 is a cadence-mismatched behavioral reference; "
                "never present CreditTTT-vs-K50 as the primary memory claim."
            ),
            "primary_comparison_scope": "matched_cadence_only",
            "primary_baselines": ["clean_ttt", NATIVE_VARIANT_K1],
            "exploratory_cadence_mismatched_baselines": [NATIVE_VARIANT_CHUNK],
            "persistent_reset_pair": True,
            "fixed_episode_seeds_across_methods": True,
        },
        "mechanism_audits": {
            # Delay bins are frozen per task.  Short episodes only support the
            # 1--16 bin; longer bins are included only by profiles whose task
            # metadata declares enough temporal extent.  Never extrapolate a
            # delay-bin result beyond the declared task support.
            "delay_bins_by_task": {
                str(task["id"]): list(task.get("delay_bins_present", []))
                for task in tasks
            },
            # Union retained for consumers that expect one flat list.  The
            # task-specific map above is authoritative for tables/plots.
            "delay_bins": _union_delay_bins(tasks),
            "pairwise_teacher_effect": {
                "metric": "pairwise_high_utility_delta_a_cosine",
                "group_by": ["task_id", "delay_bin"],
                "required_bins": {
                    str(task["id"]): list(task.get("delay_bins_present", []))
                    for task in tasks
                },
                "report_confidence_interval": True,
                "target": "final_slot0_action",
            },
            "intervention_retrieval": {
                "metric": "history_intervention_recall_at_k_minus_random",
                # K is frozen in the protocol and must not be selected on test
                # seeds.  A caller may report additional K values, but this
                # one is the pre-registered primary retrieval point.
                "k": 8,
                "group_by": ["task_id"],
                "random_reference": "uniform_eligible_events",
            },
            "deployment_interventions": {
                "required_conditions": ["correct", "wrong", "reset", "irrelevant"],
                "primary_selectivity_metric": "deployment_correct_minus_irrelevant_action_drift",
                "counterfactual_metric": "deployment_wrong_reset_action_drift",
                "action": "final_slot0_action",
            },
            "required": [check["id"] for check in DEFAULT_GO_NO_GO],
            "go_no_go": list(DEFAULT_GO_NO_GO),
            "interpretation": (
                "Unknown means the audit was not supplied; it is inconclusive by default, "
                "not evidence of success."
            ),
        },
        "results_layout": {
            "eval_json": "<results_root>/<method_id>/train_seed_<seed>/<task_id>/eval.json",
            "per_episode_required": True,
            "manifest_embedded_in_summary": True,
        },
        "results_root": str(args.results_root),
    }

    commands: list[dict[str, Any]] = []
    for method in methods:
        method_id = str(method["id"])
        method_seeds: Sequence[str | int]
        if method.get("replicate_policy") == "fixed_checkpoint":
            # Native-SmolVLA is a single fixed checkpoint, not three
            # independently trained students.  Repeating it under three
            # labels would be pseudo-replication; one fixed cluster is later
            # broadcast for paired comparisons with each student seed.
            method_seeds = ("fixed",)
        else:
            method_seeds = train_seeds
        for task in tasks:
            task_id = str(task["id"])
            checkpoint = task_checkpoint_maps[method_id][task_id]
            for train_seed in method_seeds:
                argv = _eval_command(
                    repo_root=str(args.repo_root),
                    method=method,
                    task=task,
                    checkpoint=checkpoint,
                    results_root=str(args.results_root),
                    train_seed=train_seed,
                    n_episodes=int(args.n_episodes),
                    start_seed=int(args.start_seed),
                    torch_seed=int(args.torch_seed),
                    sim_backend=str(args.sim_backend),
                    python_bin=str(args.python_bin),
                )
                commands.append(
                    {
                        "method_id": method_id,
                        "task_id": task_id,
                        "train_seed": train_seed,
                        "argv": argv,
                        "shell": shlex.join(argv),
                        "output": argv[-1],
                    }
                )
    manifest["commands"] = commands
    return _manifest_with_hash(manifest)


def _print_manifest_plan(manifest: Mapping[str, Any]) -> None:
    print(f"protocol={manifest['protocol_id']} version={manifest['protocol_version']}")
    if manifest.get("task_set"):
        print(f"task_set={manifest['task_set']}")
    if manifest.get("checkpoint_scope"):
        print(f"checkpoint_scope={json.dumps(manifest['checkpoint_scope'], sort_keys=True)}")
    print(f"manifest_sha256={manifest['manifest_sha256']}")
    evaluation = manifest["evaluation"]
    print(
        "evaluation="
        f"episodes:{evaluation['n_episodes']} "
        f"seeds:{evaluation['start_seed']}.."
        f"{evaluation['start_seed'] + evaluation['n_episodes'] - 1} "
        f"torch_seed:{evaluation['torch_seed_base']}+i"
    )
    for command in manifest.get("commands", []):
        print(
            f"[{command['method_id']} | {command['task_id']} | "
            f"train_seed={command['train_seed']}] {command['shell']}"
        )


def _path_train_seed(path: Path) -> str:
    match = re.search(r"train[_-]seed[_-]([^/\\]+)", str(path))
    return match.group(1) if match else "unknown"


def _method_from_metadata(result: Mapping[str, Any], path: Path) -> tuple[str, str | None]:
    model = result.get("model")
    model = model if isinstance(model, Mapping) else {}
    method = str(model.get("method") or result.get("method") or "")
    # Nested canonical provenance is the authoritative V3 marker.  Use its
    # scalar protocol value for classification; passing the mapping itself to
    # ``str`` would make a malformed object look like a marker.
    nested_protocol_values: list[Any] = []
    for container in (result, model):
        identity = container.get("credit_ttt_protocol")
        if isinstance(identity, Mapping):
            nested_protocol_values.append(identity.get("protocol"))
        elif identity is not None:
            nested_protocol_values.append(identity)
    protocol_candidates = (
        model.get("protocol_version"),
        model.get("protocol_id"),
        model.get("protocol"),
        *nested_protocol_values,
        model.get("hd_attribution_protocol"),
        model.get("format"),
        result.get("protocol_version"),
        result.get("protocol_id"),
        result.get("protocol"),
    )
    has_v3_marker = any(_is_v3_marker(value) for value in protocol_candidates)
    protocol = (
        next((value for value in protocol_candidates if _is_v3_marker(value)), None)
        if has_v3_marker
        else next((value for value in protocol_candidates if value is not None), None)
    )
    protocol_text = None if protocol is None else str(protocol)
    normalized = method.strip().lower().replace("_", "-")

    if method == "CreditTTT" or has_v3_marker:
        # A V3 result must identify itself explicitly.  In particular, an old
        # HD-TTT result with a path named credit_ttt is not accepted.
        # ``hd_attribution_protocol=credit_ttt_v3_query_effect`` is accepted
        # as a transitional marker while the evaluator is being upgraded;
        # it still cannot be confused with legacy/v2 strings.
        if not has_v3_marker or normalized not in {"creditttt", "hd-ttt", ""}:
            raise ValueError(
                f"{path}: CreditTTT result lacks explicit V3 metadata; "
                "legacy HD-TTT/V2 JSON cannot be relabeled as CreditTTT"
            )
        return "credit_ttt", protocol_text
    if normalized in {"hd-ttt", "hindsight-distilled-ttt", "v2", "v1"}:
        return "legacy_rejected", protocol_text
    if normalized in {
        "smolvla-k1",
        "native-smolvla-k1",
        "native-smolvla-receding",
    }:
        return NATIVE_VARIANT_K1, protocol_text
    if normalized in {"smolvla", "native-smolvla"}:
        # The baseline evaluator emits an explicit variant/cadence marker for
        # both native modes.  Do not infer K=1 merely from a directory name or
        # a truncated success vector: that would allow a K=50 result to be
        # relabeled as the matched-cadence control.
        variant = model.get("benchmark_variant") or result.get("benchmark_variant")
        cadence = model.get("execution_cadence") or result.get("execution_cadence")
        if variant == NATIVE_VARIANT_K1 or cadence == NATIVE_CADENCE_RECEDING:
            return NATIVE_VARIANT_K1, protocol_text
        return NATIVE_VARIANT_CHUNK, protocol_text
    if normalized in {"clean-ttt", "clean-ttt-kvb", "cleanttt"}:
        return "clean_ttt", protocol_text
    if normalized in {"utility-kvb", "utility-kvb-baseline"}:
        return "utility_kvb", protocol_text

    # A path-level method directory is accepted only if the JSON has at least
    # a method field; this keeps accidental summary files from being treated
    # as evaluation results.
    path_parts = {part.lower() for part in path.parts}
    if "credit_ttt" in path_parts or "creditttt" in path_parts:
        raise ValueError(
            f"{path}: result is in a CreditTTT directory but metadata method={method!r} "
            "is not the explicit V3 marker"
        )
    if method:
        return "unknown", protocol_text
    return "unknown", protocol_text


def _iter_result_records(payload: Any, path: Path) -> Iterable[dict[str, Any]]:
    """Yield per-task records from either adapter JSON envelope."""

    if isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
        inherited = {
            key: payload[key]
            for key in (
                "model",
                "benchmark_protocol",
                "benchmark_commit",
                "protocol_id",
                "protocol_version",
                "credit_ttt_protocol",
                "benchmark_variant",
                "execution_cadence",
                "execution_action_steps",
                "model_action_horizon",
            )
            if key in payload
        }
        for item in payload["results"]:
            if isinstance(item, Mapping):
                record = dict(item)
                # Some lightweight evaluators put provenance on the envelope
                # rather than repeating it for each task.  Inherit only
                # absent keys; a per-task value remains authoritative.
                for key, value in inherited.items():
                    record.setdefault(key, value)
                yield record
        return
    if isinstance(payload, Mapping) and "successes" in payload and "env_id" in payload:
        yield dict(payload)
        return
    # A summary file has no per-episode vector and must never be aggregated.
    raise ValueError(f"{path}: expected an eval JSON with results[] or env_id+successes")


def _coherent_result_field(
    result: Mapping[str, Any],
    model: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> Any:
    """Read a provenance field while rejecting contradictory envelope copies."""

    values = [
        container[key]
        for container in (result, model)
        if key in container and container[key] is not None
    ]
    if not values:
        return None
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(
            f"{path}: contradictory {key!r} values on result/model envelopes: {values!r}"
        )
    return first


def _validate_eval_record(
    result: Mapping[str, Any],
    *,
    path: Path,
    expected_method: str,
    expected_task: Mapping[str, Any] | None,
    expected_episode_seeds: Sequence[int],
) -> dict[str, Any]:
    method, protocol = _method_from_metadata(result, path)
    if method == "legacy_rejected":
        raise ValueError(
            f"{path}: legacy HD-TTT/V1/V2 metadata is not a valid CreditTTT result; "
            "run the complete V3 protocol instead"
        )
    if method != expected_method:
        raise ValueError(
            f"{path}: metadata resolves to method {method!r}, but result directory/manifest "
            f"expects {expected_method!r}"
        )
    canonical_identity: dict[str, Any] | None = None
    if expected_method == "credit_ttt":
        if not _is_v3_marker(protocol):
            raise ValueError(
                f"{path}: CreditTTT requires an explicit V3 protocol marker; "
                f"got {protocol!r}"
            )
        # A scalar marker is useful for human-readable summaries, but it is
        # not sufficient provenance.  Require and validate the complete
        # canonical identity object emitted by the V3 evaluator.
        canonical_identity = _canonical_identity_from_result(
            result,
            path=path,
            required=True,
        )
    model = result.get("model")
    model = model if isinstance(model, Mapping) else {}
    if expected_method in {"native_smolvla", "clean_ttt"} and bool(model.get("hd_ttt_enabled", False)):
        raise ValueError(
            f"{path}: {expected_method} result advertises hd_ttt_enabled=true; "
            "use an explicit CreditTTT method marker or a clean checkpoint"
        )
    benchmark_protocol = str(result.get("benchmark_protocol") or "")
    if not benchmark_protocol:
        raise ValueError(
            f"{path}: missing benchmark_protocol; only the official MIKASA runner "
            "is admissible for the frozen protocol"
        )
    if OFFICIAL_PROTOCOL not in benchmark_protocol:
        raise ValueError(
            f"{path}: unsupported benchmark_protocol={benchmark_protocol!r}; "
            f"expected {OFFICIAL_PROTOCOL!r}"
        )
    if expected_task is not None:
        expected_env = str(expected_task["env_id"])
        if str(result.get("env_id")) != expected_env:
            raise ValueError(
                f"{path}: env_id={result.get('env_id')!r} does not match task {expected_env!r}"
            )
    successes = result.get("successes")
    if not isinstance(successes, list) or not successes:
        raise ValueError(f"{path}: successes must be a non-empty per-episode list")
    # JSON ``bool`` values are the canonical schema.  Accept numeric 0/1 for
    # interoperability with a few metric exporters, but reject arbitrary
    # truthy strings (for example ``"false"``), which would silently turn a
    # failed episode into a success under Python's ``bool`` conversion.
    successes_bool: list[bool] = []
    for index, value in enumerate(successes):
        if isinstance(value, bool):
            successes_bool.append(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
                raise ValueError(
                    f"{path}: successes[{index}] must be bool or numeric 0/1, got {value!r}"
                )
            successes_bool.append(bool(int(numeric)))
        else:
            raise ValueError(
                f"{path}: successes[{index}] must be bool or numeric 0/1, got {value!r}"
            )
    seeds = result.get("episode_seeds")
    if seeds is None:
        start = result.get("start_seed")
        seeds = [int(start) + i for i in range(len(successes_bool))] if start is not None else None
    if not isinstance(seeds, list) or len(seeds) != len(successes_bool):
        raise ValueError(f"{path}: episode_seeds must align one-to-one with successes")
    seeds_int = [int(seed) for seed in seeds]
    if len(set(seeds_int)) != len(seeds_int):
        raise ValueError(f"{path}: duplicate episode seeds")
    expected_set = {int(seed) for seed in expected_episode_seeds}
    if set(seeds_int) != expected_set:
        raise ValueError(
            f"{path}: episode seed set {sorted(seeds_int)[:3]}... does not match frozen "
            f"set {sorted(expected_set)[:3]}..."
        )
    chunk = result.get("action_chunk_size")
    if chunk is None:
        raise ValueError(
            f"{path}: missing action_chunk_size; cadence must be explicit for the "
            "Native-SmolVLA versus TTT comparison"
        )
    try:
        chunk_int = int(chunk)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: action_chunk_size must be an integer, got {chunk!r}") from exc
    if expected_method in {NATIVE_VARIANT_CHUNK, NATIVE_VARIANT_K1}:
        # Native controls share the same 50-slot model prediction.  The
        # result's action_chunk_size is the runner cadence and is therefore
        # checked independently from model_action_horizon.
        expected_spec = next(
            (item for item in METHODS if item["id"] == expected_method),
            {},
        )
        expected_chunk = int(expected_spec.get("expected_action_chunk_size", -1))
        if chunk_int != expected_chunk:
            raise ValueError(
                f"{path}: {expected_method} must report action_chunk_size={expected_chunk}, "
                f"got {chunk}"
            )
        if bool(model.get("ttt_enabled", False)) or str(model.get("policy_type", "")) != "smolvla":
            raise ValueError(
                f"{path}: {expected_method} must be an original SmolVLA model without TTT"
            )
        variant = _coherent_result_field(result, model, "benchmark_variant", path=path)
        cadence = _coherent_result_field(result, model, "execution_cadence", path=path)
        model_horizon = _coherent_result_field(
            result, model, "model_action_horizon", path=path
        )
        execution_steps = _coherent_result_field(
            result, model, "execution_action_steps", path=path
        )
        if variant != expected_spec.get("benchmark_variant"):
            raise ValueError(
                f"{path}: {expected_method} requires benchmark_variant="
                f"{expected_spec.get('benchmark_variant')!r}, got {variant!r}"
            )
        if cadence != expected_spec.get("expected_execution_cadence"):
            raise ValueError(
                f"{path}: {expected_method} requires execution_cadence="
                f"{expected_spec.get('expected_execution_cadence')!r}, got {cadence!r}"
            )
        if model_horizon is None or int(model_horizon) != NATIVE_MODEL_ACTION_HORIZON:
            raise ValueError(
                f"{path}: {expected_method} must report model_action_horizon="
                f"{NATIVE_MODEL_ACTION_HORIZON}, got {model_horizon!r}"
            )
        if execution_steps is None or int(execution_steps) != int(
            expected_spec.get("expected_execution_action_steps", expected_chunk)
        ):
            raise ValueError(
                f"{path}: {expected_method} must report execution_action_steps="
                f"{expected_spec.get('expected_execution_action_steps')!r}, got {execution_steps!r}"
            )
    if expected_method in {"clean_ttt", "credit_ttt", "utility_kvb"}:
        if chunk_int != 1:
            raise ValueError(f"{path}: TTT methods must report action_chunk_size=1, got {chunk}")
        expected_ttt_horizon = NATIVE_MODEL_ACTION_HORIZON
        model_horizon = _coherent_result_field(
            result, model, "model_action_horizon", path=path
        )
        execution_steps = _coherent_result_field(
            result, model, "execution_action_steps", path=path
        )
        cadence = _coherent_result_field(result, model, "execution_cadence", path=path)
        if model_horizon is None or int(model_horizon) != expected_ttt_horizon:
            raise ValueError(
                f"{path}: {expected_method} must report model_action_horizon="
                f"{expected_ttt_horizon}, got {model_horizon!r}"
            )
        if execution_steps is None or int(execution_steps) != 1:
            raise ValueError(
                f"{path}: {expected_method} must report execution_action_steps=1, "
                f"got {execution_steps!r}"
            )
        if cadence != NATIVE_CADENCE_RECEDING:
            raise ValueError(
                f"{path}: {expected_method} must report execution_cadence="
                f"{NATIVE_CADENCE_RECEDING!r}, got {cadence!r}"
            )
    order = np.argsort(np.asarray(seeds_int, dtype=np.int64))
    return {
        "method_id": expected_method,
        "env_id": str(result.get("env_id")),
        "successes": [successes_bool[int(index)] for index in order],
        "episode_seeds": [seeds_int[int(index)] for index in order],
        "returns": result.get("returns"),
        "source": str(path),
        "protocol_version": protocol,
        "credit_ttt_protocol": canonical_identity,
        "train_seed": str(result.get("train_seed") or _path_train_seed(path)),
        "raw": dict(result),
    }


def _discover_eval_files(results_root: Path) -> list[Path]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    candidates = sorted(results_root.rglob("*.json"))
    return [path for path in candidates if path.name not in {"summary.json", "manifest.json", "aggregate.json"}]


def collect_eval_runs(
    manifest: Mapping[str, Any],
    results_root: Path,
    *,
    allow_incomplete: bool = False,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    methods = {str(item["id"]): item for item in manifest["methods"]}
    tasks = {str(item["id"]): item for item in manifest["tasks"]}
    env_to_task = {str(item["env_id"]): task_id for task_id, item in tasks.items()}
    expected_seeds = [int(seed) for seed in manifest["evaluation"]["episode_seeds"]]
    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for path in _discover_eval_files(results_root):
        try:
            payload = _read_json(path)
            records = list(_iter_result_records(payload, path))
            for record in records:
                env_id = str(record.get("env_id"))
                task_id = env_to_task.get(env_id)
                if task_id is None:
                    # Ignore unrelated diagnostics under a shared root, but
                    # do not silently ignore an explicitly named task folder.
                    continue
                method_id, _ = _method_from_metadata(record, path)
                if method_id == "legacy_rejected":
                    raise ValueError(
                        "legacy HD-TTT/V1/V2 metadata cannot be aggregated as CreditTTT"
                    )
                if method_id not in methods:
                    continue
                validated = _validate_eval_record(
                    record,
                    path=path,
                    expected_method=method_id,
                    expected_task=tasks[task_id],
                    expected_episode_seeds=expected_seeds,
                )
                train_seed = validated["train_seed"]
                key = (method_id, task_id, train_seed)
                if key in runs:
                    raise ValueError(f"Duplicate evaluation run for {key}: {path} and {runs[key]['source']}")
                validated["task_id"] = task_id
                runs[key] = validated
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
    if errors and not allow_incomplete:
        joined = "\n".join(f"- {error}" for error in errors[:20])
        more = "" if len(errors) <= 20 else f"\n(and {len(errors) - 20} more errors)"
        raise ValueError(f"Evaluation provenance validation failed:\n{joined}{more}")
    return runs


def _clusters_for_method_task(
    runs: Mapping[tuple[str, str, str], Mapping[str, Any]],
    method_id: str,
    task_id: str,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for (method, task, train_seed), run in runs.items():
        if method == method_id and task == task_id:
            result[str(train_seed)] = np.asarray(run["successes"], dtype=np.float64)
    return result


def _bootstrap_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return math.nan, math.nan, math.nan
    if values.size == 1:
        observed = float(values[0])
        return observed, observed, observed
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(int(n_bootstrap), values.size))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _hierarchical_sr_ci(
    clusters: Mapping[str, np.ndarray],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap train-seed clusters, then episodes within each cluster."""

    if not clusters:
        return math.nan, math.nan, math.nan
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in clusters.values()]
    if any(array.size == 0 for array in arrays):
        return math.nan, math.nan, math.nan
    observed = float(np.mean([array.mean() for array in arrays]))
    rng = np.random.default_rng(seed)
    cluster_indices = rng.integers(0, len(arrays), size=(int(n_bootstrap), len(arrays)))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for row, indices in enumerate(cluster_indices):
        cluster_means: list[float] = []
        for index in indices:
            array = arrays[int(index)]
            sampled = array[rng.integers(0, array.size, size=array.size)]
            cluster_means.append(float(sampled.mean()))
        boot[row] = float(np.mean(cluster_means))
    return observed, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _paired_episode_vectors(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    paired: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    common = set(left) & set(right)
    if common:
        clusters: Iterable[str] = sorted(common)
        for cluster in clusters:
            a = np.asarray(left[cluster], dtype=np.float64)
            b = np.asarray(right[cluster], dtype=np.float64)
            if a.shape != b.shape:
                raise ValueError(f"Cannot pair cluster {cluster}: shapes {a.shape} and {b.shape}")
            paired[cluster] = (a, b)
        return paired

    # A fixed native checkpoint has one deterministic evaluation cluster.  It
    # is statistically valid to compare that same cluster against each
    # independently trained student seed; treating the repeated native vector
    # as three *training* replicates would instead understate uncertainty.
    if len(left) == 1:
        fixed_left = np.asarray(next(iter(left.values())), dtype=np.float64)
        for cluster, value in sorted(right.items()):
            right_value = np.asarray(value, dtype=np.float64)
            if fixed_left.shape != right_value.shape:
                raise ValueError(f"Cannot broadcast fixed cluster: shapes {fixed_left.shape} and {right_value.shape}")
            paired[f"{cluster}|fixed_left"] = (fixed_left, right_value)
        return paired
    if len(right) == 1:
        baseline = np.asarray(next(iter(right.values())), dtype=np.float64)
        for cluster, value in sorted(left.items()):
            student = np.asarray(value, dtype=np.float64)
            if baseline.shape != student.shape:
                raise ValueError(f"Cannot broadcast fixed cluster: shapes {student.shape} and {baseline.shape}")
            paired[f"{cluster}|fixed_right"] = (student, baseline)
        return paired
    return paired


def _hierarchical_paired_ci(
    paired: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    if not paired:
        return math.nan, math.nan, math.nan, 0
    arrays = [np.asarray(a) - np.asarray(b) for a, b in paired.values()]
    observed = float(np.mean([array.mean() for array in arrays]))
    n_clusters = len(arrays)
    rng = np.random.default_rng(seed)
    cluster_indices = rng.integers(0, n_clusters, size=(int(n_bootstrap), n_clusters))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for row, indices in enumerate(cluster_indices):
        means: list[float] = []
        for index in indices:
            array = arrays[int(index)]
            means.append(float(array[rng.integers(0, array.size, size=array.size)].mean()))
        boot[row] = float(np.mean(means))
    count = int(sum(array.size for array in arrays))
    return observed, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), count


def _exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    if left_bool.shape != right_bool.shape:
        raise ValueError(f"McNemar vectors have different shapes {left_bool.shape} and {right_bool.shape}")
    b = int(np.logical_and(left_bool, ~right_bool).sum())
    c = int(np.logical_and(~left_bool, right_bool).sum())
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2.0**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "ours_success_baseline_fail": b,
        "ours_fail_baseline_success": c,
        "discordant": discordant,
        "p_two_sided": float(p_value),
    }


def _paired_test_report(
    paired: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Report McNemar without pseudo-replicating a fixed checkpoint.

    When a single native checkpoint is broadcast against several student
    seeds, pooling all copied baseline vectors would count the same baseline
    episode multiple times.  Keep a per-student-cluster exact test instead
    and make the limitation explicit in the JSON.
    """

    if any("|fixed_" in str(cluster) for cluster in paired):
        return {
            "pooled": None,
            "status": "per_cluster_only_fixed_checkpoint_broadcast",
            "per_cluster": {
                str(cluster): _exact_mcnemar(left, right)
                for cluster, (left, right) in paired.items()
            },
        }
    left = np.concatenate([a for a, _ in paired.values()])
    right = np.concatenate([b for _, b in paired.values()])
    return {"pooled": _exact_mcnemar(left, right), "status": "pooled_common_clusters"}


def _macro_paired_ci(
    paired_by_task: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Task-stratified hierarchical bootstrap for the official macro SR."""

    task_items = [(task, pairs) for task, pairs in sorted(paired_by_task.items()) if pairs]
    if not task_items:
        return math.nan, math.nan, math.nan, 0
    observed_task_deltas = []
    count = 0
    for _, pairs in task_items:
        arrays = [np.asarray(a) - np.asarray(b) for a, b in pairs.values()]
        observed_task_deltas.append(float(np.mean([array.mean() for array in arrays])))
        count += sum(array.size for array in arrays)
    observed = float(np.mean(observed_task_deltas))
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for row in range(int(n_bootstrap)):
        task_deltas: list[float] = []
        task_indices = rng.integers(0, len(task_items), size=len(task_items))
        for task_index in task_indices:
            _, pairs = task_items[int(task_index)]
            arrays = [np.asarray(a) - np.asarray(b) for a, b in pairs.values()]
            cluster_indices = rng.integers(0, len(arrays), size=len(arrays))
            cluster_means = []
            for cluster_index in cluster_indices:
                array = arrays[int(cluster_index)]
                cluster_means.append(float(array[rng.integers(0, array.size, size=array.size)].mean()))
            task_deltas.append(float(np.mean(cluster_means)))
        boot[row] = float(np.mean(task_deltas))
    return observed, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), count


def aggregate_results(
    manifest: Mapping[str, Any],
    results_root: Path,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 1_729,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    runs = collect_eval_runs(manifest, results_root, allow_incomplete=allow_incomplete)
    methods = {str(item["id"]): item for item in manifest["methods"]}
    tasks = {str(item["id"]): item for item in manifest["tasks"]}
    method_task: dict[str, dict[str, Any]] = {}
    for method_id, method in methods.items():
        if bool(method.get("optional")) and not any(key[0] == method_id for key in runs):
            continue
        method_task[method_id] = {}
        for task_id in tasks:
            clusters = _clusters_for_method_task(runs, method_id, task_id)
            if not clusters:
                continue
            observed, low, high = _hierarchical_sr_ci(
                clusters,
                n_bootstrap=n_bootstrap,
                seed=seed + _stable_seed(method_id, task_id),
            )
            method_task[method_id][task_id] = {
                "sr": observed,
                "ci95": [low, high],
                "n_train_seed_clusters": len(clusters),
                "episodes_per_cluster": sorted({int(array.size) for array in clusters.values()}),
                "train_seeds": sorted(clusters),
            }

    # Only cadence-matched baselines are eligible for the primary method
    # claim.  Native K=50 is retained as a useful behavioral reference but is
    # intentionally routed to a separate exploratory namespace so its action
    # persistence/observation-frequency advantage cannot be mistaken for a
    # memory gain.
    primary_baselines = ["clean_ttt", NATIVE_VARIANT_K1, "utility_kvb"]
    exploratory_baselines = [NATIVE_VARIANT_CHUNK]
    pairwise: dict[str, Any] = {}
    pairwise_exploratory: dict[str, Any] = {}
    ours_runs = _clusters_for_method_task  # keep the local name readable below

    def _make_pairwise_entry(baseline: str, *, primary: bool) -> dict[str, Any] | None:
        if "credit_ttt" not in method_task or baseline not in method_task:
            return None
        per_task: dict[str, Any] = {}
        paired_by_task: dict[str, Mapping[str, tuple[np.ndarray, np.ndarray]]] = {}
        for task_id in tasks:
            ours = ours_runs(runs, "credit_ttt", task_id)
            base = ours_runs(runs, baseline, task_id)
            paired = _paired_episode_vectors(ours, base)
            if not paired:
                continue
            delta, low, high, count = _hierarchical_paired_ci(
                paired,
                n_bootstrap=n_bootstrap,
                seed=seed + _stable_seed("credit_ttt", baseline, task_id),
            )
            per_task[task_id] = {
                "delta_sr_credit_minus_baseline": delta,
                "ci95": [low, high],
                "n_paired_episodes": count,
                "mcnemar": _paired_test_report(paired),
                "paired_train_seed_clusters": sorted(paired),
            }
            paired_by_task[task_id] = paired
        if not per_task:
            return None
        macro, low, high, count = _macro_paired_ci(
            paired_by_task,
            n_bootstrap=n_bootstrap,
            seed=seed + _stable_seed("credit_ttt", baseline, "macro"),
        )
        baseline_spec = methods.get(baseline, {})
        scope = "matched_cadence" if primary else "cadence_mismatched_reference"
        entry: dict[str, Any] = {
            "per_task": per_task,
            "macro_delta_sr": macro,
            "macro_ci95": [low, high],
            "macro_n_paired_episodes": count,
            "primary": bool(primary),
            "comparison_scope": scope,
            "credit_ttt_action_chunk_size": 1,
            "baseline_action_chunk_size": baseline_spec.get("expected_action_chunk_size"),
        }
        if not primary:
            entry["warning"] = (
                "Native K=50 consumes a different number of actions per query; "
                "this comparison is descriptive only and cannot support a causal "
                "memory-improvement claim."
            )
        return entry

    for baseline in primary_baselines:
        entry = _make_pairwise_entry(baseline, primary=True)
        if entry is not None:
            pairwise[f"CreditTTT_vs_{baseline}"] = entry
    for baseline in exploratory_baselines:
        entry = _make_pairwise_entry(baseline, primary=False)
        if entry is not None:
            pairwise_exploratory[f"CreditTTT_vs_{baseline}"] = entry

    required_tasks = set(tasks)
    native_k1_tasks = set(method_task.get(NATIVE_VARIANT_K1, {}))
    missing_native_k1_tasks = sorted(required_tasks - native_k1_tasks)
    fairness = {
        "primary_scope": "matched_cadence_only",
        "required_native_k1_control": True,
        "native_k1_control_id": NATIVE_VARIANT_K1,
        "native_k1_complete": not missing_native_k1_tasks,
        "native_k1_missing_tasks": missing_native_k1_tasks,
        "native_k50_reference_id": NATIVE_VARIANT_CHUNK,
        "native_k50_is_primary": False,
        "method_cadence": {
            method_id: {
                "runner_action_chunk_size": method.get("expected_action_chunk_size"),
                "model_action_horizon": method.get("expected_model_action_horizon"),
                "comparison_scope": method.get("comparison_scope"),
            }
            for method_id, method in methods.items()
        },
    }

    payload: dict[str, Any] = {
        "protocol_id": manifest["protocol_id"],
        "protocol_version": manifest["protocol_version"],
        "credit_ttt_protocol": dict(manifest["credit_ttt_protocol"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "results_root": str(results_root),
        "bootstrap": {"replicates": int(n_bootstrap), "seed": int(seed)},
        "runs_discovered": len(runs),
        "per_method_task": method_task,
        "pairwise": pairwise,
        "pairwise_exploratory": pairwise_exploratory,
        "fairness": fairness,
        "primary_metric": "success_once",
        "debug_metric": "mean_return",
        "legacy_v1_v2_included": False,
    }
    return payload


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, list):
        # Mechanism audits often store one value per delay bin/seed.  Expose
        # the mean under the field name as a deterministic default.  For a
        # two-element confidence interval, retain explicit ``_low``/``_high``
        # aliases as well; this lets a go/no-go gate require a positive lower
        # bound instead of accepting a positive point estimate alone.
        numeric_values = [
            float(item)
            for item in value
            if isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
        ]
        if numeric_values and len(numeric_values) == len(value):
            flattened[prefix] = float(np.mean(numeric_values))
            if len(numeric_values) == 2:
                flattened[f"{prefix}_low"] = numeric_values[0]
                flattened[f"{prefix}_high"] = numeric_values[1]
        else:
            for index, child in enumerate(value):
                child_prefix = f"{prefix}.{index}" if prefix else str(index)
                flattened.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        flattened[prefix] = float(value)
    return flattened


def _find_alias(flattened: Mapping[str, float], aliases: Sequence[str]) -> tuple[str, float] | None:
    # Dotted paths are emitted when a metric is nested in a JSON object;
    # treating dots like underscores allows a paper-facing alias such as
    # ``deployment.correct_action_drift`` to be declared once without making
    # callers know the exact envelope nesting.
    normalized = {
        key.lower().replace("-", "_").replace(".", "_"): (key, value)
        for key, value in flattened.items()
    }
    for alias in aliases:
        alias_norm = alias.lower().replace("-", "_").replace(".", "_")
        if alias_norm in normalized:
            return normalized[alias_norm]
        # Accept a nested key ending in the declared metric name, but reject
        # ambiguous matches rather than silently selecting a random field.
        matches = [
            item
            for key, item in normalized.items()
            if key.endswith("_" + alias_norm) or key.endswith("." + alias_norm)
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _evaluate_check(value: float, operator: str, threshold: Any) -> bool:
    if operator == ">=":
        return value >= float(threshold)
    if operator == ">":
        return value > float(threshold)
    if operator == "==":
        return math.isclose(value, float(threshold), abs_tol=1e-9)
    if operator == "between":
        low, high = threshold
        return float(low) <= value <= float(high)
    raise ValueError(f"Unsupported go/no-go operator {operator!r}")


def run_go_no_go_checks(
    mechanism_payload: Mapping[str, Any],
    *,
    aggregate_payload: Mapping[str, Any] | None = None,
    strict: bool = False,
    protocol_id: str = PROTOCOL_ID,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    flattened = _flatten_numeric(mechanism_payload)
    checks: list[dict[str, Any]] = []
    for spec in DEFAULT_GO_NO_GO:
        found = _find_alias(flattened, spec["aliases"])
        if found is None:
            status = "FAIL" if strict and spec.get("required", False) else "UNKNOWN"
            checks.append(
                {
                    "id": spec["id"],
                    "status": status,
                    "value": None,
                    "operator": spec["operator"],
                    "threshold": spec["threshold"],
                    "reason": spec["reason"],
                }
            )
            continue
        key, value = found
        passed = _evaluate_check(value, spec["operator"], spec["threshold"])
        checks.append(
            {
                "id": spec["id"],
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "source_key": key,
                "operator": spec["operator"],
                "threshold": spec["threshold"],
                "reason": spec["reason"],
            }
        )

    # Benchmark evidence is deliberately a separate family from mechanism
    # checks.  A positive SR difference is necessary for the paper claim but
    # cannot substitute for proving the causal mechanism.  The aggregate
    # writer places only cadence-matched comparisons in ``pairwise``; native
    # K=50 comparisons live in ``pairwise_exploratory`` and are never turned
    # into a primary gate here.
    if aggregate_payload is not None:
        fairness = aggregate_payload.get("fairness")
        if isinstance(fairness, Mapping):
            native_k1_complete = fairness.get("native_k1_complete")
            missing_tasks = fairness.get("native_k1_missing_tasks", [])
            if native_k1_complete is True:
                fairness_status = "PASS"
            elif strict:
                fairness_status = "FAIL"
            else:
                fairness_status = "INCONCLUSIVE"
            checks.append(
                {
                    "id": "benchmark_native_k1_cadence_control",
                    "status": fairness_status,
                    "value": native_k1_complete,
                    "missing_tasks": list(missing_tasks)
                    if isinstance(missing_tasks, Sequence)
                    and not isinstance(missing_tasks, (str, bytes))
                    else missing_tasks,
                    "reason": (
                        "A K=1 native receding-horizon control is required before "
                        "credit gains can be attributed to memory rather than action cadence."
                    ),
                }
            )
        else:
            checks.append(
                {
                    "id": "benchmark_native_k1_cadence_control",
                    "status": "FAIL" if strict else "INCONCLUSIVE",
                    "value": None,
                    "missing_tasks": None,
                    "reason": (
                        "Aggregate artifact has no fairness contract; regenerate it with "
                        "the V3 coordinator so Native K=50 cannot be confused with K=1."
                    ),
                }
            )
        for pair_name, pair in aggregate_payload.get("pairwise", {}).items():
            low_high = pair.get("macro_ci95")
            if not isinstance(low_high, list) or len(low_high) != 2:
                continue
            low, high = float(low_high[0]), float(low_high[1])
            if low > 0:
                status = "PASS"
            elif high < 0:
                status = "FAIL"
            else:
                status = "INCONCLUSIVE"
            checks.append(
                {
                    "id": f"benchmark_{pair_name}",
                    "status": status,
                    "value": pair.get("macro_delta_sr"),
                    "ci95": low_high,
                    "reason": "CreditTTT must improve the paired macro SR without seed cherry-picking.",
                }
            )

    statuses = [item["status"] for item in checks]
    if "FAIL" in statuses:
        overall = "NO-GO"
    elif "UNKNOWN" in statuses or "INCONCLUSIVE" in statuses:
        overall = "INCONCLUSIVE"
    else:
        overall = "GO"
    return {
        "protocol_id": str(protocol_id),
        "protocol_version": str(protocol_version),
        "strict": bool(strict),
        "overall": overall,
        "checks": checks,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    _verify_manifest(manifest)
    if manifest.get("protocol_id") not in SUPPORTED_PROTOCOL_IDS:
        raise ValueError(f"Unsupported protocol_id={manifest.get('protocol_id')!r}")
    # Manifests generated before the profile split have no ``task_set`` field;
    # infer the legacy profile from their envelope ID.  New manifests record
    # the profile explicitly and are checked against the envelope ID.
    protocol_id = str(manifest.get("protocol_id"))
    inferred_task_set = (
        "legacy_two" if protocol_id == LEGACY_TWO_TASK_PROTOCOL_ID else "published_four"
    )
    task_set = _normalize_task_set(manifest.get("task_set", inferred_task_set))
    expected_protocol = TASK_SET_PROTOCOL_IDS[task_set]
    if protocol_id != expected_protocol:
        raise ValueError(
            f"Manifest task_set={task_set!r} conflicts with protocol_id={protocol_id!r}"
        )
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)) or not raw_tasks:
        raise ValueError("Manifest must contain a non-empty tasks sequence")
    _validate_canonical_v3_identity(
        manifest.get("credit_ttt_protocol"),
        path=f"{path}:credit_ttt_protocol",
    )
    _validate_cadence_manifest(manifest, path=path)
    _validate_checkpoint_manifest(manifest, path=path)
    return dict(manifest)


def _validate_cadence_manifest(manifest: Mapping[str, Any], *, path: Path | str) -> None:
    """Fail closed on manifests that can mix native K=50 and TTT K=1.

    A pre-fairness manifest can still have a valid SHA256 and V3 model
    identity, yet its aggregate would report the cadence-mismatched native
    result as a primary baseline.  Requiring the explicit K=1 control and
    comparison scopes here prevents that old envelope from being reused by
    simply editing a results directory.
    """

    raw_methods = manifest.get("methods")
    if not isinstance(raw_methods, Sequence) or isinstance(raw_methods, (str, bytes)):
        raise ValueError(f"{path}: manifest methods must be a sequence")
    methods = {
        str(item.get("id")): item
        for item in raw_methods
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    for method_id in (NATIVE_VARIANT_CHUNK, NATIVE_VARIANT_K1, "clean_ttt", "credit_ttt"):
        if method_id not in methods:
            raise ValueError(
                f"{path}: fairness manifest is missing required method {method_id!r}; "
                "regenerate it with the current coordinator"
            )
    native_k50 = methods[NATIVE_VARIANT_CHUNK]
    native_k1 = methods[NATIVE_VARIANT_K1]
    if native_k50.get("comparison_scope") != "cadence_mismatched_reference":
        raise ValueError(f"{path}: Native-SmolVLA K=50 must be exploratory-only")
    if native_k1.get("comparison_scope") != "matched_cadence":
        raise ValueError(f"{path}: Native-SmolVLA-K1 must be a matched-cadence control")
    if int(native_k50.get("expected_action_chunk_size", -1)) != NATIVE_MODEL_ACTION_HORIZON:
        raise ValueError(f"{path}: Native-SmolVLA K=50 cadence metadata is malformed")
    if int(native_k1.get("expected_action_chunk_size", -1)) != 1:
        raise ValueError(f"{path}: Native-SmolVLA-K1 cadence metadata is malformed")
    controls = manifest.get("fairness_controls")
    if not isinstance(controls, Mapping):
        raise ValueError(f"{path}: manifest is missing fairness_controls")
    if controls.get("primary_comparison_scope") != "matched_cadence_only":
        raise ValueError(f"{path}: primary comparisons must be matched-cadence-only")
    if controls.get("native_k1_control") != NATIVE_VARIANT_K1:
        raise ValueError(f"{path}: fairness manifest must name Native-SmolVLA-K1 control")
    primary = controls.get("primary_baselines")
    if not isinstance(primary, Sequence) or isinstance(primary, (str, bytes)):
        raise ValueError(f"{path}: fairness primary_baselines must be a sequence")
    if NATIVE_VARIANT_CHUNK in primary or NATIVE_VARIANT_K1 not in primary:
        raise ValueError(
            f"{path}: primary_baselines must include K=1 and exclude native K=50"
        )


def _validate_checkpoint_manifest(manifest: Mapping[str, Any], *, path: Path | str) -> None:
    """Validate optional per-task checkpoint provenance in a frozen manifest.

    Older manifests predate ``checkpoints_by_task`` and remain valid; new
    manifests carry both the map and an explicit ``checkpoint_scope`` so a
    reviewer can tell whether one student checkpoint was intentionally shared
    or independently trained for each task.
    """

    raw_maps = manifest.get("checkpoints_by_task")
    raw_scope = manifest.get("checkpoint_scope")
    if raw_maps is None and raw_scope is None:
        return
    if not isinstance(raw_maps, Mapping) or not isinstance(raw_scope, Mapping):
        raise ValueError(f"{path}: checkpoints_by_task and checkpoint_scope must both be objects")
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
        raise ValueError(f"{path}: cannot validate checkpoint map without tasks")
    task_ids = {str(task["id"]) for task in raw_tasks if isinstance(task, Mapping) and "id" in task}
    if not task_ids:
        raise ValueError(f"{path}: manifest has no task IDs for checkpoint map")
    required_methods = {
        str(item["id"])
        for item in manifest.get("methods", [])
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    for method_id in required_methods:
        scope = raw_scope.get(method_id)
        if scope not in {"shared", "per_task"}:
            raise ValueError(
                f"{path}: checkpoint_scope[{method_id!r}] must be 'shared' or 'per_task'"
            )
        mapping = raw_maps.get(method_id)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{path}: checkpoints_by_task[{method_id!r}] must be an object")
        keys = {str(key) for key in mapping}
        if keys != task_ids:
            raise ValueError(
                f"{path}: checkpoint map for {method_id!r} must cover exactly task IDs "
                f"{sorted(task_ids)}, got {sorted(keys)}"
            )
        if any(not isinstance(value, str) or not value.strip() for value in mapping.values()):
            raise ValueError(f"{path}: checkpoint map for {method_id!r} contains an empty path")
        if scope == "shared" and len(set(str(value) for value in mapping.values())) != 1:
            raise ValueError(
                f"{path}: checkpoint_scope={scope!r} for {method_id!r} but paths differ"
            )


def _cmd_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(args)
    _write_json(Path(args.output), manifest)
    _print_manifest_plan(manifest)
    print(f"wrote {args.output}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    if args.output:
        output = Path(args.output)
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        lines.append("# Generated by benchmark_credit_ttt_v3.py; inspect before running.")
        for command in manifest.get("commands", []):
            lines.append(f"# {command['method_id']} / {command['task_id']} / seed {command['train_seed']}")
            lines.append(command["shell"])
            lines.append("")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        output.chmod(0o755)
        print(f"wrote {output}")
    else:
        _print_manifest_plan(manifest)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if not args.execute:
        print("Refusing to launch jobs. Use `run --execute` only after reviewing the generated manifest.")
        return 2
    manifest = _load_manifest(Path(args.manifest))
    unresolved = [
        command
        for command in manifest.get("commands", [])
        if any(token.startswith("<") and token.endswith(">") for token in command["argv"])
    ]
    if unresolved:
        raise ValueError(
            "Manifest contains unresolved checkpoint placeholders; edit the manifest/checkpoint map "
            "before executing"
        )
    for index, command in enumerate(manifest.get("commands", []), start=1):
        print(f"[{index}/{len(manifest['commands'])}] {command['shell']}", flush=True)
        subprocess.run(command["argv"], check=True, cwd=args.cwd)
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    payload = aggregate_results(
        manifest,
        Path(args.results_root),
        n_bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        allow_incomplete=bool(args.allow_incomplete),
    )
    _write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    mechanism = _read_json(Path(args.mechanism_json))
    if not isinstance(mechanism, Mapping):
        raise ValueError("mechanism JSON must be an object")
    mechanism_identity = _canonical_identity_from_artifact(
        mechanism,
        path=Path(args.mechanism_json),
        required=bool(args.strict),
    )
    aggregate = None
    if args.aggregate_json:
        aggregate = _read_json(Path(args.aggregate_json))
        if not isinstance(aggregate, Mapping):
            raise ValueError("aggregate JSON must be an object")
        if aggregate.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise ValueError("aggregate JSON was produced from a different manifest")
    payload = run_go_no_go_checks(
        mechanism,
        aggregate_payload=aggregate,
        strict=bool(args.strict),
        protocol_id=str(manifest["protocol_id"]),
        protocol_version=str(manifest["protocol_version"]),
    )
    payload["manifest_sha256"] = manifest["manifest_sha256"]
    payload["credit_ttt_protocol"] = mechanism_identity
    _write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["overall"] == "GO" else 1


def _cmd_self_check(_: argparse.Namespace) -> int:
    """Run fast, dependency-light checks without touching a dataset or GPU."""

    parser = _build_parser()
    args = parser.parse_args(
        [
            "manifest",
            "--output",
            "/tmp/credit_ttt_v3_self_check_manifest.json",
            "--repo-root",
            os.getcwd(),
            "--python-bin",
            "python",
            "--task-set",
            "legacy_two",
        ]
    )
    manifest = build_manifest(args)
    assert manifest["manifest_sha256"] == sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    assert _validate_canonical_v3_identity(
        manifest["credit_ttt_protocol"], path="self-check manifest"
    ) == CANONICAL_V3_PROTOCOL_IDENTITY
    assert manifest["task_set"] == "legacy_two"
    assert manifest["protocol_id"] == LEGACY_TWO_TASK_PROTOCOL_ID
    assert all(scope == "shared" for scope in manifest["checkpoint_scope"].values())
    _validate_checkpoint_manifest(manifest, path="self-check manifest")
    assert sum(command["method_id"] == NATIVE_VARIANT_CHUNK for command in manifest["commands"]) == 2
    assert sum(command["method_id"] == NATIVE_VARIANT_K1 for command in manifest["commands"]) == 2
    assert all(
        command["train_seed"] == "fixed"
        for command in manifest["commands"]
        if command["method_id"] in {NATIVE_VARIANT_CHUNK, NATIVE_VARIANT_K1}
    )
    native_k1_commands = [
        command
        for command in manifest["commands"]
        if command["method_id"] == NATIVE_VARIANT_K1
    ]
    assert all("--execution-action-steps" in command["argv"] for command in native_k1_commands)
    assert all(
        "--execution-action-steps" not in command["argv"]
        for command in manifest["commands"]
        if command["method_id"] == NATIVE_VARIANT_CHUNK
    )
    assert manifest["fairness_controls"]["primary_comparison_scope"] == "matched_cadence_only"
    assert manifest["fairness_controls"]["native_k1_control"] == NATIVE_VARIANT_K1
    cadence_audit = run_go_no_go_checks(
        {},
        aggregate_payload={
            "fairness": {"native_k1_complete": True, "native_k1_missing_tasks": []},
            "pairwise": {},
            "pairwise_exploratory": {
                "CreditTTT_vs_native_smolvla": {
                    "macro_delta_sr": 1.0,
                    "macro_ci95": [1.0, 1.0],
                    "primary": False,
                }
            },
        },
        strict=False,
    )
    assert not any(
        item["id"] == "benchmark_CreditTTT_vs_native_smolvla"
        for item in cadence_audit["checks"]
    )
    assert any(
        item["id"] == "benchmark_native_k1_cadence_control"
        and item["status"] == "PASS"
        for item in cadence_audit["checks"]
    )
    bins = manifest["mechanism_audits"]["delay_bins_by_task"]
    assert bins["color"] == ["1-16"]
    assert "1025+" not in bins["shuffle_long"]
    seeds = np.arange(4, dtype=np.float64)
    paired = _paired_episode_vectors({"1000": seeds}, {"fixed": seeds[::-1]})
    assert len(paired) == 1 and next(iter(paired.values()))[0].shape == (4,)
    # A native K=1 control must be identified by explicit cadence provenance;
    # a K=50 result in a directory named ``native_smolvla_k1`` is not enough.
    native_common = {
        "benchmark_protocol": OFFICIAL_PROTOCOL,
        "env_id": DEFAULT_TASKS[0]["env_id"],
        "successes": [True],
        "episode_seeds": [DEFAULT_START_SEED],
        "action_chunk_size": 1,
        "model_action_horizon": NATIVE_MODEL_ACTION_HORIZON,
        "execution_action_steps": 1,
        "execution_cadence": NATIVE_CADENCE_RECEDING,
        "model": {
            "method": "SmolVLA",
            "policy_type": "smolvla",
            "ttt_enabled": False,
            "benchmark_variant": NATIVE_VARIANT_K1,
            "model_action_horizon": NATIVE_MODEL_ACTION_HORIZON,
            "execution_action_steps": 1,
            "execution_cadence": NATIVE_CADENCE_RECEDING,
        },
    }
    assert _method_from_metadata(native_common, Path("native_smolvla_k1/eval.json"))[0] == NATIVE_VARIANT_K1
    _validate_eval_record(
        native_common,
        path=Path("native_smolvla_k1/eval.json"),
        expected_method=NATIVE_VARIANT_K1,
        expected_task=DEFAULT_TASKS[0],
        expected_episode_seeds=[DEFAULT_START_SEED],
    )
    malformed_native_k1 = dict(native_common)
    malformed_native_k1["action_chunk_size"] = 50
    malformed_native_k1["model"] = dict(native_common["model"])
    malformed_native_k1["model"]["benchmark_variant"] = NATIVE_VARIANT_CHUNK
    malformed_native_k1["model"]["execution_cadence"] = NATIVE_CADENCE_CHUNK
    malformed_native_k1["execution_cadence"] = NATIVE_CADENCE_CHUNK
    malformed_native_k1["execution_action_steps"] = 50
    malformed_native_k1["model"]["execution_action_steps"] = 50
    try:
        _validate_eval_record(
            malformed_native_k1,
            path=Path("native_smolvla_k1/relabeled_k50.json"),
            expected_method=NATIVE_VARIANT_K1,
            expected_task=DEFAULT_TASKS[0],
            expected_episode_seeds=[DEFAULT_START_SEED],
        )
    except ValueError:
        pass
    else:  # pragma: no cover - defensive assertion for the self-check itself
        raise AssertionError("K=50 native result was accepted as the K=1 control")
    legacy_method, _ = _method_from_metadata(
        {"model": {"method": "HD-TTT", "hd_attribution_protocol": "v2_relative_antithetic_robust"}},
        Path("legacy.json"),
    )
    assert legacy_method == "legacy_rejected"
    # The benchmark envelope's own version must not be sufficient to relabel
    # an otherwise unproven model as CreditTTT.  Only canonical model markers
    # (or the explicit CreditTTT method plus one of them) qualify.
    try:
        envelope_only = {
            "model": {"method": "CreditTTT"},
            "protocol_version": "credit_ttt_v3_baseline_protocol_1",
        }
        _method_from_metadata(envelope_only, Path("envelope_only.json"))
        _validate_eval_record(
            {
                **envelope_only,
                "benchmark_protocol": OFFICIAL_PROTOCOL,
                "env_id": DEFAULT_TASKS[0]["env_id"],
                "successes": [True],
                "episode_seeds": [DEFAULT_START_SEED],
                "action_chunk_size": 1,
            },
            path=Path("envelope_only.json"),
            expected_method="credit_ttt",
            expected_task=DEFAULT_TASKS[0],
            expected_episode_seeds=[DEFAULT_START_SEED],
        )
    except ValueError:
        pass
    else:  # pragma: no cover - defensive assertion for the self-check itself
        raise AssertionError("benchmark envelope version must not authenticate a V3 model")
    canonical_record = {
        "model": {
            "method": "CreditTTT",
            "protocol_id": "credit_ttt_v3",
            "protocol_version": "creditttt_qh2l_v3",
            "credit_ttt_protocol": dict(CANONICAL_V3_PROTOCOL_IDENTITY),
        },
        "benchmark_protocol": OFFICIAL_PROTOCOL,
        "env_id": DEFAULT_TASKS[0]["env_id"],
        "successes": [True],
        "episode_seeds": [DEFAULT_START_SEED],
        "action_chunk_size": 1,
        "model_action_horizon": NATIVE_MODEL_ACTION_HORIZON,
        "execution_action_steps": 1,
        "execution_cadence": NATIVE_CADENCE_RECEDING,
    }
    validated = _validate_eval_record(
        canonical_record,
        path=Path("canonical.json"),
        expected_method="credit_ttt",
        expected_task=DEFAULT_TASKS[0],
        expected_episode_seeds=[DEFAULT_START_SEED],
    )
    assert validated["credit_ttt_protocol"] == CANONICAL_V3_PROTOCOL_IDENTITY
    assert _canonical_identity_from_artifact(
        {"metadata": dict(CANONICAL_V3_PROTOCOL_IDENTITY)},
        path="self-check artifact",
        required=True,
    ) == CANONICAL_V3_PROTOCOL_IDENTITY
    for field, bad_value in (
        ("target", "denoising_velocity"),
        ("causal", 1),
        ("intervention", "content_replacement"),
    ):
        malformed = dict(canonical_record)
        malformed_model = dict(canonical_record["model"])
        malformed_identity = dict(CANONICAL_V3_PROTOCOL_IDENTITY)
        malformed_identity[field] = bad_value
        malformed_model["credit_ttt_protocol"] = malformed_identity
        malformed["model"] = malformed_model
        try:
            _validate_eval_record(
                malformed,
                path=Path(f"malformed_{field}.json"),
                expected_method="credit_ttt",
                expected_task=DEFAULT_TASKS[0],
                expected_episode_seeds=[DEFAULT_START_SEED],
            )
        except ValueError:
            pass
        else:  # pragma: no cover - defensive assertion for the self-check itself
            raise AssertionError(f"malformed canonical field {field!r} was accepted")
    print("CreditTTT V3 benchmark self-check: PASS")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="freeze protocol and emit evaluation commands")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--repo-root", default=os.getcwd())
    manifest.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", "/workspace/MIKASA-Robo/.venv/bin/python"),
        help="Python interpreter used in generated evaluation commands",
    )
    manifest.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    manifest.add_argument(
        "--task-set",
        choices=tuple(sorted(TASK_SET_ALIASES)),
        default=DEFAULT_TASK_SET,
        help=(
            "Task profile: published_four (default; SGT/IM/RC3/RC9) or "
            "legacy_two (historical color/shuffle_long). Aliases are accepted."
        ),
    )
    manifest.add_argument("--native-checkpoint", default="<CHECKPOINT_NATIVE_SMOLVLA>")
    manifest.add_argument("--clean-checkpoint", default="<CHECKPOINT_CLEAN_TTT>")
    manifest.add_argument("--credit-checkpoint", default="<CHECKPOINT_CREDIT_TTT>")
    manifest.add_argument("--utility-checkpoint", default="<CHECKPOINT_UTILITY_KVB>")
    manifest.add_argument(
        "--native-checkpoints-json",
        default=None,
        help="Optional JSON object/file mapping task_id (or env_id) to native checkpoint paths.",
    )
    manifest.add_argument(
        "--clean-checkpoints-json",
        default=None,
        help="Optional JSON object/file mapping task_id (or env_id) to Clean-TTT checkpoints.",
    )
    manifest.add_argument(
        "--credit-checkpoints-json",
        default=None,
        help="Optional JSON object/file mapping task_id (or env_id) to CreditTTT checkpoints.",
    )
    manifest.add_argument(
        "--utility-checkpoints-json",
        default=None,
        help="Optional JSON object/file mapping task_id (or env_id) to Utility-KVB checkpoints.",
    )
    manifest.add_argument("--color-dataset-root", default=None)
    manifest.add_argument("--shuffle-dataset-root", default=None)
    manifest.add_argument("--shell-touch-dataset-root", default=None)
    manifest.add_argument("--intercept-medium-dataset-root", default=None)
    manifest.add_argument("--remember-color3-dataset-root", default=None)
    manifest.add_argument("--remember-color9-dataset-root", default=None)
    manifest.add_argument("--train-seeds", default=",".join(map(str, DEFAULT_TRAIN_SEEDS)))
    manifest.add_argument("--n-episodes", type=int, default=DEFAULT_EPISODES)
    manifest.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    manifest.add_argument("--torch-seed", type=int, default=DEFAULT_TORCH_SEED)
    manifest.add_argument("--sim-backend", choices=("cpu", "gpu"), default="gpu")
    manifest.add_argument("--include-optional", action="store_true")
    manifest.set_defaults(func=_cmd_manifest)

    plan = subparsers.add_parser("plan", help="print or write the frozen shell command plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--output", type=Path, default=None)
    plan.set_defaults(func=_cmd_plan)

    run = subparsers.add_parser("run", help="execute a reviewed plan (opt-in)")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--execute", action="store_true", help="required safety acknowledgement")
    run.add_argument("--cwd", default=None)
    run.set_defaults(func=_cmd_run)

    aggregate = subparsers.add_parser("aggregate", help="aggregate paired per-episode evaluation JSON files")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--results-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap", type=int, default=10_000)
    aggregate.add_argument("--seed", type=int, default=1_729)
    aggregate.add_argument("--allow-incomplete", action="store_true")
    aggregate.set_defaults(func=_cmd_aggregate)

    check = subparsers.add_parser("check", help="run mechanistic go/no-go checks")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--mechanism-json", type=Path, required=True)
    check.add_argument("--aggregate-json", type=Path, default=None)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--strict", action="store_true")
    check.set_defaults(func=_cmd_check)

    self_check = subparsers.add_parser(
        "self-check",
        help="run fast manifest/seed/provenance checks without a dataset or GPU",
    )
    self_check.set_defaults(func=_cmd_self_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
