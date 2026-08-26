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
labels.  ``CreditTTT`` is the proposed method; ``Native-SmolVLA`` and
``Clean-TTT`` are the primary baselines.  A legacy result whose metadata says
``HD-TTT`` (or an old v1/v2 protocol) is rejected when it is supplied as a
CreditTTT result.  This prevents an old checkpoint from silently becoming a
reported V3 result.

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


PROTOCOL_ID = "credit_ttt_v3_mikasa_two_task"
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
DEFAULT_START_SEED = 4_242_424_242
DEFAULT_EPISODES = 50
DEFAULT_TORCH_SEED = 7_000
DEFAULT_TRAIN_SEEDS = (1000, 1001, 1002)
DEFAULT_RESULTS_ROOT = "benchmark_results/credit_ttt_v3"

DEFAULT_TASKS: tuple[dict[str, Any], ...] = (
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

# These names are intentionally stable and human-readable in tables.  The
# optional Utility-KVB entry is a mechanism baseline, not a version of our
# method; it is omitted from the primary comparison unless result files exist.
METHODS: tuple[dict[str, Any], ...] = (
    {
        "id": "native_smolvla",
        "label": "Native-SmolVLA",
        "role": "primary_baseline",
        "evaluator": "examples/mikasa/evaluate_smolvla_baseline.py",
        "expected_action_chunk_size": 50,
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


def _task_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = [dict(task) for task in DEFAULT_TASKS]
    if getattr(args, "color_dataset_root", None):
        tasks[0]["dataset_root"] = str(Path(args.color_dataset_root))
    if getattr(args, "shuffle_dataset_root", None):
        tasks[1]["dataset_root"] = str(Path(args.shuffle_dataset_root))
    return tasks


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
    return command


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.n_episodes) <= 0:
        raise ValueError("--n-episodes must be positive")
    if int(args.start_seed) < 0 or int(args.torch_seed) < 0:
        raise ValueError("seeds must be non-negative")
    train_seeds = _parse_int_list(args.train_seeds)
    tasks = _task_specs(args)
    methods = _method_specs(args.include_optional)
    checkpoint_map = {
        "native_smolvla": str(args.native_checkpoint),
        "clean_ttt": str(args.clean_checkpoint),
        "credit_ttt": str(args.credit_checkpoint),
    }
    if args.include_optional:
        checkpoint_map["utility_kvb"] = str(args.utility_checkpoint)

    manifest: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
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
                "train_and_label": "episodes [0, 199]",
                "offline_validation": "episodes [200, 249]",
                "test": "simulator seeds only; no test-seed tuning",
            },
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
            "native_chunk_control_required": True,
            "native_chunk_control_note": (
                "Native K=50 is the canonical baseline. Add a K=1 receding-horizon "
                "native control before attributing gains to memory rather than observation cadence."
            ),
            "persistent_reset_pair": True,
            "fixed_episode_seeds_across_methods": True,
        },
        "mechanism_audits": {
            # Delay bins are frozen per task.  Color episodes are only a few
            # dozen frames long, so reporting 65+ bins for that task would be
            # a fabricated extrapolation.  Shuffle-Long is the only task in
            # this manifest with enough frames for the longer bins.
            "delay_bins_by_task": {
                "color": ["1-16"],
                "shuffle_long": ["1-16", "17-64", "65-256", "257-1024"],
            },
            # Union retained for consumers that expect one flat list.  The
            # task-specific map above is authoritative for tables/plots.
            "delay_bins": ["1-16", "17-64", "65-256", "257-1024"],
            "pairwise_teacher_effect": {
                "metric": "pairwise_high_utility_delta_a_cosine",
                "group_by": ["task_id", "delay_bin"],
                "required_bins": {
                    "color": ["1-16"],
                    "shuffle_long": ["1-16", "17-64", "65-256", "257-1024"],
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
        checkpoint = checkpoint_map[method_id]
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
                        "task_id": str(task["id"]),
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
    if normalized in {"smolvla", "native-smolvla"}:
        return "native_smolvla", protocol_text
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
    if expected_method == "native_smolvla" and int(chunk) != 50:
        raise ValueError(f"{path}: Native-SmolVLA must report canonical action_chunk_size=50, got {chunk}")
    if expected_method in {"clean_ttt", "credit_ttt", "utility_kvb"} and int(chunk) != 1:
        raise ValueError(f"{path}: TTT methods must report action_chunk_size=1, got {chunk}")
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

    primary_baselines = ["native_smolvla", "clean_ttt", "utility_kvb"]
    pairwise: dict[str, Any] = {}
    ours_runs = _clusters_for_method_task  # keep the local name readable below
    for baseline in primary_baselines:
        if "credit_ttt" not in method_task or baseline not in method_task:
            continue
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
        if per_task:
            macro, low, high, count = _macro_paired_ci(
                paired_by_task,
                n_bootstrap=n_bootstrap,
                seed=seed + _stable_seed("credit_ttt", baseline, "macro"),
            )
            pairwise[f"CreditTTT_vs_{baseline}"] = {
                "per_task": per_task,
                "macro_delta_sr": macro,
                "macro_ci95": [low, high],
                "macro_n_paired_episodes": count,
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
    # cannot substitute for proving the causal mechanism.
    if aggregate_payload is not None:
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
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "strict": bool(strict),
        "overall": overall,
        "checks": checks,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    _verify_manifest(manifest)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Unsupported protocol_id={manifest.get('protocol_id')!r}")
    _validate_canonical_v3_identity(
        manifest.get("credit_ttt_protocol"),
        path=f"{path}:credit_ttt_protocol",
    )
    return dict(manifest)


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
    payload = run_go_no_go_checks(mechanism, aggregate_payload=aggregate, strict=bool(args.strict))
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
        ]
    )
    manifest = build_manifest(args)
    assert manifest["manifest_sha256"] == sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    assert _validate_canonical_v3_identity(
        manifest["credit_ttt_protocol"], path="self-check manifest"
    ) == CANONICAL_V3_PROTOCOL_IDENTITY
    assert sum(command["method_id"] == "native_smolvla" for command in manifest["commands"]) == 2
    assert all(
        command["train_seed"] == "fixed"
        for command in manifest["commands"]
        if command["method_id"] == "native_smolvla"
    )
    bins = manifest["mechanism_audits"]["delay_bins_by_task"]
    assert bins["color"] == ["1-16"]
    assert "1025+" not in bins["shuffle_long"]
    seeds = np.arange(4, dtype=np.float64)
    paired = _paired_episode_vectors({"1000": seeds}, {"fixed": seeds[::-1]})
    assert len(paired) == 1 and next(iter(paired.values()))[0].shape == (4,)
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
    manifest.add_argument("--native-checkpoint", default="<CHECKPOINT_NATIVE_SMOLVLA>")
    manifest.add_argument("--clean-checkpoint", default="<CHECKPOINT_CLEAN_TTT>")
    manifest.add_argument("--credit-checkpoint", default="<CHECKPOINT_CREDIT_TTT>")
    manifest.add_argument("--utility-checkpoint", default="<CHECKPOINT_UTILITY_KVB>")
    manifest.add_argument("--color-dataset-root", default=None)
    manifest.add_argument("--shuffle-dataset-root", default=None)
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
