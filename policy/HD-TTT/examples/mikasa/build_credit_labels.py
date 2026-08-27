#!/usr/bin/env python
"""Build the frame-aligned CreditTTT V3 pair-label artifact.

This builder emits only the CreditTTT V3 contract:

* an explicit causal ``FullHistoryActionTeacher`` supplies the full and
  event-write-deleted action predictions;
* ``compute_pairwise_control_credit`` computes one utility/effect for every
  event--future pair; and
* a per-event delay/stratum sampler selects a deterministic, fixed-size set of
  pairs for each source frame.

The input is the feature cache written by ``train_full_history_teacher.py``.
It contains one observation event token and one *previous executed slot-0*
action per physical frame.  A cache may additionally contain
``future_query_q`` and ``future_action_tail_h`` columns (both ``[T,D]``); when
present they are gathered into the artifact for the online QH2L reader.  The
builder never invents those tensors: ``--require-query-features`` can be used
for the paper run to fail fast when the student query cache is absent.

The output is a flat, frame-aligned tensor mapping.  Each required
``hd_v3_pair_*`` column has shape ``[N,K]`` (the effect column is
``[N,K,A]``), where ``N`` is the concatenated number of physical frames and
``K`` is the fixed number of sampled future queries per event.  Episode-local
index semantics and concatenation slices are recorded in metadata.

The teacher head is trained against the normalized executed slot-0 action, so
the target is recorded as ``normalized_executed_slot0_action``.  It is not a
velocity target and must not be relabelled as one.  The canonical intervention
is an event-write deletion, which is exactly the before/after fast-state
intervention consumed by the student.  A future flow-integrated adapter or a
donor-content replacement must be identified as a separate protocol variant,
not mixed into the canonical labels.

Example::

    PYTHONPATH=src python examples/mikasa/build_credit_labels.py \
      --features-input /workspace/credit_ttt/features.pt \
      --teacher-checkpoint /workspace/credit_ttt/teacher.pt \
      --output /workspace/credit_ttt/credit_pairs.pt \
      --pair-k 5 --intervention delete --seed 1000 \
      --require-query-features

The command only reads feature/checkpoint files and writes ``--output``.  It
does not modify either protected policy directory or the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.smolvla_ttt.credit_ttt_v3 import (
    CREDIT_TTT_DELAY_EDGES,
    CREDIT_TTT_V3_FORMAT,
    CREDIT_TTT_V3_INTERVENTION,
    CREDIT_TTT_V3_INTERVENTION_SCOPE,
    CREDIT_TTT_V3_PAIR_SCHEMA,
    CREDIT_TTT_V3_PROTOCOL,
    CREDIT_TTT_V3_STATE,
    CREDIT_TTT_V3_TARGET,
    DEFAULT_CREDIT_TTT_PROTOCOL,
)
from lerobot.policies.smolvla_ttt.history_teacher import (
    FULL_HISTORY_TEACHER_FORMAT,
    FullHistoryActionTeacher,
    PairwiseControlCredit,
    compute_pairwise_control_credit,
    history_teacher_state_sha256,
    load_full_history_teacher_checkpoint,
)


LOGGER = logging.getLogger("credit_ttt.build_labels")
FEATURE_FORMAT = "credit_ttt_v3_prefix_features_v1"
# The on-disk artifact deliberately uses the protocol identity as its top
# level format.  ``lerobot_train`` validates this exact value before attaching
# columns to a dataset.  The more descriptive implementation revision is kept
# separately in ``artifact_revision`` below.
LABEL_FORMAT = CREDIT_TTT_V3_FORMAT
ARTIFACT_REVISION = "credit_ttt_v3_pair_labels_v1"
QUERY_Q_KEYS = ("future_query_q", "query_q", "future_query_q_j")
QUERY_H_KEYS = ("future_action_tail_h", "action_tail_h", "future_action_tail_feature_h_j")
# Keep the publication bins invariant across shards/episodes.  In particular,
# do not replace the final ``1025+`` boundary with ``max_episode_length+1``:
# doing so makes delay-bin IDs incomparable between artifacts.
DEFAULT_DELAY_EDGES = CREDIT_TTT_DELAY_EDGES
# Utility is a normalized relative degradation in [0, 1-ish] after the
# confidence factor.  Keep the positive/null split tied to the student
# protocol's default (``hd_v3_null_threshold``) instead of inheriting the
# generic sampler's zero threshold.  A zero threshold would turn arbitrarily
# tiny numerical effects into positive writer targets while the online path
# treats those same pairs as null/invariance examples.
DEFAULT_POSITIVE_THRESHOLD = 0.05
# Counterfactual teacher replay is an execution-only optimisation.  The
# canonical label schema and all sampling seeds remain unchanged when this
# value is changed.  A value of zero is reserved for the original one-event
# loop and is useful as a bitwise/reference implementation in audits.
DEFAULT_COUNTERFACTUAL_BATCH_SIZE = 64
COUNTERFACTUAL_BATCH_ENV = "CREDIT_TTT_TEACHER_REPLAY_BATCH_SIZE"


def _load_torch(path: str | Path) -> Any:
    """Load a tensor-only artifact with a safe PyTorch path."""

    path = Path(path).expanduser()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - PyTorch < 2.0 compatibility
        return torch.load(path, map_location="cpu")


def _as_float_tensor(value: Any, *, name: str) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if not tensor.is_floating_point():
        tensor = tensor.float()
    tensor = tensor.detach().cpu().float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _as_index_tensor(value: Any, *, name: str) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.ndim == 0:
        raise ValueError(f"{name} must contain one value per frame")
    return tensor.detach().cpu().to(dtype=torch.int64)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _load_feature_cache(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and validate ``train_full_history_teacher`` feature episodes."""

    payload = _load_torch(path)
    if not isinstance(payload, Mapping) or payload.get("format") != FEATURE_FORMAT:
        raise ValueError(
            f"Unsupported feature cache {path!s}; expected format {FEATURE_FORMAT!r}"
        )
    metadata = payload.get("metadata")
    raw_episodes = payload.get("episodes")
    if not isinstance(metadata, Mapping) or not isinstance(raw_episodes, Sequence):
        raise ValueError("Feature cache needs mapping metadata and an episodes sequence")
    episodes: list[dict[str, Any]] = []
    required = {
        "event_tokens",
        "previous_executed_actions",
        "target_actions",
        "global_indices",
        "episode_indices",
        "frame_indices",
    }
    for episode_index, raw in enumerate(raw_episodes):
        if not isinstance(raw, Mapping) or not required.issubset(raw):
            raise ValueError(f"Feature episode {episode_index} is missing required columns")
        row: dict[str, Any] = {}
        row["event_tokens"] = _as_float_tensor(raw["event_tokens"], name="event_tokens")
        row["previous_executed_actions"] = _as_float_tensor(
            raw["previous_executed_actions"], name="previous_executed_actions"
        )
        row["target_actions"] = _as_float_tensor(raw["target_actions"], name="target_actions")
        row["global_indices"] = _as_index_tensor(raw["global_indices"], name="global_indices")
        row["episode_indices"] = _as_index_tensor(raw["episode_indices"], name="episode_indices")
        row["frame_indices"] = _as_index_tensor(raw["frame_indices"], name="frame_indices")
        length = int(row["event_tokens"].shape[0])
        if row["event_tokens"].ndim != 2:
            raise ValueError(f"Feature episode {episode_index} event_tokens must be [T,D]")
        if row["previous_executed_actions"].ndim != 2 or row["target_actions"].ndim != 2:
            raise ValueError(
                f"Feature episode {episode_index} actions must be [T,A]"
            )
        for key in ("previous_executed_actions", "target_actions", "global_indices", "episode_indices", "frame_indices"):
            if row[key].shape[0] != length:
                raise ValueError(
                    f"Feature episode {episode_index} column {key!r} has length "
                    f"{row[key].shape[0]}, expected {length}"
                )
        if row["previous_executed_actions"].shape[-1] != row["target_actions"].shape[-1]:
            raise ValueError(
                f"Feature episode {episode_index} previous/target action widths disagree"
            )
        # Event spans are optional in the cache.  A missing span means one
        # physical interaction per frame, which is the canonical teacher cache.
        for key in ("event_starts", "event_ends", "event_start", "event_end"):
            if key in raw:
                value = _as_index_tensor(raw[key], name=key)
                if value.numel() not in {1, length}:
                    raise ValueError(
                        f"Feature episode {episode_index} {key!r} must be scalar or [T]"
                    )
                row[key] = value.reshape(-1)
        for key_group, keys in (("future_query_q", QUERY_Q_KEYS), ("future_action_tail_h", QUERY_H_KEYS)):
            value = _first_present(raw, keys)
            if value is not None:
                query = _as_float_tensor(value, name=key_group)
                if query.ndim != 2 or query.shape[0] != length:
                    raise ValueError(
                        f"Feature episode {episode_index} {key_group} must have [T,D] shape"
                    )
                row[key_group] = query
        # Replacement tokens can be supplied by a phase-matched donor cache.
        replacement = _first_present(
            raw, ("replacement_event_tokens", "donor_event_tokens", "event_replacements")
        )
        if replacement is not None:
            replacement = _as_float_tensor(replacement, name="replacement_event_tokens")
            if replacement.shape != row["event_tokens"].shape:
                raise ValueError(
                    f"Feature episode {episode_index} replacement_event_tokens must match [T,D]"
                )
            row["replacement_event_tokens"] = replacement
        episodes.append(row)
    if not episodes:
        raise ValueError("Feature cache contains no episodes")
    return dict(metadata), episodes


def _event_spans(row: Mapping[str, Any], length: int, block_size: int) -> tuple[Tensor, Tensor]:
    """Resolve half-open event spans while enforcing causal ordering."""

    starts_value = row.get("event_starts", row.get("event_start"))
    ends_value = row.get("event_ends", row.get("event_end"))
    if starts_value is None and ends_value is None:
        starts = torch.arange(length, dtype=torch.int64)
        ends = (starts + int(block_size)).clamp_max(length)
    elif starts_value is None or ends_value is None:
        raise ValueError("event_starts and event_ends must be supplied together")
    else:
        starts = torch.as_tensor(starts_value, dtype=torch.int64).reshape(-1)
        ends = torch.as_tensor(ends_value, dtype=torch.int64).reshape(-1)
        if starts.numel() == 1:
            starts = starts.expand(length)
        if ends.numel() == 1:
            ends = ends.expand(length)
        if starts.numel() != length or ends.numel() != length:
            raise ValueError("event span columns must have one value per frame")
    if bool((starts < 0).any()) or bool((ends <= starts).any()) or bool((ends > length).any()):
        raise ValueError("event spans must satisfy 0 <= start < end <= episode length")
    if length > 1 and not bool((starts[1:] >= starts[:-1]).all()):
        raise ValueError("event_starts must be non-decreasing")
    return starts, ends


def _fixed_delay_edges(max_delay: int, requested: Sequence[int]) -> Tensor:
    """Return invariant, inclusive-left delay-bin edges.

    ``max_delay`` is checked only to catch malformed/overflowing episodes.  It
    must *not* determine the final edge: doing that would silently change the
    meaning of a bin when two shards have different episode lengths.  The
    default protocol therefore always ends in the fixed ``1025+`` sentinel;
    custom edge lists receive the same large sentinel when they omit one.
    """

    if int(max_delay) < 0:
        raise ValueError("max_delay must be non-negative")
    sentinel = int(CREDIT_TTT_DELAY_EDGES[-1])
    if int(max_delay) >= sentinel:
        raise ValueError(
            f"max_delay={max_delay} exceeds the protocol delay sentinel {sentinel}"
        )
    values = sorted({int(edge) for edge in requested})
    if not values or values[0] != 1:
        raise ValueError("delay edges must start at exactly 1")
    if any(edge <= 0 for edge in values):
        raise ValueError("delay edges must be positive integers")
    if values[-1] > sentinel:
        raise ValueError(
            f"delay edge {values[-1]} exceeds the protocol sentinel {sentinel}"
        )
    if values[-1] != sentinel:
        values.append(sentinel)
    if len(values) < 2:
        raise ValueError("delay edges must contain at least one finite bin and a final sentinel")
    return torch.tensor(values, dtype=torch.int64)


def _file_sha256(path: str | Path) -> str:
    """Hash an input artifact by bytes for strict provenance checking."""

    digest = hashlib.sha256()
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"feature artifact does not exist: {source}")
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_teacher_file(path: str | Path) -> Path:
    """Resolve the single checkpoint file accepted by the teacher loader."""

    checkpoint = Path(path).expanduser()
    if not checkpoint.is_dir():
        return checkpoint
    candidates = [
        checkpoint / name
        for name in ("teacher.pt", "checkpoint.pt", "full_history_teacher.pt")
        if (checkpoint / name).is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Teacher checkpoint directory must contain exactly one of "
            "teacher.pt/checkpoint.pt/full_history_teacher.pt"
        )
    return candidates[0]


def _episode_seed(seed: int, episode_number: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{int(episode_number)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _resolve_counterfactual_batch_size(value: int | None) -> int:
    """Resolve the execution-only teacher replay batch size.

    The pair-label protocol does not depend on this value: it only changes
    how many independent event interventions are evaluated in one teacher
    call.  Keeping the legacy ``0`` escape hatch makes it possible to compare
    the vectorized path against the historical one-event loop.  An environment
    override is intentionally read only when the API/CLI argument is omitted,
    so an explicit command-line value always wins.
    """

    if value is None:
        raw = os.environ.get(COUNTERFACTUAL_BATCH_ENV)
        if raw is None or not raw.strip():
            return DEFAULT_COUNTERFACTUAL_BATCH_SIZE
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{COUNTERFACTUAL_BATCH_ENV} must be an integer >= 0, got {raw!r}"
            ) from exc
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"counterfactual_batch_size must be an integer >= 0, got {value!r}"
        ) from exc
    if resolved < 0:
        raise ValueError(
            f"counterfactual_batch_size must be >= 0 (0 selects the legacy loop), got {resolved}"
        )
    return resolved


def _counterfactual_teacher_replays(
    teacher: FullHistoryActionTeacher,
    events: Tensor,
    previous: Tensor,
    starts: Tensor,
    ends: Tensor,
    *,
    intervention: str,
    replacement: Tensor | None,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Evaluate one event-write intervention per row, in deterministic order.

    The historical implementation invoked the causal teacher once for every
    event.  For an episode of length ``T`` that launches ``T`` separate
    ``T``-step recurrences.  This helper groups consecutive interventions into
    independent batch rows.  ``FullHistoryActionTeacher`` already accepts
    ``[B,T,D]`` events and ``[B,T]`` masks, so the recurrence and intervention
    semantics are unchanged; only the batch dimension is larger.  The output
    row ``r`` is still the counterfactual for event ``r``.

    ``batch_size=0`` deliberately executes the old loop and is retained for
    regression tests/reference audits.  A positive size is an execution-only
    memory/throughput bound; it does not alter pair sampling, delay bins, or
    any artifact metadata.
    """

    if intervention not in {"delete", "replace"}:
        raise ValueError(f"unsupported intervention {intervention!r}")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if events.ndim != 2 or previous.ndim != 2:
        raise ValueError("events and previous must have [T,D]/[T,A] shape")
    length = int(events.shape[0])
    if length <= 0:
        raise ValueError("events must contain at least one timestep")
    if previous.shape[0] != length:
        raise ValueError("events and previous must have the same sequence length")
    if starts.numel() != length or ends.numel() != length:
        raise ValueError("event spans must contain one start/end per timestep")
    if intervention == "replace" and replacement is None:
        raise ValueError("intervention='replace' requires replacement event tokens")

    # ``events``/``previous`` are already detached CPU feature rows when this
    # helper is called.  Move once per episode, not once per event.
    events_device = events.to(device=device)
    previous_device = previous.to(device=device)
    starts_device = starts.to(device=device, dtype=torch.long)
    ends_device = ends.to(device=device, dtype=torch.long)
    replacement_device = None if replacement is None else replacement.to(device=device)

    # Keep the exact historical path available for bitwise comparison.  The
    # old loop used rank-1 masks for a single episode; preserving that spelling
    # avoids introducing a shape-dependent numerical change in the reference
    # branch.
    if batch_size == 0:
        counterfactual: list[Tensor] = []
        for start, end in zip(starts_device.tolist(), ends_device.tolist(), strict=True):
            if intervention == "delete":
                mask = torch.zeros(length, dtype=torch.float32, device=device)
                mask[int(start) : int(end)] = 1.0
                branch = teacher(events_device, previous_device, delete_mask=mask)
            else:
                assert replacement_device is not None
                mask = torch.zeros(length, dtype=torch.bool, device=device)
                mask[int(start) : int(end)] = True
                branch = teacher(
                    events_device,
                    previous_device,
                    replacement_event_tokens=replacement_device,
                    replace_mask=mask,
                )
            counterfactual.append(branch.actions)
        if not counterfactual:  # defensive; length is checked above
            raise ValueError("episode contains no events")
        return torch.stack(counterfactual, dim=0)

    # ``batch_size`` bounds the temporary [C,T,D] event view and the teacher's
    # [C,T,H] recurrent state.  ``expand`` avoids copying the common episode
    # rows; every intervention still receives its own independent recurrent
    # state inside the teacher call.
    chunk_size = min(int(batch_size), length)
    chunks: list[Tensor] = []
    time_index = torch.arange(length, device=device, dtype=torch.long)
    for chunk_start in range(0, length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, length)
        chunk_events = events_device.unsqueeze(0).expand(chunk_end - chunk_start, -1, -1)
        chunk_previous = previous_device.unsqueeze(0).expand(chunk_end - chunk_start, -1, -1)
        chunk_starts = starts_device[chunk_start:chunk_end]
        chunk_ends = ends_device[chunk_start:chunk_end]
        # A row-wise interval mask is equivalent to the old per-event slice,
        # including custom multi-frame replacement/delete ablations.
        interval_mask = (time_index[None, :] >= chunk_starts[:, None]) & (
            time_index[None, :] < chunk_ends[:, None]
        )
        if intervention == "delete":
            branch = teacher(
                chunk_events,
                chunk_previous,
                delete_mask=interval_mask.to(dtype=torch.float32),
            )
        else:
            assert replacement_device is not None
            chunk_replacement = replacement_device.unsqueeze(0).expand(
                chunk_end - chunk_start, -1, -1
            )
            branch = teacher(
                chunk_events,
                chunk_previous,
                replacement_event_tokens=chunk_replacement,
                replace_mask=interval_mask,
            )
        actions = branch.actions
        if actions.ndim != 3 or actions.shape[0] != chunk_end - chunk_start:
            raise ValueError(
                "Batched teacher replay returned unexpected action shape "
                f"{tuple(actions.shape)}"
            )
        chunks.append(actions)
    return torch.cat(chunks, dim=0)


def _teacher_replays(
    teacher: FullHistoryActionTeacher,
    events: Tensor,
    previous: Tensor,
    target_actions: Tensor,
    starts: Tensor,
    ends: Tensor,
    *,
    intervention: str,
    replacement: Tensor | None,
    device: torch.device,
    counterfactual_batch_size: int = DEFAULT_COUNTERFACTUAL_BATCH_SIZE,
) -> tuple[Tensor, Tensor, PairwiseControlCredit]:
    """Run full and one event-write intervention per event."""

    events_device = events.to(device=device)
    previous_device = previous.to(device=device)
    target_device = target_actions.to(device=device)
    resolved_batch_size = _resolve_counterfactual_batch_size(counterfactual_batch_size)
    with torch.no_grad():
        full_actions = teacher(events_device, previous_device).actions
        # Group independent event interventions into batched teacher calls.
        # ``counterfactual_batch_size=0`` retains the exact historical loop
        # for regression/audit runs; positive values only change execution
        # parallelism and preserve event row order.
        cf_actions = _counterfactual_teacher_replays(
            teacher,
            events,
            previous,
            starts,
            ends,
            intervention=intervention,
            replacement=replacement,
            device=device,
            batch_size=resolved_batch_size,
        )
        credit = compute_pairwise_control_credit(
            full_actions,
            cf_actions,
            target_device,
            event_ends=ends.to(device=device),
        )
    return full_actions, cf_actions, credit


def _sample_frame_pairs(
    credit: PairwiseControlCredit,
    *,
    starts: Tensor,
    ends: Tensor,
    pair_k: int,
    delay_edges: Tensor,
    seed: int,
    positive_threshold: float,
    full_actions: Tensor | None = None,
    counterfactual_actions: Tensor | None = None,
    expert_actions: Tensor | None = None,
) -> dict[str, Tensor]:
    """Sample and scatter pairs into ``[T,K]`` frame-aligned columns.

    Sampling is performed *per event*.  A global delay-balanced draw followed
    by scattering can leave almost every event without a future query (the
    selected rows happen to concentrate on a few early events), which defeats
    the local-writer supervision.  Here each event receives up to ``K`` pairs,
    balanced first across the fixed delay bins and then across positive/null
    strata.  Late events with fewer than ``K`` causal futures retain all
    available pairs and explicit padding; they are never duplicated merely to
    satisfy a tensor shape.
    """

    # ``compute_pairwise_control_credit`` returns [1,I,J] for an unbatched
    # episode.  The sampler consumes [I,J] and enforces the same causal mask;
    # passing the explicit mask preserves event blocks longer than one frame.
    # Sampling/serialization are CPU-side and deterministic.  Moving only
    # the detached label tensors here avoids an unnecessary CUDA↔CPU copy per
    # selected scalar while leaving the teacher replay device configurable.
    utility = credit.utility[0].detach().cpu()
    pair_mask = credit.pair_mask[0].detach().cpu()
    credit_confidence = credit.confidence[0].detach().cpu()
    credit_effect = credit.action_effect[0].detach().cpu()
    length = int(starts.numel())
    action_suffix = tuple(credit_effect.shape[2:])
    effect_shape = (length, pair_k, *action_suffix)
    # These names are part of the public V3 data contract.  Prefixing every
    # training column with ``hd_v3_pair_`` prevents the generic HD loader from
    # confusing a pair artifact with legacy per-frame velocity labels.
    out: dict[str, Tensor] = {
        "hd_v3_pair_event_index": torch.full((length, pair_k), -1, dtype=torch.int64),
        "hd_v3_pair_future_index": torch.full((length, pair_k), -1, dtype=torch.int64),
        "hd_v3_pair_event_start": torch.full((length, pair_k), -1, dtype=torch.int64),
        "hd_v3_pair_event_end": torch.full((length, pair_k), -1, dtype=torch.int64),
        "hd_v3_pair_delay": torch.full((length, pair_k), -1, dtype=torch.int64),
        "hd_v3_pair_delay_bin": torch.full((length, pair_k), -1, dtype=torch.int64),
        "hd_v3_pair_utility": torch.zeros((length, pair_k), dtype=utility.dtype),
        "hd_v3_pair_confidence": torch.zeros((length, pair_k), dtype=credit_confidence.dtype),
        "hd_v3_pair_valid": torch.zeros((length, pair_k), dtype=torch.bool),
        "hd_v3_pair_positive": torch.zeros((length, pair_k), dtype=torch.bool),
        "hd_v3_pair_effect": torch.zeros(effect_shape, dtype=credit_effect.dtype),
    }

    def _normalise_action_replay(
        value: Tensor | None,
        *,
        name: str,
        event_axis: bool,
    ) -> Tensor | None:
        if value is None:
            return None
        tensor = value.detach().cpu().float()
        if not event_axis and tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if event_axis and tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        expected_prefix = (length, length) if event_axis else (length,)
        if tensor.shape[: len(expected_prefix)] != expected_prefix:
            raise ValueError(
                f"{name} must start with {expected_prefix}, got {tuple(tensor.shape)}"
            )
        if tensor.shape[-1] != action_suffix[-1]:
            raise ValueError(
                f"{name} action dim {tensor.shape[-1]} disagrees with effect dim {action_suffix[-1]}"
            )
        return tensor

    full_replay = _normalise_action_replay(
        full_actions, name="full_actions", event_axis=False
    )
    counterfactual_replay = _normalise_action_replay(
        counterfactual_actions,
        name="counterfactual_actions",
        event_axis=True,
    )
    expert_replay = _normalise_action_replay(
        expert_actions, name="expert_actions", event_axis=False
    )
    if any(value is not None for value in (full_replay, counterfactual_replay, expert_replay)):
        if any(value is None for value in (full_replay, counterfactual_replay, expert_replay)):
            raise ValueError(
                "full_actions, counterfactual_actions, and expert_actions must be supplied together"
            )
        out["hd_v3_pair_teacher_full_action"] = torch.zeros(effect_shape, dtype=torch.float32)
        out["hd_v3_pair_teacher_counterfactual_action"] = torch.zeros(
            effect_shape, dtype=torch.float32
        )
        out["hd_v3_pair_expert_action"] = torch.zeros(effect_shape, dtype=torch.float32)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def _balanced_take(
        candidate_futures: Tensor,
        candidate_bins: Tensor,
        count: int,
    ) -> list[int]:
        """Round-robin one stratum across available delay bins."""

        if count <= 0 or candidate_futures.numel() == 0:
            return []
        queues: dict[int, list[int]] = {}
        for bin_id in sorted(set(int(value) for value in candidate_bins.tolist())):
            rows = torch.nonzero(candidate_bins == bin_id, as_tuple=False).flatten()
            order = torch.randperm(rows.numel(), generator=generator)
            queues[bin_id] = candidate_futures.index_select(0, rows.index_select(0, order)).tolist()
        chosen: list[int] = []
        while len(chosen) < count:
            progressed = False
            for bin_id in sorted(queues):
                if queues[bin_id] and len(chosen) < count:
                    chosen.append(int(queues[bin_id].pop()))
                    progressed = True
            if not progressed:
                break
        return chosen

    for event in range(length):
        futures = torch.nonzero(pair_mask[event], as_tuple=False).flatten()
        if futures.numel() == 0:
            continue
        # Delay is measured from the first frame of the intervened event.  For
        # the canonical one-frame protocol this is simply ``future-event``;
        # using ``starts[event]`` keeps the artifact mathematically correct for
        # explicitly named multi-frame offline ablations as well.
        delays = futures - starts[event]
        bins = torch.bucketize(delays, delay_edges[1:-1], right=False)
        values = utility[event].index_select(0, futures)
        is_positive = values > float(positive_threshold)
        positive_quota = (pair_k + 1) // 2
        null_quota = pair_k - positive_quota
        positive_futures = futures[is_positive]
        positive_bins = bins[is_positive]
        null_futures = futures[~is_positive]
        null_bins = bins[~is_positive]
        chosen = _balanced_take(positive_futures, positive_bins, positive_quota)
        chosen.extend(_balanced_take(null_futures, null_bins, null_quota))
        # If one stratum is absent, back-fill without replacement from the
        # remaining futures while retaining the same round-robin delay rule.
        if len(chosen) < min(pair_k, int(futures.numel())):
            chosen_set = set(chosen)
            remaining_mask = torch.tensor(
                [int(value) not in chosen_set for value in futures.tolist()],
                dtype=torch.bool,
            )
            chosen.extend(
                _balanced_take(
                    futures[remaining_mask],
                    bins[remaining_mask],
                    min(pair_k, int(futures.numel())) - len(chosen),
                )
            )

        for slot, future in enumerate(chosen[:pair_k]):
            # The sampler's delay is based on the event start.  For a one-frame
            # event this is the paper's ``j-i`` convention; for a named
            # multi-frame ablation it remains a true physical-frame delay.
            # Recompute defensively in case a custom sampler returns stale data.
            delay = future - starts[event]
            out["hd_v3_pair_event_index"][event, slot] = event
            out["hd_v3_pair_future_index"][event, slot] = future
            out["hd_v3_pair_event_start"][event, slot] = starts[event]
            out["hd_v3_pair_event_end"][event, slot] = ends[event]
            out["hd_v3_pair_delay"][event, slot] = delay
            out["hd_v3_pair_delay_bin"][event, slot] = int(
                torch.bucketize(torch.tensor(delay), delay_edges[1:-1], right=False).item()
            )
            out["hd_v3_pair_utility"][event, slot] = utility[event, future]
            out["hd_v3_pair_confidence"][event, slot] = credit_confidence[event, future]
            out["hd_v3_pair_valid"][event, slot] = True
            out["hd_v3_pair_positive"][event, slot] = bool(
                utility[event, future] > float(positive_threshold)
            )
            out["hd_v3_pair_effect"][event, slot] = credit_effect[event, future]
            if full_replay is not None:
                assert counterfactual_replay is not None and expert_replay is not None
                out["hd_v3_pair_teacher_full_action"][event, slot] = full_replay[future]
                out["hd_v3_pair_teacher_counterfactual_action"][event, slot] = (
                    counterfactual_replay[event, future]
                )
                out["hd_v3_pair_expert_action"][event, slot] = expert_replay[future]
    out["hd_v3_pair_null"] = out["hd_v3_pair_valid"] & ~out["hd_v3_pair_positive"]
    return out


def _gather_query_features(
    pair_fields: dict[str, Tensor],
    row: Mapping[str, Any],
    *,
    require: bool,
) -> None:
    """Gather optional future ``q_j``/``h_j`` into frame-aligned pair fields.

    These fields are diagnostic/cache columns for a query-replay backend.  The
    current SmolVLA student captures its exact query projection online, so the
    QH2L loss does not trust a detached cached query as a substitute.
    """

    q = row.get("future_query_q")
    h = row.get("future_action_tail_h")
    if (q is None or h is None) and require:
        raise ValueError(
            "--require-query-features requested but feature cache lacks both "
            "future_query_q and future_action_tail_h"
        )
    if q is None or h is None:
        return
    valid = pair_fields["hd_v3_pair_valid"]
    future = pair_fields["hd_v3_pair_future_index"].clamp_min(0)
    pair_fields["hd_v3_pair_query"] = q[future]
    pair_fields["hd_v3_pair_action_tail"] = h[future]
    pair_fields["hd_v3_pair_query"] = torch.where(
        valid.unsqueeze(-1),
        pair_fields["hd_v3_pair_query"],
        torch.zeros_like(pair_fields["hd_v3_pair_query"]),
    )
    pair_fields["hd_v3_pair_action_tail"] = torch.where(
        valid.unsqueeze(-1),
        pair_fields["hd_v3_pair_action_tail"],
        torch.zeros_like(pair_fields["hd_v3_pair_action_tail"]),
    )


def build_labels(
    features_path: str | Path,
    teacher_checkpoint: str | Path,
    output_path: str | Path,
    *,
    pair_k: int = 5,
    intervention: str = "delete",
    event_block_size: int = 1,
    delay_edges: Sequence[int] = DEFAULT_DELAY_EDGES,
    seed: int = 1000,
    device: str | torch.device = "cpu",
    require_query_features: bool = False,
    dataset_repo_id: str | None = None,
    fps: int | None = None,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    allow_custom_delay_edges: bool = False,
    counterfactual_batch_size: int | None = None,
) -> dict[str, Any]:
    """Build and save a deterministic CreditTTT V3 pair artifact.

    ``dataset_repo_id`` and ``fps`` are intentionally part of the function
    contract rather than inferred from a path.  The training loader checks
    both values before attaching labels; callers may omit them only when the
    feature cache already contains the corresponding provenance fields.
    """

    if pair_k <= 0:
        raise ValueError("pair_k must be positive")
    resolved_counterfactual_batch_size = _resolve_counterfactual_batch_size(
        counterfactual_batch_size
    )
    if not math.isfinite(float(positive_threshold)) or float(positive_threshold) < 0:
        raise ValueError("positive_threshold must be finite and non-negative")
    if intervention not in {"delete", "replace"}:
        raise ValueError("intervention must be 'delete' or 'replace'")
    if event_block_size <= 0:
        raise ValueError("event_block_size must be positive")
    feature_path = Path(features_path).expanduser()
    feature_metadata, rows = _load_feature_cache(feature_path)
    target_mode = str(feature_metadata.get("target", "normalized_executed_slot0_action"))
    if target_mode != "normalized_executed_slot0_action":
        raise ValueError(
            "This direct FullHistoryActionTeacher builder expects feature target "
            "'normalized_executed_slot0_action'; got "
            f"{target_mode!r}"
        )
    if feature_metadata.get("causal_previous_action") is not True:
        raise ValueError(
            "Feature cache must explicitly declare causal_previous_action=true; "
            "regenerate it with train_full_history_teacher.py"
        )
    resolved_dataset_id = dataset_repo_id or feature_metadata.get("dataset_repo_id")
    if resolved_dataset_id is None or not str(resolved_dataset_id).strip():
        raise ValueError(
            "CreditTTT V3 provenance requires dataset_repo_id; pass --dataset-repo-id "
            "or include it in the feature cache metadata"
        )
    resolved_fps: int | None
    if fps is None:
        raw_fps = feature_metadata.get("fps")
        if raw_fps is None:
            raise ValueError(
                "CreditTTT V3 provenance requires fps; pass --fps or include fps "
                "in the feature cache metadata"
            )
        try:
            fps_float = float(raw_fps)
            resolved_fps = int(fps_float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Malformed feature-cache fps={raw_fps!r}") from exc
        if not math.isfinite(fps_float) or fps_float != resolved_fps:
            raise ValueError(f"fps must be a positive integer, got {raw_fps!r}")
    else:
        resolved_fps = int(fps)
    if resolved_fps <= 0:
        raise ValueError(f"fps must be positive, got {resolved_fps}")
    feature_file_hash = _file_sha256(feature_path)
    device_obj = torch.device(device)
    teacher, teacher_manifest = load_full_history_teacher_checkpoint(
        teacher_checkpoint,
        map_location=device_obj,
    )
    teacher = teacher.to(device_obj).eval()
    teacher_hash = history_teacher_state_sha256(teacher)
    if teacher_manifest.get("format") != FULL_HISTORY_TEACHER_FORMAT:
        raise ValueError("Loaded teacher manifest is not an explicit full-history teacher")
    event_dim = int(rows[0]["event_tokens"].shape[-1])
    action_dim = int(rows[0]["target_actions"].shape[-1])
    declared_event_dim = feature_metadata.get("event_dim")
    declared_action_dim = feature_metadata.get("action_dim")
    if declared_event_dim is not None and int(declared_event_dim) != event_dim:
        raise ValueError(
            f"Feature metadata event_dim={declared_event_dim} disagrees with tensors ({event_dim})"
        )
    if declared_action_dim is not None and int(declared_action_dim) != action_dim:
        raise ValueError(
            f"Feature metadata action_dim={declared_action_dim} disagrees with tensors ({action_dim})"
        )
    if teacher.event_dim != event_dim or teacher.action_dim != action_dim:
        raise ValueError(
            f"Teacher dimensions ({teacher.event_dim},{teacher.action_dim}) disagree with "
            f"features ({event_dim},{action_dim})"
        )
    if teacher.action_horizon != 1:
        raise ValueError(
            "The frame-aligned slot-0 builder requires a horizon-one teacher; "
            f"got action_horizon={teacher.action_horizon}"
        )
    max_delay = max(1, max(int(row["event_tokens"].shape[0]) - 1 for row in rows))
    requested_edges = tuple(int(edge) for edge in delay_edges)
    canonical_edges = tuple(int(edge) for edge in CREDIT_TTT_DELAY_EDGES)
    if requested_edges != canonical_edges and not allow_custom_delay_edges:
        raise ValueError(
            "Canonical CreditTTT requires the frozen delay edges "
            f"{canonical_edges}; custom edges are a separately named ablation "
            "and require allow_custom_delay_edges=True"
        )
    edges = _fixed_delay_edges(max_delay, delay_edges)
    episode_outputs: list[dict[str, Any]] = []
    query_presence = [
        ("future_query_q" in row, "future_action_tail_h" in row)
        for row in rows
    ]
    if any(q != h for q, h in query_presence):
        raise ValueError(
            "Each feature episode must provide both future_query_q and "
            "future_action_tail_h, or neither; partial query caches are ambiguous"
        )
    query_available = bool(query_presence and all(query_presence[0]))
    if require_query_features and not query_available:
        raise ValueError(
            "--require-query-features requested but the feature cache lacks "
            "query features (future_query_q/future_action_tail_h) for every episode"
        )
    episode_slices: list[dict[str, int]] = []
    concatenation_offset = 0
    seen_global: set[int] = set()
    LOGGER.info(
        "counterfactual teacher replay batch size=%d (set %s=0 for the legacy reference loop)",
        resolved_counterfactual_batch_size,
        COUNTERFACTUAL_BATCH_ENV,
    )
    for episode_number, row in enumerate(rows):
        events = row["event_tokens"]
        previous = row["previous_executed_actions"]
        targets = row["target_actions"]
        length = int(events.shape[0])
        starts, ends = _event_spans(row, length, event_block_size)
        # The production student traces one fast-weight transition per
        # physical frame.  A multi-frame deletion would therefore remove a
        # different transition set from the one used by QH2L/CMD while still
        # looking like a canonical artifact in the metadata.  Keep block
        # interventions available for explicitly named offline ablations, but
        # fail closed for the canonical event-write protocol.
        if intervention == "delete":
            expected_starts = torch.arange(length, dtype=torch.int64)
            expected_ends = expected_starts + 1
            if not torch.equal(starts.cpu(), expected_starts) or not torch.equal(
                ends.cpu(), expected_ends
            ):
                raise ValueError(
                    "Canonical CreditTTT event-write deletion requires one "
                    "event span per frame ([i, i+1)); multi-frame/custom "
                    "spans need a separately implemented block-state replay"
                )
        replacement = row.get("replacement_event_tokens")
        if intervention == "replace" and replacement is None:
            # Replacement is an explicitly named ablation.  The canonical
            # event-write-deletion protocol never enters this branch.  Feature
            # caches may provide a curated paired replacement; when they do
            # not, deterministically map the same normalized episode phase
            # from another demonstration.  The source is recorded below, and
            # the V3 trainer rejects this variant until donor-state replay is
            # implemented on the student side.
            if len(rows) < 2:
                raise ValueError(
                    "intervention='replace' needs replacement_event_tokens or at least "
                    "two episodes for phase-matched cross-episode donors"
                )
            donor = rows[(episode_number + 1) % len(rows)]["event_tokens"]
            if donor.shape[-1] != events.shape[-1]:
                raise ValueError("Phase-matched donor event width disagrees with the target episode")
            if length == 1:
                donor_indices = torch.zeros(1, dtype=torch.long)
            else:
                donor_indices = torch.linspace(
                    0,
                    max(int(donor.shape[0]) - 1, 0),
                    length,
                    dtype=torch.float32,
                ).round().to(torch.long)
            replacement = donor.index_select(0, donor_indices)
        # Targets are deliberately separate from the recurrent call so no
        # current/future expert action can leak into the teacher state.
        full_actions, cf_actions, credit = _teacher_replays(
            teacher,
            events,
            previous,
            targets,
            starts,
            ends,
            intervention=intervention,
            replacement=replacement,
            device=device_obj,
            counterfactual_batch_size=resolved_counterfactual_batch_size,
        )
        pair_fields = _sample_frame_pairs(
            credit,
            starts=starts,
            ends=ends,
            pair_k=pair_k,
            delay_edges=edges,
            seed=_episode_seed(seed, int(row["episode_indices"][0].item()) if row["episode_indices"].numel() else episode_number),
            positive_threshold=float(positive_threshold),
            full_actions=full_actions,
            counterfactual_actions=cf_actions,
            expert_actions=targets,
        )
        _gather_query_features(pair_fields, row, require=require_query_features)
        pair_fields.update(
            {
                "global_index": row["global_indices"].clone(),
                "episode_index": row["episode_indices"].clone(),
                "frame_index": row["frame_indices"].clone(),
                # Retain the full direct-action teacher prediction for audits;
                # pair training consumes only the event-centric columns.
                "teacher_full_action": full_actions.detach().cpu().float(),
                "expert_target_action": targets.clone().float(),
            }
        )
        episode_outputs.append(pair_fields)
        global_indices = row["global_indices"].clone().to(dtype=torch.int64)
        if global_indices.numel() != torch.unique(global_indices).numel():
            raise ValueError(f"Episode {episode_number} contains duplicate global_index values")
        overlap = seen_global.intersection(int(value) for value in global_indices.tolist())
        if overlap:
            raise ValueError(
                "Feature episodes contain duplicate global_index values; first overlap="
                f"{min(overlap)}"
            )
        seen_global.update(int(value) for value in global_indices.tolist())
        episode_id = int(row["episode_indices"][0].item())
        episode_slices.append(
            {
                "episode_index": episode_id,
                "row_start": concatenation_offset,
                "row_end": concatenation_offset + length,
                "length": length,
            }
        )
        concatenation_offset += length
        LOGGER.info(
            "episode %d: %d frames, %d valid pairs",
            episode_number,
            length,
            int(pair_fields["hd_v3_pair_valid"].sum().item()),
        )

    field_names = [
        "hd_v3_pair_event_index",
        "hd_v3_pair_future_index",
        "hd_v3_pair_event_start",
        "hd_v3_pair_event_end",
        "hd_v3_pair_delay",
        "hd_v3_pair_delay_bin",
        "hd_v3_pair_utility",
        "hd_v3_pair_confidence",
        "hd_v3_pair_effect",
        "hd_v3_pair_valid",
        "hd_v3_pair_positive",
        "hd_v3_pair_null",
        "hd_v3_pair_teacher_full_action",
        "hd_v3_pair_teacher_counterfactual_action",
        "hd_v3_pair_expert_action",
    ]
    if query_available:
        field_names.extend(("hd_v3_pair_query", "hd_v3_pair_action_tail"))
    episode_lengths = [int(row["event_tokens"].shape[0]) for row in rows]
    # ``teacher_manifest`` is generated by ``FullHistoryActionTeacher`` and
    # contains a parameter hash.  Keep both the parameter and file hashes:
    # the former identifies the numerical model, while the latter detects a
    # replaced/edited checkpoint container.
    teacher_file = _resolve_teacher_file(teacher_checkpoint)
    teacher_file_hash = _file_sha256(teacher_file)
    metadata: dict[str, Any] = {
        "format": LABEL_FORMAT,
        "artifact_revision": ARTIFACT_REVISION,
        "version": 3,
        "protocol": CREDIT_TTT_V3_PROTOCOL,
        "pair_schema": CREDIT_TTT_V3_PAIR_SCHEMA,
        "attribution_protocol": "credit_ttt_v3_query_effect",
        "state": CREDIT_TTT_V3_STATE,
        "dataset_repo_id": str(resolved_dataset_id),
        "fps": int(resolved_fps),
        # ``intervention`` is the protocol-level schema identity.  The
        # concrete branch (delete vs paired replacement) is kept separately
        # so ``CreditTTTProtocol.from_dict`` remains valid for either mode.
        "intervention": CREDIT_TTT_V3_INTERVENTION,
        # ``intervention`` is the immutable method identity.  The concrete
        # branch is recorded separately so replacement artifacts remain
        # inspectable without being mistaken for the canonical deletion
        # protocol by the student trainer.
        "intervention_schema": CREDIT_TTT_V3_INTERVENTION,
        "intervention_type": intervention,
        "intervention_mode": intervention,
        "intervention_scope": (
            CREDIT_TTT_V3_INTERVENTION_SCOPE
            if intervention == "delete"
            else "event_content_replacement_previous_executed_action_held_fixed"
        ),
        "replacement_source": (
            "phase_matched_cross_episode_or_curated"
            if intervention == "replace"
            else "none_event_write_skip"
        ),
        "protocol_variant": (
            "canonical_event_write_deletion"
            if intervention == "delete" and tuple(edges.tolist()) == tuple(CREDIT_TTT_DELAY_EDGES)
            else (
                "content_replacement_ablation_not_consumed_by_student"
                if intervention == "replace"
                else "custom_delay_bins_ablation"
            )
        ),
        "canonical_delay_edges": tuple(edges.tolist()) == tuple(CREDIT_TTT_DELAY_EDGES),
        "target": CREDIT_TTT_V3_TARGET,
        "target_mode": "normalized_executed_slot0_action",
        "causal": True,
        "denoise_steps": int(getattr(DEFAULT_CREDIT_TTT_PROTOCOL, "denoise_steps", 10)),
        "antithetic_noise": False,
        "includes_previous_executed_action": True,
        "teacher_adapter": "causal_action_head",
        "antithetic_note": "Direct recurrent teacher cache; flow antithetic replay is an adapter-level extension.",
        "pair_k": int(pair_k),
        "event_block_size": int(event_block_size),
        "delay_edges": edges.tolist(),
        "delay_bin_labels": [
            f"[{edges[index].item()},{edges[index + 1].item()})"
            for index in range(edges.numel() - 1)
        ],
        "seed": int(seed),
        "positive_threshold": float(positive_threshold),
        "source_features": str(feature_path),
        "source_feature_format": FEATURE_FORMAT,
        "feature_artifact_sha256": feature_file_hash,
        "feature_cache_declared_sha256": feature_metadata.get("features_sha256"),
        "teacher_checkpoint": str(Path(teacher_checkpoint).expanduser()),
        "teacher_checkpoint_file": str(teacher_file),
        # The checkpoint hash names the serialized file; the numerical state
        # hash is recorded separately so a metadata-only rewrite can be
        # distinguished from a changed teacher function.
        "teacher_checkpoint_sha256": teacher_file_hash,
        "teacher_checkpoint_file_sha256": teacher_file_hash,
        "teacher_format": teacher_manifest.get("format"),
        "teacher_parameter_sha256": teacher_hash,
        "teacher_state_sha256": teacher_hash,
        "teacher_provenance": dict(teacher_manifest),
        "event_dim": event_dim,
        "action_dim": action_dim,
        "query_features_available": bool(query_available),
        "query_features_required": bool(require_query_features),
        # The pair artifact is defined on complete episodes.  These fields
        # make that assumption machine-checkable at the student trainer
        # boundary instead of leaving it only in the shell recipe/README.
        "history_mode": "full_episode_replay",
        "min_sequence_length": int(max(episode_lengths)),
        "sequence_stride_policy": "equal_sequence_length",
        "max_windows_per_episode": 1,
        "ttt_history_warmup_length": None,
        "sequence_offset_policy": "episode_local_zero",
        "frame_aligned": True,
        "pair_axis": "K",
        "index_semantics": "episode_local_frame_index",
        "delay_semantics": "future_index_minus_event_start",
        "episode_slices": episode_slices,
        "episode_lengths": episode_lengths,
        "flow_target_available": False,
        "direct_teacher_note": (
            "This artifact uses a causal recurrent action teacher. It is not a "
            "flow velocity target; target_mode names the normalized executed slot-0 action."
        ),
        "fields": field_names,
    }
    # Fail closed before serializing an artifact if any locally assembled
    # metadata diverges from the immutable CreditTTT method identity.  The
    # canonical deletion branch is validated literally.  Replacement is an
    # offline ablation with a different intervention scope, so validate its
    # shared protocol fields against a temporary canonical scope and retain
    # the concrete scope in the artifact for auditability; the student trainer
    # rejects that variant before optimization.
    validation_metadata = metadata
    if intervention != "delete":
        validation_metadata = dict(metadata)
        validation_metadata["intervention_scope"] = CREDIT_TTT_V3_INTERVENTION_SCOPE
    DEFAULT_CREDIT_TTT_PROTOCOL.validate(validation_metadata)
    # The canonical artifact is flat and frame-aligned so
    # ``HindsightLabelDataset`` can attach one row per physical frame.  The
    # event/future axes remain inside each row as K-sized tensors; no temporal
    # pair is collapsed into a scalar.  Episode slices in metadata preserve
    # auditability without introducing a second, ambiguous storage format.
    all_keys = sorted({key for episode in episode_outputs for key in episode})
    payload: dict[str, Any] = {"format": LABEL_FORMAT, "metadata": metadata}
    for key in all_keys:
        values = [episode[key] for episode in episode_outputs if key in episode]
        if len(values) != len(episode_outputs):
            # Optional query fields are either present for all episodes or
            # rejected above; any other missing column is a malformed replay.
            raise ValueError(f"Episode outputs disagree on field {key!r}")
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError(f"Label field {key!r} must be tensor-valued")
        payload[key] = torch.cat([value.detach().cpu() for value in values], dim=0)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    metadata["artifact_sha256"] = _artifact_sha256(payload)
    # Save the hash in a sidecar-free way by rewriting the small mapping.  The
    # tensors are already detached CPU values, so this does not rerun replay.
    torch.save({**payload, "metadata": metadata}, output)
    LOGGER.info("Wrote CreditTTT pair labels to %s", output)
    return {"format": LABEL_FORMAT, "metadata": metadata, "output": str(output)}


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    """Hash metadata and tensor bytes without serializing pickle internals."""

    digest = hashlib.sha256()
    metadata = dict(payload.get("metadata", {}))
    metadata.pop("artifact_sha256", None)
    digest.update(json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str).encode())
    # Hash the flat frame columns in a stable key order.  The metadata hash
    # above intentionally excludes the self-referential artifact hash.
    for key in sorted(payload):
        if key in {"metadata", "format"}:
            continue
        value = payload[key]
        digest.update(str(key).encode())
        if isinstance(value, Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, default=str, sort_keys=True).encode())
    return digest.hexdigest()


def _parse_edges(value: str) -> list[int]:
    try:
        edges = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("delay edges must be comma-separated integers") from exc
    if not edges:
        raise argparse.ArgumentTypeError("at least one delay edge is required")
    return edges


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--features-input", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-k", type=int, default=5)
    parser.add_argument(
        "--intervention",
        choices=("delete", "replace"),
        default="delete",
        help=(
            "History intervention. The canonical paper protocol deletes the "
            "event write; phase-matched content replacement is an explicit "
            "offline ablation and is not consumed by the V3 student."
        ),
    )
    parser.add_argument("--event-block-size", type=int, default=1)
    parser.add_argument(
        "--delay-edges",
        type=_parse_edges,
        default=list(DEFAULT_DELAY_EDGES),
        help="Inclusive-left delay boundaries, e.g. 1,17,65,257,1025",
    )
    parser.add_argument(
        "--allow-custom-delay-edges",
        action="store_true",
        help=(
            "Permit a non-canonical delay schedule as a separately named ablation. "
            "The strict V3 student trainer will reject that artifact."
        ),
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=DEFAULT_POSITIVE_THRESHOLD,
        help=(
            "Normalized utility threshold for positive QH2L pairs; values at or below "
            f"the default {DEFAULT_POSITIVE_THRESHOLD:g} are null/invariance pairs"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Dataset identity for provenance (defaults to feature-cache metadata)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Dataset FPS for provenance (defaults to feature-cache metadata)",
    )
    parser.add_argument(
        "--require-query-features",
        action="store_true",
        help="Fail unless the feature cache contains future q_j and action-tail h_j",
    )
    parser.add_argument(
        "--counterfactual-batch-size",
        type=int,
        default=None,
        help=(
            "Execution-only batch size for independent event deletion/replacement replays. "
            f"Defaults to {DEFAULT_COUNTERFACTUAL_BATCH_SIZE} (or ${COUNTERFACTUAL_BATCH_ENV}); "
            "use 0 for the legacy one-event reference loop."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_labels(
        args.features_input,
        args.teacher_checkpoint,
        args.output,
        pair_k=args.pair_k,
        intervention=args.intervention,
        event_block_size=args.event_block_size,
        delay_edges=args.delay_edges,
        seed=args.seed,
        positive_threshold=args.positive_threshold,
        device=args.device,
        require_query_features=args.require_query_features,
        dataset_repo_id=args.dataset_repo_id,
        fps=args.fps,
        allow_custom_delay_edges=args.allow_custom_delay_edges,
        counterfactual_batch_size=args.counterfactual_batch_size,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
