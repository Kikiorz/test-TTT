"""Core, tensor-only primitives for CreditTTT (V3).

This module contains the small pieces shared by the offline hindsight-credit
pipeline and the online query-conditioned writer.  It intentionally has no
dependency on SmolVLA or on a particular fast-weight implementation.  Keeping
the protocol here tensor-only makes the supervision easy to audit and lets us
unit-test the causal/data contracts without loading a VLM.

The V3 protocol represents supervision as event--future pairs ``(i, j)``:

* ``i`` is an interaction that may write to fast weights;
* ``j`` is a strictly later query (same episode); and
* the teacher label is the effect of deleting event ``i``'s fast-weight write
  on the *final executed slot-0 action*, not an intermediate denoising
  velocity.  A donor-content replacement is retained only as an explicitly
  named ablation and is not consumed by the canonical student path.

At deployment only the local writer update and the action read are retained.
The teacher, pair labels, and counterfactual branches are training-time
artifacts.  Functions in this file therefore detach teacher labels by
construction and never mutate a caller's fast-weight state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Generic, Literal, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Versioned protocol constants
# ---------------------------------------------------------------------------

# Keep these strings stable: they are written into label metadata and are
# checked when a checkpoint/artifact is attached to a training run.
CREDIT_TTT_V3_FORMAT = "credit_ttt_v3"
CREDIT_TTT_V3_PROTOCOL = "creditttt_qh2l_v3"
CREDIT_TTT_V3_PAIR_SCHEMA = "event_future_control_pair_v3"
# The canonical student effect is the difference between the fast state just
# before and just after event ``i`` is written.  Therefore the offline teacher
# must use the same intervention: skip that event write while holding the
# future observation and expert action fixed.  Content replacement remains a
# separately named ablation; it is intentionally not the publication
# protocol because the current student trace does not replay donor writes.
CREDIT_TTT_V3_INTERVENTION = "event_write_deletion"
# The canonical counterfactual removes only the event's write transition.  The
# demonstrated predecessor action remains fixed in both branches; making this
# scope explicit prevents the label from being described as deletion of the
# whole physical interaction (which would be a different causal experiment).
CREDIT_TTT_V3_INTERVENTION_SCOPE = "event_write_only_previous_executed_action_held_fixed"
CREDIT_TTT_V3_TARGET = "final_slot0_action"
CREDIT_TTT_V3_STATE = "causal_fast_weights"
# Protocol delay bins are expressed as half-open integer edges.  The final
# edge is a large sentinel representing the open-ended ``1025+`` bin.
CREDIT_TTT_DELAY_EDGES = (1, 17, 65, 257, 1025, 2**31 - 1)
CREDIT_TTT_DELAY_BIN_LABELS = ("1-16", "17-64", "65-256", "257-1024", "1025+")

# More verbose aliases are useful to callers that prefer the terminology used
# in the paper.  They are aliases, rather than duplicate values, so metadata
# comparisons remain exact.
CREDIT_TTT_FORMAT = CREDIT_TTT_V3_FORMAT
CREDIT_TTT_PROTOCOL = CREDIT_TTT_V3_PROTOCOL
CREDIT_TTT_PAIR_SCHEMA = CREDIT_TTT_V3_PAIR_SCHEMA
FULL_HISTORY_CONTROL_ATTRIBUTION = "full_history_control_attribution"
QUERY_CONDITIONED_LOCAL_TTT = "query_conditioned_local_ttt"
CAUSAL_MEMORY_DEPLOYMENT = "causal_memory_deployment"

_EPS = 1e-8


@dataclass(frozen=True)
class CreditTTTProtocol:
    """Immutable description of the V3 label/update contract.

    The defaults are deliberately tied to the method definition, rather than
    to a particular task.  Dataset-specific choices (for example the number
    of sampled pairs) live in the sampler call and are recorded separately in
    an experiment config.
    """

    format: str = CREDIT_TTT_V3_FORMAT
    protocol: str = CREDIT_TTT_V3_PROTOCOL
    version: int = 3
    pair_schema: str = CREDIT_TTT_V3_PAIR_SCHEMA
    intervention: str = CREDIT_TTT_V3_INTERVENTION
    intervention_scope: str = CREDIT_TTT_V3_INTERVENTION_SCOPE
    target: str = CREDIT_TTT_V3_TARGET
    state: str = CREDIT_TTT_V3_STATE
    causal: bool = True
    denoise_steps: int = 10
    # The public protocol is agnostic to the teacher implementation.  The
    # shipped lightweight adapter predicts the executed slot-0 action directly
    # (and consequently has no flow noise); a flow-integrated adapter may set
    # this flag true.  Recording the choice prevents a direct-action pilot from
    # being reported as an antithetic-flow experiment.
    antithetic_noise: bool = False
    includes_previous_executed_action: bool = True
    teacher_adapter: str = "causal_action_head"
    flow_target_available: bool = False

    def __post_init__(self) -> None:
        """Reject silently incompatible metadata at construction time."""

        if self.format != CREDIT_TTT_V3_FORMAT:
            raise ValueError(f"unsupported CreditTTT format: {self.format!r}")
        if self.protocol != CREDIT_TTT_V3_PROTOCOL:
            raise ValueError(f"unsupported CreditTTT protocol: {self.protocol!r}")
        if self.version != 3:
            raise ValueError("CreditTTT V3 requires version=3")
        if self.pair_schema != CREDIT_TTT_V3_PAIR_SCHEMA:
            raise ValueError(f"unsupported pair schema: {self.pair_schema!r}")
        if self.intervention != CREDIT_TTT_V3_INTERVENTION:
            raise ValueError(f"unsupported intervention: {self.intervention!r}")
        if self.intervention_scope != CREDIT_TTT_V3_INTERVENTION_SCOPE:
            raise ValueError(
                "unsupported intervention scope: "
                f"{self.intervention_scope!r}"
            )
        if self.target != CREDIT_TTT_V3_TARGET:
            raise ValueError(f"unsupported teacher target: {self.target!r}")
        if self.state != CREDIT_TTT_V3_STATE:
            raise ValueError(f"unsupported state representation: {self.state!r}")
        if not self.causal:
            raise ValueError("CreditTTT V3 pair/update protocol must be causal")
        if self.denoise_steps <= 0:
            raise ValueError("denoise_steps must be positive")
        if not isinstance(self.teacher_adapter, str) or not self.teacher_adapter:
            raise ValueError("teacher_adapter must be a non-empty string")
        if type(self.flow_target_available) is not bool:
            raise ValueError("flow_target_available must be a boolean")

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata for a label artifact/checkpoint."""

        return {
            "format": self.format,
            "protocol": self.protocol,
            "version": self.version,
            "pair_schema": self.pair_schema,
            "intervention": self.intervention,
            "intervention_scope": self.intervention_scope,
            "target": self.target,
            "state": self.state,
            "causal": self.causal,
            "denoise_steps": self.denoise_steps,
            "antithetic_noise": self.antithetic_noise,
            "includes_previous_executed_action": self.includes_previous_executed_action,
            "teacher_adapter": self.teacher_adapter,
            "flow_target_available": self.flow_target_available,
        }

    # ``to_dict`` is a conventional spelling used by the other policy
    # metadata helpers.
    to_dict = as_dict

    def validate(self, metadata: Mapping[str, Any]) -> None:
        """Validate protocol identity fields in an artifact manifest."""

        if not isinstance(metadata, Mapping):
            raise TypeError("CreditTTT metadata must be a mapping")
        expected = self.as_dict()
        for key in (
            "format",
            "protocol",
            "version",
            "pair_schema",
            "intervention",
            "intervention_scope",
            "target",
            "state",
            "causal",
        ):
            if key not in metadata:
                raise ValueError(f"CreditTTT metadata is missing required field {key!r}")
            if metadata[key] != expected[key]:
                raise ValueError(
                    f"CreditTTT metadata field {key!r}={metadata[key]!r} does not match "
                    f"expected {expected[key]!r}"
                )

    @classmethod
    def from_dict(cls, metadata: Mapping[str, Any]) -> "CreditTTTProtocol":
        """Construct and validate a protocol from JSON metadata."""

        if not isinstance(metadata, Mapping):
            raise TypeError("CreditTTT metadata must be a mapping")
        names = {
            "format",
            "protocol",
            "version",
            "pair_schema",
            "intervention",
            "intervention_scope",
            "target",
            "state",
            "causal",
            "denoise_steps",
            "antithetic_noise",
            "includes_previous_executed_action",
            "teacher_adapter",
            "flow_target_available",
        }
        protocol = cls(**{name: metadata[name] for name in names if name in metadata})
        protocol.validate(metadata)
        return protocol


DEFAULT_CREDIT_TTT_PROTOCOL = CreditTTTProtocol()


@dataclass(frozen=True)
class InteractionFuturePair:
    """A single causal event/future pair.

    ``utility`` is the signed, symmetric relative teacher utility.  Positive
    utility means that removing/replacing the event write worsens the future
    action prediction.  Negative utility is retained as a harmful-write signal
    and is normally routed to the null/invariance branch.
    """

    event_index: int
    future_index: int
    delay: int
    utility: float | Tensor
    batch_index: int = 0
    delay_bin: int = 0
    positive: bool = False

    @property
    def i(self) -> int:
        """Equation-style alias for the event index."""

        return self.event_index

    @property
    def j(self) -> int:
        """Equation-style alias for the future query index."""

        return self.future_index


@dataclass(frozen=True)
class DelayBalancedPairBatch:
    """Tensor batch returned by :func:`sample_delay_balanced_pairs`.

    All per-pair fields are one-dimensional and have the same length.  If ``pad_to``
    was requested, invalid padded rows use ``-1`` indices and
    ``valid_mask=False``.  Flattening the batch dimension is intentional: the
    event and query encoders can gather arbitrary physical frames while
    ``batch_index`` prevents cross-episode pairs.  ``delay_edges`` records the
    temporal stratification bins when present.
    """

    batch_index: Tensor
    event_index: Tensor
    future_index: Tensor
    delay: Tensor
    delay_bin: Tensor
    utility: Tensor
    positive_mask: Tensor
    null_mask: Tensor
    valid_mask: Tensor
    # Optional final-action effect labels gathered alongside indices.
    teacher_effect: Tensor | None = None
    # Bin edges used by the sampler; retained for reproducible audit logs.
    delay_edges: Tensor | None = None

    def __post_init__(self) -> None:
        fields = (
            self.batch_index,
            self.event_index,
            self.future_index,
            self.delay,
            self.delay_bin,
            self.utility,
            self.positive_mask,
            self.null_mask,
            self.valid_mask,
        )
        if any(not isinstance(value, Tensor) for value in fields):
            raise TypeError("pair batch fields must be tensors")
        shapes = {tuple(value.shape) for value in fields}
        if len(shapes) != 1 or self.batch_index.ndim != 1:
            raise ValueError("pair batch index/mask fields must be one-dimensional and aligned")
        if self.teacher_effect is not None:
            if not isinstance(self.teacher_effect, Tensor) or self.teacher_effect.ndim == 0:
                raise ValueError("teacher_effect must be a tensor with a feature dimension")
            if self.teacher_effect.shape[0] != self.batch_index.shape[0]:
                raise ValueError("teacher_effect first dimension must match pair count")
        if self.delay_edges is not None:
            if not isinstance(self.delay_edges, Tensor) or self.delay_edges.ndim != 1:
                raise ValueError("delay_edges must be a one-dimensional tensor")
            if self.delay_edges.numel() > 1 and bool((self.delay_edges[1:] <= self.delay_edges[:-1]).any().item()):
                raise ValueError("delay_edges must be strictly increasing")
        if self.event_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("event_index must be an integer tensor")
        if self.future_index.dtype != self.event_index.dtype:
            raise TypeError("event_index and future_index dtypes must match")

    @property
    def num_pairs(self) -> int:
        """Number of rows, including optional padded rows."""

        return int(self.event_index.numel())

    @property
    def num_valid(self) -> int:
        """Number of sampled (non-padding) rows."""

        return int(self.valid_mask.to(dtype=torch.int64).sum().item())

    @property
    def pairs(self) -> list[InteractionFuturePair]:
        """Materialize valid pairs for logging or artifact serialization."""

        result: list[InteractionFuturePair] = []
        valid = self.valid_mask.detach().to(device="cpu", dtype=torch.bool)
        for row in valid.nonzero(as_tuple=False).flatten().tolist():
            utility = self.utility[row].detach().cpu()
            if utility.numel() == 1:
                utility_value: float | Tensor = float(utility.item())
            else:
                utility_value = utility
            result.append(
                InteractionFuturePair(
                    event_index=int(self.event_index[row].item()),
                    future_index=int(self.future_index[row].item()),
                    delay=int(self.delay[row].item()),
                    utility=utility_value,
                    batch_index=int(self.batch_index[row].item()),
                    delay_bin=int(self.delay_bin[row].item()),
                    positive=bool(self.positive_mask[row].item()),
                )
            )
        return result

    def compact(self) -> "DelayBalancedPairBatch":
        """Drop padded rows while preserving tensor/device/dtype metadata."""

        index = self.valid_mask.nonzero(as_tuple=False).flatten()
        effect = None if self.teacher_effect is None else self.teacher_effect.index_select(0, index)
        return DelayBalancedPairBatch(
            batch_index=self.batch_index.index_select(0, index),
            event_index=self.event_index.index_select(0, index),
            future_index=self.future_index.index_select(0, index),
            delay=self.delay.index_select(0, index),
            delay_bin=self.delay_bin.index_select(0, index),
            utility=self.utility.index_select(0, index),
            positive_mask=self.positive_mask.index_select(0, index),
            null_mask=self.null_mask.index_select(0, index),
            valid_mask=self.valid_mask.index_select(0, index),
            teacher_effect=effect,
            delay_edges=self.delay_edges,
        )

    def to(self, *args: Any, **kwargs: Any) -> "DelayBalancedPairBatch":
        """Move all tensors to a device/dtype (masks remain boolean)."""

        def move(value: Tensor, *, bool_dtype: bool = False, preserve_dtype: bool = False) -> Tensor:
            moved = value.to(*args, **kwargs)
            if preserve_dtype:
                moved = moved.to(dtype=value.dtype)
            return moved.to(dtype=torch.bool) if bool_dtype else moved

        return DelayBalancedPairBatch(
            batch_index=move(self.batch_index, preserve_dtype=True),
            event_index=move(self.event_index, preserve_dtype=True),
            future_index=move(self.future_index, preserve_dtype=True),
            delay=move(self.delay, preserve_dtype=True),
            delay_bin=move(self.delay_bin, preserve_dtype=True),
            utility=move(self.utility),
            positive_mask=move(self.positive_mask, bool_dtype=True),
            null_mask=move(self.null_mask, bool_dtype=True),
            valid_mask=move(self.valid_mask, bool_dtype=True),
            teacher_effect=None if self.teacher_effect is None else move(self.teacher_effect),
            delay_edges=None
            if self.delay_edges is None
            else move(self.delay_edges, preserve_dtype=True),
        )


# Short aliases make the schema discoverable without breaking the longer name.
CreditPairBatch = DelayBalancedPairBatch
InteractionFuturePairBatch = DelayBalancedPairBatch


def symmetric_relative_utility(
    reference: Tensor,
    counterfactual: Tensor,
    *,
    eps: float = _EPS,
    dim: int | Sequence[int] | None = None,
    reduction: Literal["mean", "sum", "norm"] = "mean",
    clip: float | None = None,
) -> Tensor:
    """Compute a signed, scale-free intervention utility.

    For scalar losses this is

    ``(counterfactual - reference) / (0.5*(|counterfactual| + |reference|)+eps)``.

    Positive values mean the counterfactual is worse.  ``dim`` optionally
    reduces action-feature dimensions before the ratio; the default preserves
    the input shape.  A zero/zero pair returns zero, avoiding an artificial
    positive label for numerically empty padded rows.  ``clip`` is opt-in and
    therefore cannot silently change the protocol's signed target.
    """

    if not isinstance(reference, Tensor) or not isinstance(counterfactual, Tensor):
        raise TypeError("reference and counterfactual must be tensors")
    if not torch.is_floating_point(reference) or not torch.is_floating_point(counterfactual):
        raise TypeError("reference and counterfactual must be floating-point tensors")
    if eps <= 0:
        raise ValueError("eps must be positive")
    reference, counterfactual = torch.broadcast_tensors(reference, counterfactual)
    if dim is not None:
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        normalized_dims = tuple(d if d >= 0 else reference.ndim + d for d in dims)
        if any(d < 0 or d >= reference.ndim for d in normalized_dims):
            raise ValueError(f"dim={dim!r} is out of range for shape {tuple(reference.shape)}")
        if reduction == "norm":
            reference = reference.square().sum(dim=normalized_dims).sqrt()
            counterfactual = counterfactual.square().sum(dim=normalized_dims).sqrt()
        elif reduction == "mean":
            reference = reference.mean(dim=normalized_dims)
            counterfactual = counterfactual.mean(dim=normalized_dims)
        elif reduction == "sum":
            reference = reference.sum(dim=normalized_dims)
            counterfactual = counterfactual.sum(dim=normalized_dims)
        else:
            raise ValueError(f"unknown reduction: {reduction!r}")
    elif reduction not in {"mean", "sum", "norm"}:
        raise ValueError(f"unknown reduction: {reduction!r}")
    result = (counterfactual - reference) / (
        0.5 * (counterfactual.abs() + reference.abs())
    ).clamp_min(eps)
    finite = torch.isfinite(result)
    result = torch.where(finite, result, torch.zeros_like(result))
    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive")
        result = result.clamp(-clip, clip)
    return result


# Names used in notes/equations and by earlier prototypes.
compute_symmetric_relative_utility = symmetric_relative_utility
relative_utility = symmetric_relative_utility


# ---------------------------------------------------------------------------
# Delay-balanced event/future sampling
# ---------------------------------------------------------------------------


def _broadcast_pair_mask(mask: Tensor | None, shape: torch.Size, *, device: torch.device) -> Tensor:
    """Broadcast a pair mask and normalize it to bool."""

    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not isinstance(mask, Tensor):
        mask = torch.as_tensor(mask, device=device)
    else:
        mask = mask.to(device=device)
    try:
        return torch.broadcast_to(mask.to(dtype=torch.bool), shape)
    except RuntimeError as exc:
        raise ValueError(f"pair_mask shape {tuple(mask.shape)} is not broadcastable to {tuple(shape)}") from exc


def _sample_indices(
    candidates: Tensor,
    count: int,
    *,
    replacement: bool,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample rows without relying on a global RNG state."""

    if count <= 0 or candidates.numel() == 0:
        return candidates.new_empty((0,), dtype=torch.long)
    candidates = candidates.to(dtype=torch.long)
    if replacement:
        # ``torch.randint`` supports a CPU generator on every PyTorch build;
        # use the candidate device only when no generator was supplied.
        if generator is None:
            choice = torch.randint(candidates.numel(), (count,), device=candidates.device)
        else:
            try:
                choice = torch.randint(
                    candidates.numel(), (count,), generator=generator, device=candidates.device
                )
            except RuntimeError:
                choice = torch.randint(candidates.numel(), (count,), generator=generator, device="cpu").to(
                    candidates.device
                )
        return candidates.index_select(0, choice)
    count = min(count, candidates.numel())
    if generator is None:
        order = torch.randperm(candidates.numel(), device=candidates.device)[:count]
    else:
        try:
            order = torch.randperm(candidates.numel(), generator=generator, device=candidates.device)[:count]
        except RuntimeError:
            order = torch.randperm(candidates.numel(), generator=generator, device="cpu")[:count].to(
                candidates.device
            )
    return candidates.index_select(0, order)


def _delay_bin_ids(delays: Tensor, num_bins: int, delay_edges: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Return integer bin IDs and the (inclusive-left) edges used."""

    if delays.numel() == 0:
        empty_edges = torch.arange(0, max(1, num_bins) + 1, device=delays.device, dtype=torch.long)
        return delays.to(dtype=torch.long), empty_edges
    if delay_edges is not None:
        edges = torch.as_tensor(delay_edges, device=delays.device, dtype=torch.long).flatten()
        if edges.numel() < 2 or bool((edges[1:] <= edges[:-1]).any().item()):
            raise ValueError("delay_edges must contain at least two strictly increasing integers")
        # bucketize with right=False yields [edge[k], edge[k+1]) bins.  Values
        # below the first/above the last edge are clipped into boundary bins.
        bins = torch.bucketize(delays, edges[1:-1], right=False)
        return bins.to(torch.long), edges
    if num_bins <= 0:
        raise ValueError("num_delay_bins must be positive")
    # Five bins with no explicit edges use the protocol's fixed ranges.  Other
    # bin counts remain quantile-balanced, which is useful for small custom
    # diagnostics without changing the publication recipe.
    if num_bins == len(CREDIT_TTT_DELAY_BIN_LABELS):
        protocol_edges = torch.tensor(CREDIT_TTT_DELAY_EDGES, device=delays.device, dtype=torch.long)
        bins = torch.bucketize(delays, protocol_edges[1:-1], right=False)
        return bins.to(torch.long), protocol_edges
    minimum = int(delays.min().item())
    maximum = int(delays.max().item())
    if minimum == maximum:
        edges = torch.tensor([minimum, maximum + 1], device=delays.device, dtype=torch.long)
        return torch.zeros_like(delays, dtype=torch.long), edges
    # Integer delays are most naturally split by quantile ranks.  This keeps
    # bins approximately equally populated even when a long episode has a
    # highly non-uniform set of valid event/query pairs.  Repeated delay values
    # may straddle a boundary; this is preferable to an empty tail bin and is
    # deterministic for a fixed candidate ordering.
    bins_count = min(int(num_bins), int(delays.numel()))
    order = torch.argsort(delays, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(delays.numel(), device=delays.device)
    bins = (ranks * bins_count // delays.numel()).clamp_max(bins_count - 1)
    # Edges are metadata for logging.  Their exact values are the smallest
    # observed delay in each rank bucket, followed by one past the maximum.
    starts = delays[order][
        torch.linspace(0, delays.numel() - 1, bins_count, device=delays.device).round().to(torch.long)
    ]
    starts = torch.unique_consecutive(starts)
    edges = torch.cat((starts, (delays.max() + 1).reshape(1))).to(torch.long)
    return bins.to(torch.long), edges


def sample_delay_balanced_pairs(
    utility: Tensor,
    *,
    pair_mask: Tensor | None = None,
    num_pairs: int | None = None,
    pairs_per_bin: int | None = 1,
    num_delay_bins: int = len(CREDIT_TTT_DELAY_BIN_LABELS),
    delay_edges: Tensor | Sequence[int] | None = None,
    event_block_size: int = 1,
    positive_threshold: float = 0.0,
    include_null: bool = True,
    replacement: bool = False,
    pad_to: int | None = None,
    generator: torch.Generator | None = None,
    teacher_effect: Tensor | None = None,
) -> DelayBalancedPairBatch:
    """Sample causal event/future pairs with balanced temporal delays.

    ``utility`` has shape ``[I, J]`` or ``[B, I, J]``.  A default mask keeps
    only ``j >= i + event_block_size``; callers can pass a content-intervention
    mask when event blocks are not consecutive.  Sampling is stratified by
    delay bins and, when ``include_null=True``, by positive versus null
    utility.  The latter is important for QH2L: high-utility pairs teach the
    writer the desired effect while null/harmful pairs teach invariance.

    The returned rows are flattened across episodes and carry ``batch_index``
    so no sampled pair can accidentally cross an episode boundary.  Sampling
    is without replacement by default.  If fewer candidates exist than the
    requested count, all available candidates are returned (or padded when
    ``pad_to`` is set).  No global random state is touched when a generator is
    supplied.
    """

    if not isinstance(utility, Tensor) or utility.ndim not in (2, 3):
        raise ValueError("utility must have shape [I,J] or [B,I,J]")
    if not torch.is_floating_point(utility):
        raise TypeError("utility must be a floating-point tensor")
    if event_block_size <= 0:
        raise ValueError("event_block_size must be positive")
    if pairs_per_bin is not None and pairs_per_bin < 0:
        raise ValueError("pairs_per_bin must be non-negative or None")
    if num_pairs is not None and num_pairs < 0:
        raise ValueError("num_pairs must be non-negative")
    if num_delay_bins <= 0:
        raise ValueError("num_delay_bins must be positive")
    if pad_to is not None and pad_to < 0:
        raise ValueError("pad_to must be non-negative")
    if positive_threshold < 0:
        raise ValueError("positive_threshold must be non-negative")
    validated_delay_edges: Tensor | None = None
    if delay_edges is not None:
        validated_delay_edges = torch.as_tensor(delay_edges, device=utility.device, dtype=torch.long).flatten()
        if validated_delay_edges.numel() < 2 or bool(
            (validated_delay_edges[1:] <= validated_delay_edges[:-1]).any().item()
        ):
            raise ValueError("delay_edges must contain at least two strictly increasing integers")

    squeezed = utility.ndim == 2
    values = utility.unsqueeze(0) if squeezed else utility
    batch_size, events, futures = values.shape
    device = values.device
    finite = torch.isfinite(values)
    mask = _broadcast_pair_mask(pair_mask, values.shape, device=device) & finite
    event = torch.arange(events, device=device).view(1, events, 1)
    future = torch.arange(futures, device=device).view(1, 1, futures)
    causal = future >= event + event_block_size
    mask &= causal

    # Flatten candidates but preserve batch IDs.  Invalid entries are never
    # materialized, so an episode with no valid future contributes no rows.
    candidate = mask.nonzero(as_tuple=False)
    if candidate.numel() == 0:
        selected = candidate.new_empty((0, 3))
        bins = torch.empty((0,), device=device, dtype=torch.long)
        used_edges = (
            validated_delay_edges
            if validated_delay_edges is not None
            else torch.empty((0,), device=device, dtype=torch.long)
        )
    else:
        candidate_values = values[candidate[:, 0], candidate[:, 1], candidate[:, 2]]
        delays = candidate[:, 2] - candidate[:, 1]
        bins, used_edges = _delay_bin_ids(
            delays,
            num_delay_bins,
            validated_delay_edges,
        )
        positive = candidate_values > positive_threshold
        # Determine the requested count per temporal bin.  ``num_pairs`` takes
        # precedence and is distributed round-robin, which avoids a final bin
        # receiving all of the remainder.
        actual_bins = int(bins.max().item()) + 1 if bins.numel() else 0
        if num_pairs is not None:
            if actual_bins:
                base, remainder = divmod(int(num_pairs), actual_bins)
                bin_counts = [base + (index < remainder) for index in range(actual_bins)]
            else:
                bin_counts = []
        else:
            if pairs_per_bin is None:
                # ``None`` means exhaustive sampling, which is convenient for
                # small diagnostic episodes and avoids an accidental empty
                # artifact when both count knobs are omitted.
                bin_counts = [int((bins == bin_id).sum().item()) for bin_id in range(actual_bins)]
            else:
                count = int(pairs_per_bin)
                bin_counts = [count] * actual_bins

        selected_rows: list[Tensor] = []
        for bin_id, requested in enumerate(bin_counts):
            if requested <= 0:
                continue
            in_bin = bins == bin_id
            positive_rows = torch.nonzero(in_bin & positive, as_tuple=False).flatten()
            null_rows = torch.nonzero(in_bin & ~positive, as_tuple=False).flatten()
            if include_null:
                # Split as evenly as possible, then back-fill from the other
                # stratum if one side is unavailable.
                positive_count = (requested + 1) // 2
                null_count = requested - positive_count
                picked_positive = _sample_indices(
                    positive_rows,
                    positive_count,
                    replacement=replacement,
                    generator=generator,
                )
                picked_null = _sample_indices(
                    null_rows,
                    null_count,
                    replacement=replacement,
                    generator=generator,
                )
                missing = requested - picked_positive.numel() - picked_null.numel()
                if missing > 0:
                    used = torch.cat((picked_positive, picked_null))
                    # Construct the remaining rows directly from the global
                    # candidate positions in this delay bin.
                    all_rows = torch.nonzero(in_bin, as_tuple=False).flatten()
                    if used.numel():
                        keep = ~torch.isin(all_rows, used)
                        all_rows = all_rows[keep]
                    picked_extra = _sample_indices(
                        all_rows,
                        missing,
                        replacement=replacement,
                        generator=generator,
                    )
                    picked = torch.cat((picked_positive, picked_null, picked_extra))
                else:
                    picked = torch.cat((picked_positive, picked_null))
            else:
                picked = _sample_indices(
                    torch.nonzero(in_bin, as_tuple=False).flatten(),
                    requested,
                    replacement=replacement,
                    generator=generator,
                )
            if picked.numel():
                selected_rows.append(picked)
        selected_indices = (
            torch.cat(selected_rows) if selected_rows else torch.empty((0,), device=device, dtype=torch.long)
        )
        selected = candidate.index_select(0, selected_indices)
        selected_bins = bins.index_select(0, selected_indices)

    if candidate.numel() == 0:
        selected_bins = torch.empty((0,), device=device, dtype=torch.long)
    selected_count = int(selected.shape[0])
    target_length = selected_count if pad_to is None else max(selected_count, int(pad_to))
    if pad_to is not None and pad_to < selected_count:
        # Explicitly truncating would make the requested delay balance
        # dependent on incidental candidate order; reject it instead.
        raise ValueError(f"pad_to={pad_to} is smaller than sampled count {selected_count}")

    batch_index = torch.full((target_length,), -1, device=device, dtype=torch.long)
    event_index = torch.full((target_length,), -1, device=device, dtype=torch.long)
    future_index = torch.full((target_length,), -1, device=device, dtype=torch.long)
    delay = torch.full((target_length,), -1, device=device, dtype=torch.long)
    delay_bin = torch.full((target_length,), -1, device=device, dtype=torch.long)
    utility_out = torch.zeros((target_length,), device=device, dtype=utility.dtype)
    positive_mask = torch.zeros((target_length,), device=device, dtype=torch.bool)
    null_mask = torch.zeros((target_length,), device=device, dtype=torch.bool)
    valid_mask = torch.zeros((target_length,), device=device, dtype=torch.bool)
    effect_out: Tensor | None = None
    if teacher_effect is not None:
        if not isinstance(teacher_effect, Tensor) or teacher_effect.ndim < 2:
            raise ValueError("teacher_effect must have leading shape [B,I,J] or [I,J]")
        # Pair labels are offline teacher artifacts; never retain a gradient
        # graph through the gather operation.
        raw_effect = teacher_effect.to(device=device).detach()
        # A scalar effect can be stored as [I,J] / [B,I,J]; vector actions add
        # feature dimensions after those leading axes.  Add a batch axis for
        # the unbatched utility form and broadcast singleton batch/event axes
        # where a collator intentionally shared a label across episodes.
        if squeezed:
            if raw_effect.shape[:2] != (events, futures):
                raise ValueError(
                    "unbatched teacher_effect must start with [I,J], got "
                    f"{tuple(raw_effect.shape[:2])} vs {(events, futures)}"
                )
            effect_values = raw_effect.unsqueeze(0)
        else:
            if raw_effect.ndim < 3:
                raise ValueError("batched teacher_effect must have leading shape [B,I,J]")
            effect_values = raw_effect
        if effect_values.ndim == 3:
            effect_values = effect_values.unsqueeze(-1)
        feature_shape = effect_values.shape[3:]
        try:
            effect_values = torch.broadcast_to(
                effect_values,
                (batch_size, events, futures, *feature_shape),
            )
        except RuntimeError as exc:
            raise ValueError(
                "teacher_effect leading shape must broadcast to utility [B,I,J], got "
                f"{tuple(effect_values.shape[:3])} vs {(batch_size, events, futures)}"
            ) from exc
        effect_out = torch.zeros(
            (target_length, *effect_values.shape[3:]), device=effect_values.device, dtype=effect_values.dtype
        )

    if selected_count:
        batch_index[:selected_count] = selected[:, 0]
        event_index[:selected_count] = selected[:, 1]
        future_index[:selected_count] = selected[:, 2]
        delay[:selected_count] = selected[:, 2] - selected[:, 1]
        delay_bin[:selected_count] = selected_bins
        utility_out[:selected_count] = values[selected[:, 0], selected[:, 1], selected[:, 2]]
        positive_mask[:selected_count] = utility_out[:selected_count] > positive_threshold
        null_mask[:selected_count] = ~positive_mask[:selected_count]
        valid_mask[:selected_count] = True
        if effect_out is not None:
            effect_out[:selected_count] = effect_values[
                selected[:, 0], selected[:, 1], selected[:, 2]
            ]

    return DelayBalancedPairBatch(
        batch_index=batch_index,
        event_index=event_index,
        future_index=future_index,
        delay=delay,
        delay_bin=delay_bin,
        utility=utility_out,
        positive_mask=positive_mask,
        null_mask=null_mask,
        valid_mask=valid_mask,
        teacher_effect=effect_out,
        delay_edges=used_edges,
    )


# A concise alias used by the benchmark scripts.
delay_balanced_pair_sampling = sample_delay_balanced_pairs


# ---------------------------------------------------------------------------
# Query-conditioned local effect objective
# ---------------------------------------------------------------------------


def _broadcast_prefix(value: Tensor, prefix: torch.Size, *, name: str) -> Tensor:
    """Broadcast a per-pair tensor to ``prefix`` with useful diagnostics."""

    if value.ndim == 0:
        return value.expand(prefix)
    if tuple(value.shape) == tuple(prefix):
        return value
    candidates = [value]
    if value.ndim < len(prefix):
        # A leading [B] or trailing [T] vector is a common collator form.
        candidates.append(value.reshape(*value.shape, *([1] * (len(prefix) - value.ndim))))
        candidates.append(value.reshape(*([1] * (len(prefix) - value.ndim)), *value.shape))
    for candidate in candidates:
        try:
            return torch.broadcast_to(candidate, prefix)
        except RuntimeError:
            continue
    raise ValueError(f"{name} shape {tuple(value.shape)} is not broadcastable to {tuple(prefix)}")


def _broadcast_effect(effect: Tensor, target_shape: torch.Size, *, name: str) -> Tensor:
    """Broadcast an effect tensor while accepting a singleton pair axis."""

    if effect.ndim == len(target_shape) - 1 and effect.shape[-1] == target_shape[-1]:
        effect = effect.unsqueeze(-2)
    try:
        return torch.broadcast_to(effect, target_shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} shape {tuple(effect.shape)} is not broadcastable to {tuple(target_shape)}") from exc


def _robust_effect_scale(effect: Tensor, *, eps: float, floor: Tensor | float | None) -> Tensor:
    """Return detached per-row scales with a robust numerical floor."""

    if effect.ndim == 0:
        raise ValueError("effect must have a feature dimension")
    rms = effect.detach().square().mean(dim=-1).sqrt()
    if floor is None:
        positive = rms.reshape(-1)
        positive = positive[torch.isfinite(positive) & (positive > eps)]
        base = positive.median() if positive.numel() else effect.new_tensor(1.0)
        base = base.clamp_min(effect.new_tensor(1e-3))
    else:
        base = torch.as_tensor(floor, device=effect.device, dtype=effect.dtype).detach()
        if base.numel() != 1 or not bool(torch.isfinite(base).item()) or not bool((base > 0).item()):
            raise ValueError("normalization_floor must be finite and positive")
        base = base.reshape(()).clamp_min(effect.new_tensor(1e-3))
    return rms.clamp_min(base).unsqueeze(-1)


def _huber_per_pair(error: Tensor, *, delta: float) -> Tensor:
    """Mean Huber penalty over the final feature dimension."""

    if delta <= 0:
        raise ValueError("huber_delta must be positive")
    magnitude = error.abs()
    quadratic = 0.5 * magnitude.square() / delta
    linear = magnitude - 0.5 * delta
    return torch.where(magnitude <= delta, quadratic, linear).mean(dim=-1)


@dataclass(frozen=True)
class LocalEffectLossBreakdown:
    """Inspectable components of :func:`query_conditioned_local_effect_loss`."""

    total: Tensor
    positive: Tensor
    null: Tensor
    positive_count: Tensor
    null_count: Tensor
    student_effect: Tensor
    target_effect: Tensor


@dataclass(frozen=True)
class CausalMemoryDeploymentBreakdown:
    """Reader-side CMD objective and its auditable components.

    ``correct_action`` and ``wrong_action`` are generated by the deployed
    denoising flow under two *read-only* memory states.  The event write is
    therefore not trained by this objective; QH2L owns that credit path.
    Keeping the four terms separate makes it possible to report whether a
    gain comes from action distillation, causal effect matching, ranking, or
    null-memory invariance.
    """

    full: Tensor
    effect: Tensor
    rank: Tensor
    null: Tensor
    total: Tensor


def query_conditioned_local_effect_loss(
    student_before: Tensor,
    student_after: Tensor,
    teacher_effect: Tensor | None = None,
    *,
    utility: Tensor | float | None = None,
    positive_mask: Tensor | None = None,
    null_mask: Tensor | None = None,
    valid_mask: Tensor | None = None,
    positive_weight: Tensor | float | None = None,
    null_weight: Tensor | float | None = None,
    positive_denominator: Tensor | float | None = None,
    null_denominator: Tensor | float | None = None,
    null_loss_weight: Tensor | float = 1.0,
    normalization_floor: Tensor | float | None = None,
    relative: bool = True,
    huber_delta: float = 1.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
    return_components: bool = False,
    eps: float = _EPS,
) -> Tensor | LocalEffectLossBreakdown:
    """Train a writer from query-conditioned before/after readouts.

    ``student_before`` is the future-query readout before applying event ``i``'s
    local update; ``student_after`` is the readout after that update.  Their
    difference is matched to the detached hindsight teacher effect on positive
    pairs.  Null (zero/negative utility) pairs instead enforce an invariant
    readout, preventing arbitrary memory writes from changing unrelated
    actions.  Both branches are normalized by a detached robust action-effect
    scale, so the objective is stable across tasks and denoising coordinates.

    ``utility`` is optional.  If supplied, positive rows are ``utility > 0``
    and null rows are the complement; its positive magnitude softly weights the
    matching branch after clamping to ``[0, 1]``.  Explicit masks take
    precedence and are useful when labels already contain a sampled stratum.
    At least one of ``teacher_effect`` or no-positive rows is required: a
    positive pair cannot be supervised without a teacher target.

    The two non-empty strata are averaged separately and then averaged, making
    the loss insensitive to the arbitrary number of null rows sampled per
    delay bin.  ``positive_denominator`` and ``null_denominator`` optionally
    supply denominators computed over a complete episode/window.  They are
    used by TBPTT callers so summing segment losses is exactly equivalent to
    one full-window objective.  ``null_loss_weight`` is an explicit coefficient
    on the null/invariance stratum; unlike a per-row sampling weight it does
    not cancel from the normalized mean.  ``reduction='none'`` returns one
    loss per pair (feature axis removed), with invalid rows set to zero.
    """

    if not isinstance(student_before, Tensor) or not isinstance(student_after, Tensor):
        raise TypeError("student_before and student_after must be tensors")
    if student_before.ndim == 0 or student_after.ndim == 0:
        raise ValueError("student readouts must have a feature dimension")
    if student_before.shape != student_after.shape:
        raise ValueError(
            "student_before and student_after must have identical shapes, got "
            f"{tuple(student_before.shape)} and {tuple(student_after.shape)}"
        )
    if not torch.is_floating_point(student_before) or not torch.is_floating_point(student_after):
        raise TypeError("student readouts must be floating-point tensors")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"unknown reduction: {reduction!r}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    prefix = student_before.shape[:-1]
    device = student_before.device
    dtype = student_before.dtype
    student_effect = student_after - student_before
    if teacher_effect is None:
        target_effect = torch.zeros_like(student_effect)
    else:
        if not isinstance(teacher_effect, Tensor) or teacher_effect.ndim == 0:
            raise ValueError("teacher_effect must be a tensor with a feature dimension")
        target_effect = _broadcast_effect(teacher_effect, student_before.shape, name="teacher_effect")
        target_effect = target_effect.detach().to(device=device, dtype=dtype)
    target_effect = torch.where(torch.isfinite(target_effect), target_effect, torch.zeros_like(target_effect))

    def _mask(value: Tensor | None, *, name: str) -> Tensor | None:
        if value is None:
            return None
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, device=device)
        return _broadcast_prefix(value.to(device=device, dtype=torch.bool), prefix, name=name)

    valid = _mask(valid_mask, name="valid_mask")
    if valid is None:
        valid = torch.ones(prefix, dtype=torch.bool, device=device)
    valid = valid & torch.isfinite(student_effect).all(dim=-1)
    valid = valid & torch.isfinite(target_effect).all(dim=-1)

    utility_tensor: Tensor | None = None
    if utility is not None:
        utility_tensor = torch.as_tensor(utility, device=device, dtype=dtype)
        utility_tensor = _broadcast_prefix(utility_tensor, prefix, name="utility").detach()
        utility_tensor = torch.where(torch.isfinite(utility_tensor), utility_tensor, torch.zeros_like(utility_tensor))

    positive = _mask(positive_mask, name="positive_mask")
    null = _mask(null_mask, name="null_mask")
    positive_was_explicit = positive is not None
    null_was_explicit = null is not None
    if positive is None:
        if null_was_explicit and utility_tensor is None:
            positive = ~null
        elif utility_tensor is not None:
            positive = utility_tensor > 0
        else:
            positive = (
                torch.ones(prefix, dtype=torch.bool, device=device)
                if teacher_effect is not None
                else torch.zeros(prefix, dtype=torch.bool, device=device)
            )
    if null is None:
        if positive_was_explicit and utility_tensor is None:
            null = ~positive
        else:
            null = ~positive if utility_tensor is not None or teacher_effect is None else torch.zeros_like(positive)
    # An explicitly supplied stratum mask wins over an inferred utility sign.
    # This is useful when the label builder reserves a small uncertainty band
    # around zero and marks it as null by construction.
    if null_was_explicit and not positive_was_explicit:
        positive = positive & ~null
    if positive_was_explicit and not null_was_explicit:
        null = null & ~positive
    if bool((positive & null).any().item()):
        raise ValueError("positive_mask and null_mask must be disjoint")
    # Rows not explicitly assigned to either branch are ignored, which lets a
    # caller retain a third 'uncertain' stratum without inventing a target.
    positive = positive & valid
    null = null & valid

    if bool(positive.any().item()) and teacher_effect is None:
        raise ValueError("teacher_effect is required when positive pairs are present")

    if relative:
        scale = _robust_effect_scale(target_effect, eps=eps, floor=normalization_floor)
    else:
        scale = torch.ones_like(target_effect[..., :1])
    effect_error = _huber_per_pair((student_effect - target_effect) / scale, delta=huber_delta)
    null_error = _huber_per_pair(student_effect / scale, delta=huber_delta)

    if positive_weight is None:
        pos_weight = positive.to(dtype=dtype)
        if utility_tensor is not None:
            # Symmetric relative utility is normally in [-2, 2].  Clamping,
            # rather than introducing a temperature, keeps weighting bounded.
            pos_weight = utility_tensor.clamp_min(0).clamp_max(1) * positive.to(dtype=dtype)
    else:
        pos_weight = _broadcast_prefix(
            torch.as_tensor(positive_weight, device=device, dtype=dtype), prefix, name="positive_weight"
        )
        pos_weight = pos_weight.clamp_min(0) * positive.to(dtype=dtype)
    if null_weight is None:
        null_weight_tensor = null.to(dtype=dtype)
    else:
        null_weight_tensor = _broadcast_prefix(
            torch.as_tensor(null_weight, device=device, dtype=dtype), prefix, name="null_weight"
        )
        null_weight_tensor = null_weight_tensor.clamp_min(0) * null.to(dtype=dtype)

    null_loss_weight_tensor = torch.as_tensor(
        null_loss_weight, device=device, dtype=dtype
    )
    if (
        null_loss_weight_tensor.numel() != 1
        or not bool(torch.isfinite(null_loss_weight_tensor).item())
    ):
        raise ValueError("null_loss_weight must be a finite scalar")
    null_loss_weight_tensor = null_loss_weight_tensor.reshape(()).clamp_min(0)

    positive_term = effect_error * pos_weight
    null_term = null_error * null_weight_tensor
    if reduction == "none":
        total = positive_term + null_term
        positive_out = positive_term
        null_out = null_term
        positive_count = positive.to(dtype=dtype)
        null_count = null.to(dtype=dtype)
    else:
        if positive_denominator is None:
            pos_den = pos_weight.sum().clamp_min(1)
            global_has_pos = pos_weight.sum() > 0
        else:
            raw_pos_den = torch.as_tensor(positive_denominator, device=device, dtype=dtype)
            if raw_pos_den.numel() != 1 or not bool(torch.isfinite(raw_pos_den).item()):
                raise ValueError("positive_denominator must be a finite scalar")
            raw_pos_den = raw_pos_den.reshape(())
            if bool((raw_pos_den < 0).item()):
                raise ValueError("positive_denominator must be non-negative")
            pos_den = raw_pos_den.clamp_min(1)
            global_has_pos = raw_pos_den > 0
        if null_denominator is None:
            null_den = null_weight_tensor.sum().clamp_min(1)
            global_has_null = null_weight_tensor.sum() > 0
        else:
            raw_null_den = torch.as_tensor(null_denominator, device=device, dtype=dtype)
            if raw_null_den.numel() != 1 or not bool(torch.isfinite(raw_null_den).item()):
                raise ValueError("null_denominator must be a finite scalar")
            raw_null_den = raw_null_den.reshape(())
            if bool((raw_null_den < 0).item()):
                raise ValueError("null_denominator must be non-negative")
            null_den = raw_null_den.clamp_min(1)
            global_has_null = raw_null_den > 0
        pos_value = positive_term.sum() / pos_den
        null_value = null_term.sum() / null_den
        if bool(global_has_pos.item()) and bool(global_has_null.item()):
            total = 0.5 * (pos_value + null_loss_weight_tensor * null_value)
        elif bool(global_has_pos.item()):
            total = pos_value
        elif bool(global_has_null.item()):
            total = null_loss_weight_tensor * null_value
        else:
            # Keep a zero connected to the student graph so an empty/padded
            # minibatch can safely call backward.
            total = student_effect.sum() * 0.0
        if reduction == "sum":
            # ``sum`` is defined as the sum of the two normalized strata; it is
            # still invariant to duplicate null rows and is useful for logs.
            total = pos_value + null_loss_weight_tensor * null_value
        positive_out = pos_value
        null_out = null_value
        positive_count = positive.to(dtype=dtype).sum()
        null_count = null.to(dtype=dtype).sum()

    if return_components:
        return LocalEffectLossBreakdown(
            total=total,
            positive=positive_out,
            null=null_out,
            positive_count=positive_count,
            null_count=null_count,
            student_effect=student_effect,
            target_effect=target_effect,
        )
    return total


def causal_memory_deployment_loss(
    correct_action: Tensor,
    wrong_action: Tensor,
    *,
    teacher_full_action: Tensor | None = None,
    teacher_wrong_action: Tensor | None = None,
    teacher_effect: Tensor | None = None,
    expert_action: Tensor | None = None,
    utility: Tensor | float | None = None,
    positive_mask: Tensor | None = None,
    null_mask: Tensor | None = None,
    valid_mask: Tensor | None = None,
    margin: float = 0.05,
    null_weight: float = 0.25,
    full_denominator: Tensor | float | None = None,
    positive_denominator: Tensor | float | None = None,
    null_denominator: Tensor | float | None = None,
    huber_delta: float = 1.0,
    reduction: Literal["mean", "sum"] = "mean",
    return_components: bool = False,
    eps: float = _EPS,
) -> Tensor | CausalMemoryDeploymentBreakdown:
    """Train/audit the causal reader with read-only memory interventions.

    The two action tensors are generated by the *same* denoising flow and
    differ only in the memory supplied to its final selected TTT layer.  The
    function deliberately treats that state as detached from the event writer:
    gradients train the reader/action tail, while QH2L separately trains the
    event write.  ``teacher_full_action`` and ``teacher_wrong_action`` are
    optional because older pair artifacts contain only ``teacher_effect``.

    Four terms implement the CMD contract from the method definition:

    ``full``
        Distill the correct-memory action from the full-history teacher.
    ``effect``
        Match the correct-minus-wrong action effect on positive pairs.
    ``rank``
        Require the correct memory to be at least ``margin`` better than the
        wrong memory on the fixed expert action target.
    ``null``
        Enforce invariance to memory on low-utility/irrelevant pairs.

    No term introduces a global memory-off scalar.  Empty strata return a
    graph-connected zero, which is important for delay-stratified minibatches.
    Optional complete-window denominators make the sum of per-segment CMD
    losses invariant to the TBPTT partition.
    """

    if not isinstance(correct_action, Tensor) or not isinstance(wrong_action, Tensor):
        raise TypeError("correct_action and wrong_action must be tensors")
    if correct_action.shape != wrong_action.shape or correct_action.ndim == 0:
        raise ValueError(
            "correct_action and wrong_action must have identical non-scalar shapes; "
            f"got {tuple(correct_action.shape)} and {tuple(wrong_action.shape)}"
        )
    if not correct_action.is_floating_point() or not wrong_action.is_floating_point():
        raise TypeError("CMD actions must be floating-point tensors")
    if margin < 0 or null_weight < 0 or huber_delta <= 0 or eps <= 0:
        raise ValueError("margin/null_weight must be non-negative and huber_delta/eps positive")
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"unknown reduction: {reduction!r}")

    prefix = correct_action.shape[:-1]
    device, dtype = correct_action.device, correct_action.dtype

    def aligned(value: Tensor | None, name: str) -> Tensor | None:
        if value is None:
            return None
        if not isinstance(value, Tensor) or value.ndim == 0:
            raise ValueError(f"{name} must be a non-scalar tensor")
        result = _broadcast_effect(value, correct_action.shape, name=name)
        return result.detach().to(device=device, dtype=dtype)

    teacher_full = aligned(teacher_full_action, "teacher_full_action")
    teacher_wrong = aligned(teacher_wrong_action, "teacher_wrong_action")
    target_effect = aligned(teacher_effect, "teacher_effect")
    if target_effect is None and teacher_full is not None and teacher_wrong is not None:
        target_effect = teacher_full - teacher_wrong
    if target_effect is None:
        target_effect = torch.zeros_like(correct_action)
    target_effect = torch.where(
        torch.isfinite(target_effect), target_effect, torch.zeros_like(target_effect)
    )

    def mask(value: Tensor | None, name: str, default: bool) -> Tensor:
        if value is None:
            return torch.full(prefix, default, device=device, dtype=torch.bool)
        return _broadcast_prefix(
            torch.as_tensor(value, device=device, dtype=torch.bool), prefix, name=name
        )

    valid = mask(valid_mask, "valid_mask", True)
    valid = valid & torch.isfinite(correct_action).all(dim=-1) & torch.isfinite(wrong_action).all(dim=-1)
    utility_tensor: Tensor | None = None
    if utility is not None:
        utility_tensor = _broadcast_prefix(
            torch.as_tensor(utility, device=device, dtype=dtype), prefix, name="utility"
        ).detach()
        utility_tensor = torch.where(
            torch.isfinite(utility_tensor), utility_tensor, torch.zeros_like(utility_tensor)
        )
    positive = mask(positive_mask, "positive_mask", False) if positive_mask is not None else (
        utility_tensor > 0 if utility_tensor is not None else valid
    )
    null = mask(null_mask, "null_mask", False) if null_mask is not None else ~positive
    positive = positive & valid
    null = null & valid & ~positive

    def per_row_squared(error: Tensor) -> Tensor:
        return error.square().mean(dim=-1)

    # Correct-memory full-history distillation is useful even when an artifact
    # has no explicit counterfactual action.  In that case the term is zero,
    # while effect/null/rank still provide a valid reader objective.
    full_error = (
        per_row_squared(correct_action - teacher_full)
        if teacher_full is not None
        else correct_action.sum(dim=-1) * 0.0
    )
    effect = correct_action - wrong_action
    effect_scale = _robust_effect_scale(target_effect, eps=eps, floor=None)
    effect_error = _huber_per_pair((effect - target_effect) / effect_scale, delta=huber_delta)
    null_error = _huber_per_pair(effect / effect_scale, delta=huber_delta)

    # Action ranking is defined only when an expert target is available.  The
    # target is detached by design (it is demonstration supervision, not a
    # deploy-time signal).
    if expert_action is not None:
        expert = aligned(expert_action, "expert_action")
        assert expert is not None
        correct_distance = per_row_squared(correct_action - expert)
        wrong_distance = per_row_squared(wrong_action - expert)
        rank_error = F.relu(float(margin) + correct_distance - wrong_distance)
    else:
        rank_error = correct_action.sum(dim=-1) * 0.0

    if utility_tensor is None:
        positive_weights = positive.to(dtype=dtype)
    else:
        positive_weights = utility_tensor.clamp_min(0).clamp_max(1) * positive.to(dtype=dtype)
    null_weights = null.to(dtype=dtype)
    full_weights = valid.to(dtype=dtype)

    def weighted_mean(
        values: Tensor,
        weights: Tensor,
        denominator: Tensor | float | None = None,
    ) -> Tensor:
        numerator = (values * weights).sum()
        if denominator is None:
            normalizer = weights.sum().clamp_min(1.0)
        else:
            raw_normalizer = torch.as_tensor(denominator, device=device, dtype=dtype)
            if (
                raw_normalizer.numel() != 1
                or not bool(torch.isfinite(raw_normalizer).item())
                or bool((raw_normalizer < 0).item())
            ):
                raise ValueError("CMD normalization denominators must be finite and non-negative")
            normalizer = raw_normalizer.reshape(()).clamp_min(1.0)
        return numerator / normalizer

    full_term = weighted_mean(full_error, full_weights, full_denominator)
    effect_term = weighted_mean(effect_error, positive_weights, positive_denominator)
    rank_term = weighted_mean(rank_error, positive_weights, positive_denominator)
    null_term = weighted_mean(null_error, null_weights, null_denominator)
    # Do not let an absent stratum contribute a synthetic constant.  The
    # connected zero keeps autograd safe while preserving the declared terms.
    if not bool((positive_weights > 0).any().item()):
        effect_term = effect_term * 0.0
        rank_term = rank_term * 0.0
    if not bool(null.any().item()):
        null_term = null_term * 0.0
    total = full_term + effect_term + rank_term + float(null_weight) * null_term
    if reduction == "sum":
        total = total * max(int(valid.sum().item()), 1)
    if return_components:
        return CausalMemoryDeploymentBreakdown(
            full=full_term,
            effect=effect_term,
            rank=rank_term,
            null=null_term,
            total=total,
        )
    return total


# Naming variants used in the manuscript and benchmark scripts.
cmd_loss = causal_memory_deployment_loss
causal_memory_loss = causal_memory_deployment_loss


# Names used in the manuscript and in the first benchmark draft.
qh2l_loss = query_conditioned_local_effect_loss
query_conditioned_effect_loss = query_conditioned_local_effect_loss


# ---------------------------------------------------------------------------
# Functional, differentiable local TTT update/read helper
# ---------------------------------------------------------------------------


FastState = TypeVar("FastState")


def _flatten_state(state: Any) -> tuple[list[Tensor], Any]:
    """Flatten a tensor pytree and return a reconstruction specification.

    Fast-weight implementations in this repository use both tuples and
    dataclasses, while small experiments often use a dictionary.  Supporting
    tensors, mappings, dataclasses, lists, and tuples here keeps the QH2L primitive generic
    without depending on PyTorch's private pytree API.
    """

    leaves: list[Tensor] = []

    def visit(node: Any) -> Any:
        if isinstance(node, Tensor):
            index = len(leaves)
            leaves.append(node)
            return ("tensor", index)
        if isinstance(node, Mapping):
            return (
                "mapping",
                type(node),
                tuple((key, visit(value)) for key, value in node.items()),
            )
        if isinstance(node, tuple):
            return ("tuple", type(node), tuple(visit(value) for value in node))
        if isinstance(node, list):
            return ("list", tuple(visit(value) for value in node))
        if is_dataclass(node) and not isinstance(node, type):
            return (
                "dataclass",
                type(node),
                tuple((field.name, visit(getattr(node, field.name))) for field in fields(node)),
            )
        # Non-tensor leaves (typically ``None`` or an integer position) are
        # carried through unchanged.  They are state metadata, not updateable
        # fast weights, and accepting them lets callers pass TTTFastState
        # directly without a bespoke adapter.
        return ("constant", node)

    spec = visit(state)
    return leaves, spec


def _unflatten_state(spec: Any, leaves: Sequence[Tensor]) -> Any:
    """Reconstruct a state from :func:`_flatten_state`'s specification."""

    kind = spec[0]
    if kind == "tensor":
        return leaves[spec[1]]
    if kind == "mapping":
        mapping_type = spec[1]
        items = [(key, _unflatten_state(child, leaves)) for key, child in spec[2]]
        try:
            return mapping_type(items)
        except (TypeError, ValueError):
            return dict(items)
    if kind == "tuple":
        values = tuple(_unflatten_state(child, leaves) for child in spec[2])
        tuple_type = spec[1]
        if tuple_type is tuple:
            return values
        try:
            # Named tuples expect positional arguments rather than one tuple.
            return tuple_type(*values)
        except TypeError:
            return tuple_type(values)
    if kind == "list":
        return [_unflatten_state(child, leaves) for child in spec[1]]
    if kind == "dataclass":
        dataclass_type = spec[1]
        values = {name: _unflatten_state(child, leaves) for name, child in spec[2]}
        return dataclass_type(**values)
    if kind == "constant":
        return spec[1]
    raise RuntimeError(f"unknown state reconstruction tag: {kind!r}")


def functional_local_ttt_update(
    state: FastState,
    update_loss: Tensor,
    learning_rate: float | Tensor,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
    allow_unused: bool = True,
) -> FastState:
    """Apply one functional gradient update to a tensor fast-weight state.

    The returned state is a new pytree; the input is never modified.  When
    ``create_graph=True`` (the default), gradients of a future action loss can
    flow through the local update into the writer parameters, which is the
    differentiable QH2L path.  ``allow_unused`` leaves unrelated leaves
    unchanged rather than manufacturing NaNs.

    ``learning_rate`` may itself be a scalar tensor, allowing a caller to
    learn or schedule it.  It is intentionally not detached.  A non-scalar
    ``update_loss`` is reduced by mean to make accidental per-token losses
    safe at this boundary.
    """

    if not isinstance(update_loss, Tensor):
        raise TypeError("update_loss must be a tensor")
    if update_loss.ndim:
        update_loss = update_loss.mean()
    if not torch.is_floating_point(update_loss):
        raise TypeError("update_loss must be floating-point")
    leaves, spec = _flatten_state(state)
    requires_grad = [leaf for leaf in leaves if leaf.requires_grad]
    if not requires_grad or not update_loss.requires_grad:
        return state
    if isinstance(learning_rate, Tensor):
        lr = learning_rate.to(device=update_loss.device, dtype=update_loss.dtype)
    else:
        lr = torch.as_tensor(learning_rate, device=update_loss.device, dtype=update_loss.dtype)
    if lr.numel() != 1:
        raise ValueError("learning_rate must be a scalar")
    lr = lr.reshape(())
    if not bool(torch.isfinite(lr.detach()).item()) or not bool((lr.detach() >= 0).item()):
        raise ValueError("learning_rate must be finite and non-negative")
    gradients = torch.autograd.grad(
        update_loss,
        requires_grad,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )
    gradient_iter = iter(gradients)
    updated_leaves: list[Tensor] = []
    for leaf in leaves:
        if leaf.requires_grad:
            gradient = next(gradient_iter)
            if gradient is None:
                updated_leaves.append(leaf)
            else:
                updated_leaves.append(leaf - lr * gradient)
        else:
            updated_leaves.append(leaf)
    return _unflatten_state(spec, updated_leaves)


@dataclass(frozen=True)
class LocalTTTReadBeforeAfter(Generic[FastState]):
    """Result of one local update and a query-conditioned read."""

    state_before: FastState
    state_after: FastState
    read_before: Tensor
    read_after: Tensor
    update_loss: Tensor

    @property
    def effect(self) -> Tensor:
        """The differentiable action/readout effect of the local update."""

        return self.read_after - self.read_before

    @property
    def student_before(self) -> Tensor:
        """Alias consumed by :func:`query_conditioned_local_effect_loss`."""

        return self.read_before

    @property
    def student_after(self) -> Tensor:
        """Alias consumed by :func:`query_conditioned_local_effect_loss`."""

        return self.read_after


ReadFn = Callable[[FastState, Any], Tensor]


def local_update_read_before_after(
    state: FastState,
    query: Any,
    update_loss: Tensor | Callable[[FastState], Tensor],
    read_fn: ReadFn,
    learning_rate: float | Tensor,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
    allow_unused: bool = True,
) -> LocalTTTReadBeforeAfter[FastState]:
    """Compute a differentiable before/after read for one future query.

    ``read_fn`` receives ``(state, query)`` and should return an action-tail or
    other query-conditioned tensor.  ``update_loss`` may be precomputed or a
    callable evaluated on the unmodified state.  The helper applies a single
    gradient step to a cloned functional state, then reads the *same* query
    from both states.  It is therefore a direct implementation of the local
    QH2L target and does not replay intervening future events.
    """

    if not callable(read_fn):
        raise TypeError("read_fn must be callable")
    loss = update_loss(state) if callable(update_loss) else update_loss
    if not isinstance(loss, Tensor):
        raise TypeError("update_loss callable must return a tensor")
    before = read_fn(state, query)
    after_state = functional_local_ttt_update(
        state,
        loss,
        learning_rate,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )
    after = read_fn(after_state, query)
    if not isinstance(before, Tensor) or not isinstance(after, Tensor):
        raise TypeError("read_fn must return a tensor")
    if before.shape != after.shape:
        raise ValueError(
            f"read_fn returned different shapes before/after update: {tuple(before.shape)} vs {tuple(after.shape)}"
        )
    return LocalTTTReadBeforeAfter(
        state_before=state,
        state_after=after_state,
        read_before=before,
        read_after=after,
        update_loss=loss,
    )


# Explicit aliases make the operation easy to discover from either equation
# notation or implementation terminology.
differentiable_local_update = functional_local_ttt_update
local_ttt_read_before_after = local_update_read_before_after
query_conditioned_local_readout = local_update_read_before_after


__all__ = [
    "CREDIT_TTT_V3_FORMAT",
    "CREDIT_TTT_V3_PROTOCOL",
    "CREDIT_TTT_V3_PAIR_SCHEMA",
    "CREDIT_TTT_V3_INTERVENTION",
    "CREDIT_TTT_V3_INTERVENTION_SCOPE",
    "CREDIT_TTT_V3_TARGET",
    "CREDIT_TTT_V3_STATE",
    "CREDIT_TTT_DELAY_EDGES",
    "CREDIT_TTT_DELAY_BIN_LABELS",
    "CREDIT_TTT_FORMAT",
    "CREDIT_TTT_PROTOCOL",
    "CREDIT_TTT_PAIR_SCHEMA",
    "FULL_HISTORY_CONTROL_ATTRIBUTION",
    "QUERY_CONDITIONED_LOCAL_TTT",
    "CAUSAL_MEMORY_DEPLOYMENT",
    "CreditTTTProtocol",
    "DEFAULT_CREDIT_TTT_PROTOCOL",
    "InteractionFuturePair",
    "DelayBalancedPairBatch",
    "CreditPairBatch",
    "InteractionFuturePairBatch",
    "symmetric_relative_utility",
    "compute_symmetric_relative_utility",
    "relative_utility",
    "sample_delay_balanced_pairs",
    "delay_balanced_pair_sampling",
    "LocalEffectLossBreakdown",
    "query_conditioned_local_effect_loss",
    "qh2l_loss",
    "query_conditioned_effect_loss",
    "functional_local_ttt_update",
    "differentiable_local_update",
    "LocalTTTReadBeforeAfter",
    "local_update_read_before_after",
    "local_ttt_read_before_after",
    "query_conditioned_local_readout",
]
