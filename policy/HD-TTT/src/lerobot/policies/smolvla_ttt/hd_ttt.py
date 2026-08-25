"""Tensor-only building blocks for Hindsight-Distilled TTT (HD-TTT).

The functions in this module deliberately do not know anything about SmolVLA,
the action expert, or a particular fast-weight implementation.  They operate on
already-computed tensors and are therefore useful both in the offline teacher
pipeline and in unit tests.  The deployment code can keep only the local K/V/B
objective; hindsight tensors and counterfactual branches are training-only.

Conventions
------------
An attribution matrix uses the final two dimensions ``[event, future]``.  Any
dimensions before those two (normally a batch dimension) are preserved.  A
velocity tensor uses the same event/future dimensions followed by one or more
action dimensions, for example ``[B, I, J, action_dim]``.  ``rho`` is indexed
by future time and ``u`` by event time.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor


_EPS = 1e-8


def _as_tensor(value: Tensor | Sequence[int] | int, *, device: torch.device | None = None) -> Tensor:
    """Convert a length-like value without needlessly copying tensors."""

    if isinstance(value, Tensor):
        return value.to(device=device) if device is not None else value
    return torch.as_tensor(value, device=device)


def _broadcast_mask(mask: Tensor, shape: torch.Size, *, name: str) -> Tensor:
    """Broadcast a boolean mask to ``shape`` with a useful error message."""

    mask = mask.to(dtype=torch.bool)
    try:
        return torch.broadcast_to(mask, shape)
    except RuntimeError as exc:
        raise ValueError(f"{name} with shape {tuple(mask.shape)} is not broadcastable to {tuple(shape)}") from exc


def _broadcast_prefix_weight(weight: Tensor, prefix: torch.Size, *, name: str) -> Tensor:
    """Broadcast a sample/token weight to a tensor without a feature dimension.

    A one-dimensional ``[B]`` weight is interpreted as a batch weight when the
    loss has shape ``[B, T]``; a one-dimensional ``[T]`` weight is interpreted as
    a token weight when its length matches the final prefix dimension.
    """

    if weight.ndim == 0:
        return weight.expand(prefix)
    if tuple(weight.shape) == tuple(prefix):
        return weight

    candidates: list[Tensor] = [weight]
    # Common batch and token forms.
    if weight.ndim == 1 and len(prefix) > 1:
        if weight.shape[0] == prefix[0]:
            candidates.append(weight.reshape(weight.shape[0], *([1] * (len(prefix) - 1))))
        if weight.shape[0] == prefix[-1]:
            candidates.append(weight.reshape(*([1] * (len(prefix) - 1)), weight.shape[0]))
    if weight.ndim < len(prefix):
        candidates.append(weight.reshape(*([1] * (len(prefix) - weight.ndim)), *weight.shape))
        candidates.append(weight.reshape(*weight.shape, *([1] * (len(prefix) - weight.ndim))))

    for candidate in candidates:
        try:
            return torch.broadcast_to(candidate, prefix)
        except RuntimeError:
            continue
    raise ValueError(f"{name} with shape {tuple(weight.shape)} is not broadcastable to {tuple(prefix)}")


def normalize_scores(
    scores: Tensor,
    mask: Tensor | None = None,
    *,
    dim: int = -1,
    mode: Literal["max", "sum", "mean", "softmax"] = "max",
    eps: float = _EPS,
) -> Tensor:
    """Normalize non-negative scores along ``dim`` while respecting a mask.

    ``max`` (the default) maps the largest valid score to one, ``sum`` and
    ``mean`` provide mass-normalized alternatives, and ``softmax`` is useful
    when a probability distribution over events is desired.  Invalid entries
    are always returned as zero.  Empty rows are handled safely and return
    zeros rather than NaNs.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    if scores.ndim == 0:
        return scores.clone()
    dim = dim if dim >= 0 else scores.ndim + dim
    if not 0 <= dim < scores.ndim:
        raise ValueError(f"dim={dim} is out of range for scores with {scores.ndim} dimensions")
    valid = torch.ones_like(scores, dtype=torch.bool) if mask is None else _broadcast_mask(mask, scores.shape, name="mask")
    safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
    if mode == "max":
        denominator = safe_scores.amax(dim=dim, keepdim=True)
        normalized = safe_scores / denominator.clamp_min(eps)
    elif mode == "sum":
        denominator = safe_scores.sum(dim=dim, keepdim=True)
        normalized = safe_scores / denominator.clamp_min(eps)
    elif mode == "mean":
        denominator = safe_scores.sum(dim=dim, keepdim=True)
        count = valid.sum(dim=dim, keepdim=True).to(dtype=scores.dtype).clamp_min(1)
        normalized = safe_scores / (denominator / count).clamp_min(eps)
    elif mode == "softmax":
        logits = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        normalized = torch.softmax(logits, dim=dim)
        normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}")
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def normalize_importance(
    scores: Tensor,
    mask: Tensor | None = None,
    *,
    dim: int = -1,
    mode: Literal["max", "sum", "mean", "softmax"] = "max",
    eps: float = _EPS,
) -> Tensor:
    """Alias with terminology used for ``u`` and ``rho`` weights."""

    return normalize_scores(scores, mask, dim=dim, mode=mode, eps=eps)


# British spelling is convenient when this utility is used from papers/code
# written with ``normalise``; retaining the alias costs nothing and avoids
# duplicate implementations.
normalise_scores = normalize_scores
normalise_importance = normalize_importance


@dataclass(frozen=True)
class EpisodeEvent:
    """A half-open event interval ``[start, end)`` inside one episode."""

    start: int
    end: int
    episode_index: int = 0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Expected 0 <= start < end, got [{self.start}, {self.end})")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class HindsightPair:
    """One sparse event/future attribution pair.

    ``score`` is kept as a scalar tensor when produced by
    :func:`iter_hindsight_pairs`; this preserves dtype/device until a caller
    explicitly serializes the pair.
    """

    event_index: int
    future_index: int
    score: Tensor | float
    batch_index: int | None = None

    @property
    def i(self) -> int:
        """Short event-index alias used in equations."""

        return self.event_index

    @property
    def j(self) -> int:
        """Short future-index alias used in equations."""

        return self.future_index


def _lengths_to_episode_ids(lengths: Tensor, *, max_steps: int | None = None) -> tuple[Tensor, Tensor]:
    """Create packed episode ids and a valid-time mask from episode lengths."""

    if lengths.ndim == 0:
        lengths = lengths.reshape(1)
    if lengths.ndim != 1:
        raise ValueError(f"episode lengths must be one-dimensional, got {tuple(lengths.shape)}")
    if lengths.numel() == 0:
        raise ValueError("episode lengths cannot be empty")
    if lengths.dtype.is_floating_point:
        if not torch.equal(lengths, lengths.round()):
            raise ValueError("episode lengths must contain integers")
        lengths = lengths.round().to(torch.long)
    else:
        lengths = lengths.to(torch.long)
    if (lengths <= 0).any():
        raise ValueError("episode lengths must be positive")
    inferred_steps = int(lengths.max().item())
    steps = inferred_steps if max_steps is None else int(max_steps)
    if steps <= 0 or steps < inferred_steps:
        raise ValueError("max_steps must be at least the largest episode length")
    positions = torch.arange(steps, device=lengths.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    # Every padded position receives -1 and therefore cannot pair with anything.
    episode_ids = torch.arange(lengths.numel(), device=lengths.device).unsqueeze(1).expand(-1, steps)
    episode_ids = torch.where(valid, episode_ids, torch.full_like(episode_ids, -1))
    return episode_ids, valid


def build_episode_event_block_mask(
    episode_lengths: Tensor | Sequence[int] | int | None = None,
    *,
    episode_ids: Tensor | None = None,
    valid_mask: Tensor | None = None,
    max_steps: int | None = None,
    event_block_size: int = 1,
    include_event_end: bool = True,
) -> Tensor:
    """Build a safe causal event-to-future pair mask.

    The returned mask has shape ``[B, I, J]`` (or ``[I, J]`` when a one-
    dimensional ``episode_ids`` is supplied).  Event ``i`` represents the
    half-open block ``[i, i + event_block_size)`` clipped to the episode.  A
    future index is valid only when it belongs to the same episode and starts
    at/after the block end.  Thus the default ``event_block_size=1`` permits
    ``j > i`` and never leaks the current action into its own attribution.

    ``episode_lengths`` describes a padded batch of episodes.  Alternatively,
    pass explicit ``episode_ids`` with shape ``[B, T]`` (or ``[T]``) and an
    optional ``valid_mask``.  Padded ids should be ``-1``; they are always
    masked out.  This function is intentionally conservative: malformed
    lengths, negative block sizes, and mismatched masks raise ``ValueError``.
    """

    if event_block_size <= 0:
        raise ValueError("event_block_size must be positive")
    squeezed = False
    if episode_ids is None:
        if episode_lengths is None:
            raise ValueError("provide episode_lengths or episode_ids")
        lengths = _as_tensor(episode_lengths)
        episode_ids, inferred_valid = _lengths_to_episode_ids(lengths, max_steps=max_steps)
        valid_mask = inferred_valid if valid_mask is None else valid_mask
    else:
        if episode_ids.ndim not in (1, 2):
            raise ValueError(f"episode_ids must have shape [T] or [B, T], got {tuple(episode_ids.shape)}")
        if episode_ids.ndim == 1:
            episode_ids = episode_ids.unsqueeze(0)
            squeezed = True
        episode_ids = episode_ids.to(torch.long)
        inferred_valid = episode_ids >= 0
        valid_mask = inferred_valid if valid_mask is None else valid_mask

    if valid_mask is None:
        valid_mask = episode_ids >= 0
    if valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)
    valid_mask = _broadcast_mask(valid_mask, episode_ids.shape, name="valid_mask")
    if valid_mask.device != episode_ids.device:
        valid_mask = valid_mask.to(device=episode_ids.device)

    batch_size, steps = episode_ids.shape
    indices = torch.arange(steps, device=episode_ids.device)
    event_indices = indices[:, None]
    future_indices = indices[None, :]
    # ``end`` is exclusive. For a one-step event this is i+1, giving j > i.
    block_end = (event_indices + event_block_size).clamp_max(steps)
    temporal = future_indices >= block_end if include_event_end else future_indices > block_end
    same_episode = episode_ids[:, :, None] == episode_ids[:, None, :]
    mask = temporal.unsqueeze(0) & same_episode & valid_mask[:, :, None] & valid_mask[:, None, :]
    return mask[0] if squeezed else mask


def build_event_block_mask(*args: Any, **kwargs: Any) -> Tensor:
    """Compatibility alias for :func:`build_episode_event_block_mask`."""

    return build_episode_event_block_mask(*args, **kwargs)


def safe_episode_event_block_mask(*args: Any, **kwargs: Any) -> Tensor:
    """Explicitly named alias emphasizing leakage-safe masking."""

    return build_episode_event_block_mask(*args, **kwargs)


def _default_pair_mask(
    shape: torch.Size,
    *,
    event_block_size: int,
    include_event_end: bool,
    device: torch.device,
) -> Tensor:
    """Create a causal mask for an arbitrary leading batch shape."""

    if len(shape) < 2:
        raise ValueError("attribution tensors need at least event and future dimensions")
    events, futures = shape[-2:]
    if event_block_size <= 0:
        raise ValueError("event_block_size must be positive")
    event = torch.arange(events, device=device)[:, None]
    future = torch.arange(futures, device=device)[None, :]
    end = (event + event_block_size).clamp_max(futures)
    mask_2d = future >= end if include_event_end else future > end
    return mask_2d.reshape(*([1] * (len(shape) - 2)), events, futures).expand(shape)


def _infer_velocity_event_dim(velocity: Tensor, target: Tensor) -> int:
    """Infer the event axis from a velocity/target pair.

    Targets conventionally omit the event axis (``[..., future, action...]``),
    while velocity tensors contain it (``[..., event, future, action...]``).
    Removing each candidate axis and checking broadcastability handles both
    scalar actions and vector/multi-dimensional actions.  Candidates are
    visited from the right so a leading batch dimension with the same size as
    an event dimension is not mistaken for the event axis.
    """

    if velocity.ndim < 2:
        raise ValueError("teacher velocity tensors need event and future dimensions")
    candidates: list[tuple[int, int]] = []
    for index in range(velocity.ndim - 2, -1, -1):
        reduced_shape = velocity.shape[:index] + velocity.shape[index + 1 :]
        try:
            torch.broadcast_shapes(reduced_shape, target.shape)
        except RuntimeError:
            continue
        # The dimension immediately after an event is the future dimension;
        # all dimensions after that are action features.  Prefer the candidate
        # with the longest matching action suffix.  Without this tie-break, a
        # vector target [J, D] and a square [B, I=J, J, D] could incorrectly
        # remove the future (J) axis instead of the event (I) axis.
        action_rank = velocity.ndim - index - 2
        if target.ndim < action_rank + 1:
            continue
        target_future_index = target.ndim - action_rank - 1
        if velocity.shape[index + 1] != target.shape[target_future_index]:
            continue
        candidates.append((index, action_rank))
    if target.ndim < velocity.ndim and candidates:
        return max(candidates, key=lambda candidate: (candidate[1], -candidate[0]))[0]
    # A target with the same rank is already event-aligned, or the caller has
    # supplied scalar losses masquerading as velocities.  In both cases the
    # final two dimensions are the safest convention.
    event_dim = velocity.ndim - 2
    if event_dim < 0:
        raise ValueError("unable to infer event axis from velocity shape")
    return event_dim


def _reduce_velocity_loss(velocity: Tensor, target: Tensor) -> Tensor:
    """Convert velocity tensors to a scalar-loss matrix preserving leading dims."""

    if velocity.ndim < 2:
        raise ValueError("teacher velocity tensors need event and future dimensions")
    if target.ndim < 1:
        raise ValueError("target velocity must have at least one dimension")
    event_dim = _infer_velocity_event_dim(velocity, target)
    future_dim = event_dim + 1
    if target.ndim < velocity.ndim:
        # Target convention: [..., future, action...]. Broadcast it to the
        # velocity shape with the event dimension removed, then insert that
        # event axis. This also accepts a target shared across batch elements,
        # e.g. [J, D] for velocity [B, I, J, D].
        reduced_shape = velocity.shape[:event_dim] + velocity.shape[event_dim + 1 :]
        try:
            target = torch.broadcast_to(target, reduced_shape).unsqueeze(event_dim)
        except RuntimeError as exc:
            raise ValueError(
                f"target velocity shape {tuple(target.shape)} is not broadcastable to {tuple(velocity.shape)}"
            ) from exc
    elif target.ndim != velocity.ndim:
        raise ValueError(
            f"target velocity rank {target.ndim} is incompatible with velocity rank {velocity.ndim}"
        )
    try:
        target = torch.broadcast_to(target, velocity.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"target velocity shape {tuple(target.shape)} is not broadcastable to {tuple(velocity.shape)}"
        ) from exc
    squared = (velocity - target) ** 2
    # Reduce action dimensions only; preserve all dimensions through future.
    action_dims = tuple(range(future_dim + 1, squared.ndim))
    if action_dims:
        squared = squared.mean(dim=action_dims)
    return squared


@dataclass(frozen=True)
class HindsightAttribution:
    """Result of hindsight control attribution.

    ``C`` is the positive hinge ``relu(masked - full)`` and has shape
    ``[..., event, future]``.  ``u`` and ``rho`` are raw sums over future and
    event dimensions respectively unless ``normalize=True`` was requested.
    ``pair_mask`` records causal/episode/sparsity selection and is useful for
    weighting later writer updates.  The object is iterable, so existing code
    can write ``C, u, rho = compute_hindsight_attribution(...)``.
    """

    C: Tensor
    u: Tensor
    rho: Tensor
    pair_mask: Tensor
    full_history_loss: Tensor
    masked_history_loss: Tensor
    row_importance_raw: Tensor | None = None
    column_dependency_raw: Tensor | None = None

    @property
    def attribution(self) -> Tensor:
        """Long-form alias for ``C``."""

        return self.C

    @property
    def C_ij(self) -> Tensor:
        """Equation-style alias for the event/future credit matrix."""

        return self.C

    @property
    def row_importance(self) -> Tensor:
        """Long-form alias for ``u``."""

        return self.u

    @property
    def u_i(self) -> Tensor:
        """Equation-style alias for row/event importance."""

        return self.u

    @property
    def column_dependency(self) -> Tensor:
        """Long-form alias for ``rho``."""

        return self.rho

    @property
    def rho_j(self) -> Tensor:
        """Equation-style alias for future-column dependency."""

        return self.rho

    def __iter__(self) -> Iterable[Tensor]:
        yield self.C
        yield self.u
        yield self.rho

    def pairs(self, *, batch_index: int | None = None) -> list[HindsightPair]:
        """Materialize positive sparse pairs on the CPU for logging/scheduling."""

        return list(iter_hindsight_pairs(self.C, self.pair_mask, batch_index=batch_index))


# More explicit type name for callers that prefer to distinguish the result
# object from the computation function.  ``HindsightAttribution`` remains the
# short name used in the equations and README.
HindsightAttributionResult = HindsightAttribution


def iter_hindsight_pairs(
    attribution: Tensor,
    pair_mask: Tensor | None = None,
    *,
    batch_index: int | None = None,
) -> Iterable[HindsightPair]:
    """Yield non-zero event/future pairs from a possibly batched matrix."""

    if attribution.ndim < 2:
        raise ValueError("attribution must have event and future dimensions")
    mask = attribution > 0 if pair_mask is None else _broadcast_mask(pair_mask, attribution.shape, name="pair_mask")
    mask = mask & (attribution > 0)
    leading = attribution.shape[:-2]
    if not leading:
        indices = mask.nonzero(as_tuple=False)
        for i, j in indices.tolist():
            yield HindsightPair(i, j, attribution[i, j], batch_index=batch_index)
        return
    flat_attr = attribution.reshape(-1, *attribution.shape[-2:])
    flat_mask = mask.reshape_as(flat_attr)
    for flat_index, matrix_mask in enumerate(flat_mask):
        leading_index = flat_index
        if batch_index is not None and leading_index != batch_index:
            continue
        for i, j in matrix_mask.nonzero(as_tuple=False).tolist():
            yield HindsightPair(i, j, flat_attr[flat_index, i, j], batch_index=leading_index)


def _resolve_hindsight_inputs(
    full_history_loss: Tensor | None,
    masked_history_loss: Tensor | None,
    target_velocity: Tensor | None,
    *,
    full_teacher_velocity: Tensor | None,
    masked_teacher_velocity: Tensor | None,
    teacher_target_velocity: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Return full/masked scalar losses, computing them from velocities if needed."""

    velocity_target = teacher_target_velocity if teacher_target_velocity is not None else target_velocity
    # A third positional ``target_velocity`` is a convenient shorthand for the
    # two positional velocity tensors.  It is unambiguous because a target is
    # not needed when scalar losses are supplied.
    if full_teacher_velocity is None and masked_teacher_velocity is None and velocity_target is not None:
        if full_history_loss is None or masked_history_loss is None:
            raise ValueError("full and masked teacher velocities are both required")
        full_teacher_velocity, masked_teacher_velocity = full_history_loss, masked_history_loss
        full_history_loss = masked_history_loss = None

    if full_teacher_velocity is not None or masked_teacher_velocity is not None:
        if full_teacher_velocity is None or masked_teacher_velocity is None or velocity_target is None:
            raise ValueError("full_teacher_velocity, masked_teacher_velocity, and target_velocity are required together")
        if full_teacher_velocity.shape[: -1] != masked_teacher_velocity.shape[: -1]:
            raise ValueError("full and masked teacher velocity tensors must have compatible shapes")
        # Locate event/future dimensions from the pair itself.  For canonical
        # [.., I, J, D] this is the final two dimensions before D.
        if full_teacher_velocity.ndim < 2:
            raise ValueError("teacher velocity tensors need event and future dimensions")
        full_loss = _reduce_velocity_loss(full_teacher_velocity, velocity_target)
        masked_loss = _reduce_velocity_loss(masked_teacher_velocity, velocity_target)
        return full_loss, masked_loss

    if full_history_loss is None or masked_history_loss is None:
        raise ValueError("provide scalar losses or full/masked teacher velocities")
    if full_history_loss.shape != masked_history_loss.shape:
        raise ValueError(
            f"full_history_loss shape {tuple(full_history_loss.shape)} does not match "
            f"masked_history_loss shape {tuple(masked_history_loss.shape)}"
        )
    if full_history_loss.ndim < 2:
        raise ValueError("scalar loss tensors need event and future dimensions")
    return full_history_loss, masked_history_loss


def compute_hindsight_attribution(
    full_history_loss: Tensor | None = None,
    masked_history_loss: Tensor | None = None,
    target_velocity: Tensor | None = None,
    *,
    full_teacher_velocity: Tensor | None = None,
    masked_teacher_velocity: Tensor | None = None,
    teacher_target_velocity: Tensor | None = None,
    pair_mask: Tensor | None = None,
    valid_pair_mask: Tensor | None = None,
    episode_lengths: Tensor | Sequence[int] | int | None = None,
    episode_ids: Tensor | None = None,
    valid_mask: Tensor | None = None,
    event_block_size: int = 1,
    include_event_end: bool = True,
    top_k: int | None = None,
    threshold: float | None = None,
    normalize: bool = False,
    normalization: Literal["max", "sum", "mean", "softmax"] = "max",
    eps: float = _EPS,
) -> HindsightAttribution:
    """Compute leakage-safe hindsight control credit.

    For scalar losses, ``C[i, j] = relu(masked_history_loss[i, j] -
    full_history_loss[i, j])``.  To use teacher velocities instead, pass
    ``full_teacher_velocity``, ``masked_teacher_velocity`` and
    ``teacher_target_velocity``; their squared action error is reduced over
    action dimensions before applying the same hinge.  A shorthand positional
    form is ``compute_hindsight_attribution(full_velocity, masked_velocity,
    target_velocity)``.

    By default only causal pairs after an event block are retained.  ``pair_mask``
    (or its explicit alias ``valid_pair_mask``) can further restrict pairs.
    ``top_k``
    selects at most ``k`` positive events **per future column**, while
    ``threshold`` removes scores below the supplied hinge value.  Both filters
    are applied before calculating ``u`` and ``rho``.  Set ``normalize=True``
    to max/sum/mean/softmax-normalize the two aggregates; raw aggregates remain
    available as ``row_importance_raw`` and ``column_dependency_raw``.
    """

    full_loss, masked_loss = _resolve_hindsight_inputs(
        full_history_loss,
        masked_history_loss,
        target_velocity,
        full_teacher_velocity=full_teacher_velocity,
        masked_teacher_velocity=masked_teacher_velocity,
        teacher_target_velocity=teacher_target_velocity,
    )
    if not torch.is_floating_point(full_loss) or not torch.is_floating_point(masked_loss):
        raise TypeError("history losses/velocities must be floating-point tensors")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if threshold is not None and threshold < 0:
        raise ValueError("threshold must be non-negative because C is a positive hinge")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    full_loss, masked_loss = torch.broadcast_tensors(full_loss, masked_loss)
    if pair_mask is not None and valid_pair_mask is not None:
        raise ValueError("pass at most one of pair_mask and valid_pair_mask")
    if valid_pair_mask is not None:
        pair_mask = valid_pair_mask
    C = torch.relu(masked_loss - full_loss)
    if pair_mask is None:
        if episode_ids is not None or episode_lengths is not None or valid_mask is not None:
            # Episode utility naturally returns [B, I, J].  If the loss has
            # extra leading dimensions, broadcasting below handles them.
            mask = build_episode_event_block_mask(
                episode_lengths,
                episode_ids=episode_ids,
                valid_mask=valid_mask,
                max_steps=C.shape[-1],
                event_block_size=event_block_size,
                include_event_end=include_event_end,
            )
            mask = mask.to(device=C.device)
            mask = _broadcast_mask(mask, C.shape, name="episode pair mask")
        else:
            mask = _default_pair_mask(
                C.shape,
                event_block_size=event_block_size,
                include_event_end=include_event_end,
                device=C.device,
            )
    else:
        mask = _broadcast_mask(pair_mask.to(device=C.device), C.shape, name="pair_mask")
        # A supplied mask is still intersected with the safe causal relation.
        mask = mask & _default_pair_mask(
            C.shape,
            event_block_size=event_block_size,
            include_event_end=include_event_end,
            device=C.device,
        )
    mask = mask & torch.isfinite(C) & (C > 0)
    if threshold is not None:
        mask = mask & (C >= threshold)

    if top_k is not None:
        # Select the strongest event rows separately for every future column.
        # Flatten leading dimensions to keep this independent of batch rank.
        flat_C = C.reshape(-1, C.shape[-2], C.shape[-1])
        flat_mask = mask.reshape_as(flat_C)
        sparse_mask = torch.zeros_like(flat_mask)
        k = min(top_k, C.shape[-2])
        values = flat_C.masked_fill(~flat_mask, torch.finfo(flat_C.dtype).min)
        top_values, top_indices = torch.topk(values, k=k, dim=-2)
        sparse_mask.scatter_(-2, top_indices, top_values > torch.finfo(flat_C.dtype).min / 2)
        mask = sparse_mask.reshape_as(C)
    C = torch.where(mask, C, torch.zeros_like(C))

    row_raw = C.sum(dim=-1)
    column_raw = C.sum(dim=-2)
    if normalize:
        row_valid = mask.any(dim=-1)
        column_valid = mask.any(dim=-2)
        u = normalize_importance(row_raw, row_valid, dim=-1, mode=normalization, eps=eps)
        rho = normalize_importance(column_raw, column_valid, dim=-1, mode=normalization, eps=eps)
    else:
        u, rho = row_raw, column_raw
    return HindsightAttribution(
        C=C,
        u=u,
        rho=rho,
        pair_mask=mask,
        full_history_loss=full_loss,
        masked_history_loss=masked_loss,
        row_importance_raw=row_raw,
        column_dependency_raw=column_raw,
    )


class HindsightAttributionComputer:
    """Configurable callable wrapper around :func:`compute_hindsight_attribution`."""

    def __init__(self, **defaults: Any) -> None:
        self.defaults = dict(defaults)

    def __call__(self, *args: Any, **kwargs: Any) -> HindsightAttribution:
        options = {**self.defaults, **kwargs}
        return compute_hindsight_attribution(*args, **options)


def local_kvb_loss(
    query: Tensor | None,
    key: Tensor,
    value: Tensor,
    prediction: Tensor,
    write_gate: Tensor | None = None,
    *,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Return the local deployable K/V/B writer objective.

    The objective is ``g / 2 * ||prediction - stop_gradient(value)||^2``.
    ``key`` and ``query`` are accepted to make the K/V/B call site explicit;
    ``prediction`` is assumed to be ``f_W(key)`` and the query is used later by
    the reader, so neither is differentiated through by this utility.  Feature
    dimensions may differ (the fast map can be rectangular), but all prefix
    dimensions must agree.  ``write_gate`` may be scalar, ``[B]``, ``[T]``, or
    any shape broadcastable to the prefix dimensions.

    ``reduction='none'`` returns one loss per prefix element (features are
    averaged); the other reductions return a scalar.
    """

    if key.ndim == 0 or value.ndim == 0 or prediction.ndim == 0:
        raise ValueError("key, value, and prediction must have a feature dimension")
    if value.shape != prediction.shape:
        raise ValueError(f"value shape {tuple(value.shape)} must match prediction shape {tuple(prediction.shape)}")
    if key.shape[:-1] != value.shape[:-1]:
        raise ValueError(
            f"key prefix {tuple(key.shape[:-1])} must match value/prediction prefix {tuple(value.shape[:-1])}"
        )
    if query is not None and query.shape[:-1] != key.shape[:-1]:
        raise ValueError(f"query prefix {tuple(query.shape[:-1])} must match key prefix {tuple(key.shape[:-1])}")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unknown reduction: {reduction!r}")
    prefix = value.shape[:-1]
    per_element = 0.5 * (prediction - value.detach()).square().mean(dim=-1)
    if write_gate is not None:
        gate = _broadcast_prefix_weight(write_gate.to(device=per_element.device, dtype=per_element.dtype), prefix, name="write_gate")
        per_element = per_element * gate
    if reduction == "none":
        return per_element
    return per_element.mean() if reduction == "mean" else per_element.sum()


@dataclass(frozen=True)
class CounterfactualGroundingBreakdown:
    """Optional components returned by :func:`counterfactual_grounding_loss`."""

    total: Tensor
    direction: Tensor
    invariance: Tensor


def counterfactual_grounding_loss(
    student_true: Tensor,
    student_wrong: Tensor,
    teacher_true: Tensor,
    teacher_wrong: Tensor,
    rho: Tensor | float,
    margin: float | Tensor = 0.0,
    *,
    reduction: Literal["mean", "sum", "none"] = "mean",
    return_components: bool = False,
) -> Tensor | CounterfactualGroundingBreakdown:
    """Ground the reader with correct/wrong-memory counterfactuals.

    Let ``d_s = student_true - student_wrong`` and ``d_t =
    stop_gradient(teacher_true - teacher_wrong)``.  High-dependency pairs use
    direction matching ``||d_s - d_t||²``; low-dependency pairs enforce
    invariance ``||d_s||²``.  ``rho`` is clamped to ``[0, 1]`` and broadcasts
    over all non-feature dimensions.  ``margin`` is a non-negative tolerance:
    errors within that element-wise radius are ignored (``margin=0`` gives the
    exact squared objectives above).  Teacher tensors are detached before any
    operation, so no gradient can flow into the hindsight teacher.
    """

    tensors = (student_true, student_wrong, teacher_true, teacher_wrong)
    if any(t.ndim == 0 for t in tensors):
        raise ValueError("student/teacher outputs must have a feature dimension")
    if not (student_true.shape == student_wrong.shape == teacher_true.shape == teacher_wrong.shape):
        raise ValueError(
            "student_true, student_wrong, teacher_true, and teacher_wrong must have identical shapes"
        )
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unknown reduction: {reduction!r}")
    if isinstance(margin, Tensor):
        if (margin < 0).any():
            raise ValueError("margin must be non-negative")
        margin_tensor = margin.to(device=student_true.device, dtype=student_true.dtype)
    else:
        if margin < 0:
            raise ValueError("margin must be non-negative")
        margin_tensor = torch.as_tensor(margin, device=student_true.device, dtype=student_true.dtype)

    delta_student = student_true - student_wrong
    delta_teacher = teacher_true.detach() - teacher_wrong.detach()
    direction_error = (delta_student - delta_teacher).abs().sub(margin_tensor).clamp_min(0).square().mean(dim=-1)
    invariance_error = delta_student.abs().sub(margin_tensor).clamp_min(0).square().mean(dim=-1)
    rho_tensor = torch.as_tensor(rho, device=student_true.device, dtype=student_true.dtype).clamp(0, 1)
    rho_tensor = _broadcast_prefix_weight(rho_tensor, direction_error.shape, name="rho")
    direction = rho_tensor * direction_error
    invariance = (1 - rho_tensor) * invariance_error
    per_element = direction + invariance
    if reduction == "none":
        total: Tensor = per_element
        direction_out, invariance_out = direction, invariance
    elif reduction == "sum":
        total = per_element.sum()
        direction_out, invariance_out = direction.sum(), invariance.sum()
    else:
        total = per_element.mean()
        direction_out, invariance_out = direction.mean(), invariance.mean()
    if return_components:
        return CounterfactualGroundingBreakdown(total, direction_out, invariance_out)
    return total


__all__ = [
    "CounterfactualGroundingBreakdown",
    "EpisodeEvent",
    "HindsightAttribution",
    "HindsightAttributionComputer",
    "HindsightAttributionResult",
    "HindsightPair",
    "build_episode_event_block_mask",
    "build_event_block_mask",
    "compute_hindsight_attribution",
    "counterfactual_grounding_loss",
    "iter_hindsight_pairs",
    "local_kvb_loss",
    "normalize_importance",
    "normalize_scores",
    "normalise_importance",
    "normalise_scores",
    "safe_episode_event_block_mask",
]
