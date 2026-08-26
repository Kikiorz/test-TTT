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


def _broadcast_prefix_mask(mask: Tensor, prefix: torch.Size, *, name: str) -> Tensor:
    """Broadcast a validity mask over a loss prefix.

    Dataset collators occasionally retain a singleton action/event axis (for
    example ``[B,T,1]`` for a per-frame mask) after the feature reduction has
    produced ``[B,T]`` losses.  Treat only trailing singleton axes as
    structural and remove them before ordinary PyTorch broadcasting; a
    non-singleton mismatch still raises instead of being guessed.
    """

    if not isinstance(mask, Tensor):
        mask = torch.as_tensor(mask)
    # Validity masks may arrive from numpy/JSON collators as float 0/1
    # tensors.  Normalize the dtype at the public boundary so ``torch.where``
    # and boolean indexing behave identically for bool and numeric masks.
    mask = mask.to(dtype=torch.bool)
    while mask.ndim > len(prefix) and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if tuple(mask.shape) == tuple(prefix):
        return mask

    # A reduced loss commonly has prefix ``[B,T]`` while a caller keeps an
    # explicit event/slot axis in the mask, whose target shape is
    # ``[B,T,K]``.  PyTorch aligns broadcast dimensions from the right, so a
    # bare ``[B,T]`` would incorrectly try to match ``T`` with ``K``.  Treat
    # a shape matching the *leading* prefix dimensions as a per-event mask and
    # append singleton axes explicitly.  The symmetric suffix form keeps
    # token-only masks (``[T]``) useful without guessing non-singleton axes.
    candidates: list[Tensor] = [mask]
    if mask.ndim < len(prefix):
        if tuple(mask.shape) == tuple(prefix[: mask.ndim]):
            candidates.append(mask.reshape(*mask.shape, *([1] * (len(prefix) - mask.ndim))))
        if mask.ndim and tuple(mask.shape) == tuple(prefix[-mask.ndim :]):
            candidates.append(mask.reshape(*([1] * (len(prefix) - mask.ndim)), *mask.shape))
        if mask.ndim == 1 and len(prefix) > 1:
            if mask.shape[0] == prefix[0]:
                candidates.append(mask.reshape(mask.shape[0], *([1] * (len(prefix) - 1))))
            if mask.shape[0] == prefix[-1]:
                candidates.append(mask.reshape(*([1] * (len(prefix) - 1)), mask.shape[0]))
    for candidate in candidates:
        try:
            return torch.broadcast_to(candidate, prefix)
        except RuntimeError:
            continue
    raise ValueError(f"{name} with shape {tuple(mask.shape)} is not broadcastable to {tuple(prefix)}")


def _broadcast_prefix_weight(weight: Tensor, prefix: torch.Size, *, name: str) -> Tensor:
    """Broadcast a sample/token weight to a tensor without a feature dimension.

    A one-dimensional ``[B]`` weight is interpreted as a batch weight when the
    loss has shape ``[B, T]``; a one-dimensional ``[T]`` weight is interpreted as
    a token weight when its length matches the final prefix dimension.
    """

    if weight.ndim == 0:
        return weight.expand(prefix)
    # Callers often retain a singleton feature/event axis (``[B,T,1]``)
    # while the reduced loss has prefix ``[B,T]``.  Remove only trailing
    # singleton axes; squeezing a non-singleton axis would silently change the
    # intended batch/token alignment.
    while weight.ndim > len(prefix) and weight.shape[-1] == 1:
        weight = weight.squeeze(-1)
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


def symmetric_relative_credit(
    full_loss: Tensor,
    masked_loss: Tensor,
    *,
    eps: float = _EPS,
) -> Tensor:
    """Return a scale-free signed counterfactual degradation.

    ``masked_loss - full_loss`` is the natural causal intervention score, but
    its magnitude changes with the task/action scale and with the denoising
    phase.  The symmetric denominator keeps the sign while bounding ordinary
    changes to a comparable range::

        d = (masked - full) / (0.5 * (|masked| + |full|) + eps)

    Positive values mean that removing the memory write hurts the expert
    action prediction (the history is useful); negative values mean that the
    intervention helped or that the write is harmful.  The operation is
    deliberately symmetric, so swapping the two branches changes only the
    sign.  It is useful for comparing episodes and is not a replacement for
    the raw loss difference, which should still be retained for auditing.
    """

    if not isinstance(full_loss, Tensor) or not isinstance(masked_loss, Tensor):
        raise TypeError("full_loss and masked_loss must be tensors")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not torch.is_floating_point(full_loss) or not torch.is_floating_point(masked_loss):
        raise TypeError("full_loss and masked_loss must be floating-point tensors")
    full_loss, masked_loss = torch.broadcast_tensors(full_loss, masked_loss)
    denominator = 0.5 * (full_loss.abs() + masked_loss.abs())
    return (masked_loss - full_loss) / denominator.clamp_min(eps)


def adaptive_topk_mean(
    values: Tensor,
    mask: Tensor | None = None,
    *,
    dim: int = -1,
) -> Tensor:
    """Average the strongest ``ceil(sqrt(n_valid))`` entries along ``dim``.

    Hindsight credit is often sparse and a plain maximum is extremely
    sensitive to one noisy future frame.  A fixed user-selected ``k`` has the
    opposite problem: its effect changes with episode length.  This reducer
    chooses the geometric, data-dependent compromise ``ceil(sqrt(n))``.  It
    is deterministic, has no task-specific tuning knob, and approaches a
    mean as the valid horizon grows while remaining robust to isolated
    outliers.  Invalid entries and empty rows contribute zero.
    """

    if values.ndim == 0:
        raise ValueError("values must have at least one dimension")
    dim = dim if dim >= 0 else values.ndim + dim
    if not 0 <= dim < values.ndim:
        raise ValueError(f"dim={dim} is out of range for values with {values.ndim} dimensions")
    valid = torch.ones_like(values, dtype=torch.bool) if mask is None else _broadcast_mask(
        mask, values.shape, name="mask"
    )
    finite = torch.isfinite(values)
    valid = valid & finite
    moved = values.movedim(dim, -1)
    moved_valid = valid.movedim(dim, -1)
    length = moved.shape[-1]
    if length == 0:
        return values.new_zeros(values.shape[:dim] + values.shape[dim + 1 :])
    safe = moved.masked_fill(~moved_valid, torch.finfo(moved.dtype).min)
    sorted_values = safe.sort(dim=-1, descending=True).values
    count = moved_valid.sum(dim=-1)
    k = torch.ceil(count.to(dtype=torch.float32).sqrt()).to(torch.long).clamp_min(1)
    ranks = torch.arange(length, device=values.device).reshape(
        *([1] * (sorted_values.ndim - 1)), length
    )
    selected = (ranks < k.unsqueeze(-1)) & (ranks < count.unsqueeze(-1))
    selected_values = torch.where(selected, sorted_values, torch.zeros_like(sorted_values))
    result = selected_values.sum(dim=-1) / k.to(dtype=values.dtype).clamp_min(1)
    result = torch.where(count > 0, result, torch.zeros_like(result))
    # ``movedim`` is its own inverse, but the reduced axis has disappeared;
    # the remaining dimensions are already in their original order.
    return result


def _masked_quantile(
    values: Tensor,
    mask: Tensor,
    *,
    dim: int,
    quantile: float,
) -> Tensor:
    """Compute a quantile while ignoring invalid entries without NaNs."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    dim = dim if dim >= 0 else values.ndim + dim
    moved = values.movedim(dim, -1)
    moved_mask = mask.movedim(dim, -1)
    length = moved.shape[-1]
    if length == 0:
        return values.new_zeros(values.shape[:dim] + values.shape[dim + 1 :])
    # Sorting ascending with -inf sentinels makes the valid values occupy the
    # final ``count`` positions.  Gather the rank relative to that suffix.
    safe = moved.masked_fill(~moved_mask, torch.finfo(moved.dtype).min)
    sorted_values = safe.sort(dim=-1, descending=False).values
    count = moved_mask.sum(dim=-1)
    rank = torch.floor((count.to(dtype=torch.float32) - 1).clamp_min(0) * quantile).to(torch.long)
    index = (length - count + rank).clamp(0, length - 1).unsqueeze(-1)
    result = sorted_values.gather(-1, index).squeeze(-1)
    return torch.where(count > 0, result, torch.zeros_like(result))


def robust_percentile_normalize(
    scores: Tensor,
    mask: Tensor | None = None,
    *,
    dim: int = -1,
    quantile: float = 0.9,
    eps: float = _EPS,
) -> Tensor:
    """Map non-negative scores to ``[0, 1]`` using a robust percentile scale.

    The denominator is the requested percentile of *valid positive* values,
    rather than the episode maximum.  Consequently one unusually large
    intervention cannot rescale every other event.  ``quantile=0.9`` is the
    protocol default; callers may expose it for ablations without changing
    the underlying causal definition.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    valid = torch.ones_like(scores, dtype=torch.bool) if mask is None else _broadcast_mask(
        mask, scores.shape, name="mask"
    )
    valid = valid & torch.isfinite(scores) & (scores > 0)
    scale = _masked_quantile(scores, valid, dim=dim, quantile=quantile).unsqueeze(dim)
    normalized = scores / scale.clamp_min(eps)
    return torch.where(valid, normalized.clamp(0, 1), torch.zeros_like(normalized))


def robust_signed_normalize(
    scores: Tensor,
    mask: Tensor | None = None,
    *,
    dim: int = -1,
    quantile: float = 0.9,
    eps: float = _EPS,
) -> Tensor:
    """Robustly normalize signed scores by a percentile of their magnitude."""

    magnitude = scores.abs()
    valid = torch.ones_like(scores, dtype=torch.bool) if mask is None else _broadcast_mask(
        mask, scores.shape, name="mask"
    )
    scale = _masked_quantile(magnitude, valid & torch.isfinite(magnitude) & (magnitude > 0), dim=dim, quantile=quantile)
    normalized = scores / scale.unsqueeze(dim).clamp_min(eps)
    return torch.where(valid & torch.isfinite(scores), normalized.clamp(-1, 1), torch.zeros_like(scores))


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
    # v2 signed/robust fields. Optional defaults preserve the legacy API.
    signed_C: Tensor | None = None
    harm_C: Tensor | None = None
    harm_u: Tensor | None = None
    harm_rho: Tensor | None = None
    normalization_quantile: float | None = None
    robust: bool = False

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

    @property
    def signed_attribution(self) -> Tensor:
        """Return signed pair credit, or ``C`` for legacy results."""

        return self.C if self.signed_C is None else self.signed_C

    @property
    def harmful_attribution(self) -> Tensor:
        """Return the non-negative harmful-write component."""

        if self.harm_C is None:
            return torch.zeros_like(self.C)
        return self.harm_C

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


def compute_robust_hindsight_attribution(
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
    quantile: float = 0.9,
    eps: float = _EPS,
) -> HindsightAttribution:
    """Compute the v2 scale-free, signed hindsight control attribution.

    This follows the same leakage-safe causal masking contract as
    :func:`compute_hindsight_attribution`, but replaces the raw positive hinge
    with :func:`symmetric_relative_credit` and robust aggregation.  For each
    event/future pair it retains the signed relative degradation; ``C`` remains
    the positive (useful-memory) part for backward-compatible gate labels, and
    the negative/harmful part is exposed as ``harm_C``.  Event and future
    aggregates use :func:`adaptive_topk_mean`, and their normalized forms use
    a robust percentile rather than an episode maximum.

    ``pair_mask`` is expected to encode the exact event intervals when events
    are not one-step consecutive (as in the MIKASA block collector).  When no
    explicit mask is supplied, the ordinary causal event-block mask is built
    exactly as in the legacy function.  If an explicit mask is supplied it is
    trusted after finite-value filtering; callers constructing it from event
    starts/ends should therefore ensure it is causal.

    The returned object is still a :class:`HindsightAttribution`, so old code
    can unpack ``C, u, rho``.  New code can inspect ``signed_C``, ``harm_C``,
    ``harm_u`` and ``harm_rho``.
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
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    full_loss, masked_loss = torch.broadcast_tensors(full_loss, masked_loss)
    if pair_mask is not None and valid_pair_mask is not None:
        raise ValueError("pass at most one of pair_mask and valid_pair_mask")
    if valid_pair_mask is not None:
        pair_mask = valid_pair_mask

    if pair_mask is None:
        if episode_ids is not None or episode_lengths is not None or valid_mask is not None:
            mask = build_episode_event_block_mask(
                episode_lengths,
                episode_ids=episode_ids,
                valid_mask=valid_mask,
                max_steps=full_loss.shape[-1],
                event_block_size=event_block_size,
                include_event_end=include_event_end,
            ).to(device=full_loss.device)
            mask = _broadcast_mask(mask, full_loss.shape, name="episode pair mask")
        else:
            mask = _default_pair_mask(
                full_loss.shape,
                event_block_size=event_block_size,
                include_event_end=include_event_end,
                device=full_loss.device,
            )
    else:
        # Unlike the generic legacy helper, an explicit mask may describe
        # non-consecutive event blocks.  Do not intersect it with an index-
        # based default relation; this is what lets the offline builder pass
        # the true ``event_end`` boundaries without leaking the current block.
        mask = _broadcast_mask(pair_mask.to(device=full_loss.device), full_loss.shape, name="pair_mask")
    signed = symmetric_relative_credit(full_loss, masked_loss, eps=eps)
    mask = mask & torch.isfinite(signed)
    positive = torch.where(mask, signed.clamp_min(0), torch.zeros_like(signed))
    harm = torch.where(mask, (-signed).clamp_min(0), torch.zeros_like(signed))
    positive_mask = mask & (positive > 0)
    harm_mask = mask & (harm > 0)

    # The adaptive reducer is applied to positive/harmful magnitudes
    # separately.  This preserves the sign information without allowing a
    # large harmful branch to cancel a useful branch before normalization.
    # Count only non-zero evidence when choosing the adaptive top-sqrt set.
    # Including every causal zero would make ``k`` depend on the remaining
    # episode horizon and dilute a sparse but real effect; positive and harmful
    # evidence are therefore reduced under their own masks.
    row_raw = adaptive_topk_mean(positive, positive_mask, dim=-1)
    column_raw = adaptive_topk_mean(positive, positive_mask, dim=-2)
    harm_row_raw = adaptive_topk_mean(harm, harm_mask, dim=-1)
    harm_column_raw = adaptive_topk_mean(harm, harm_mask, dim=-2)
    row = robust_percentile_normalize(row_raw, row_raw > 0, dim=-1, quantile=quantile, eps=eps)
    column = robust_percentile_normalize(
        column_raw, column_raw > 0, dim=-1, quantile=quantile, eps=eps
    )
    harm_row = robust_percentile_normalize(
        harm_row_raw, harm_row_raw > 0, dim=-1, quantile=quantile, eps=eps
    )
    harm_column = robust_percentile_normalize(
        harm_column_raw, harm_column_raw > 0, dim=-1, quantile=quantile, eps=eps
    )
    # Keep only causal/finite entries in the exposed matrices.  The positive
    # ``C`` field retains exactly the old semantic expected by HCA/gate code.
    signed = torch.where(mask, signed, torch.zeros_like(signed))
    positive = torch.where(mask, positive, torch.zeros_like(positive))
    harm = torch.where(mask, harm, torch.zeros_like(harm))
    return HindsightAttribution(
        C=positive,
        u=row,
        rho=column,
        pair_mask=mask,
        full_history_loss=full_loss,
        masked_history_loss=masked_loss,
        row_importance_raw=row_raw,
        column_dependency_raw=column_raw,
        signed_C=signed,
        harm_C=harm,
        harm_u=harm_row,
        harm_rho=harm_column,
        normalization_quantile=float(quantile),
        robust=True,
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


@dataclass(frozen=True)
class ActionEffectDistillationBreakdown:
    """Components of the scale-normalized action-effect objective.

    ``effect`` is the high-dependency branch matching term and ``invariance``
    is the low-dependency term.  Keeping the two values separate makes it
    possible to report whether a run learned *what* a memory changes or merely
    collapsed the true/wrong branches to the same output.
    """

    total: Tensor
    effect: Tensor
    invariance: Tensor


def compute_action_effect_normalization_floor(
    teacher_effect: Tensor,
    *,
    eps: float = _EPS,
) -> Tensor:
    """Return the detached robust RMS floor used by action-effect distillation.

    The statistic is intentionally separable from
    :func:`action_effect_distillation_loss`: a sequence trainer can compute it
    once over the complete physical window and reuse it for every TBPTT
    segment.  Rows retain their own teacher RMS; this scalar only lower-bounds
    those per-row scales.  Consequently truncation boundaries cannot change a
    row's normalization while large effects still normalize by their own
    magnitude.

    Non-finite coordinates are treated as padded zeros, matching the public
    loss.  The median is taken over positive row-wise RMS values.  An all-zero
    window uses unit scale, and the small numerical lower bound preserves the
    historical behavior for sub-quantization effects.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    if not isinstance(teacher_effect, Tensor) or teacher_effect.ndim == 0:
        raise ValueError("teacher_effect must have a feature dimension")
    safe_effect = teacher_effect.detach()
    safe_effect = torch.where(
        torch.isfinite(safe_effect),
        safe_effect,
        torch.zeros_like(safe_effect),
    )
    teacher_rms = safe_effect.square().mean(dim=-1).sqrt()
    flat_rms = teacher_rms.reshape(-1)
    positive_rms = flat_rms[torch.isfinite(flat_rms) & (flat_rms > eps)]
    positive_floor = safe_effect.new_tensor(1e-3)
    if positive_rms.numel():
        floor = positive_rms.median().clamp_min(positive_floor)
    else:
        floor = safe_effect.new_tensor(1.0)
    return floor.detach()


def action_effect_distillation_loss(
    student_true: Tensor,
    student_wrong: Tensor,
    teacher_true: Tensor | None = None,
    teacher_wrong: Tensor | None = None,
    *,
    teacher_effect: Tensor | None = None,
    importance: Tensor | float = 1.0,
    valid_mask: Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
    return_components: bool = False,
    normalization_floor: Tensor | float | None = None,
    eps: float = _EPS,
) -> Tensor | ActionEffectDistillationBreakdown:
    """Distill *how memory changes the action*, with writer gradients intact.

    This is the H2L content/effect term.  Unlike
    :func:`counterfactual_grounding_loss`, it is intended to be called on
    student branches whose fast-weight update graph is still connected to the
    writer.  The teacher is detached, while both student branches receive
    gradients.  A dimensionless robust scale (the median non-zero teacher
    effect RMS) prevents a single large action coordinate or task from
    dominating the objective.  ``normalization_floor`` may provide that
    scalar from the complete trajectory; when omitted, it is computed from
    this call's teacher batch for backward compatibility.

    ``d_s = student_true - student_wrong`` and
    ``d_t = teacher_true - teacher_wrong`` (or the explicit ``teacher_effect``)
    are compared when ``importance`` is high.  For low-importance rows the
    objective instead enforces ``d_s ~= 0``; this prevents arbitrary memory
    writes from changing actions.  ``importance`` broadcasts over every
    non-feature dimension and is clamped to ``[0, 1]``.  The optional
    ``valid_mask`` is useful for padded action slots and multi-event branches.

    The loss uses a unit-beta Huber penalty after robust per-batch scaling.  No
    task/action-unit hyperparameter is needed, and changing the outer loss
    weight changes optimization emphasis rather than the causal target.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    tensors = (student_true, student_wrong)
    if any(not isinstance(value, Tensor) or value.ndim == 0 for value in tensors):
        raise ValueError("student_true and student_wrong must have a feature dimension")
    if student_true.shape != student_wrong.shape:
        raise ValueError(
            "student_true and student_wrong must have identical shapes, got "
            f"{tuple(student_true.shape)} and {tuple(student_wrong.shape)}"
        )
    has_teacher_pair = teacher_true is not None and teacher_wrong is not None
    if (teacher_effect is not None and has_teacher_pair) or (
        teacher_effect is None and not has_teacher_pair
    ):
        # Exactly one representation must be supplied.  The explicit effect is
        # convenient for compact [B,T,K,D] label artifacts.
        raise ValueError("provide teacher_effect or both teacher_true and teacher_wrong")
    if teacher_effect is None:
        assert teacher_true is not None and teacher_wrong is not None
        if teacher_true.shape != teacher_wrong.shape:
            raise ValueError("teacher_true and teacher_wrong must have identical shapes")
        teacher_effect = teacher_true - teacher_wrong
    if not isinstance(teacher_effect, Tensor) or teacher_effect.ndim == 0:
        raise ValueError("teacher_effect must have a feature dimension")
    # A single selected-event label is commonly stored as ``[..., D]`` while
    # a v2 student may evaluate a fixed event axis ``[..., K, D]``.  Insert
    # that singleton event axis before broadcasting; all other rank changes
    # are rejected rather than guessed.
    if teacher_effect.ndim == student_true.ndim - 1 and teacher_effect.shape[-1] == student_true.shape[-1]:
        teacher_effect = teacher_effect.unsqueeze(-2)
    try:
        teacher_effect = torch.broadcast_to(teacher_effect, student_true.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"teacher_effect shape {tuple(teacher_effect.shape)} is not broadcastable to "
            f"student shape {tuple(student_true.shape)}"
        ) from exc
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unknown reduction: {reduction!r}")

    student_effect = student_true - student_wrong
    teacher_effect = teacher_effect.detach().to(
        device=student_effect.device,
        dtype=student_effect.dtype,
    )
    # Offline artifacts are expected to be finite, but sanitizing here keeps a
    # malformed padded row from poisoning every normalized row in the segment.
    # The corresponding ``valid_mask`` still decides whether that row
    # contributes to the objective.
    teacher_effect = torch.where(
        torch.isfinite(teacher_effect),
        teacher_effect,
        torch.zeros_like(teacher_effect),
    )
    # A robust floor is computed from the teacher batch only and detached from
    # the optimization graph.  It prevents tiny effects from generating huge
    # gradients while retaining relative magnitude information for meaningful
    # branches.  ``median`` is stable under a few outlier interventions.  The
    # action head is trained in a normalized coordinate system.  A small
    # fixed numerical floor (1e-3) keeps ordinary effects such as 0.02 from
    # being washed out, while preventing sub-quantization noise from creating
    # huge gradients.  In particular, an all-zero teacher effect must *not*
    # use ``eps`` as its scale: doing so would turn the useful low-dependency
    # invariance term into an enormous gradient and make the result appear
    # hyper-parameter sensitive.  For that degenerate batch we use unit scale,
    # which remains finite and still teaches the student that an intervention
    # with no measured effect should not change the action.
    teacher_rms = teacher_effect.square().mean(dim=-1).sqrt()
    positive_floor = teacher_effect.new_tensor(1e-3)
    if normalization_floor is None:
        floor = compute_action_effect_normalization_floor(
            teacher_effect,
            eps=eps,
        )
    else:
        floor = torch.as_tensor(
            normalization_floor,
            device=teacher_effect.device,
            dtype=teacher_effect.dtype,
        ).detach()
        if floor.numel() != 1:
            raise ValueError("normalization_floor must be a scalar")
        floor = floor.reshape(())
        if not bool(torch.isfinite(floor).item()) or not bool((floor > 0).item()):
            raise ValueError("normalization_floor must be finite and positive")
        floor = floor.clamp_min(positive_floor)
    scale = teacher_rms.clamp_min(floor).unsqueeze(-1)
    normalized_delta_error = (student_effect - teacher_effect) / scale
    normalized_invariance = student_effect / scale

    def _unit_huber(error: Tensor) -> Tensor:
        magnitude = error.abs()
        return torch.where(magnitude < 1, 0.5 * magnitude.square(), magnitude - 0.5).mean(dim=-1)

    effect_error = _unit_huber(normalized_delta_error)
    invariance_error = _unit_huber(normalized_invariance)
    importance_tensor = torch.as_tensor(
        importance,
        device=effect_error.device,
        dtype=effect_error.dtype,
    ).clamp(0, 1)
    importance_tensor = _broadcast_prefix_weight(
        importance_tensor,
        effect_error.shape,
        name="importance",
    )
    per_element = importance_tensor * effect_error + (1 - importance_tensor) * invariance_error
    valid: Tensor | None = None
    if valid_mask is not None:
        valid = _broadcast_prefix_mask(
            valid_mask.to(device=per_element.device),
            per_element.shape,
            name="valid_mask",
        )
        per_element = torch.where(valid, per_element, torch.zeros_like(per_element))
        weights = valid.to(dtype=per_element.dtype)
    else:
        weights = None
    if reduction == "none":
        total = per_element
        effect_out = torch.where(
            valid if valid is not None else torch.ones_like(effect_error, dtype=torch.bool),
            importance_tensor * effect_error,
            torch.zeros_like(effect_error),
        )
        invariance_out = torch.where(
            valid if valid is not None else torch.ones_like(invariance_error, dtype=torch.bool),
            (1 - importance_tensor) * invariance_error,
            torch.zeros_like(invariance_error),
        )
    elif reduction == "sum":
        total = per_element.sum()
        effect_out = (importance_tensor * effect_error * (weights if weights is not None else 1)).sum()
        invariance_out = ((1 - importance_tensor) * invariance_error * (weights if weights is not None else 1)).sum()
    else:
        if weights is None:
            denominator = per_element.new_tensor(float(per_element.numel())).clamp_min(1)
            total = per_element.sum() / denominator
            effect_out = (importance_tensor * effect_error).mean()
            invariance_out = ((1 - importance_tensor) * invariance_error).mean()
        else:
            denominator = weights.sum().clamp_min(1)
            total = per_element.sum() / denominator
            effect_out = (importance_tensor * effect_error * weights).sum() / denominator
            invariance_out = ((1 - importance_tensor) * invariance_error * weights).sum() / denominator
    if return_components:
        return ActionEffectDistillationBreakdown(total, effect_out, invariance_out)
    return total


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
    "ActionEffectDistillationBreakdown",
    "CounterfactualGroundingBreakdown",
    "EpisodeEvent",
    "HindsightAttribution",
    "HindsightAttributionComputer",
    "HindsightAttributionResult",
    "HindsightPair",
    "build_episode_event_block_mask",
    "build_event_block_mask",
    "compute_hindsight_attribution",
    "compute_robust_hindsight_attribution",
    "compute_action_effect_normalization_floor",
    "counterfactual_grounding_loss",
    "action_effect_distillation_loss",
    "adaptive_topk_mean",
    "iter_hindsight_pairs",
    "local_kvb_loss",
    "normalize_importance",
    "normalize_scores",
    "normalise_importance",
    "normalise_scores",
    "robust_percentile_normalize",
    "robust_signed_normalize",
    "symmetric_relative_credit",
    "safe_episode_event_block_mask",
]
