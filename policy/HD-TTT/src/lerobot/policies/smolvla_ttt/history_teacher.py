"""Causal full-history teacher utilities for HD-TTT.

This module is deliberately independent from the deployed SmolVLA-TTT model.
It provides a small, explicit history encoder that can be used *only* while
building hindsight labels (or while training a teacher).  The encoder receives
one event token per physical observation, updates a recurrent state in causal
order, and emits one memory token that a caller may append to an action-expert
prefix.

The separation from :mod:`modeling_smolvla_ttt` is intentional:

* the ordinary student/checkpoint has no additional parameters or inputs;
* a teacher replay can branch by cloning a state and changing ``write_mask``;
* a long episode can be processed in chunks without resetting the history;
* no action, denoising noise, timestep, or future observation is consumed by
  this API.  The caller must construct ``event_tokens`` from the current
  observation/prefix only.

The core recurrence is a masked GRU update.  For a valid frame ``t`` it is

``h_t = h_{t-1} + m_t (GRU(e_t, h_{t-1}) - h_{t-1})``

where ``m_t`` is ``write_mask``.  Thus setting a block's mask to zero performs
an explicit event deletion while preserving the physical clock.  The output
memory token is produced from the post-update state, so it is available to the
action at the same frame; callers can request the pre-update state when an
exclude-current variant is needed.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


HISTORY_TEACHER_FORMAT = "causal_history_teacher_v1"
HISTORY_EVENT_SCHEMA = "masked_mean_observation_prefix"
HISTORY_DELETION_SCHEMA = "interpolated_gru_write_mask"

# The compact ``CausalHistoryTeacher`` above predates the V3 method and is
# intentionally kept byte-for-byte compatible with its original artifact
# format.  V3 uses a separate, action-supervised teacher.  Keeping a distinct
# format/schema makes it impossible to accidentally consume a memory-only GRU
# checkpoint as a full-history action teacher (or vice versa).
FULL_HISTORY_TEACHER_FORMAT = "explicit_full_history_action_teacher_v1"
FULL_HISTORY_EVENT_SCHEMA = "observation_prefix_plus_previous_executed_action"
FULL_HISTORY_INTERVENTION_SCHEMA = "event_content_replacement_or_deletion"


def history_teacher_state_sha256(teacher: nn.Module) -> str:
    """Hash a teacher's tensor state for label/checkpoint provenance.

    The digest is independent of parameter insertion order and device.  Names,
    dtypes, shapes, and raw contiguous bytes are included, so two teachers with
    the same architecture but different learned history dynamics cannot share
    a label artifact accidentally.
    """

    digest = hashlib.sha256()
    for name, tensor in sorted(teacher.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        # Viewing bytes avoids NumPy dtype limitations (notably bfloat16) and
        # keeps the digest valid for mixed-precision teacher checkpoints.
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _as_bool_mask(
    value: Tensor | None,
    shape: tuple[int, int],
    *,
    device: torch.device,
    name: str,
    default: bool,
) -> Tensor:
    """Broadcast a ``[B,T]`` mask and give errors at the API boundary."""

    if value is None:
        return torch.full(shape, default, dtype=torch.bool, device=device)
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, device=device)
    else:
        value = value.to(device=device)
    if value.ndim == 0:
        value = value.expand(shape)
    elif value.ndim == 1:
        if value.shape[0] == shape[0]:
            value = value[:, None].expand(shape)
        elif shape[0] == 1 and value.shape[0] == shape[1]:
            value = value[None, :]
        else:
            raise ValueError(
                f"{name} rank-1 value must match batch (or timesteps when B=1); "
                f"got {tuple(value.shape)} for {shape}"
            )
    elif value.shape != shape:
        try:
            value = torch.broadcast_to(value, shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name} with shape {tuple(value.shape)} is not broadcastable to {shape}"
            ) from error
    return value.bool()


def _as_write_mask(
    value: Tensor | None,
    shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Broadcast and validate a differentiable event write mask."""

    if value is None:
        return torch.ones(shape, dtype=dtype, device=device)
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=dtype, device=device)
    else:
        value = value.to(device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.expand(shape)
    elif value.ndim == 1:
        if value.shape[0] == shape[0]:
            value = value[:, None].expand(shape)
        elif shape[0] == 1 and value.shape[0] == shape[1]:
            value = value[None, :]
        else:
            raise ValueError(
                "write_mask rank-1 value must match batch "
                f"(or timesteps when B=1); got {tuple(value.shape)} for {shape}"
            )
    elif value.shape != shape:
        try:
            value = torch.broadcast_to(value, shape)
        except RuntimeError as error:
            raise ValueError(
                f"write_mask with shape {tuple(value.shape)} is not broadcastable to {shape}"
            ) from error
    if not torch.isfinite(value).all():
        raise ValueError("write_mask must contain only finite values")
    if bool((value < 0).any()) or bool((value > 1).any()):
        raise ValueError("write_mask values must lie in [0, 1]")
    return value


@dataclass(frozen=True)
class CausalHistoryState:
    """Recurrent state carried between chunks of one or more episodes.

    ``hidden`` has shape ``[B,H]`` and ``position`` has shape ``[B]``.  The
    position counts valid physical observations and is diagnostic only; it is
    intentionally retained so a caller can verify that chunked replay did not
    reset or skip history.
    """

    hidden: Tensor
    position: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.hidden.shape[0])

    def detach(self, *, requires_grad: bool = False) -> "CausalHistoryState":
        hidden = self.hidden.detach()
        if requires_grad and hidden.is_floating_point():
            hidden = hidden.requires_grad_(True)
        return CausalHistoryState(hidden, self.position.detach())

    def clone(
        self, *, detach: bool = False, requires_grad: bool = False
    ) -> "CausalHistoryState":
        hidden = self.hidden.detach().clone() if detach else self.hidden.clone()
        if requires_grad and hidden.is_floating_point():
            hidden.requires_grad_(True)
        return CausalHistoryState(hidden, self.position.detach().clone())


@dataclass(frozen=True)
class CausalHistoryOutput:
    """Output of :class:`CausalHistoryTeacher`.

    ``event_tokens`` are the projected current-prefix summaries and have shape
    ``[B,T,H]``.  ``memory_tokens`` have shape ``[B,T,M]`` and are suitable for
    appending as one extra prefix token per frame.  ``pre_memory_tokens`` is
    optional and is populated when ``return_pre_update=True``.
    """

    event_tokens: Tensor
    memory_tokens: Tensor
    state: CausalHistoryState
    pre_memory_tokens: Tensor | None = None


def summarize_prefix(
    prefix_embeddings: Tensor,
    prefix_valid_mask: Tensor | None = None,
) -> Tensor:
    """Reduce current observation-prefix embeddings to one event token.

    Args:
        prefix_embeddings: ``[B,P,D]`` for one frame or ``[B,T,P,D]`` for a
            sequence.  The returned shape is ``[B,D]`` or ``[B,T,D]``.
        prefix_valid_mask: Boolean mask over ``P`` (or ``[B,T,P]``).  ``True``
            means a real prefix token.  Padding is excluded from the mean.

    This helper intentionally has no argument for actions/noise/time.  It is
    therefore difficult for a caller to accidentally construct a teacher event
    from the denoising suffix while using the standard API.
    """

    if prefix_embeddings.ndim not in (3, 4):
        raise ValueError(
            "prefix_embeddings must have shape [B,P,D] or [B,T,P,D], got "
            f"{tuple(prefix_embeddings.shape)}"
        )
    if prefix_embeddings.shape[-2] <= 0:
        raise ValueError("prefix_embeddings must contain at least one prefix token")
    if not prefix_embeddings.is_floating_point():
        raise ValueError("prefix_embeddings must be a floating-point tensor")
    mask = prefix_valid_mask
    if mask is None:
        return prefix_embeddings.mean(dim=-2)
    if not isinstance(mask, Tensor):
        mask = torch.as_tensor(mask, device=prefix_embeddings.device)
    mask = mask.to(device=prefix_embeddings.device, dtype=torch.bool)
    expected = prefix_embeddings.shape[:-1]
    if mask.shape != expected:
        raise ValueError(
            "prefix_valid_mask must match all prefix_embeddings dimensions "
            f"except feature dim; got {tuple(mask.shape)} vs {tuple(expected)}"
        )
    weights = mask.to(dtype=prefix_embeddings.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=-2).clamp_min(1.0)
    return (prefix_embeddings * weights).sum(dim=-2) / denominator


class CausalHistoryTeacher(nn.Module):
    """Explicit causal recurrent teacher over observation-prefix event tokens.

    The module is *functional* with respect to trajectory state: no hidden
    state is stored on the object.  Pass the returned ``state`` into the next
    call to continue a long episode, or clone it to create an intervention
    branch.  This makes event-deletion counterfactuals deterministic and safe
    under dataloader workers/DDP.

    ``event_dim`` is the feature width of the current-prefix summary.  The
    hidden state is ``hidden_dim`` wide and is projected to ``memory_dim`` for
    the action-expert prefix.  A single GRUCell update is used per physical
    frame; there is no bidirectional or pooled future path.
    """

    def __init__(
        self,
        event_dim: int,
        hidden_dim: int,
        memory_dim: int | None = None,
        *,
        include_current: bool = True,
        deletion_mode: str = "skip",
    ) -> None:
        super().__init__()
        for name, value in (
            ("event_dim", event_dim),
            ("hidden_dim", hidden_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if memory_dim is None:
            memory_dim = hidden_dim
        if int(memory_dim) <= 0:
            raise ValueError("memory_dim must be positive")
        if deletion_mode not in {"skip", "null"}:
            raise ValueError("deletion_mode must be 'skip' or 'null'")

        self.event_dim = int(event_dim)
        self.hidden_dim = int(hidden_dim)
        self.memory_dim = int(memory_dim)
        self.include_current = bool(include_current)
        self.deletion_mode = deletion_mode

        self.event_norm = nn.LayerNorm(self.event_dim)
        self.event_projection = nn.Linear(self.event_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        # A learned null event is useful for the optional ``null`` deletion
        # ablation.  The default ``skip`` mode does not write it at all.
        self.null_event = nn.Parameter(torch.zeros(self.event_dim))
        self.memory_projection = nn.Linear(self.hidden_dim, self.memory_dim)
        self.memory_norm = nn.LayerNorm(self.memory_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> CausalHistoryState:
        """Create a zero state whose device/dtype follows the teacher by default."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        position = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        return CausalHistoryState(hidden, position)

    def _validate_state(
        self,
        state: CausalHistoryState | None,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CausalHistoryState:
        if state is None:
            return self.initial_state(batch_size, device=device, dtype=dtype)
        if state.hidden.ndim != 2 or state.hidden.shape != (batch_size, self.hidden_dim):
            raise ValueError(
                "state.hidden must have shape "
                f"[{batch_size}, {self.hidden_dim}], got {tuple(state.hidden.shape)}"
            )
        if state.position.ndim != 1 or state.position.shape[0] != batch_size:
            raise ValueError(
                f"state.position must have shape [{batch_size}], got {tuple(state.position.shape)}"
            )
        if state.hidden.device != device or state.position.device != device:
            raise ValueError("state and event_tokens must be on the same device")
        if state.hidden.dtype != dtype:
            state = CausalHistoryState(state.hidden.to(dtype=dtype), state.position)
        return state

    def forward(
        self,
        event_tokens: Tensor,
        *,
        state: CausalHistoryState | None = None,
        valid_mask: Tensor | None = None,
        reset_mask: Tensor | None = None,
        write_mask: Tensor | None = None,
        return_pre_update: bool = False,
    ) -> CausalHistoryOutput:
        """Encode a causal event sequence and return memory tokens.

        Args:
            event_tokens: ``[B,T,D]`` projected current-prefix events.  A
                ``[T,D]`` tensor is accepted as a convenience and treated as
                ``B=1``.
            state: State from the preceding contiguous chunk.  It is never
                mutated in-place.
            valid_mask: ``[B,T]`` physical-frame mask.  Invalid/padded rows do
                not advance state and emit a zero memory token.
            reset_mask: Reset before processing rows marked true.  Use this at
                episode boundaries when a batch contains multiple episodes.
            write_mask: Differentiable event-retention mask in ``[0,1]``.  A
                zero block is an event-deletion intervention.  The physical
                position still advances, while the hidden state is unchanged
                in ``skip`` mode (or processes ``null_event`` in ``null`` mode).
            return_pre_update: Also return the memory visible before each
                current event.  The standard teacher uses post-update memory.

        The recurrence is evaluated in Python over ``T`` so no operation can
        accidentally attend to a future event.  This is intentionally a small
        teacher-side cost relative to the VLM replay.
        """

        if event_tokens.ndim == 2:
            event_tokens = event_tokens.unsqueeze(0)
            squeezed = True
        elif event_tokens.ndim == 3:
            squeezed = False
        else:
            raise ValueError(
                f"event_tokens must have shape [T,D] or [B,T,D], got {tuple(event_tokens.shape)}"
            )
        if not event_tokens.is_floating_point():
            raise ValueError("event_tokens must be floating-point")
        batch_size, steps, event_dim = event_tokens.shape
        if steps <= 0:
            raise ValueError("event_tokens must contain at least one timestep")
        if event_dim != self.event_dim:
            raise ValueError(
                f"event_tokens feature dim must be {self.event_dim}, got {event_dim}"
            )
        device = event_tokens.device
        compute_dtype = self.event_projection.weight.dtype
        # Keep the recurrent state in the module's parameter dtype.  This is
        # stable under autocast and converts the final memory back to the input
        # dtype for painless prefix concatenation.
        events = event_tokens.to(dtype=compute_dtype)
        valid = _as_bool_mask(
            valid_mask,
            (batch_size, steps),
            device=device,
            name="valid_mask",
            default=True,
        )
        reset = _as_bool_mask(
            reset_mask,
            (batch_size, steps),
            device=device,
            name="reset_mask",
            default=False,
        )
        writes = _as_write_mask(
            write_mask,
            (batch_size, steps),
            device=device,
            dtype=compute_dtype,
        )
        state = self._validate_state(
            state,
            batch_size,
            device=device,
            dtype=compute_dtype,
        )

        hidden = state.hidden
        position = state.position
        event_outputs: list[Tensor] = []
        memory_outputs: list[Tensor] = []
        pre_memory_outputs: list[Tensor] = []
        for index in range(steps):
            valid_t = valid[:, index]
            reset_t = reset[:, index]
            write_t = writes[:, index] * valid_t.to(dtype=compute_dtype)

            # Reset is applied before reading the current event.  Clone rather
            # than mutate a caller-owned state so intervention branches remain
            # independent.
            hidden_before = torch.where(reset_t[:, None], torch.zeros_like(hidden), hidden)
            position_before = torch.where(
                reset_t,
                torch.full_like(position, -1),
                position,
            )
            pre_memory = self.memory_norm(self.memory_projection(hidden_before))

            current_event = events[:, index]
            if self.deletion_mode == "null":
                # In null mode a deleted event is replaced by a learned null
                # token and still performs a recurrent update.  ``skip`` is
                # the canonical HCA event-deletion semantics.
                null_event = self.null_event.to(dtype=compute_dtype, device=device)
                current_event = torch.where(
                    (write_t <= 0)[:, None],
                    null_event[None, :],
                    current_event,
                )
            normalized_event = self.event_norm(current_event)
            projected_event = self.event_projection(normalized_event)
            candidate = self.gru(projected_event, hidden_before)
            if self.deletion_mode == "null":
                # ``write_t`` controls whether a real event is retained.  A
                # null replacement is considered valid and gets a full update
                # when the physical frame is valid.
                update_weight = valid_t.to(dtype=compute_dtype)
            else:
                update_weight = write_t
            hidden_after = hidden_before + update_weight[:, None] * (candidate - hidden_before)
            # Invalid rows must not leak a previous episode's memory into a
            # padded prefix.  Their state/position remain unchanged.
            hidden = torch.where(valid_t[:, None], hidden_after, hidden_before)
            position = position_before + valid_t.to(dtype=position.dtype)

            visible_hidden = hidden if self.include_current else hidden_before
            memory = self.memory_norm(self.memory_projection(visible_hidden))
            memory = memory * valid_t[:, None].to(dtype=memory.dtype)
            event_outputs.append(projected_event)
            memory_outputs.append(memory)
            if return_pre_update:
                pre_memory_outputs.append(pre_memory * valid_t[:, None].to(dtype=pre_memory.dtype))

        event_output = torch.stack(event_outputs, dim=1)
        memory_output = torch.stack(memory_outputs, dim=1)
        pre_memory_output = (
            torch.stack(pre_memory_outputs, dim=1) if return_pre_update else None
        )
        output = CausalHistoryOutput(
            event_tokens=event_output,
            memory_tokens=memory_output,
            state=CausalHistoryState(hidden, position),
            pre_memory_tokens=pre_memory_output,
        )
        if squeezed:
            # Preserve the convenient [T,...] form for one unbatched episode;
            # the state remains batched because it is always consumed by the
            # next functional call.
            return CausalHistoryOutput(
                event_tokens=output.event_tokens[0],
                memory_tokens=output.memory_tokens[0],
                state=output.state,
                pre_memory_tokens=(
                    None
                    if output.pre_memory_tokens is None
                    else output.pre_memory_tokens[0]
                ),
            )
        return output

    def encode_prefix(
        self,
        prefix_embeddings: Tensor,
        prefix_valid_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> CausalHistoryOutput:
        """Summarize one/ many current prefixes and run :meth:`forward`."""

        events = summarize_prefix(prefix_embeddings, prefix_valid_mask)
        return self.forward(events, **kwargs)

    @staticmethod
    def deletion_write_mask(
        length: int,
        start: int,
        end: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return an all-write mask with the half-open event block deleted."""

        if length <= 0:
            raise ValueError("length must be positive")
        if not 0 <= start < end <= length:
            raise ValueError(f"expected 0 <= start < end <= {length}, got [{start}, {end})")
        mask = torch.ones(length, dtype=dtype, device=device)
        mask[start:end] = 0
        return mask

    def provenance(
        self,
        *,
        teacher_checkpoint: str | Path | None = None,
        source_policy_checkpoint: str | Path | None = None,
        event_schema: str = HISTORY_EVENT_SCHEMA,
    ) -> dict[str, Any]:
        """Return JSON-safe metadata that must accompany generated labels."""

        return {
            "format": HISTORY_TEACHER_FORMAT,
            "event_schema": str(event_schema),
            "deletion_schema": HISTORY_DELETION_SCHEMA,
            "causal": True,
            "include_current": self.include_current,
            "deletion_mode": self.deletion_mode,
            "event_dim": self.event_dim,
            "hidden_dim": self.hidden_dim,
            "memory_dim": self.memory_dim,
            "parameter_sha256": history_teacher_state_sha256(self),
            "teacher_checkpoint": None if teacher_checkpoint is None else str(teacher_checkpoint),
            "source_policy_checkpoint": (
                None if source_policy_checkpoint is None else str(source_policy_checkpoint)
            ),
        }


@dataclass(frozen=True)
class FullHistoryActionOutput:
    """Output of :class:`FullHistoryActionTeacher`.

    ``action_predictions`` has shape ``[B,T,A]`` when ``action_horizon=1``
    and ``[B,T,H,A]`` otherwise.  A ``[T,...]`` input is accepted by the
    teacher as a convenience and produces the corresponding unbatched output;
    the recurrent ``state`` remains batched so it can be passed to a later
    chunk.  ``hidden_states`` are the post-update causal states and are useful
    for auditing whether an intervention changed the latent memory rather than
    merely changing the action head.
    """

    event_tokens: Tensor
    previous_action_tokens: Tensor
    hidden_states: Tensor
    memory_tokens: Tensor
    action_predictions: Tensor
    state: CausalHistoryState
    pre_memory_tokens: Tensor | None = None
    pre_action_predictions: Tensor | None = None

    @property
    def actions(self) -> Tensor:
        """Short alias used by label-generation code."""

        return self.action_predictions

    @property
    def predicted_actions(self) -> Tensor:
        """Descriptive alias for :attr:`action_predictions`."""

        return self.action_predictions


@dataclass(frozen=True)
class PairwiseControlCredit:
    """Pairwise hindsight credit for event/future interventions.

    The canonical shapes are ``utility``/``raw_degradation``/``pair_mask``
    ``[B,I,J]`` and ``action_effect`` ``[B,I,J,...]``.  ``I`` is the number of
    intervened events and ``J`` the number of future queries.  ``utility`` is
    the positive, confidence-weighted increase in expert action loss; the
    signed raw degradation is retained so harmful interventions are auditable.
    """

    utility: Tensor
    action_effect: Tensor
    pair_mask: Tensor
    full_loss: Tensor
    counterfactual_loss: Tensor
    raw_degradation: Tensor
    confidence: Tensor

    @property
    def u_ij(self) -> Tensor:
        return self.utility

    @property
    def delta_a(self) -> Tensor:
        return self.action_effect


def _normalise_action_sequence(value: Tensor, *, name: str) -> tuple[Tensor, bool]:
    """Return an action sequence as ``[B,T,...]`` and whether it was squeezed."""

    if not isinstance(value, Tensor) or value.ndim < 2:
        raise ValueError(f"{name} must have shape [T,A] or [B,T,A(,...)]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim == 2:
        return value.unsqueeze(0), True
    return value, False


def _normalise_previous_action_sequence(
    value: Tensor,
    *,
    batch_size: int,
    steps: int,
    name: str = "previous_executed_actions",
) -> tuple[Tensor, bool]:
    """Normalize executed-action history to ``[B,T,A]``.

    Dataset action chunks are often stored as ``[B,T,H,A]`` even though only
    slot ``0`` was physically executed before the next observation.  The
    teacher's interaction token is explicitly that executed slot, so selecting
    it here is safer than flattening a chunk (which would silently change the
    feature width).  A rank-2 ``[T,A]`` tensor remains the convenient
    unbatched form.
    """

    if not isinstance(value, Tensor) or value.ndim < 2:
        raise ValueError(f"{name} must have shape [T,A], [B,T,A], or [B,T,H,A]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim == 2:
        if batch_size != 1 or value.shape[0] != steps:
            raise ValueError(
                f"unbatched {name} must have [{steps},A] shape; got {tuple(value.shape)}"
            )
        return value.unsqueeze(0), True
    if value.ndim == 3:
        if value.shape[:2] == (batch_size, steps):
            return value, False
        if batch_size == 1 and value.shape[0] == steps:
            # Unbatched chunk form [T,H,A].  This is necessarily ambiguous
            # when T=1; the canonical [B,T,A] interpretation wins above.
            if value.shape[1] <= 0:
                raise ValueError(f"{name} action chunks must contain at least one slot")
            return value[:, 0, :].unsqueeze(0), True
        raise ValueError(
            f"{name} must share [B,T]=[{batch_size},{steps}], got {tuple(value.shape[:2])}"
        )
    if value.ndim == 4:
        # Canonical batched action chunks: [B,T,H,A].  Keep the executed
        # first slot only; all later denoising slots are future predictions,
        # not causal writer inputs.
        if value.shape[:2] != (batch_size, steps) or value.shape[2] <= 0:
            raise ValueError(
                f"{name} chunk must have [B,T,H,A]=[{batch_size},{steps},H,A], "
                f"got {tuple(value.shape)}"
            )
        return value[:, :, 0, :], False
    raise ValueError(f"{name} must have rank 2, 3, or 4; got rank {value.ndim}")


def _align_action_target(
    predictions: Tensor,
    target: Tensor,
    *,
    name: str = "target_actions",
) -> Tensor:
    """Broadcast a target to an action prediction tensor without guessing axes."""

    if not isinstance(target, Tensor) or target.ndim == 0:
        raise ValueError(f"{name} must be a floating-point action tensor")
    if not target.is_floating_point():
        raise ValueError(f"{name} must be a floating-point action tensor")
    if target.shape[-1] != predictions.shape[-1]:
        raise ValueError(
            f"{name} feature dim {target.shape[-1]} does not match prediction dim "
            f"{predictions.shape[-1]}"
        )
    rank_difference = predictions.ndim - target.ndim
    if rank_difference == 1:
        if target.shape[:-1] == predictions.shape[:-2]:
            # [B,T,A] -> [B,T,1,A] for a multi-step horizon.
            target = target.unsqueeze(-2)
        elif target.ndim == 2 and target.shape[0] == predictions.shape[1]:
            # [T,A] -> [1,T,A] for one-batch predictions.
            target = target.unsqueeze(0)
    elif rank_difference == 2 and target.ndim == 2 and target.shape[0] == predictions.shape[1]:
        # [T,A] -> [1,T,1,A].
        target = target.unsqueeze(0).unsqueeze(-2)
    # A horizon-one target [B,T,A] is intentionally broadcast over a predicted
    # multi-step horizon [B,T,H,A]; all other rank changes are rejected by the
    # regular broadcast check below.
    try:
        return torch.broadcast_to(target, predictions.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} shape {tuple(target.shape)} is not broadcastable to predictions "
            f"shape {tuple(predictions.shape)}"
        ) from exc


def _align_sequence_mask(
    value: Tensor | None,
    shape: tuple[int, int],
    *,
    device: torch.device,
    name: str,
    default: bool,
) -> Tensor:
    """Use the existing strict mask helper with a descriptive local name."""

    return _as_bool_mask(value, shape, device=device, name=name, default=default)


class FullHistoryActionTeacher(nn.Module):
    """Trainable causal teacher that predicts actions from complete history.

    This module is deliberately independent from the deployed SmolVLA-TTT
    model.  It is used during training/label mining only.  At physical time
    ``t`` it consumes exactly the current observation event token and the
    action executed at ``t-1``; no current noisy action, denoising timestep, or
    future observation/action is accepted by the recurrence.  A GRUCell keeps
    the implementation inexpensive for the 145--513 frame MIKASA episodes,
    while the functional state API permits chunked replay and counterfactual
    branches.

    ``write_mask`` is a causal intervention control.  In the canonical
    ``deletion_mode='skip'`` setting, a zero value deletes the event write but
    still advances the physical clock.  ``replacement_event_tokens`` together
    with ``replace_mask`` implements a paired, in-distribution cue
    replacement.  Both branches can therefore be replayed from the same state
    without mutating one another.
    """

    def __init__(
        self,
        event_dim: int,
        action_dim: int,
        hidden_dim: int,
        memory_dim: int | None = None,
        *,
        previous_action_dim: int | None = None,
        action_horizon: int = 1,
        action_steps: int | None = None,
        action_head_hidden_dim: int | None = None,
        include_current: bool = True,
        deletion_mode: str = "skip",
        target_mode: str = "executed_action",
    ) -> None:
        super().__init__()
        for name, value in (
            ("event_dim", event_dim),
            ("action_dim", action_dim),
            ("hidden_dim", hidden_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if memory_dim is None:
            memory_dim = hidden_dim
        if int(memory_dim) <= 0:
            raise ValueError("memory_dim must be positive")
        if previous_action_dim is None:
            previous_action_dim = action_dim
        if int(previous_action_dim) <= 0:
            raise ValueError("previous_action_dim must be positive")
        if action_steps is not None:
            if action_horizon != 1 and int(action_horizon) != int(action_steps):
                raise ValueError("action_horizon and action_steps disagree")
            action_horizon = int(action_steps)
        if int(action_horizon) <= 0:
            raise ValueError("action_horizon must be positive")
        if action_head_hidden_dim is None:
            action_head_hidden_dim = hidden_dim
        if int(action_head_hidden_dim) <= 0:
            raise ValueError("action_head_hidden_dim must be positive")
        if deletion_mode not in {"skip", "null"}:
            raise ValueError("deletion_mode must be 'skip' or 'null'")
        if not isinstance(target_mode, str) or not target_mode:
            raise ValueError("target_mode must be a non-empty string")

        self.event_dim = int(event_dim)
        self.action_dim = int(action_dim)
        self.previous_action_dim = int(previous_action_dim)
        self.hidden_dim = int(hidden_dim)
        self.memory_dim = int(memory_dim)
        self.action_horizon = int(action_horizon)
        self.action_head_hidden_dim = int(action_head_hidden_dim)
        self.include_current = bool(include_current)
        self.deletion_mode = deletion_mode
        self.target_mode = target_mode

        self.event_norm = nn.LayerNorm(self.event_dim)
        self.event_projection = nn.Linear(self.event_dim, self.hidden_dim)
        # The validity bit distinguishes a true all-zero executed action from
        # the synthetic start-of-episode action.  It is causal metadata, not a
        # future target shortcut.
        self.previous_action_projection = nn.Linear(
            self.previous_action_dim + 1,
            self.hidden_dim,
        )
        self.fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.null_event = nn.Parameter(torch.zeros(self.event_dim))
        self.memory_projection = nn.Linear(self.hidden_dim, self.memory_dim)
        self.memory_norm = nn.LayerNorm(self.memory_dim)
        self.action_norm = nn.LayerNorm(self.hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.action_head_hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.action_head_hidden_dim,
                self.action_horizon * self.action_dim,
            ),
        )

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Action shape per physical frame (horizon is omitted when one)."""

        if self.action_horizon == 1:
            return (self.action_dim,)
        return (self.action_horizon, self.action_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> CausalHistoryState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        position = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        return CausalHistoryState(hidden, position)

    def _validate_state(
        self,
        state: CausalHistoryState | None,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CausalHistoryState:
        if state is None:
            return self.initial_state(batch_size, device=device, dtype=dtype)
        if state.hidden.shape != (batch_size, self.hidden_dim):
            raise ValueError(
                f"state.hidden must have shape [{batch_size},{self.hidden_dim}], "
                f"got {tuple(state.hidden.shape)}"
            )
        if state.position.shape != (batch_size,):
            raise ValueError(
                f"state.position must have shape [{batch_size}], got {tuple(state.position.shape)}"
            )
        if state.hidden.device != device or state.position.device != device:
            raise ValueError("state and event_tokens must be on the same device")
        if state.hidden.dtype != dtype:
            state = CausalHistoryState(state.hidden.to(dtype=dtype), state.position)
        return state

    @staticmethod
    def _action_shape(predictions: Tensor, *, batch_size: int, steps: int) -> tuple[int, ...]:
        if predictions.shape[:2] != (batch_size, steps):
            raise ValueError(
                f"action head returned an invalid prefix {tuple(predictions.shape[:2])}; "
                f"expected {(batch_size, steps)}"
            )
        return tuple(predictions.shape[2:])

    def _predict_actions(self, hidden: Tensor) -> Tensor:
        """Decode hidden states and restore the configured horizon shape."""

        flat = self.action_head(self.action_norm(hidden))
        return flat.reshape(
            hidden.shape[0],
            self.action_horizon,
            self.action_dim,
        )

    def forward(
        self,
        event_tokens: Tensor,
        previous_executed_actions: Tensor | None = None,
        *,
        previous_actions: Tensor | None = None,
        previous_action_valid: Tensor | None = None,
        state: CausalHistoryState | None = None,
        valid_mask: Tensor | None = None,
        reset_mask: Tensor | None = None,
        write_mask: Tensor | None = None,
        replacement_event_tokens: Tensor | None = None,
        replace_mask: Tensor | None = None,
        delete_mask: Tensor | None = None,
        return_pre_update: bool = False,
    ) -> FullHistoryActionOutput:
        """Run a strictly causal full-history replay.

        ``previous_executed_actions[:,t]`` denotes the action executed before
        observation ``t``.  It is never shifted internally; callers therefore
        cannot accidentally pass the current expert target and claim it is a
        causal input.  A stored action chunk ``[B,T,H,A]`` is reduced to its
        physically executed slot ``H=0``.  ``previous_action_valid`` optionally
        marks the first frame of an episode (or another unavailable action)
        explicitly.  When omitted, the first row is treated as invalid for a
        fresh state and as valid for a chunk whose incoming state already
        contains history.
        """

        if previous_executed_actions is not None and previous_actions is not None:
            raise ValueError("pass at most one of previous_executed_actions and previous_actions")
        if previous_executed_actions is None:
            previous_executed_actions = previous_actions
        has_previous_actions = previous_executed_actions is not None

        if event_tokens.ndim == 2:
            event_tokens = event_tokens.unsqueeze(0)
            squeezed = True
        elif event_tokens.ndim == 3:
            squeezed = False
        else:
            raise ValueError(
                "event_tokens must have shape [T,D] or [B,T,D], got "
                f"{tuple(event_tokens.shape)}"
            )
        if not event_tokens.is_floating_point():
            raise ValueError("event_tokens must be floating-point")
        batch_size, steps, event_dim = event_tokens.shape
        if steps <= 0:
            raise ValueError("event_tokens must contain at least one timestep")
        if event_dim != self.event_dim:
            raise ValueError(f"event_tokens feature dim must be {self.event_dim}, got {event_dim}")

        if previous_executed_actions is None:
            previous_executed_actions = torch.zeros(
                batch_size,
                steps,
                self.previous_action_dim,
                device=event_tokens.device,
                dtype=event_tokens.dtype,
            )
            inferred_previous_valid = torch.zeros(
                batch_size, steps, dtype=torch.bool, device=event_tokens.device
            )
        else:
            previous_executed_actions, action_squeezed = _normalise_previous_action_sequence(
                previous_executed_actions,
                batch_size=batch_size,
                steps=steps,
                name="previous_executed_actions",
            )
            if previous_executed_actions.shape[:2] != (batch_size, steps):
                raise ValueError(
                    "previous_executed_actions must share [B,T] with event_tokens; got "
                    f"{tuple(previous_executed_actions.shape[:2])} vs {(batch_size, steps)}"
                )
            if previous_executed_actions.shape[-1] != self.previous_action_dim:
                raise ValueError(
                    f"previous_executed_actions feature dim must be {self.previous_action_dim}, "
                    f"got {previous_executed_actions.shape[-1]}"
                )
            inferred_previous_valid = torch.ones(
                batch_size, steps, dtype=torch.bool, device=event_tokens.device
            )
            # A new episode has no prior executed action at its first frame.
            # For chunked replay, an incoming state with position >=0 means
            # the first row really does have a preceding action.
            if state is None:
                inferred_previous_valid[:, 0] = False
            else:
                inferred_previous_valid[:, 0] = state.position.to(
                    device=event_tokens.device
                ) >= 0

        device = event_tokens.device
        compute_dtype = self.event_projection.weight.dtype
        events = event_tokens.to(dtype=compute_dtype)
        previous = previous_executed_actions.to(device=device, dtype=compute_dtype)
        valid = _align_sequence_mask(
            valid_mask,
            (batch_size, steps),
            device=device,
            name="valid_mask",
            default=True,
        )
        reset = _align_sequence_mask(
            reset_mask,
            (batch_size, steps),
            device=device,
            name="reset_mask",
            default=False,
        )
        # When validity is not supplied explicitly, an action can be a causal
        # predecessor only if the immediately preceding physical row was
        # valid.  This matters for padded/interleaved episodes: blindly
        # marking every t>0 row valid would let a zero-filled padded action
        # leak into the next real frame.  ``previous_action_valid`` remains an
        # explicit escape hatch for datasets that carry a richer validity
        # column.
        if has_previous_actions and steps > 1:
            inferred_previous_valid[:, 1:] = valid[:, :-1]
        writes = _as_write_mask(
            write_mask,
            (batch_size, steps),
            device=device,
            dtype=compute_dtype,
        )
        # ``delete_mask`` is a readable alias for the common binary
        # intervention.  Combining it multiplicatively also supports a soft
        # delete mask used in differentiable ablations.
        if delete_mask is not None:
            deletes = delete_mask if isinstance(delete_mask, Tensor) else torch.as_tensor(delete_mask)
            deletes = deletes.to(device=device, dtype=compute_dtype)
            if deletes.ndim == 0:
                deletes = deletes.expand(batch_size, steps)
            elif deletes.ndim == 1:
                if deletes.shape[0] == steps and batch_size == 1:
                    deletes = deletes[None, :]
                elif deletes.shape[0] == batch_size:
                    deletes = deletes[:, None].expand(batch_size, steps)
            try:
                deletes = torch.broadcast_to(deletes, (batch_size, steps))
            except RuntimeError as exc:
                raise ValueError("delete_mask is not broadcastable to [B,T]") from exc
            if not torch.isfinite(deletes).all() or bool((deletes < 0).any()) or bool((deletes > 1).any()):
                raise ValueError("delete_mask values must lie in [0,1]")
            writes = writes * (1.0 - deletes)

        replacement = None
        replace = torch.zeros(batch_size, steps, dtype=torch.bool, device=device)
        if replacement_event_tokens is not None:
            replacement = replacement_event_tokens
            if replacement.ndim == 1:
                replacement = replacement.reshape(1, 1, -1).expand(batch_size, steps, -1)
            elif replacement.ndim == 2:
                if replacement.shape == (steps, event_dim) and batch_size == 1:
                    replacement = replacement.unsqueeze(0)
                elif replacement.shape == (batch_size, event_dim):
                    replacement = replacement[:, None, :].expand(batch_size, steps, -1)
                else:
                    replacement = replacement.unsqueeze(0)
            if replacement.ndim != 3 or replacement.shape != (batch_size, steps, event_dim):
                raise ValueError(
                    "replacement_event_tokens must be [D], [T,D], [B,D], or [B,T,D] "
                    f"matching {(batch_size, steps, event_dim)}, got {tuple(replacement.shape)}"
                )
            replacement = replacement.to(device=device, dtype=compute_dtype)
            replace = _align_sequence_mask(
                replace_mask,
                (batch_size, steps),
                device=device,
                name="replace_mask",
                # Supplying replacement tokens without a mask means replace
                # every event; an explicit mask can select a sparse donor
                # interval.
                default=True,
            )
        elif replace_mask is not None:
            raise ValueError("replace_mask requires replacement_event_tokens")

        # The validity marker is causal metadata, not a public future-action
        # target.  A caller may override the inferred start-of-episode marker
        # when a dataset carries an explicit previous-action validity column.
        previous_valid = (
            inferred_previous_valid
            if previous_action_valid is None
            else _align_sequence_mask(
                previous_action_valid,
                (batch_size, steps),
                device=device,
                name="previous_action_valid",
                default=False,
            )
        ).to(device=device)
        # A reset marks the first observation of a new episode; an action from
        # the preceding episode must not be presented as its predecessor.
        previous_valid = previous_valid & ~reset
        state = self._validate_state(
            state,
            batch_size,
            device=device,
            dtype=compute_dtype,
        )
        hidden = state.hidden
        position = state.position
        event_outputs: list[Tensor] = []
        previous_outputs: list[Tensor] = []
        hidden_outputs: list[Tensor] = []
        memory_outputs: list[Tensor] = []
        action_outputs: list[Tensor] = []
        pre_memory_outputs: list[Tensor] = []
        pre_action_outputs: list[Tensor] = []

        for index in range(steps):
            valid_t = valid[:, index]
            reset_t = reset[:, index]
            write_t = writes[:, index] * valid_t.to(dtype=compute_dtype)
            hidden_before = torch.where(reset_t[:, None], torch.zeros_like(hidden), hidden)
            position_before = torch.where(reset_t, torch.full_like(position, -1), position)
            pre_memory = self.memory_norm(self.memory_projection(hidden_before))
            pre_action = self._predict_actions(hidden_before)

            current_event = events[:, index]
            if replacement is not None:
                current_event = torch.where(
                    replace[:, index, None], replacement[:, index], current_event
                )
            if self.deletion_mode == "null":
                current_event = torch.where(
                    (write_t <= 0)[:, None],
                    self.null_event.to(device=device, dtype=compute_dtype)[None, :],
                    current_event,
                )
            event_feature = self.event_projection(self.event_norm(current_event))
            previous_feature_input = torch.cat(
                [
                    # Zero an unavailable predecessor as well as exposing
                    # its validity bit.  The bit lets the network distinguish
                    # a genuine all-zero action from the start-of-episode
                    # sentinel, while masking prevents arbitrary padded
                    # values from becoming a hidden non-causal side channel.
                    previous[:, index]
                    * previous_valid[:, index, None].to(dtype=compute_dtype),
                    previous_valid[:, index, None].to(dtype=compute_dtype),
                ],
                dim=-1,
            )
            previous_feature = self.previous_action_projection(previous_feature_input)
            fused = self.fusion_norm(self.fusion(torch.cat([event_feature, previous_feature], dim=-1)))
            candidate = self.gru(fused, hidden_before)
            update_weight = (
                valid_t.to(dtype=compute_dtype)
                if self.deletion_mode == "null"
                else write_t
            )
            hidden_after = hidden_before + update_weight[:, None] * (candidate - hidden_before)
            hidden = torch.where(valid_t[:, None], hidden_after, hidden_before)
            position = position_before + valid_t.to(dtype=position.dtype)
            visible_hidden = hidden if self.include_current else hidden_before
            memory = self.memory_norm(self.memory_projection(visible_hidden))
            actions = self._predict_actions(visible_hidden)
            valid_action = valid_t[:, None]
            memory = memory * valid_action.to(dtype=memory.dtype)
            actions = actions * valid_action.reshape(batch_size, 1, 1).to(dtype=actions.dtype)

            event_outputs.append(event_feature)
            previous_outputs.append(previous_feature)
            hidden_outputs.append(visible_hidden)
            memory_outputs.append(memory)
            action_outputs.append(actions)
            if return_pre_update:
                pre_memory_outputs.append(pre_memory * valid_action.to(dtype=pre_memory.dtype))
                pre_action_outputs.append(
                    pre_action * valid_action.reshape(batch_size, 1, 1).to(dtype=pre_action.dtype)
                )

        event_output = torch.stack(event_outputs, dim=1)
        previous_output = torch.stack(previous_outputs, dim=1)
        hidden_output = torch.stack(hidden_outputs, dim=1)
        memory_output = torch.stack(memory_outputs, dim=1)
        action_output = torch.stack(action_outputs, dim=1)
        # Keep the horizon-one API ergonomic while retaining an explicit
        # horizon axis for multi-step action targets.
        if self.action_horizon == 1:
            action_output = action_output[:, :, 0, :]
        pre_memory_output = torch.stack(pre_memory_outputs, dim=1) if return_pre_update else None
        pre_action_output = torch.stack(pre_action_outputs, dim=1) if return_pre_update else None
        output = FullHistoryActionOutput(
            event_tokens=event_output,
            previous_action_tokens=previous_output,
            hidden_states=hidden_output,
            memory_tokens=memory_output,
            action_predictions=action_output,
            state=CausalHistoryState(hidden, position),
            pre_memory_tokens=pre_memory_output,
            pre_action_predictions=(
                None
                if pre_action_output is None
                else (pre_action_output[:, :, 0, :] if self.action_horizon == 1 else pre_action_output)
            ),
        )
        if not squeezed:
            return output
        return FullHistoryActionOutput(
            event_tokens=output.event_tokens[0],
            previous_action_tokens=output.previous_action_tokens[0],
            hidden_states=output.hidden_states[0],
            memory_tokens=output.memory_tokens[0],
            action_predictions=output.action_predictions[0],
            state=output.state,
            pre_memory_tokens=(
                None if output.pre_memory_tokens is None else output.pre_memory_tokens[0]
            ),
            pre_action_predictions=(
                None if output.pre_action_predictions is None else output.pre_action_predictions[0]
            ),
        )

    def forward_from_prefix(
        self,
        prefix_embeddings: Tensor,
        previous_executed_actions: Tensor | None = None,
        *,
        prefix_valid_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> FullHistoryActionOutput:
        """Summarize observation prefixes and run the causal action teacher.

        ``prefix_embeddings`` may be ``[B,T,P,D]`` or ``[B,P,D]`` for one
        physical frame.  The method intentionally accepts no expert/future
        action argument; the only action input is explicitly named
        ``previous_executed_actions``.
        """

        if prefix_embeddings.ndim == 3:
            # ``[B,P,D]`` denotes one frame for each batch member, not an
            # unbatched ``[T,D]`` sequence.  Add an explicit time axis before
            # entering ``forward`` so a batch of B=1 is not ambiguous.
            events = summarize_prefix(prefix_embeddings, prefix_valid_mask)[:, None, :]
            if previous_executed_actions is not None:
                if previous_executed_actions.ndim == 1:
                    previous_executed_actions = previous_executed_actions[None, None, :]
                elif previous_executed_actions.ndim == 2:
                    previous_executed_actions = previous_executed_actions[:, None, :]
                elif previous_executed_actions.ndim == 3:
                    if previous_executed_actions.shape[0] != prefix_embeddings.shape[0]:
                        raise ValueError(
                            "batched previous action chunks must share the prefix batch; got "
                            f"{tuple(previous_executed_actions.shape)}"
                        )
                    if previous_executed_actions.shape[1] <= 0:
                        raise ValueError("previous action chunks must contain at least one slot")
                    # [B,H,A] is a one-frame action chunk; only slot 0 was
                    # executed before this observation.
                    previous_executed_actions = previous_executed_actions[:, :1, :]
            output = self.forward(events, previous_executed_actions, **kwargs)
            return FullHistoryActionOutput(
                event_tokens=output.event_tokens[:, 0],
                previous_action_tokens=output.previous_action_tokens[:, 0],
                hidden_states=output.hidden_states[:, 0],
                memory_tokens=output.memory_tokens[:, 0],
                action_predictions=output.action_predictions[:, 0],
                state=output.state,
                pre_memory_tokens=(
                    None if output.pre_memory_tokens is None else output.pre_memory_tokens[:, 0]
                ),
                pre_action_predictions=(
                    None if output.pre_action_predictions is None else output.pre_action_predictions[:, 0]
                ),
            )
        if prefix_embeddings.ndim == 4:
            events = summarize_prefix(prefix_embeddings, prefix_valid_mask)
        else:
            raise ValueError(
                "prefix_embeddings must have shape [B,P,D] or [B,T,P,D], got "
                f"{tuple(prefix_embeddings.shape)}"
            )
        return self.forward(events, previous_executed_actions, **kwargs)

    # Explicit aliases make the training/label scripts self-documenting while
    # preserving one implementation of the recurrence.
    encode_prefix = forward_from_prefix
    replay = forward

    def replay_pair(
        self,
        event_tokens: Tensor,
        previous_executed_actions: Tensor | None = None,
        *,
        intervention_mask: Tensor | None = None,
        replacement_event_tokens: Tensor | None = None,
        delete_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[FullHistoryActionOutput, FullHistoryActionOutput]:
        """Return full and counterfactual causal replays from one initial state.

        ``intervention_mask`` selects replacement tokens when
        ``replacement_event_tokens`` is supplied; otherwise it is interpreted
        as a soft/binary event deletion mask.  The initial state is cloned when
        necessary, and neither replay mutates caller-owned tensors.
        """

        if intervention_mask is not None and delete_mask is not None:
            raise ValueError("pass at most one of intervention_mask and delete_mask")
        initial_state = kwargs.get("state")
        full = self.forward(
            event_tokens,
            previous_executed_actions,
            **kwargs,
        )
        counterfactual_kwargs = dict(kwargs)
        if initial_state is not None:
            counterfactual_kwargs["state"] = initial_state.clone(detach=False)
        if replacement_event_tokens is not None:
            counterfactual_kwargs["replacement_event_tokens"] = replacement_event_tokens
            counterfactual_kwargs["replace_mask"] = intervention_mask
            counterfactual_kwargs.pop("delete_mask", None)
        elif intervention_mask is not None:
            counterfactual_kwargs["delete_mask"] = intervention_mask
        elif delete_mask is not None:
            counterfactual_kwargs["delete_mask"] = delete_mask
        counterfactual = self.forward(
            event_tokens,
            previous_executed_actions,
            **counterfactual_kwargs,
        )
        return full, counterfactual

    causal_replay_pair = replay_pair

    @staticmethod
    def action_loss(
        predictions: Tensor,
        target_actions: Tensor,
        valid_mask: Tensor | None = None,
        *,
        reduction: str = "mean",
        loss_type: str = "smooth_l1",
    ) -> Tensor:
        """Compute a masked action loss for teacher training.

        The target is consumed *after* the causal replay and therefore cannot
        influence the teacher state.  A horizon-one target may supervise every
        predicted horizon slot by ordinary broadcasting.
        """

        if predictions.ndim < 2 or target_actions.ndim < 2:
            raise ValueError("predictions and target_actions need [B,T,...] dimensions")
        if predictions.ndim == 2:
            predictions = predictions.unsqueeze(0)
            if target_actions.ndim == 2:
                target_actions = target_actions.unsqueeze(0)
        target = _align_action_target(predictions, target_actions, name="target_actions")
        if loss_type == "mse":
            per_feature = (predictions - target).square()
        elif loss_type in {"smooth_l1", "huber"}:
            per_feature = F.smooth_l1_loss(predictions, target, reduction="none")
        else:
            raise ValueError("loss_type must be 'smooth_l1'/'huber' or 'mse'")
        feature_dims = tuple(range(2, per_feature.ndim))
        per_step = per_feature.mean(dim=feature_dims) if feature_dims else per_feature
        if valid_mask is not None:
            mask = valid_mask if isinstance(valid_mask, Tensor) else torch.as_tensor(valid_mask)
            mask = mask.to(device=per_step.device, dtype=per_step.dtype)
            if mask.ndim == 1 and per_step.ndim == 2 and mask.shape[0] == per_step.shape[1]:
                mask = mask[None, :]
            # A horizon mask [B,T,H] is reduced to a per-step fraction before
            # weighting, avoiding a hidden preference for longer horizons.
            if mask.ndim > per_step.ndim:
                mask = mask.mean(dim=tuple(range(per_step.ndim, mask.ndim)))
            try:
                mask = torch.broadcast_to(mask, per_step.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"valid_mask shape {tuple(mask.shape)} is not broadcastable to {tuple(per_step.shape)}"
                ) from exc
            per_step = per_step * mask
            denominator = mask.sum().clamp_min(1.0)
        else:
            denominator = per_step.new_tensor(float(per_step.numel())).clamp_min(1.0)
        if reduction == "none":
            return per_step
        if reduction == "sum":
            return per_step.sum()
        if reduction != "mean":
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        return per_step.sum() / denominator

    def compute_loss(
        self,
        event_tokens: Tensor,
        target_actions: Tensor,
        previous_executed_actions: Tensor | None = None,
        *,
        valid_mask: Tensor | None = None,
        return_output: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, float]] | tuple[Tensor, dict[str, float], FullHistoryActionOutput]:
        """Run a causal replay and supervise its action head."""

        output = self.forward(
            event_tokens,
            previous_executed_actions,
            valid_mask=valid_mask,
            **kwargs,
        )
        loss = self.action_loss(
            output.action_predictions,
            target_actions,
            valid_mask=valid_mask,
        )
        metrics = {"loss": float(loss.detach().item())}
        if return_output:
            return loss, metrics, output
        return loss, metrics

    def config_dict(self) -> dict[str, Any]:
        """Return constructor arguments suitable for checkpoint recreation."""

        return {
            "event_dim": self.event_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "memory_dim": self.memory_dim,
            "previous_action_dim": self.previous_action_dim,
            "action_horizon": self.action_horizon,
            "action_head_hidden_dim": self.action_head_hidden_dim,
            "include_current": self.include_current,
            "deletion_mode": self.deletion_mode,
            "target_mode": self.target_mode,
        }

    def provenance(
        self,
        *,
        teacher_checkpoint: str | Path | None = None,
        source_policy_checkpoint: str | Path | None = None,
        dataset_id: str | None = None,
        target_mode: str | None = None,
    ) -> dict[str, Any]:
        """Return a strict, JSON-safe provenance manifest for this teacher."""

        return {
            "format": FULL_HISTORY_TEACHER_FORMAT,
            "event_schema": FULL_HISTORY_EVENT_SCHEMA,
            "intervention_schema": FULL_HISTORY_INTERVENTION_SCHEMA,
            "causal": True,
            "include_current": self.include_current,
            "deletion_mode": self.deletion_mode,
            "event_dim": self.event_dim,
            "previous_action_dim": self.previous_action_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "hidden_dim": self.hidden_dim,
            "memory_dim": self.memory_dim,
            "target_mode": self.target_mode if target_mode is None else str(target_mode),
            "dataset_id": dataset_id,
            "parameter_sha256": history_teacher_state_sha256(self),
            "teacher_checkpoint": None if teacher_checkpoint is None else str(teacher_checkpoint),
            "source_policy_checkpoint": (
                None if source_policy_checkpoint is None else str(source_policy_checkpoint)
            ),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save weights, constructor config, and a provenance manifest."""

        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.provenance(teacher_checkpoint=str(output))
        if metadata is not None:
            supplied = dict(metadata)
            supplied_digest = supplied.get("parameter_sha256")
            if supplied_digest is not None and supplied_digest != manifest["parameter_sha256"]:
                raise ValueError("metadata parameter_sha256 does not match teacher state")
            # Core identity fields cannot be changed by an auxiliary manifest.
            for key in ("format", "event_schema", "intervention_schema", "causal"):
                if key in supplied and supplied[key] != manifest[key]:
                    raise ValueError(f"metadata cannot override core provenance field {key!r}")
            manifest.update(supplied)
        payload = {
            "format": FULL_HISTORY_TEACHER_FORMAT,
            "config": self.config_dict(),
            "state_dict": self.state_dict(),
            "metadata": manifest,
        }
        torch.save(payload, output)
        return manifest

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        return_metadata: bool = False,
    ) -> "FullHistoryActionTeacher" | tuple["FullHistoryActionTeacher", dict[str, Any]]:
        """Load a V3 teacher checkpoint and verify its state hash."""

        checkpoint = Path(path).expanduser()
        if checkpoint.is_dir():
            candidates = [
                checkpoint / name
                for name in ("teacher.pt", "checkpoint.pt", "full_history_teacher.pt")
            ]
            existing = [candidate for candidate in candidates if candidate.is_file()]
            if len(existing) != 1:
                raise FileNotFoundError(
                    "Teacher checkpoint directory must contain exactly one of "
                    f"{[x.name for x in candidates]}; "
                    f"found {[x.name for x in existing]}"
                )
            checkpoint = existing[0]
        try:
            payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(checkpoint, map_location=map_location)
        if not isinstance(payload, Mapping):
            raise ValueError("Full-history teacher checkpoint must contain a mapping")
        if payload.get("format") != FULL_HISTORY_TEACHER_FORMAT:
            raise ValueError(
                f"Unsupported full-history teacher format {payload.get('format')!r}; "
                f"expected {FULL_HISTORY_TEACHER_FORMAT!r}"
            )
        config = payload.get("config")
        state_dict = payload.get("state_dict")
        metadata = payload.get("metadata", {})
        if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
            raise ValueError("Full-history teacher checkpoint needs config and state_dict mappings")
        teacher = cls(**dict(config))
        missing, unexpected = teacher.load_state_dict(state_dict, strict=strict)
        if strict and (missing or unexpected):  # defensive for custom modules
            raise ValueError(f"Checkpoint keys mismatch: missing={missing}, unexpected={unexpected}")
        if isinstance(metadata, Mapping) and metadata:
            validate_full_history_teacher_provenance(metadata, teacher=teacher)
            manifest = dict(metadata)
        else:
            manifest = teacher.provenance(teacher_checkpoint=str(checkpoint))
        teacher.eval()
        if return_metadata:
            return teacher, manifest
        return teacher

    load_checkpoint = from_checkpoint


# Names used in the paper and in early experiment notes.  They intentionally
# resolve to the same implementation, while the old memory-only teacher keeps
# its own class/format above.
ExplicitFullHistoryTeacher = FullHistoryActionTeacher
CausalActionTeacher = FullHistoryActionTeacher


def compute_pairwise_control_credit(
    full_actions: Tensor,
    counterfactual_actions: Tensor,
    expert_actions: Tensor,
    *,
    event_ends: Tensor | Sequence[int] | None = None,
    pair_mask: Tensor | None = None,
    valid_mask: Tensor | None = None,
    confidence_scale: float = 1.0,
    eps: float = 1e-8,
) -> PairwiseControlCredit:
    """Compute ``u_{i->j}`` and ``Delta a_{i->j}`` from teacher replays.

    Args:
        full_actions: Full-history predictions ``[B,J,...]`` (``[J,...]`` is
            accepted for one episode).
        counterfactual_actions: Predictions for ``I`` interventions,
            canonically ``[B,I,J,...]``.  ``[I,J,...]`` is accepted for one
            episode.
        expert_actions: Fixed future expert targets ``[B,J,...]``.
        event_ends: Exclusive end index for each event, shape ``[B,I]`` or
            ``[I]``.  If omitted, event ``i`` is assumed to end at ``i+1``.
        pair_mask: Optional additional causal/data-valid mask ``[B,I,J]``.

    The helper is agnostic to whether actions came from a direct action head
    or a complete flow integration; this lets the benchmark use the exact
    deployed slot-0 action while keeping label math in one audited function.
    """

    full, _ = _normalise_action_sequence(full_actions, name="full_actions")
    if full.ndim < 3:
        raise ValueError("full_actions must contain a feature/action dimension")
    batch_size, future_steps = full.shape[:2]
    if counterfactual_actions.ndim < 3:
        raise ValueError("counterfactual_actions must have [I,J,...] or [B,I,J,...] shape")
    cf = counterfactual_actions
    if cf.ndim == full.ndim:
        # One episode [I,J,...] is the common compact form.  For a batched
        # single-event input [B,J,...], retain the batch axis explicitly.
        if batch_size == 1 and cf.shape[1] == future_steps:
            cf = cf.unsqueeze(0)
        elif cf.shape[:2] == (batch_size, future_steps):
            cf = cf.unsqueeze(1)
        else:
            raise ValueError(
                "counterfactual_actions rank matches full_actions but its first axes are ambiguous; "
                "use canonical [B,I,J,...]"
            )
    if cf.ndim != full.ndim + 1:
        raise ValueError(
            f"counterfactual_actions must have one event axis beyond full_actions; got {tuple(cf.shape)}"
        )
    if cf.shape[0] != batch_size or cf.shape[2] != future_steps:
        raise ValueError(
            "counterfactual_actions must match full batch/future dimensions; got "
            f"{tuple(cf.shape[:3])} vs batch={batch_size}, future={future_steps}"
        )
    if cf.shape[3:] != full.shape[2:]:
        raise ValueError(
            f"counterfactual action suffix {tuple(cf.shape[3:])} does not match full {tuple(full.shape[2:])}"
        )
    target = _align_action_target(full, expert_actions, name="expert_actions")
    target = target.unsqueeze(1).expand_as(cf)
    full_expanded = full.unsqueeze(1).expand_as(cf)
    full_loss = (full_expanded - target).square()
    cf_loss = (cf - target).square()
    reduce_dims = tuple(range(3, full_loss.ndim))
    if reduce_dims:
        full_loss = full_loss.mean(dim=reduce_dims)
        cf_loss = cf_loss.mean(dim=reduce_dims)
    raw_degradation = cf_loss - full_loss
    scale = float(confidence_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("confidence_scale must be finite and positive")
    confidence = torch.exp(-full_loss.detach() / max(scale, eps))
    # Keep the signed absolute degradation for auditability, but use a
    # symmetric relative degradation as the training utility.  Absolute
    # squared-error differences otherwise make long/higher-variance episodes
    # dominate the pair sampler and turn the attribution threshold into a
    # task-scale knob.  The denominator is detached from the teacher graph;
    # labels are always treated as constants by the student.
    relative_degradation = raw_degradation / (
        0.5 * (cf_loss.detach().abs() + full_loss.detach().abs()) + eps
    )
    utility = relative_degradation.clamp_min(0) * confidence

    if event_ends is None:
        ends = torch.arange(cf.shape[1], device=cf.device, dtype=torch.long).clamp_max(future_steps - 1) + 1
        ends = ends[None, :].expand(batch_size, -1)
    else:
        ends = torch.as_tensor(event_ends, device=cf.device)
        if ends.ndim == 1:
            if ends.shape[0] != cf.shape[1]:
                raise ValueError("event_ends rank-1 length must match event axis")
            ends = ends[None, :].expand(batch_size, -1)
        elif ends.shape != (batch_size, cf.shape[1]):
            raise ValueError(
                f"event_ends must have shape [{batch_size},{cf.shape[1]}], got {tuple(ends.shape)}"
            )
        if ends.dtype.is_floating_point:
            if not torch.equal(ends, ends.round()):
                raise ValueError("event_ends must contain integers")
            ends = ends.round().to(torch.long)
        else:
            ends = ends.to(torch.long)
    if bool((ends < 0).any()) or bool((ends > future_steps).any()):
        raise ValueError("event_ends must lie in [0, future_steps]")
    future_indices = torch.arange(future_steps, device=cf.device)[None, None, :]
    causal = future_indices >= ends[:, :, None]
    if valid_mask is not None:
        valid = valid_mask if isinstance(valid_mask, Tensor) else torch.as_tensor(valid_mask)
        valid = valid.to(device=cf.device, dtype=torch.bool)
        if valid.ndim == 1:
            if valid.shape[0] != future_steps:
                raise ValueError("valid_mask rank-1 length must match future axis")
            valid = valid[None, :].expand(batch_size, -1)
        valid = torch.broadcast_to(valid, (batch_size, future_steps))
        causal = causal & valid[:, None, :]
    if pair_mask is not None:
        supplied = pair_mask if isinstance(pair_mask, Tensor) else torch.as_tensor(pair_mask)
        supplied = supplied.to(device=cf.device, dtype=torch.bool)
        causal = causal & torch.broadcast_to(supplied, causal.shape)
    finite = (
        torch.isfinite(raw_degradation)
        & torch.isfinite(full_expanded).all(dim=tuple(range(3, full_expanded.ndim)))
        & torch.isfinite(cf).all(dim=tuple(range(3, cf.ndim)))
        & torch.isfinite(target).all(dim=tuple(range(3, target.ndim)))
    )
    causal = causal & finite
    utility = torch.where(causal, utility, torch.zeros_like(utility))
    raw_degradation = torch.where(causal, raw_degradation, torch.zeros_like(raw_degradation))
    effect = full_expanded - cf
    mask_shape = causal.shape + (1,) * len(full.shape[2:])
    effect = torch.where(causal.reshape(mask_shape), effect, torch.zeros_like(effect))
    return PairwiseControlCredit(
        utility=utility,
        action_effect=effect,
        pair_mask=causal,
        full_loss=full_loss,
        counterfactual_loss=cf_loss,
        raw_degradation=raw_degradation,
        confidence=confidence,
    )


pairwise_teacher_control_credit = compute_pairwise_control_credit


def validate_full_history_teacher_provenance(
    metadata: Mapping[str, Any],
    *,
    teacher: FullHistoryActionTeacher | None = None,
) -> None:
    """Validate a V3 full-history teacher manifest.

    The check is intentionally strict about causal identity and dimensions,
    but permits additional experiment fields (dataset hashes, optimizer
    settings, and so on) to be attached by a benchmark script.  When a
    teacher instance is supplied, its state hash and constructor dimensions
    are checked as well; this catches a surprisingly common error where a
    label artifact is generated with a different hidden width.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("full-history teacher provenance must be a mapping")
    required = {
        "format",
        "event_schema",
        "intervention_schema",
        "causal",
        "include_current",
        "deletion_mode",
        "event_dim",
        "previous_action_dim",
        "action_dim",
        "action_horizon",
        "hidden_dim",
        "memory_dim",
        "target_mode",
        "parameter_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Full-history teacher provenance is missing fields: {missing}")
    if metadata["format"] != FULL_HISTORY_TEACHER_FORMAT:
        raise ValueError(
            f"Unsupported full-history teacher format {metadata['format']!r}; "
            f"expected {FULL_HISTORY_TEACHER_FORMAT!r}"
        )
    if metadata["event_schema"] != FULL_HISTORY_EVENT_SCHEMA:
        raise ValueError("Full-history teacher event schema does not match this implementation")
    if metadata["intervention_schema"] != FULL_HISTORY_INTERVENTION_SCHEMA:
        raise ValueError("Full-history teacher intervention schema does not match this implementation")
    if metadata["causal"] is not True:
        raise ValueError("Full-history teacher provenance must declare causal=true")
    if type(metadata["include_current"]) is not bool:
        raise ValueError("include_current must be a JSON boolean")
    if metadata["deletion_mode"] not in {"skip", "null"}:
        raise ValueError("Full-history teacher provenance has an invalid deletion_mode")
    for name in (
        "event_dim",
        "previous_action_dim",
        "action_dim",
        "action_horizon",
        "hidden_dim",
        "memory_dim",
    ):
        try:
            if int(metadata[name]) <= 0:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"Full-history teacher provenance field {name!r} must be positive") from error
    digest = metadata["parameter_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("parameter_sha256 must be a 64-character hex string")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError("parameter_sha256 is not hexadecimal") from error
    if teacher is not None:
        expected = {
            "event_dim": teacher.event_dim,
            "previous_action_dim": teacher.previous_action_dim,
            "action_dim": teacher.action_dim,
            "action_horizon": teacher.action_horizon,
            "hidden_dim": teacher.hidden_dim,
            "memory_dim": teacher.memory_dim,
        }
        mismatches = {
            key: (int(metadata[key]), value)
            for key, value in expected.items()
            if int(metadata[key]) != value
        }
        if mismatches:
            raise ValueError(f"Full-history teacher dimensions disagree with provenance: {mismatches}")
        actual_digest = history_teacher_state_sha256(teacher)
        if actual_digest != digest:
            raise ValueError(
                "Full-history teacher state hash does not match provenance: "
                f"expected {digest}, got {actual_digest}"
            )


def save_full_history_teacher_checkpoint(
    teacher: FullHistoryActionTeacher,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Functional wrapper around :meth:`FullHistoryActionTeacher.save_checkpoint`."""

    if not isinstance(teacher, FullHistoryActionTeacher):
        raise TypeError("teacher must be a FullHistoryActionTeacher")
    return teacher.save_checkpoint(path, metadata=metadata)


def load_full_history_teacher_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[FullHistoryActionTeacher, dict[str, Any]]:
    """Load a full-history teacher and return it together with its manifest."""

    loaded = FullHistoryActionTeacher.from_checkpoint(
        path,
        map_location=map_location,
        strict=strict,
        return_metadata=True,
    )
    assert isinstance(loaded, tuple)
    return loaded


# ``CausalFullHistoryTeacher`` was exported by the original HD-TTT artifact
# as an alias of the memory-only class.  Keep that constructor/API intact for
# old checkpoints and scripts; V3 callers should use the unambiguous
# ``ExplicitFullHistoryTeacher``/``FullHistoryActionTeacher`` names below.
# Changing this historical alias would make an old call such as
# ``CausalFullHistoryTeacher(event_dim, hidden_dim, memory_dim)`` fail at
# import-time, which is worse than the small naming ambiguity.
CausalFullHistoryTeacher = CausalHistoryTeacher


def append_history_memory(
    prefix_embeddings: Tensor,
    prefix_pad_mask: Tensor,
    prefix_attention_mask: Tensor,
    memory_token: Tensor,
    *,
    memory_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Append one teacher memory token to a single action-expert prefix.

    ``prefix_embeddings`` is ``[B,P,D]``; ``memory_token`` is ``[B,M]`` or
    ``[B,1,M]`` and must already have width ``D``.  The memory token is put in
    a new prefix attention block (``att_mask=1``).  Existing prefix tokens are
    untouched, while suffix/action queries can attend to the appended token in
    the usual prefix-LM mask.
    """

    if prefix_embeddings.ndim != 3:
        raise ValueError(f"prefix_embeddings must be [B,P,D], got {tuple(prefix_embeddings.shape)}")
    prefix_pad_mask = prefix_pad_mask.to(device=prefix_embeddings.device)
    prefix_attention_mask = prefix_attention_mask.to(device=prefix_embeddings.device)
    if prefix_pad_mask.ndim != 2 or prefix_pad_mask.shape != prefix_embeddings.shape[:2]:
        raise ValueError(
            "prefix_pad_mask must have shape [B,P] matching prefix_embeddings, got "
            f"{tuple(prefix_pad_mask.shape)}"
        )
    if prefix_attention_mask.ndim != 2 or prefix_attention_mask.shape != prefix_embeddings.shape[:2]:
        raise ValueError(
            "prefix_attention_mask must have shape [B,P] matching prefix_embeddings, got "
            f"{tuple(prefix_attention_mask.shape)}"
        )
    if memory_token.ndim == 2:
        memory_token = memory_token[:, None, :]
    if memory_token.ndim != 3 or memory_token.shape[1] != 1:
        raise ValueError(
            f"memory_token must have shape [B,M] or [B,1,M], got {tuple(memory_token.shape)}"
        )
    if memory_token.shape[0] != prefix_embeddings.shape[0] or memory_token.shape[2] != prefix_embeddings.shape[2]:
        raise ValueError(
            "memory_token batch/feature dimensions must match prefix_embeddings; got "
            f"{tuple(memory_token.shape)} vs {tuple(prefix_embeddings.shape)}"
        )
    memory_token = memory_token.to(device=prefix_embeddings.device, dtype=prefix_embeddings.dtype)
    if memory_valid is None:
        memory_valid = torch.ones(
            prefix_embeddings.shape[0], 1, dtype=torch.bool, device=prefix_embeddings.device
        )
    else:
        if not isinstance(memory_valid, Tensor):
            memory_valid = torch.as_tensor(memory_valid, device=prefix_embeddings.device)
        memory_valid = memory_valid.to(device=prefix_embeddings.device, dtype=torch.bool)
        if memory_valid.ndim == 1:
            memory_valid = memory_valid[:, None]
        if memory_valid.shape != (prefix_embeddings.shape[0], 1):
            raise ValueError(
                f"memory_valid must have shape [{prefix_embeddings.shape[0]},1], got {tuple(memory_valid.shape)}"
            )
    memory_token = memory_token * memory_valid[:, :, None].to(dtype=memory_token.dtype)
    memory_attention = torch.ones(
        prefix_embeddings.shape[0],
        1,
        dtype=prefix_attention_mask.dtype,
        device=prefix_embeddings.device,
    )
    return (
        torch.cat((prefix_embeddings, memory_token), dim=1),
        torch.cat((prefix_pad_mask.to(dtype=torch.bool), memory_valid), dim=1),
        torch.cat((prefix_attention_mask, memory_attention), dim=1),
    )


class HistoryPrefixConditioner(nn.Module):
    """Project teacher memory to a prefix width and append it safely.

    This is the optional integration layer.  It does not import or modify the
    SmolVLA model; a caller can invoke it immediately after ``embed_prefix``
    and before concatenating the action suffix.
    """

    def __init__(self, memory_dim: int, prefix_dim: int) -> None:
        super().__init__()
        if memory_dim <= 0 or prefix_dim <= 0:
            raise ValueError("memory_dim and prefix_dim must be positive")
        self.memory_dim = int(memory_dim)
        self.prefix_dim = int(prefix_dim)
        self.memory_projection = (
            nn.Identity() if memory_dim == prefix_dim else nn.Linear(memory_dim, prefix_dim)
        )

    def forward(
        self,
        prefix_embeddings: Tensor,
        prefix_pad_mask: Tensor,
        prefix_attention_mask: Tensor,
        memory_token: Tensor,
        *,
        memory_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if memory_token.shape[-1] != self.memory_dim:
            raise ValueError(
                f"memory_token feature dim must be {self.memory_dim}, got {memory_token.shape[-1]}"
            )
        projected = self.memory_projection(memory_token)
        return append_history_memory(
            prefix_embeddings,
            prefix_pad_mask,
            prefix_attention_mask,
            projected,
            memory_valid=memory_valid,
        )

    def condition_sequence(
        self,
        prefix_embeddings: Tensor,
        prefix_pad_mask: Tensor,
        prefix_attention_mask: Tensor,
        memory_tokens: Tensor,
        *,
        memory_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Condition a ``[B,T]`` prefix sequence and restore its shape.

        ``SmolVLATTTFlowMatching`` flattens physical frames to ``B*T`` before
        running the VLM.  This helper keeps the history-teacher API episode
        shaped and performs that flattening locally, so a future integration
        hook cannot accidentally pair a memory token with a neighboring frame.
        The returned embeddings/masks have shape ``[B,T,P+1,*]``.
        """

        if prefix_embeddings.ndim != 4:
            raise ValueError(
                "sequence prefix_embeddings must have shape [B,T,P,D], got "
                f"{tuple(prefix_embeddings.shape)}"
            )
        batch_size, steps, prefix_length, prefix_dim = prefix_embeddings.shape
        if steps <= 0 or prefix_length <= 0:
            raise ValueError("prefix sequence must contain at least one timestep and prefix token")
        expected_mask_shape = (batch_size, steps, prefix_length)
        if prefix_pad_mask.shape != expected_mask_shape:
            raise ValueError(
                "sequence prefix_pad_mask must have shape "
                f"{expected_mask_shape}, got {tuple(prefix_pad_mask.shape)}"
            )
        if prefix_attention_mask.shape != expected_mask_shape:
            raise ValueError(
                "sequence prefix_attention_mask must have shape "
                f"{expected_mask_shape}, got {tuple(prefix_attention_mask.shape)}"
            )
        if memory_tokens.ndim != 3 or memory_tokens.shape[:2] != (batch_size, steps):
            raise ValueError(
                "memory_tokens must have shape [B,T,M] matching prefix sequence; got "
                f"{tuple(memory_tokens.shape)}"
            )
        if memory_valid is not None:
            if not isinstance(memory_valid, Tensor):
                memory_valid = torch.as_tensor(memory_valid, device=prefix_embeddings.device)
            if memory_valid.shape != (batch_size, steps):
                raise ValueError(
                    "memory_valid must have shape [B,T] matching prefix sequence; got "
                    f"{tuple(memory_valid.shape)}"
                )
        flattened_valid = None if memory_valid is None else memory_valid.reshape(batch_size * steps)
        conditioned = self.forward(
            prefix_embeddings.reshape(batch_size * steps, prefix_length, prefix_dim),
            prefix_pad_mask.reshape(batch_size * steps, prefix_length),
            prefix_attention_mask.reshape(batch_size * steps, prefix_length),
            memory_tokens.reshape(batch_size * steps, memory_tokens.shape[-1]),
            memory_valid=flattened_valid,
        )
        embeddings, pad_mask, attention_mask = conditioned
        return (
            embeddings.reshape(batch_size, steps, embeddings.shape[1], embeddings.shape[2]),
            pad_mask.reshape(batch_size, steps, pad_mask.shape[1]),
            attention_mask.reshape(batch_size, steps, attention_mask.shape[1]),
        )


def validate_history_teacher_provenance(metadata: Mapping[str, Any]) -> None:
    """Validate the minimum metadata contract before attaching labels.

    Label builders should call this on a teacher checkpoint/artifact manifest.
    It intentionally rejects a missing causal declaration or a teacher whose
    dimensions disagree with the instantiated module; silently accepting those
    cases would make event-deletion labels irreproducible.
    """

    required = {
        "format",
        "event_schema",
        "deletion_schema",
        "causal",
        "include_current",
        "deletion_mode",
        "event_dim",
        "hidden_dim",
        "memory_dim",
        "parameter_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"History-teacher provenance is missing fields: {missing}")
    if metadata["format"] != HISTORY_TEACHER_FORMAT:
        raise ValueError(
            f"Unsupported history-teacher format {metadata['format']!r}; expected {HISTORY_TEACHER_FORMAT!r}"
        )
    if metadata["causal"] is not True:
        raise ValueError("History teacher provenance must declare causal=true")
    if metadata["deletion_schema"] != HISTORY_DELETION_SCHEMA:
        raise ValueError("History teacher deletion schema does not match this implementation")
    if metadata["deletion_mode"] not in {"skip", "null"}:
        raise ValueError("History teacher provenance has an invalid deletion_mode")
    for name in ("event_dim", "hidden_dim", "memory_dim"):
        try:
            if int(metadata[name]) <= 0:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"History teacher provenance field {name!r} must be positive") from error
    digest = metadata["parameter_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("History teacher provenance parameter_sha256 must be a 64-character hex string")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError("History teacher provenance parameter_sha256 is not hexadecimal") from error


__all__ = [
    "HISTORY_DELETION_SCHEMA",
    "HISTORY_EVENT_SCHEMA",
    "HISTORY_TEACHER_FORMAT",
    "FULL_HISTORY_TEACHER_FORMAT",
    "FULL_HISTORY_EVENT_SCHEMA",
    "FULL_HISTORY_INTERVENTION_SCHEMA",
    "CausalHistoryOutput",
    "CausalHistoryState",
    "CausalFullHistoryTeacher",
    "CausalHistoryTeacher",
    "FullHistoryActionOutput",
    "FullHistoryActionTeacher",
    "ExplicitFullHistoryTeacher",
    "CausalActionTeacher",
    "PairwiseControlCredit",
    "compute_pairwise_control_credit",
    "pairwise_teacher_control_credit",
    "HistoryPrefixConditioner",
    "append_history_memory",
    "history_teacher_state_sha256",
    "validate_full_history_teacher_provenance",
    "save_full_history_teacher_checkpoint",
    "load_full_history_teacher_checkpoint",
    "summarize_prefix",
    "validate_history_teacher_provenance",
]
