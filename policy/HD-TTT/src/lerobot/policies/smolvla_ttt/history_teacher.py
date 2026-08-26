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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


HISTORY_TEACHER_FORMAT = "causal_history_teacher_v1"
HISTORY_EVENT_SCHEMA = "masked_mean_observation_prefix"
HISTORY_DELETION_SCHEMA = "interpolated_gru_write_mask"


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


# Name used in the paper/label recipe.  Keep the shorter class name as the
# implementation entry point while exposing an explicit full-history alias so
# call sites make the training-only role unambiguous.
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
    "CausalHistoryOutput",
    "CausalHistoryState",
    "CausalFullHistoryTeacher",
    "CausalHistoryTeacher",
    "HistoryPrefixConditioner",
    "append_history_memory",
    "history_teacher_state_sha256",
    "summarize_prefix",
    "validate_history_teacher_provenance",
]
