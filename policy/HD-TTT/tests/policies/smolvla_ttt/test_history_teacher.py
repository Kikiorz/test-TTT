"""Unit tests for the training-only causal history teacher."""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.smolvla_ttt.history_teacher import (
    HISTORY_DELETION_SCHEMA,
    HISTORY_EVENT_SCHEMA,
    HISTORY_TEACHER_FORMAT,
    CausalHistoryTeacher,
    HistoryPrefixConditioner,
    append_history_memory,
    summarize_prefix,
    validate_history_teacher_provenance,
)


def _teacher(event_dim: int = 5, hidden_dim: int = 7, memory_dim: int = 9) -> CausalHistoryTeacher:
    torch.manual_seed(17)
    return CausalHistoryTeacher(event_dim, hidden_dim, memory_dim)


def test_history_is_strictly_causal() -> None:
    teacher = _teacher()
    events = torch.randn(1, 6, teacher.event_dim)
    changed = events.clone()
    # Perturb one feature rather than adding a constant offset, which LayerNorm
    # would intentionally remove.
    changed[:, 4:, 0] += 10.0

    first = teacher(events).memory_tokens
    second = teacher(changed).memory_tokens

    # Future event edits cannot affect the current or earlier memory token.
    torch.testing.assert_close(first[:, :4], second[:, :4], rtol=0, atol=0)
    assert not torch.allclose(first[:, 4:], second[:, 4:])


def test_chunked_replay_matches_one_full_episode() -> None:
    teacher = _teacher()
    events = torch.randn(2, 11, teacher.event_dim)
    valid = torch.ones(2, 11, dtype=torch.bool)
    valid[1, -2:] = False
    reset = torch.zeros(2, 11, dtype=torch.bool)
    reset[:, 0] = True

    full = teacher(events, valid_mask=valid, reset_mask=reset)
    first = teacher(events[:, :5], valid_mask=valid[:, :5], reset_mask=reset[:, :5])
    second = teacher(
        events[:, 5:],
        state=first.state,
        valid_mask=valid[:, 5:],
        reset_mask=reset[:, 5:],
    )

    torch.testing.assert_close(
        torch.cat((first.memory_tokens, second.memory_tokens), dim=1),
        full.memory_tokens,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.cat((first.event_tokens, second.event_tokens), dim=1),
        full.event_tokens,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(first.state.hidden, full.state.hidden)
    torch.testing.assert_close(second.state.hidden, full.state.hidden)
    assert second.state.position.tolist() == full.state.position.tolist() == [10, 8]


def test_episode_reset_blocks_cross_episode_history() -> None:
    teacher = _teacher()
    events = torch.randn(1, 8, teacher.event_dim)
    changed = events.clone()
    changed[:, :3] += 20.0
    reset = torch.zeros(1, 8, dtype=torch.bool)
    reset[:, 0] = True
    reset[:, 4] = True

    first = teacher(events, reset_mask=reset).memory_tokens
    second = teacher(changed, reset_mask=reset).memory_tokens
    torch.testing.assert_close(first[:, 4:], second[:, 4:], rtol=0, atol=0)


def test_event_deletion_changes_only_retained_future() -> None:
    teacher = _teacher()
    events = torch.randn(1, 9, teacher.event_dim)
    full = teacher(events)
    gate = teacher.deletion_write_mask(9, 2, 5)
    deleted = teacher(events, write_mask=gate)

    # The event is still visible at the deleted frame only through the prior
    # state; all earlier memories are exactly unchanged.
    torch.testing.assert_close(full.memory_tokens[:, :2], deleted.memory_tokens[:, :2], rtol=0, atol=0)
    assert not torch.allclose(full.memory_tokens[:, 2:], deleted.memory_tokens[:, 2:])

    # A branch call never mutates the full-episode state object.
    torch.testing.assert_close(full.state.hidden, teacher(events).state.hidden)


def test_deleted_write_has_no_event_gradient_in_skip_mode() -> None:
    teacher = _teacher()
    events = torch.randn(1, 4, teacher.event_dim, requires_grad=True)
    gate = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    output = teacher(events, write_mask=gate)
    output.memory_tokens[:, 0].sum().backward()
    assert events.grad is not None
    torch.testing.assert_close(events.grad[:, 0], torch.zeros_like(events.grad[:, 0]), rtol=0, atol=0)


def test_prefix_summary_excludes_padding() -> None:
    prefix = torch.tensor(
        [[[[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]]]], dtype=torch.float32
    )
    mask = torch.tensor([[[True, True, False]]])
    summary = summarize_prefix(prefix, mask)
    torch.testing.assert_close(summary, torch.tensor([[[3.0, 5.0]]]))


def test_prefix_conditioner_appends_one_valid_memory_token() -> None:
    prefix = torch.randn(2, 4, 6)
    pad = torch.ones(2, 4, dtype=torch.bool)
    attention = torch.zeros(2, 4, dtype=torch.bool)
    memory = torch.randn(2, 5)
    conditioner = HistoryPrefixConditioner(memory_dim=5, prefix_dim=6)

    embeddings, new_pad, new_attention = conditioner(prefix, pad, attention, memory)
    assert embeddings.shape == (2, 5, 6)
    assert new_pad.shape == (2, 5)
    assert new_attention.shape == (2, 5)
    assert new_pad[:, -1].tolist() == [True, True]
    assert new_attention[:, -1].tolist() == [True, True]
    # The learned projection is the only transformation of the appended token;
    # existing prefix columns are copied exactly.
    torch.testing.assert_close(embeddings[:, :-1], prefix)


def test_append_history_memory_masks_invalid_rows() -> None:
    prefix = torch.zeros(1, 2, 3)
    pad = torch.ones(1, 2, dtype=torch.bool)
    attention = torch.zeros(1, 2, dtype=torch.bool)
    memory = torch.ones(1, 3)
    embeddings, new_pad, _ = append_history_memory(
        prefix,
        pad,
        attention,
        memory,
        memory_valid=torch.tensor([False]),
    )
    torch.testing.assert_close(embeddings[:, -1], torch.zeros(1, 3))
    assert new_pad[:, -1].tolist() == [False]


def test_sequence_conditioner_preserves_frame_order() -> None:
    conditioner = HistoryPrefixConditioner(memory_dim=4, prefix_dim=4)
    prefix = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    pad = torch.ones(2, 3, 2, dtype=torch.bool)
    attention = torch.zeros_like(pad)
    memory = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, False, True], [True, True, False]])
    embeddings, new_pad, _ = conditioner.condition_sequence(
        prefix, pad, attention, memory, memory_valid=valid
    )
    assert embeddings.shape == (2, 3, 3, 4)
    torch.testing.assert_close(embeddings[:, :, :-1], prefix)
    assert new_pad[:, :, -1].equal(valid)


def test_provenance_contract_is_json_safe_and_validated() -> None:
    teacher = _teacher()
    metadata = teacher.provenance(
        teacher_checkpoint="/tmp/history_teacher.pt",
        source_policy_checkpoint="clean-ttt-step-1000",
    )
    assert metadata["format"] == HISTORY_TEACHER_FORMAT
    assert metadata["event_schema"] == HISTORY_EVENT_SCHEMA
    assert metadata["deletion_schema"] == HISTORY_DELETION_SCHEMA
    validate_history_teacher_provenance(metadata)

    malformed = dict(metadata)
    malformed["causal"] = False
    with pytest.raises(ValueError, match="causal=true"):
        validate_history_teacher_provenance(malformed)


def test_api_rejects_future_or_malformed_masks() -> None:
    teacher = _teacher()
    events = torch.randn(1, 3, teacher.event_dim)
    with pytest.raises(ValueError, match="write_mask values"):
        teacher(events, write_mask=torch.tensor([[1.2, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="valid_mask"):
        teacher(events, valid_mask=torch.ones(1, 2, dtype=torch.bool))
