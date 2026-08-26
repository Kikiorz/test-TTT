"""Focused contracts for the explicit V3 full-history action teacher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_history_teacher_module():
    path = Path(__file__).parents[1] / "src" / "lerobot" / "policies" / "smolvla_ttt" / "history_teacher.py"
    spec = importlib.util.spec_from_file_location("history_teacher_for_test", path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_history_teacher_is_strictly_causal_and_uses_previous_action() -> None:
    module = _load_history_teacher_module()
    torch.manual_seed(4)
    teacher = module.FullHistoryActionTeacher(event_dim=5, action_dim=3, hidden_dim=8).eval()
    events = torch.randn(1, 6, 5)
    previous = torch.randn(1, 6, 3)
    with torch.no_grad():
        reference = teacher(events, previous).actions
        changed_future_event = teacher(torch.cat((events[:, :3], events[:, 3:] + 100), dim=1), previous).actions
        changed_future_action = teacher(
            events,
            torch.cat((previous[:, :3], previous[:, 3:] + 100), dim=1),
        ).actions
    # Neither future observation nor future executed action may affect an
    # earlier prediction.
    assert torch.allclose(reference[:, :3], changed_future_event[:, :3])
    assert torch.allclose(reference[:, :3], changed_future_action[:, :3])
    # The previous executed action is an actual causal input at the frame where
    # it is supplied, so changing it should be observable at that frame or
    # later (with overwhelming probability for a randomly initialized model).
    assert not torch.allclose(reference[:, 2:], changed_future_action[:, 2:])


def test_replay_pair_supports_delete_and_content_replacement() -> None:
    module = _load_history_teacher_module()
    torch.manual_seed(7)
    teacher = module.FullHistoryActionTeacher(event_dim=4, action_dim=2, hidden_dim=6).eval()
    events = torch.randn(5, 4)
    previous = torch.randn(5, 2)
    replacement = torch.randn_like(events)
    full, deleted = teacher.replay_pair(
        events,
        previous,
        intervention_mask=torch.tensor([False, False, True, False, False]),
    )
    assert full.actions.shape == (5, 2)
    assert deleted.actions.shape == (5, 2)
    assert torch.isfinite(deleted.actions).all()
    _, replaced = teacher.replay_pair(
        events,
        previous,
        replacement_event_tokens=replacement,
        intervention_mask=torch.tensor([False, False, True, False, False]),
    )
    assert not torch.allclose(full.actions[2:], replaced.actions[2:])


def test_chunked_replay_preserves_causal_state() -> None:
    module = _load_history_teacher_module()
    torch.manual_seed(11)
    teacher = module.FullHistoryActionTeacher(event_dim=3, action_dim=2, hidden_dim=7).eval()
    events = torch.randn(1, 9, 3)
    previous = torch.randn(1, 9, 2)
    with torch.no_grad():
        complete = teacher(events, previous)
        first = teacher(events[:, :4], previous[:, :4])
        second = teacher(events[:, 4:], previous[:, 4:], state=first.state)
    chunked_actions = torch.cat((first.actions, second.actions), dim=1)
    assert torch.allclose(complete.actions, chunked_actions, atol=1e-6, rtol=1e-6)
    assert torch.equal(complete.state.position, second.state.position)


def test_previous_action_validity_respects_padded_rows() -> None:
    """A padded predecessor must not become a causal action input later."""

    module = _load_history_teacher_module()
    torch.manual_seed(12)
    teacher = module.FullHistoryActionTeacher(event_dim=3, action_dim=2, hidden_dim=7).eval()
    events = torch.randn(1, 4, 3)
    previous = torch.randn(1, 4, 2)
    valid = torch.tensor([[True, False, True, True]])
    previous_changed = previous.clone()
    previous_changed[:, 1] += 100.0
    with torch.no_grad():
        reference = teacher(events, previous, valid_mask=valid).actions
        changed = teacher(events, previous_changed, valid_mask=valid).actions
    # Frame 2 follows an invalid/padded frame, so its predecessor is marked
    # unavailable and changing the padded action cannot affect that output.
    torch.testing.assert_close(reference[:, 2], changed[:, 2], rtol=0, atol=0)


def test_pairwise_control_credit_keeps_delay_mask_and_signed_effect() -> None:
    module = _load_history_teacher_module()
    # One full replay and two event interventions, with a deliberately useful
    # first event and a harmful second event.
    # The full replay is intentionally imperfect so the second intervention
    # can *improve* the expert match; this exercises signed (negative) credit.
    full = torch.ones(4, 1)
    counterfactual = torch.tensor(
        [
            [[0.0], [2.0], [2.0], [2.0]],
            [[0.0], [0.5], [0.5], [0.5]],
        ]
    )
    target = torch.zeros(4, 1)
    result = module.compute_pairwise_control_credit(
        full,
        counterfactual,
        target,
        event_ends=[1, 2],
    )
    assert result.utility.shape == (1, 2, 4)
    assert result.action_effect.shape == (1, 2, 4, 1)
    assert not bool(result.pair_mask[0, 0, 0])
    assert not bool(result.pair_mask[0, 1, :2].any())
    assert bool((result.utility[0, 0, 1:] > 0).any())
    # Counterfactual branch 1 improves the target loss, so its signed credit
    # is negative and is retained in raw_degradation rather than clipped away.
    assert bool((result.raw_degradation[0, 1, 2:] < 0).all())


def test_full_history_teacher_checkpoint_roundtrip_and_hash_validation(tmp_path: Path) -> None:
    module = _load_history_teacher_module()
    torch.manual_seed(13)
    teacher = module.FullHistoryActionTeacher(event_dim=3, action_dim=2, hidden_dim=5)
    checkpoint = tmp_path / "teacher.pt"
    manifest = teacher.save_checkpoint(checkpoint, metadata={"dataset_id": "toy"})
    loaded, loaded_manifest = module.load_full_history_teacher_checkpoint(checkpoint)
    assert manifest["format"] == module.FULL_HISTORY_TEACHER_FORMAT
    assert loaded_manifest["parameter_sha256"] == manifest["parameter_sha256"]
    events = torch.randn(1, 4, 3)
    previous = torch.randn(1, 4, 2)
    with torch.no_grad():
        assert torch.allclose(teacher(events, previous).actions, loaded(events, previous).actions)

    tampered = dict(manifest)
    tampered["parameter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="state hash"):
        module.validate_full_history_teacher_provenance(tampered, teacher=teacher)


def test_full_history_teacher_compute_loss_backpropagates() -> None:
    module = _load_history_teacher_module()
    torch.manual_seed(17)
    teacher = module.FullHistoryActionTeacher(event_dim=4, action_dim=2, hidden_dim=6)
    events = torch.randn(2, 5, 4)
    previous = torch.randn(2, 5, 2)
    targets = torch.randn(2, 5, 2)
    loss, metrics = teacher.compute_loss(events, targets, previous)
    assert torch.isfinite(loss)
    assert metrics["loss"] == float(loss.detach())
    loss.backward()
    assert any(parameter.grad is not None for parameter in teacher.parameters())
