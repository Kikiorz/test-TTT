"""Contracts for the optional padded full-history teacher execution path."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_training_module():
    """Load the example without importing LeRobot's optional policy registry.

    The training script imports only ``history_teacher`` at module import time;
    stubbing the package parents lets this focused test run in the lightweight
    test environment used by the history-teacher contracts.
    """

    root = Path(__file__).parents[1]
    package_names = ("lerobot", "lerobot.policies", "lerobot.policies.smolvla_ttt")
    previous = {name: sys.modules.get(name) for name in package_names}
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = package

    history_path = root / "src" / "lerobot" / "policies" / "smolvla_ttt" / "history_teacher.py"
    history_name = "lerobot.policies.smolvla_ttt.history_teacher"
    history_spec = importlib.util.spec_from_file_location(history_name, history_path)
    if history_spec is None or history_spec.loader is None:
        raise RuntimeError(f"could not load {history_path}")
    history_module = importlib.util.module_from_spec(history_spec)
    sys.modules[history_name] = history_module
    history_spec.loader.exec_module(history_module)

    training_path = root / "examples" / "mikasa" / "train_full_history_teacher.py"
    training_name = "train_full_history_teacher_for_batching_test"
    training_spec = importlib.util.spec_from_file_location(training_name, training_path)
    if training_spec is None or training_spec.loader is None:
        raise RuntimeError(f"could not load {training_path}")
    training_module = importlib.util.module_from_spec(training_spec)
    sys.modules[training_name] = training_module
    training_spec.loader.exec_module(training_module)

    # Leave package state as it was for neighboring tests while retaining the
    # uniquely named loaded modules used by this test module.
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    return training_module, history_module


def _toy_rows(lengths: list[int], *, event_dim: int = 5, action_dim: int = 3):
    torch.manual_seed(101)
    rows = []
    for length in lengths:
        rows.append(
            {
                "event_tokens": torch.randn(length, event_dim),
                "previous_executed_actions": torch.randn(length, action_dim),
                "target_actions": torch.randn(length, action_dim),
            }
        )
    return rows


def test_pad_episode_batch_right_pads_and_preserves_rows() -> None:
    train, _history = _load_training_module()
    rows = _toy_rows([2, 5, 3])
    events, previous, targets, valid = train._pad_episode_batch(rows, device=torch.device("cpu"))

    assert events.shape == (3, 5, 5)
    assert previous.shape == targets.shape == (3, 5, 3)
    assert valid.tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]
    for batch_index, row in enumerate(rows):
        length = row["event_tokens"].shape[0]
        torch.testing.assert_close(events[batch_index, :length], row["event_tokens"])
        torch.testing.assert_close(previous[batch_index, :length], row["previous_executed_actions"])
        torch.testing.assert_close(targets[batch_index, :length], row["target_actions"])
        assert torch.count_nonzero(events[batch_index, length:]) == 0
        assert torch.count_nonzero(previous[batch_index, length:]) == 0
        assert torch.count_nonzero(targets[batch_index, length:]) == 0


def test_masked_batch_loss_matches_frame_weighted_independent_replays() -> None:
    train, history = _load_training_module()
    torch.manual_seed(102)
    teacher = history.FullHistoryActionTeacher(event_dim=5, action_dim=3, hidden_dim=9).eval()
    rows = _toy_rows([2, 5, 3])
    events, previous, targets, valid = train._pad_episode_batch(rows, device=torch.device("cpu"))

    with torch.no_grad():
        batched = teacher(events, previous, valid_mask=valid)
        batch_loss = teacher.action_loss(batched.actions, targets, valid_mask=valid)
        weighted_numerator = torch.zeros(())
        frame_count = 0
        for row in rows:
            episode_events, episode_previous, episode_targets = train._episode_tensors(
                row, device=torch.device("cpu")
            )
            episode_output = teacher(episode_events, episode_previous)
            episode_loss = teacher.action_loss(episode_output.actions, episode_targets)
            length = int(episode_events.shape[1])
            weighted_numerator = weighted_numerator + episode_loss * length
            frame_count += length
        independent_loss = weighted_numerator / frame_count

    torch.testing.assert_close(batch_loss, independent_loss, rtol=2e-6, atol=2e-7)
    # Every real prefix is independent across batch rows; padding emits zeros
    # and therefore cannot contribute to the masked objective.
    for batch_index, row in enumerate(rows):
        length = row["event_tokens"].shape[0]
        episode_events, episode_previous, _ = train._episode_tensors(
            row, device=torch.device("cpu")
        )
        with torch.no_grad():
            expected = teacher(episode_events, episode_previous).actions
        torch.testing.assert_close(batched.actions[batch_index, :length], expected[0])
        assert torch.count_nonzero(batched.actions[batch_index, length:]) == 0


def test_batch_size_one_delegates_to_legacy_epoch() -> None:
    train, history = _load_training_module()
    rows = _toy_rows([2, 4, 3])
    torch.manual_seed(103)
    first = history.FullHistoryActionTeacher(event_dim=5, action_dim=3, hidden_dim=8)
    second = history.FullHistoryActionTeacher(event_dim=5, action_dim=3, hidden_dim=8)
    second.load_state_dict(first.state_dict())
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=1e-3)
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=1e-3)

    old_loss = train._run_teacher_epoch(first, rows, first_optimizer, device=torch.device("cpu"))
    new_loss = train._run_teacher_epoch_batched(
        second,
        rows,
        second_optimizer,
        device=torch.device("cpu"),
        episode_batch_size=1,
    )
    assert old_loss == new_loss
    for old_parameter, new_parameter in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(old_parameter, new_parameter, rtol=0, atol=0)


def test_batch_helpers_reject_empty_or_nonpositive_batch() -> None:
    train, _history = _load_training_module()
    with pytest.raises(ValueError, match="empty episode batch"):
        train._pad_episode_batch([], device=torch.device("cpu"))
    with pytest.raises(ValueError, match="episode_batch_size must be positive"):
        train._run_teacher_epoch_batched(
            object(), [], None, device=torch.device("cpu"), episode_batch_size=0
        )
