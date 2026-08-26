"""Contracts for the independent CreditTTT V3 pair-label builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_builder():
    root = Path(__file__).parents[1]
    # The builder imports the two tensor-only policy modules by their package
    # names.  The normal test environment already exposes ``src``; loading by
    # file path keeps this focused test usable even when the full LeRobot
    # dependency graph is intentionally not imported.
    import sys
    import types

    package_names = ["lerobot", "lerobot.policies", "lerobot.policies.smolvla_ttt"]
    for name in package_names:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load(
        "lerobot.policies.smolvla_ttt.credit_ttt_v3",
        root / "src/lerobot/policies/smolvla_ttt/credit_ttt_v3.py",
    )
    load(
        "lerobot.policies.smolvla_ttt.history_teacher",
        root / "src/lerobot/policies/smolvla_ttt/history_teacher.py",
    )
    return load("credit_builder_for_test", root / "examples/mikasa/build_credit_labels.py")


def _artifacts(tmp_path: Path, *, with_queries: bool = False, replacement: bool = False):
    builder = _load_builder()
    torch.manual_seed(31)
    teacher = builder.FullHistoryActionTeacher(
        event_dim=4,
        action_dim=3,
        hidden_dim=8,
        target_mode="normalized_executed_slot0_action",
    )
    checkpoint = tmp_path / "teacher.pt"
    teacher.save_checkpoint(checkpoint)
    rows = []
    for episode in range(2):
        length = 9 + episode
        row = {
            "event_tokens": torch.randn(length, 4),
            "previous_executed_actions": torch.randn(length, 3),
            "target_actions": torch.randn(length, 3),
            "global_indices": torch.arange(length) + episode * 20,
            "episode_indices": torch.full((length,), episode),
            "frame_indices": torch.arange(length),
        }
        if with_queries:
            row["future_query_q"] = torch.randn(length, 5)
            row["future_action_tail_h"] = torch.randn(length, 6)
        if replacement:
            row["replacement_event_tokens"] = torch.randn(length, 4)
        rows.append(row)
    features = tmp_path / "features.pt"
    torch.save(
        {
            "format": builder.FEATURE_FORMAT,
            "metadata": {
                "event_dim": 4,
                "action_dim": 3,
                "target": "normalized_executed_slot0_action",
                "dataset_repo_id": "synthetic-credit-v3",
                "fps": 30,
                "causal_previous_action": True,
            },
            "episodes": rows,
        },
        features,
    )
    return builder, teacher, features, checkpoint


def test_builder_emits_causal_frame_aligned_pairs_and_is_deterministic(tmp_path: Path) -> None:
    builder, _teacher, features, checkpoint = _artifacts(tmp_path)
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    builder.build_labels(features, checkpoint, first_path, pair_k=3, seed=123)
    builder.build_labels(features, checkpoint, second_path, pair_k=3, seed=123)
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    second = torch.load(second_path, map_location="cpu", weights_only=True)
    assert first["metadata"]["format"] == builder.CREDIT_TTT_V3_FORMAT
    assert first["metadata"]["pair_schema"] == builder.CREDIT_TTT_V3_PAIR_SCHEMA
    assert first["metadata"]["target_mode"] == "normalized_executed_slot0_action"
    assert first["metadata"]["antithetic_noise"] is False
    assert first["metadata"]["intervention_scope"] == (
        "event_write_only_previous_executed_action_held_fixed"
    )
    metadata = first["metadata"]
    # Full-history provenance is episode-indexed rather than merely a global
    # max length.  These invariants are consumed by the trainer's fail-closed
    # selected-episode check.
    assert metadata["min_sequence_length"] == max(metadata["episode_lengths"])
    assert len(metadata["episode_slices"]) == len(metadata["episode_lengths"])
    for item, length in zip(metadata["episode_slices"], metadata["episode_lengths"], strict=True):
        assert item["row_end"] - item["row_start"] == item["length"] == length
    for key in (
        "hd_v3_pair_event_index",
        "hd_v3_pair_future_index",
        "hd_v3_pair_delay",
        "hd_v3_pair_delay_bin",
        "hd_v3_pair_valid",
    ):
        torch.testing.assert_close(first[key], second[key])
    valid = first["hd_v3_pair_valid"]
    assert bool(
        (first["hd_v3_pair_future_index"][valid] > first["hd_v3_pair_event_index"][valid]).all()
    )
    assert first["hd_v3_pair_effect"].shape[-1] == 3


def test_builder_supports_content_replacement_and_query_features(tmp_path: Path) -> None:
    builder, _teacher, features, checkpoint = _artifacts(
        tmp_path, with_queries=True, replacement=True
    )
    output = tmp_path / "replace.pt"
    report = builder.build_labels(
        features,
        checkpoint,
        output,
        pair_k=2,
        intervention="replace",
        require_query_features=True,
    )
    assert report["metadata"]["intervention_schema"] == builder.CREDIT_TTT_V3_INTERVENTION
    assert report["metadata"]["intervention_scope"] == (
        "event_content_replacement_previous_executed_action_held_fixed"
    )
    assert report["metadata"]["query_features_available"] is True
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert "hd_v3_pair_query" in payload
    assert payload["hd_v3_pair_query"].shape[:2] == (19, 2)


def test_builder_fails_fast_when_query_features_are_required(tmp_path: Path) -> None:
    builder, _teacher, features, checkpoint = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="query features"):
        builder.build_labels(
            features,
            checkpoint,
            tmp_path / "missing.pt",
            pair_k=2,
            require_query_features=True,
        )


def test_canonical_builder_rejects_multi_frame_event_spans(tmp_path: Path) -> None:
    builder, _teacher, features, checkpoint = _artifacts(tmp_path)
    payload = torch.load(features, map_location="cpu", weights_only=True)
    payload["episodes"][0]["event_starts"] = torch.arange(9)
    payload["episodes"][0]["event_ends"] = torch.tensor([2, 2, 3, 4, 5, 6, 7, 8, 9])
    malformed = tmp_path / "multi_frame_features.pt"
    torch.save(payload, malformed)
    with pytest.raises(ValueError, match="one event span per frame"):
        builder.build_labels(
            malformed,
            checkpoint,
            tmp_path / "malformed.pt",
            pair_k=2,
        )
