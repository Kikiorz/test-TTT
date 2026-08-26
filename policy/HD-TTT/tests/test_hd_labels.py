"""Pure tests for the offline selected-event grounding contract."""

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_builder():
    path = Path(__file__).parents[1] / "examples" / "mikasa" / "build_hd_labels.py"
    spec = importlib.util.spec_from_file_location("hd_label_builder_for_test", path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_grounding_event_rejects_terminal_one_frame_winner() -> None:
    builder = _load_builder()
    # The old mean-over-all-future rule would select event 2 (.0042 over one
    # frame), despite event 0 having much larger cumulative credit.  A 64
    # frame horizon keeps the single selected wrong branch aligned with a
    # meaningful future.
    selected, mode = builder._select_grounding_event(
        [469, 401, 1],
        [1.64555, 0.52424, 0.00423],
        [1.64555 / 469, 0.52424 / 401, 0.00423],
        min_future_frames=64,
    )
    assert selected == 0
    assert mode == "min_future_horizon_mean"


def test_teacher_checkpoint_contract_preserves_prefix_writer_mode(tmp_path) -> None:
    builder = _load_builder()
    checkpoint = tmp_path / "prefix_teacher"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "smolvla_ttt",
                "hd_ttt_enabled": False,
                "hd_learned_write_gate": False,
                "ttt_layer_indices": [12, 13, 14, 15],
                "ttt_num_register_tokens": 16,
                "ttt_writer_mode": "prefix_only",
            }
        ),
        encoding="utf-8",
    )

    info = builder._validate_teacher_checkpoint(checkpoint)

    assert info["ttt_writer_mode"] == "prefix_only"
    assert info["ttt_stable_inner_update"] is False


def test_teacher_checkpoint_contract_normalizes_nullable_stable_flag(tmp_path) -> None:
    builder = _load_builder()
    checkpoint = tmp_path / "legacy_teacher"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "smolvla_ttt",
                "hd_ttt_enabled": False,
                "hd_learned_write_gate": False,
                "ttt_layer_indices": [12, 13, 14, 15],
                "ttt_num_register_tokens": 16,
                "ttt_stable_inner_update": None,
            }
        ),
        encoding="utf-8",
    )

    info = builder._validate_teacher_checkpoint(checkpoint)

    assert info["ttt_stable_inner_update"] is False


def test_selected_grounding_event_uses_total_credit_for_short_episode() -> None:
    builder = _load_builder()
    selected, mode = builder._select_grounding_event(
        [12, 8],
        [0.2, 0.5],
        [0.2 / 12, 0.5 / 8],
        min_future_frames=64,
    )
    assert selected == 1
    assert mode == "total_credit_fallback"


def test_zero_horizon_preserves_mean_credit_selection() -> None:
    builder = _load_builder()
    selected, mode = builder._select_grounding_event(
        [469, 401, 1],
        [1.64555, 0.52424, 0.00423],
        [1.64555 / 469, 0.52424 / 401, 0.00423],
        min_future_frames=0,
    )
    assert selected == 2
    assert mode == "min_future_horizon_mean"


def test_episode_label_replay_keeps_rho_and_wrong_branch_on_same_event() -> None:
    builder = _load_builder()

    class _Policy:
        config = SimpleNamespace(
            action_feature=SimpleNamespace(shape=(1,)),
            max_action_dim=1,
            chunk_size=1,
        )

        @staticmethod
        def prepare_action(prepared):
            return prepared["actions"]

    original_replay = builder._run_replay

    def fake_replay(policy, prepared, noise, time, *, frame_batch_size, write_gate=None):
        del policy, noise, time, frame_batch_size
        length = int(prepared["actions"].shape[0])
        velocity = torch.zeros(length, 1, 1)
        if write_gate is not None:
            zeros = torch.nonzero(write_gate.reshape(-1) == 0).flatten()
            event_start = int(zeros[0])
            if event_start == 0:
                velocity[4:, 0, 0] = 1.0
            elif event_start >= length - 4:
                velocity[-1, 0, 0] = 2.0
        return velocity

    builder._run_replay = fake_replay
    try:
        length = 20
        prepared = {"actions": torch.zeros(length, 1, 1)}
        labels = builder._episode_labels(
            _Policy(),
            prepared,
            torch.zeros(length, 1, 1),
            torch.ones(length),
            event_block_size=4,
            max_events=0,
            attribution_threshold=0.0,
            frame_batch_size=4,
            grounding_min_future_frames=64,
        )
    finally:
        builder._run_replay = original_replay

    assert labels["hd_selected_event"].tolist() == [0, 4]
    assert labels["hd_grounding_selection_mode"] == "total_credit_fallback"
    assert int((labels["hd_rho"] > 0).sum()) > 1


def test_v2_effect_validity_starts_after_event_end() -> None:
    """Action-effect labels must not supervise an event from its own frame."""

    builder = _load_builder()

    class _Policy:
        config = SimpleNamespace(
            action_feature=SimpleNamespace(shape=(1,)),
            max_action_dim=1,
            chunk_size=1,
        )

        @staticmethod
        def prepare_action(prepared):
            return prepared["actions"]

    original_replay = builder._run_replay

    def fake_replay(policy, prepared, noise, time, *, frame_batch_size, write_gate=None):
        del policy, noise, time, frame_batch_size
        length = int(prepared["actions"].shape[0])
        velocity = torch.zeros(length, 1, 1)
        if write_gate is not None:
            zeros = torch.nonzero(write_gate.reshape(-1) == 0).flatten()
            if zeros.numel():
                event_start = int(zeros[0])
                # Removing an event changes only its causal future.
                velocity[event_start + 4 :, 0, 0] = 1.0
        return velocity

    builder._run_replay = fake_replay
    try:
        length = 12
        labels = builder._episode_labels(
            _Policy(),
            {"actions": torch.zeros(length, 1, 1)},
            torch.zeros(length, 1, 1),
            torch.ones(length),
            event_block_size=4,
            max_events=0,
            attribution_threshold=0.0,
            frame_batch_size=4,
            grounding_min_future_frames=0,
            attribution_protocol="v2",
        )
    finally:
        builder._run_replay = original_replay

    assert labels["hd_attribution_protocol"] == builder.HD_ATTRIBUTION_PROTOCOL_V2
    events = labels["hd_effect_events"]
    valid = labels["hd_effect_valid"]
    # The selected event is branch zero and must be represented by a causal
    # mask whose first valid row is exactly the exclusive event end.
    selected_start, selected_end = events[0].tolist()
    assert selected_start == 0
    assert selected_end == 4
    assert torch.all(valid[:selected_end, 0] == 0)
    assert torch.all(valid[selected_end:, 0] == 1)
    # New v2 artifacts use one selected branch.  Older K>1 artifacts remain
    # readable, but their extra branches are intentionally ignored online.
    assert events.shape[0] == builder.V2_EFFECT_BRANCHES == 1
