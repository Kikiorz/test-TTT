"""Pure tests for the offline selected-event grounding contract."""

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
