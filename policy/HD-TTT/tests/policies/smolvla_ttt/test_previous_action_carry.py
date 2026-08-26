"""Causal predecessor handling across CreditTTT TBPTT segment boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

# Importing the full policy pulls optional LeRobot dataset/model extras.  The
# test is still useful in the normal training environment, while remaining
# collectable (as a skip) in the small static-analysis environment.
pytest.importorskip("datasets")
pytest.importorskip("transformers")

from lerobot.policies.smolvla_ttt.modeling_smolvla_ttt import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    SmolVLATTTPolicy,
)
from lerobot.policies.smolvla_ttt.configuration_smolvla_ttt import (  # noqa: E402
    SmolVLATTTConfig,
)


class _RecordingFlow(nn.Module):
    """Minimal flow adapter recording the writer predecessor it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.previous_actions: list[torch.Tensor | None] = []

    def clear_ttt_diagnostics(self) -> None:
        return None

    def forward_with_state(self, *args, previous_actions=None, fast_states=None, **kwargs):
        actions = args[5]
        self.previous_actions.append(
            None if previous_actions is None else previous_actions.detach().clone()
        )
        # Keep a non-empty numerical state so the next call is recognized as
        # a continuation rather than a fresh sequence.
        return torch.zeros_like(actions), {0: torch.zeros(1)}


class _PolicyHarness(SmolVLATTTPolicy):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            adapt_to_pi_aloha=False,
            hd_v3_include_previous_action=True,
            hd_ttt_enabled=False,
            ttt_stable_inner_update=False,
            action_feature=SimpleNamespace(shape=(2,)),
        )
        self.model = _RecordingFlow()

    def prepare_images(self, batch):
        return [], []

    def prepare_state(self, batch):
        return batch[OBS_STATE]

    def prepare_action(self, batch):
        return batch[ACTION]


def _batch(values: list[float]) -> dict[str, torch.Tensor]:
    action = torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1)
    # The policy's sequence path only needs these fields after the preprocessing
    # hooks above; language tensors are intentionally tiny.
    return {
        ACTION: action,
        OBS_STATE: torch.zeros(len(values), 1),
        OBS_LANGUAGE_TOKENS: torch.zeros(len(values), 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(len(values), 1, dtype=torch.bool),
    }


def test_previous_action_carry_is_causal_and_resets_between_sequences() -> None:
    policy = _PolicyHarness()
    # The harness has one action feature but uses a two-dimensional padded
    # action in order to exercise the exact slot-0 shape used by SmolVLA.
    policy.config.action_feature.shape = (1,)

    # Segment 0: no predecessor at the episode boundary; row 1 receives row
    # 0's executed slot-0 action.
    _, _, state = policy.forward_sequence_segment(
        _batch([1.0, 2.0]),
        sequence_shape=(1, 2),
        noise=torch.zeros(2, 1, 1),
        time=torch.zeros(2),
    )
    first = policy.model.previous_actions[0]
    assert first is not None
    torch.testing.assert_close(first, torch.tensor([[[0.0], [1.0]]]))

    # Segment 1: row 0 must use the final slot-0 action from segment 0, then
    # shift within the new segment as usual.
    _, _, _ = policy.forward_sequence_segment(
        _batch([3.0, 4.0]),
        sequence_shape=(1, 2),
        fast_states=state,
        noise=torch.zeros(2, 1, 1),
        time=torch.zeros(2),
    )
    second = policy.model.previous_actions[1]
    assert second is not None
    torch.testing.assert_close(second, torch.tensor([[[2.0], [3.0]]]))

    # ``fast_states=None`` is the explicit new-window marker; stale carry must
    # not leak across independent sampled windows.
    _, _, _ = policy.forward_sequence_segment(
        _batch([5.0]),
        sequence_shape=(1, 1),
        fast_states=None,
        noise=torch.zeros(1, 1, 1),
        time=torch.zeros(1),
    )
    third = policy.model.previous_actions[2]
    assert third is not None
    torch.testing.assert_close(third, torch.tensor([[[0.0]]]))

    # An explicit boundary argument overrides both reset and carried values.
    _, _, _ = policy.forward_sequence_segment(
        _batch([6.0]),
        sequence_shape=(1, 1),
        fast_states=None,
        previous_action_at_start=torch.tensor([[9.0]]),
        noise=torch.zeros(1, 1, 1),
        time=torch.zeros(1),
    )
    fourth = policy.model.previous_actions[3]
    assert fourth is not None
    torch.testing.assert_close(fourth, torch.tensor([[[9.0]]]))


def test_previous_action_boundary_shape_validation() -> None:
    value = torch.ones(2, 3)
    normalized = SmolVLATTTPolicy._coerce_previous_action_at_start(
        value,
        batch_size=2,
        action_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert normalized is not None
    assert normalized.shape == (2, 4)
    assert normalized.requires_grad is False

    with pytest.raises(ValueError, match="batch_size>1"):
        SmolVLATTTPolicy._coerce_previous_action_at_start(
            torch.ones(3),
            batch_size=2,
            action_dim=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_credit_ttt_config_uses_numeric_final_layer_and_excludes_v2_effect() -> None:
    common = dict(
        hd_ttt_enabled=True,
        hd_attribution_protocol="credit_ttt_v3",
        ttt_writer_mode="prefix_only",
        hd_effect_weight=0.0,
    )
    # User-provided layer order is not an architectural ordering guarantee.
    # The numerically final selected layer (15) is the effect/readout layer.
    config = SmolVLATTTConfig(**common, ttt_layer_indices=[15, 12], hd_v3_intervention="content_replacement")
    assert config.hd_v3_intervention == "replace"
    assert config.hd_v3_include_previous_action is True

    delete_config = SmolVLATTTConfig(
        **common,
        ttt_layer_indices=[12, 15],
        hd_v3_intervention="content_deletion",
    )
    assert delete_config.hd_v3_intervention == "delete"

    with pytest.raises(ValueError, match="final VLM/action-expert layer"):
        SmolVLATTTConfig(**common, ttt_layer_indices=[12, 14])
    with pytest.raises(ValueError, match="legacy v2 action-effect"):
        SmolVLATTTConfig(
            **{**common, "hd_effect_weight": 1.0},
            ttt_layer_indices=[15],
        )
    with pytest.raises(ValueError, match="at least one selected TTT layer"):
        SmolVLATTTConfig(**common, ttt_layer_indices=[])
