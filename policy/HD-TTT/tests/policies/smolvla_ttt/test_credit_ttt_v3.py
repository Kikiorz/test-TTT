"""Focused tests for the tensor-only CreditTTT V3 primitives."""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.smolvla_ttt.credit_ttt_v3 import (
    CreditTTTProtocol,
    CREDIT_TTT_V3_PAIR_SCHEMA,
    DEFAULT_CREDIT_TTT_PROTOCOL,
    functional_local_ttt_update,
    local_update_read_before_after,
    query_conditioned_local_effect_loss,
    sample_delay_balanced_pairs,
    symmetric_relative_utility,
)
from lerobot.policies.smolvla_ttt.ttt import TTTBoundedTrace, TTTMLPLayer


def test_protocol_is_versioned_and_json_safe() -> None:
    metadata = DEFAULT_CREDIT_TTT_PROTOCOL.as_dict()
    assert metadata["version"] == 3
    assert metadata["pair_schema"] == CREDIT_TTT_V3_PAIR_SCHEMA
    assert metadata["intervention_scope"] == (
        "event_write_only_previous_executed_action_held_fixed"
    )
    assert metadata["causal"] is True
    # Every value is directly serializable by the JSON artifact writer.
    import json

    json.dumps(metadata)
    assert CreditTTTProtocol.from_dict(metadata) == DEFAULT_CREDIT_TTT_PROTOCOL
    with pytest.raises(ValueError, match="protocol"):
        CreditTTTProtocol.from_dict({**metadata, "protocol": "legacy"})


def test_symmetric_relative_utility_is_signed_and_swap_symmetric() -> None:
    reference = torch.tensor([1.0, 2.0, 0.0])
    counterfactual = torch.tensor([2.0, 1.0, 0.0])
    utility = symmetric_relative_utility(reference, counterfactual)
    swapped = symmetric_relative_utility(counterfactual, reference)
    assert utility[0] > 0
    assert utility[1] < 0
    torch.testing.assert_close(utility, -swapped, rtol=0, atol=1e-6)
    assert utility[2].item() == 0.0


def test_delay_sampler_is_causal_balanced_and_reproducible() -> None:
    # A triangular utility matrix has both useful and null pairs at every
    # delay.  Explicit padding exercises the fixed-shape artifact contract.
    utility = torch.tensor(
        [
            [0.0, 1.0, 0.0, -1.0, 2.0],
            [0.0, 0.0, 1.5, 0.0, -0.5],
            [0.0, 0.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    first = sample_delay_balanced_pairs(
        utility,
        pairs_per_bin=2,
        num_delay_bins=3,
        pad_to=8,
        generator=torch.Generator().manual_seed(23),
    )
    second = sample_delay_balanced_pairs(
        utility,
        pairs_per_bin=2,
        num_delay_bins=3,
        pad_to=8,
        generator=torch.Generator().manual_seed(23),
    )
    for left, right in zip(
        (first.event_index, first.future_index, first.delay, first.valid_mask),
        (second.event_index, second.future_index, second.delay, second.valid_mask),
    ):
        torch.testing.assert_close(left, right)
    valid = first.valid_mask
    assert bool((first.future_index[valid] > first.event_index[valid]).all().item())
    assert bool((first.delay[valid] == first.future_index[valid] - first.event_index[valid]).all().item())
    assert first.positive_mask[valid].any()
    assert first.null_mask[valid].any()
    assert first.num_pairs == 8
    assert first.num_valid <= first.num_pairs


def test_delay_sampler_never_crosses_batched_episodes() -> None:
    utility = torch.zeros(2, 4, 4)
    utility[0, 0, 3] = 1.0
    utility[1, 1, 3] = 2.0
    sampled = sample_delay_balanced_pairs(
        utility,
        pairs_per_bin=4,
        num_delay_bins=2,
        generator=torch.Generator().manual_seed(7),
    )
    assert set(sampled.batch_index[sampled.valid_mask].tolist()).issubset({0, 1})
    assert all(pair.batch_index in {0, 1} for pair in sampled.pairs)


def test_qh2l_effect_loss_has_positive_and_null_branches() -> None:
    before = torch.zeros(2, 3, requires_grad=True)
    after = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    breakdown = query_conditioned_local_effect_loss(
        before,
        after,
        teacher,
        utility=torch.tensor([1.0, 0.0]),
        return_components=True,
    )
    # The positive row is exact; the null row contributes a non-zero
    # invariance penalty.  Teacher labels are detached by the objective.
    assert breakdown.positive.item() == pytest.approx(0.0, abs=1e-7)
    assert breakdown.null.item() > 0
    breakdown.total.backward()
    assert after.grad is not None
    assert teacher.grad is None

    perturbed = query_conditioned_local_effect_loss(
        before.detach(),
        (after.detach() + 0.5),
        teacher.detach(),
        positive_mask=torch.tensor([True, False]),
        null_mask=torch.tensor([False, True]),
    )
    assert perturbed.item() > breakdown.total.detach().item()


def test_qh2l_empty_batch_is_finite_and_connected() -> None:
    before = torch.randn(0, 4, requires_grad=True)
    after = torch.randn(0, 4, requires_grad=True)
    target = torch.randn(0, 4)
    loss = query_conditioned_local_effect_loss(
        before,
        after,
        target,
        positive_mask=torch.zeros(0, dtype=torch.bool),
        null_mask=torch.zeros(0, dtype=torch.bool),
    )
    assert loss.item() == 0.0
    loss.backward()
    assert after.grad is not None


def test_functional_update_is_differentiable_and_non_mutating() -> None:
    weight = torch.tensor(1.0, requires_grad=True)
    original = weight.detach().clone()
    update_loss = (weight - 2.0).square()
    updated = functional_local_ttt_update(weight, update_loss, 0.25, create_graph=True)
    torch.testing.assert_close(weight.detach(), original)
    assert updated.item() == pytest.approx(1.5)
    outer = (updated - 3.0).square()
    outer.backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad)


def test_before_after_helper_supports_nested_fast_state() -> None:
    state = {
        "weight": torch.tensor(1.0, requires_grad=True),
        "bias": (torch.tensor(0.0, requires_grad=True),),
    }
    update_loss = (state["weight"] - 2.0).square() + state["bias"][0].square()

    def read(current: dict[str, object], query: torch.Tensor) -> torch.Tensor:
        weight = current["weight"]
        bias = current["bias"][0]  # type: ignore[index]
        assert isinstance(weight, torch.Tensor)
        assert isinstance(bias, torch.Tensor)
        return weight * query + bias

    result = local_update_read_before_after(
        state,
        torch.tensor(2.0),
        update_loss,
        read,
        0.1,
    )
    assert result.read_after.item() != result.read_before.item()
    assert result.effect.item() != 0
    (result.effect.square()).backward()
    assert state["weight"].grad is not None
    assert state["bias"][0].grad is not None


def test_ttt_bounded_trace_captures_only_selected_steps_and_preserves_graph() -> None:
    torch.manual_seed(5)
    layer = TTTMLPLayer(dim=4, hidden_dim=6, base_inner_lr=0.05, second_order=True)
    layer.train()
    inputs = torch.randn(2, 4, 3, 4, requires_grad=True)
    sink = []
    output, state, trace = layer(
        inputs,
        trace_indices=[1, 3],
        trace_sink=sink,
        return_bounded_trace=True,
    )
    assert isinstance(trace, TTTBoundedTrace)
    assert trace.indices == (1, 3)
    assert len(sink) == 2
    for transition in trace.transitions:
        assert transition.state_before.position.shape == (2,)
        assert transition.state_after.position.shape == (2,)
        assert transition.query_hidden is not None
        assert transition.query_hidden.shape == (2, 3, 4)
        assert transition.read_hidden is not None
        assert transition.read_hidden.requires_grad
    # The trace is connected to the writer/read path, so an outer objective
    # can backpropagate through the selected local transition.
    outer = sum(item.read_hidden.square().mean() for item in trace.transitions)
    outer.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    # The selected event's state-after snapshot must keep the writer/meta
    # gradient.  Detaching the whole snapshot to save memory would make the
    # local effect objective a read-only diagnostic and break QH2L training.
    assert layer.k_proj.weight.grad is not None
    assert torch.isfinite(layer.k_proj.weight.grad).all()
    assert state.position.tolist() == [3, 3]


def test_ttt_bounded_trace_validates_indices_without_affecting_legacy_return() -> None:
    layer = TTTMLPLayer(dim=4, hidden_dim=6, second_order=False)
    inputs = torch.randn(1, 2, 2, 4)
    output, state = layer(inputs)
    assert output.shape == inputs.shape
    assert state.position.tolist() == [1]
    with pytest.raises(ValueError, match="trace_indices"):
        layer(inputs, trace_indices=[2])


def test_qh2l_reference_routes_all_pairs_to_final_action_replay() -> None:
    """Canonical V3 never mixes a local velocity read with action labels."""

    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    from lerobot.policies.smolvla_ttt.modeling_smolvla_ttt import SmolVLATTTPolicy

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_out_proj = torch.nn.Linear(2, 2, bias=False)
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.local_calls: list[tuple[list[int], list[int]]] = []

        def v3_local_effects_from_trace(
            self,
            _trace_collector,
            _final_hidden_collector,
            _trace_indices,
            event_indices: torch.Tensor,
            future_indices: torch.Tensor,
            _batch_indices: torch.Tensor,
        ) -> torch.Tensor:
            self.local_calls.append((event_indices.tolist(), future_indices.tolist()))
            return torch.stack(
                (event_indices.float(), future_indices.float()), dim=-1
            ) * self.scale

    class _Harness:
        _v3_qh2l_loss = SmolVLATTTPolicy._v3_qh2l_loss

        def __init__(self) -> None:
            self.model = _FakeModel()
            self.config = type("Config", (), {"hd_v3_null_weight": 0.25})()
            self.observed_student: torch.Tensor | None = None
            self.cross_calls: list[tuple[list[int], list[int]]] = []

        def _hd_active_action_dim(
            self, student: torch.Tensor, teacher: torch.Tensor
        ) -> int:
            self.observed_student = student.detach().clone()
            return min(student.shape[-1], teacher.shape[-1])

        def _v3_reference_student_effects(
            self,
            *,
            event_indices: torch.Tensor,
            future_indices_global: torch.Tensor,
            **_kwargs,
        ) -> torch.Tensor:
            self.cross_calls.append(
                (event_indices.tolist(), future_indices_global.tolist())
            )
            return torch.stack(
                (10.0 * event_indices.float(), 10.0 * future_indices_global.float()),
                dim=-1,
            ) * self.model.scale

    harness = _Harness()
    pair_labels = {
        "valid": torch.ones(4, dtype=torch.bool),
        "total_rows": torch.tensor(4),
        "event_index": torch.tensor([0, 1, 2, 3]),
        "future_index": torch.tensor([1, -1, 3, -1]),
        "event_index_global": torch.tensor([10, 11, 12, 13]),
        "future_index_global": torch.tensor([11, 16, 13, 18]),
        "batch_index": torch.zeros(4, dtype=torch.long),
        "utility": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "teacher_effect": torch.zeros(4, 2),
        "positive": torch.tensor([True, True, False, False]),
        "null": torch.tensor([False, False, True, True]),
        "delay": torch.tensor([1, 5, 1, 5]),
        "cross_segment": torch.tensor([False, True, False, True]),
    }
    loss, metrics = harness._v3_qh2l_loss(
        pair_labels,
        trace_collector={10: object()},
        final_hidden_collector={10: torch.zeros(1, 4, 2, 2)},
        trace_indices=(0, 1, 2, 3),
        reference_batch={},
    )
    assert torch.isfinite(loss)
    # A complete reference window selects the canonical final-action backend
    # for *every* pair, including futures in this TBPTT segment.  The helper
    # internally chunks these rows for memory; routing a subset through the
    # bounded trace would compare instantaneous velocity to final action.
    assert harness.model.local_calls == []
    assert harness.cross_calls == [([0, 1, 2, 3], [11, 16, 13, 18])]
    assert harness.observed_student is not None
    torch.testing.assert_close(
        harness.observed_student,
        torch.tensor([[0.0, 110.0], [10.0, 160.0], [20.0, 130.0], [30.0, 180.0]]),
    )
    assert metrics["hd_v3_cross_segment_pairs"] == 2.0
    loss.backward()
    assert harness.model.scale.grad is not None
    assert torch.isfinite(harness.model.scale.grad)


def test_qh2l_without_reference_uses_bounded_trace_diagnostic() -> None:
    """The velocity trace helper remains available outside canonical V3."""

    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    from lerobot.policies.smolvla_ttt.modeling_smolvla_ttt import SmolVLATTTPolicy

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_out_proj = torch.nn.Linear(2, 2, bias=False)
            self.local_called = False

        def v3_local_effects_from_trace(
            self,
            _trace_collector,
            _final_hidden_collector,
            _trace_indices,
            event_indices: torch.Tensor,
            future_indices: torch.Tensor,
            _batch_indices: torch.Tensor,
        ):
            self.local_called = True
            return torch.stack((event_indices.float(), future_indices.float()), dim=-1)

    class _Harness:
        _v3_qh2l_loss = SmolVLATTTPolicy._v3_qh2l_loss

        def __init__(self) -> None:
            self.model = _FakeModel()
            self.config = type("Config", (), {"hd_v3_null_weight": 0.25})()

        @staticmethod
        def _hd_active_action_dim(student: torch.Tensor, teacher: torch.Tensor) -> int:
            return min(student.shape[-1], teacher.shape[-1])

    harness = _Harness()
    pair_labels = {
        "valid": torch.ones(2, dtype=torch.bool),
        "total_rows": torch.tensor(2),
        "event_index": torch.tensor([0, 1]),
        "future_index": torch.tensor([1, 2]),
        "batch_index": torch.zeros(2, dtype=torch.long),
        "utility": torch.ones(2),
        "teacher_effect": torch.zeros(2, 2),
        "positive": torch.ones(2, dtype=torch.bool),
        "null": torch.zeros(2, dtype=torch.bool),
        "delay": torch.ones(2, dtype=torch.long),
        "cross_segment": torch.zeros(2, dtype=torch.bool),
    }
    harness._v3_qh2l_loss(
        pair_labels,
        trace_collector={10: object()},
        final_hidden_collector={10: torch.zeros(1, 3, 2, 2)},
        trace_indices=(0, 1, 2),
        reference_batch=None,
    )
    assert harness.model.local_called
