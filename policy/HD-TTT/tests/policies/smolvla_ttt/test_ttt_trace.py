"""Synthetic callback coverage for the bounded TTT trace API."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

# The policy package imports optional VLM/dataset dependencies.  Keep this
# focused test collectable in a minimal environment while exercising it when
# the normal policy test extras are installed.
pytest.importorskip("datasets")
pytest.importorskip("transformers")

from lerobot.policies.smolvla_ttt.modeling_smolvla_ttt import SmolVLATTTFlowMatching  # noqa: E402
from lerobot.policies.smolvla_ttt.ttt import TTTMLPLayer  # noqa: E402


class _CallbackHarness(SmolVLATTTFlowMatching):
    """Tiny object carrying only fields used by ``_make_expert_layer_callback``."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.config = SimpleNamespace(ttt_writer_mode="suffix")
        self.ttt_layers = nn.ModuleDict(
            {
                "2": TTTMLPLayer(dim=4, hidden_dim=6, second_order=False),
                "10": TTTMLPLayer(dim=4, hidden_dim=6, second_order=False),
            }
        )


def test_callback_collects_selected_state_and_final_hidden() -> None:
    harness = _CallbackHarness()
    traces = {}
    final_hidden = {}
    states = {}
    callback = harness._make_expert_layer_callback(
        (1, 3),
        states,
        update=True,
        create_graph=False,
        trace_indices=[1],
        trace_collector=traces,
        final_query_hidden_collector=final_hidden,
    )
    hidden = torch.randn(3, 2, 4)
    # The callback is invoked at each selected expert layer by the VLM.
    hidden = callback(2, hidden)
    hidden = callback(10, hidden)
    assert hidden.shape == (3, 2, 4)
    assert set(traces) == {2, 10}
    assert traces[2].indices == (1,)
    assert traces[10].indices == (1,)
    # Numeric max over ModuleDict keys (10, not lexical "2") is the final
    # layer whose post-TTT hidden stream is exposed.
    assert set(final_hidden) == {10}
    assert final_hidden[10].shape == (1, 1, 2, 4)
    assert set(states) == {2, 10}


def test_callback_trace_context_can_be_set_after_factory_creation() -> None:
    harness = _CallbackHarness()
    traces = {}
    final_hidden = {}
    states = {}
    callback = harness._make_expert_layer_callback(
        (1, 2),
        states,
        update=True,
        create_graph=False,
    )
    callback.set_trace_context([0], traces, final_hidden)
    callback(2, torch.randn(2, 1, 4))
    callback(10, torch.randn(2, 1, 4))
    assert traces[2].indices == (0,)
    assert set(final_hidden) == {10}


def test_callback_can_retain_only_the_final_effect_layer() -> None:
    """V3's production path avoids copying fast states at every TTT layer."""

    harness = _CallbackHarness()
    traces = {}
    final_hidden = {}
    states = {}
    callback = harness._make_expert_layer_callback(
        (1, 3),
        states,
        update=True,
        create_graph=False,
        trace_indices=[0, 2],
        trace_collector=traces,
        final_query_hidden_collector=final_hidden,
        trace_layer_indices=[10],
    )
    callback(2, torch.randn(3, 2, 4))
    callback(10, torch.randn(3, 2, 4))
    assert set(traces) == {10}
    assert traces[10].indices == (0, 2)
    assert set(final_hidden) == {10}

    # The setter also accepts the filter for callbacks created before the
    # event/future pair sampler has selected its final effect layer.
    traces.clear()
    final_hidden.clear()
    callback = harness._make_expert_layer_callback((1, 2), {}, update=True, create_graph=False)
    callback.set_trace_context([1], traces, final_hidden, trace_layer_indices=[10])
    callback(2, torch.randn(2, 1, 4))
    callback(10, torch.randn(2, 1, 4))
    assert set(traces) == {10}
    assert final_hidden[10].shape == (1, 1, 1, 4)


def test_local_effect_replay_uses_pre_read_residual_not_future_memory() -> None:
    """A pair effect must not inherit the actual intervening ``W_j`` read."""

    harness = _CallbackHarness()
    harness.config.max_action_dim = 2
    harness.config.ttt_num_register_tokens = 0

    class _ExpertLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.post_attention_layernorm = nn.Identity()
            self.mlp = nn.Identity()

    # Align VLM and expert indices so layer 10 is the final selected layer.
    harness.vlm_with_expert = SimpleNamespace(
        num_vlm_layers=11,
        num_expert_layers=11,
        expert_hidden_size=4,
        lm_expert=SimpleNamespace(
            layers=[_ExpertLayer() for _ in range(11)],
            norm=nn.Identity(),
        ),
    )
    harness.action_out_proj = nn.Linear(4, 2, bias=False)

    traces = {}
    final_hidden = {}
    callback = harness._make_expert_layer_callback(
        (1, 2),
        {},
        update=True,
        create_graph=True,
        trace_indices=[0, 1],
        trace_collector=traces,
        final_query_hidden_collector=final_hidden,
        trace_layer_indices=[10],
    )
    callback(10, torch.randn(2, 1, 4, requires_grad=True))
    effect = harness.v3_local_effects_from_trace(
        traces,
        final_hidden,
        [0, 1],
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([0]),
    )
    # Corrupting the optional post-read collector must not change the local
    # effect: the implementation should use the trace's pre-read residual.
    baseline = effect.detach().clone()
    original_final_hidden = final_hidden[10].detach().clone()
    final_hidden[10].fill_(1e6)
    replayed = harness.v3_local_effects_from_trace(
        traces,
        final_hidden,
        [0, 1],
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([0]),
    )
    torch.testing.assert_close(replayed.detach(), baseline)

    # Traces serialized by the first bounded-trace revision did not carry the
    # explicit pre-read residual.  The compatibility path must recover it as
    # ``post_read - gate * read_hidden`` rather than silently using the
    # post-read stream itself.
    legacy_trace = {
        10: type(traces[10])(
            tuple(replace(item, residual_hidden=None) for item in traces[10].transitions)
        )
    }
    recovered = harness.v3_local_effects_from_trace(
        legacy_trace,
        {10: original_final_hidden},
        [0, 1],
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([0]),
    )
    torch.testing.assert_close(recovered.detach(), baseline)
