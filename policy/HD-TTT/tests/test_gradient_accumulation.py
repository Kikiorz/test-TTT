"""Contract tests for explicit sequence-window gradient accumulation."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.scripts.lerobot_train import (
    _v3_ddp_pair_normalizers,
    update_policy_tbptt,
)
from lerobot.policies.smolvla_ttt.modeling_smolvla_ttt import SmolVLATTTPolicy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker


class _CPUAccelerator:
    device = torch.device("cpu")
    num_processes = 1
    is_main_process = True

    def autocast(self):
        return nullcontext()

    def backward(self, loss: torch.Tensor, retain_graph: bool = False) -> None:
        loss.backward(retain_graph=retain_graph)

    def clip_grad_norm_(self, parameters, max_norm: float):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model, keep_fp32_wrapper: bool = False):
        return model


class _ToySequencePolicy(nn.Module):
    """Small policy exposing the trainer's sequence-segment contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7))
        self.config = SimpleNamespace(
            hd_ttt_enabled=False,
            hd_attribution_protocol="",
            ttt_stable_inner_update=False,
        )
        self.tbptt_loss_weighting = None
        self.first_state_flags: list[bool] = []

    def forward_sequence_segment(self, batch, *, sequence_shape, fast_states=None, **kwargs):
        del sequence_shape, kwargs
        # The trainer must not carry a recurrent state from one independent
        # accumulation window into the next.  It does carry it between
        # segments of a *single* call (not exercised by this one-segment toy).
        self.first_state_flags.append(fast_states is None)
        values = batch["x"]
        loss = (self.weight * values).square().mean()
        next_state = {0: torch.ones(1)}
        return loss, {"loss_per_dim": [float(loss.detach())]}, next_state


def _tracker(accelerator: _CPUAccelerator) -> MetricsTracker:
    return MetricsTracker(
        batch_size=1,
        num_frames=8,
        num_episodes=2,
        metrics={
            "loss": AverageMeter("loss"),
            "grad_norm": AverageMeter("grad_norm"),
            "lr": AverageMeter("lr"),
            "update_s": AverageMeter("update_s"),
        },
        accelerator=accelerator,
    )


def _run_window(
    policy: _ToySequencePolicy,
    optimizer: torch.optim.Optimizer,
    accelerator: _CPUAccelerator,
    tracker: MetricsTracker,
    values: torch.Tensor,
    *,
    optimizer_step: bool,
    zero_grad_before: bool,
    gradient_scale: float,
    sequence_shape: tuple[int, int] = (1, 2),
    lr_scheduler=None,
):
    return update_policy_tbptt(
        tracker,
        policy,
        {"x": values},
        sequence_shape,
        segment_length=2,
        optimizer=optimizer,
        grad_clip_norm=0.0,
        accelerator=accelerator,
        lr_scheduler=lr_scheduler,
        optimizer_step=optimizer_step,
        zero_grad_before=zero_grad_before,
        gradient_scale=gradient_scale,
    )


def test_accumulated_independent_windows_match_one_large_mean_update() -> None:
    """Scaling every backward by 1/N is equivalent to a larger batch mean."""

    accelerator = _CPUAccelerator()
    values_a = torch.tensor([1.0, 2.0])
    values_b = torch.tensor([3.0, 5.0])

    accumulated = _ToySequencePolicy()
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.2)
    accumulated_tracker = _tracker(accelerator)
    initial = accumulated.weight.detach().clone()
    _run_window(
        accumulated,
        accumulated_optimizer,
        accelerator,
        accumulated_tracker,
        values_a,
        optimizer_step=False,
        zero_grad_before=True,
        gradient_scale=0.5,
    )
    # A deferred call must leave parameters untouched and retain gradients.
    torch.testing.assert_close(accumulated.weight.detach(), initial)
    assert accumulated.weight.grad is not None
    _run_window(
        accumulated,
        accumulated_optimizer,
        accelerator,
        accumulated_tracker,
        values_b,
        optimizer_step=True,
        zero_grad_before=False,
        gradient_scale=0.5,
    )

    direct = _ToySequencePolicy()
    direct.load_state_dict(accumulated.state_dict())
    # Restore the same pre-update parameter: accumulated now contains the
    # updated value, while direct should start from the original initial one.
    direct.weight.data.copy_(initial)
    direct_optimizer = torch.optim.SGD(direct.parameters(), lr=0.2)
    direct_tracker = _tracker(accelerator)
    _run_window(
        direct,
        direct_optimizer,
        accelerator,
        direct_tracker,
        torch.cat((values_a, values_b)),
        optimizer_step=True,
        zero_grad_before=True,
        gradient_scale=1.0,
        sequence_shape=(2, 2),
    )

    torch.testing.assert_close(accumulated.weight, direct.weight, rtol=1e-6, atol=1e-7)
    assert accumulated.first_state_flags == [True, True]


def test_deferred_sequence_call_does_not_step_scheduler_or_clear_gradients() -> None:
    accelerator = _CPUAccelerator()
    policy = _ToySequencePolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    tracker = _tracker(accelerator)

    class _Scheduler:
        def __init__(self):
            self.calls = 0

        def step(self):
            self.calls += 1

    scheduler = _Scheduler()
    _run_window(
        policy,
        optimizer,
        accelerator,
        tracker,
        torch.tensor([1.0, 2.0]),
        optimizer_step=False,
        zero_grad_before=True,
        gradient_scale=0.5,
        lr_scheduler=scheduler,
    )
    assert scheduler.calls == 0
    assert policy.weight.grad is not None

    # The toy helper does not pass a scheduler; assert the core invariant
    # directly by observing that the deferred branch left a nonzero gradient.
    assert float(policy.weight.grad.abs()) > 0

    _run_window(
        policy,
        optimizer,
        accelerator,
        tracker,
        torch.tensor([3.0, 5.0]),
        optimizer_step=True,
        zero_grad_before=False,
        gradient_scale=0.5,
        lr_scheduler=scheduler,
    )
    assert scheduler.calls == 1
    assert policy.weight.grad is None or torch.equal(policy.weight.grad, torch.zeros_like(policy.weight.grad))


class _PairPolicy:
    """Minimal pair-label adapter for the DDP normalizer helper."""

    def _v3_reference_sequence_shape(self, reference_batch):
        del reference_batch
        return (2, 3)

    def _prepare_v3_pair_labels(self, *args, **kwargs):
        del args, kwargs
        return object()

    def _v3_pair_normalizers(self, labels):
        assert labels is not None
        # Simulate one rank's local complete-window population.
        return {
            "full": torch.tensor(4.0),
            "positive": torch.tensor(2.0),
            "null": torch.tensor(2.0),
        }


class _PairReduceAccelerator:
    def __init__(self, reduced_values: tuple[float, float, float], world_size: int = 2):
        self.device = torch.device("cpu")
        self.num_processes = world_size
        self.reduced_values = iter(reduced_values)
        self.calls = 0

    def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        self.calls += 1
        return torch.tensor(next(self.reduced_values), device=value.device, dtype=value.dtype)


def test_v3_ddp_pair_normalizers_return_global_denominator_over_world_size() -> None:
    accelerator = _PairReduceAccelerator((10.0, 6.0, 4.0), world_size=2)
    normalizers = _v3_ddp_pair_normalizers(
        _PairPolicy(),
        {"_lerobot_sequence_offset": torch.tensor(0)},
        (2, 3),
        accelerator,
        enabled=True,
    )
    assert normalizers is not None
    assert accelerator.calls == 3
    assert normalizers["full"].item() == pytest.approx(5.0)
    assert normalizers["positive"].item() == pytest.approx(3.0)
    assert normalizers["null"].item() == pytest.approx(2.0)

    # If rank-local numerators are 2 and 8, passing D/P to each rank and then
    # taking the trainer's explicit mean gives (2/(10/2)+8/(10/2))/2=1,
    # exactly the global pair-weighted numerator 10/10.
    local_numerators = torch.tensor([2.0, 8.0])
    compensated_losses = local_numerators / normalizers["full"]
    assert compensated_losses.mean().item() == pytest.approx(1.0)


def test_v3_ddp_pair_normalizers_disabled_and_single_process_are_noops() -> None:
    disabled = _PairReduceAccelerator((10.0, 6.0, 4.0), world_size=2)
    assert (
        _v3_ddp_pair_normalizers(
            _PairPolicy(),
            {},
            (2, 3),
            disabled,
            enabled=False,
        )
        is None
    )
    assert disabled.calls == 0
    single = _PairReduceAccelerator((10.0, 6.0, 4.0), world_size=1)
    assert (
        _v3_ddp_pair_normalizers(
            _PairPolicy(),
            {},
            (2, 3),
            single,
            enabled=True,
        )
        is None
    )
    assert single.calls == 0


def test_v3_b1_normalizer_formula_remains_the_historical_local_value() -> None:
    labels = {
        "valid": torch.tensor([True, True, False]),
        "positive": torch.tensor([True, False, True]),
        "null": torch.tensor([False, True, False]),
        "utility": torch.tensor([0.75, -0.25, 0.5]),
    }
    normalizers = SmolVLATTTPolicy._v3_pair_normalizers(labels)
    assert normalizers is not None
    assert normalizers["full"].item() == pytest.approx(2.0)
    assert normalizers["positive"].item() == pytest.approx(0.75)
    assert normalizers["null"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_gradient_scale_must_be_finite_and_positive(bad_scale: float) -> None:
    accelerator = _CPUAccelerator()
    policy = _ToySequencePolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="gradient_scale"):
        _run_window(
            policy,
            optimizer,
            accelerator,
            _tracker(accelerator),
            torch.tensor([1.0, 2.0]),
            optimizer_step=False,
            zero_grad_before=True,
            gradient_scale=bad_scale,
        )


def test_metrics_tracker_sample_multiplier_preserves_optimizer_step_count() -> None:
    accelerator = _CPUAccelerator()
    tracker = _tracker(accelerator)
    tracker.step(sample_multiplier=4)
    assert tracker.steps == 1
    assert tracker.samples == 4
    assert tracker.epochs == pytest.approx(0.5)
