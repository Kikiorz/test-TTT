from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.scripts.lerobot_train import _hd_ttt_finite_guard


class _SingleProcessAccelerator:
    """Small stand-in for the trainer's finite-flag API."""

    device = torch.device("cpu")
    num_processes = 1


def test_hd_finite_guard_accepts_finite_loss_gradients_and_state() -> None:
    module = nn.Linear(2, 2)
    module.weight.grad = torch.ones_like(module.weight)
    module.bias.grad = torch.zeros_like(module.bias)
    state = SimpleNamespace(tensors=lambda: (torch.ones(1, 2, 2), torch.zeros(1, 2)))

    _hd_ttt_finite_guard(
        policy=module,
        loss=torch.tensor(1.0),
        grad_norm=torch.tensor(0.5),
        fast_states={0: state},
        accelerator=_SingleProcessAccelerator(),
        stage="unit test",
    )


@pytest.mark.parametrize(
    ("loss", "gradient"),
    [
        (torch.tensor(float("nan")), None),
        (torch.tensor(1.0), torch.tensor(float("inf"))),
    ],
)
def test_hd_finite_guard_raises_before_optimizer_step(
    loss: torch.Tensor,
    gradient: torch.Tensor | None,
) -> None:
    module = nn.Linear(1, 1)
    before = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    if gradient is not None:
        module.weight.grad = gradient.reshape_as(module.weight)
    else:
        module.weight.grad = torch.ones_like(module.weight)

    with pytest.raises(RuntimeError, match="HD-TTT finite guard failed before optimizer.step"):
        _hd_ttt_finite_guard(
            policy=module,
            loss=loss,
            accelerator=_SingleProcessAccelerator(),
            stage="unit test",
        )

    # The guard runs before the optimizer and must not mutate model weights.
    for name, parameter in module.named_parameters():
        assert torch.equal(parameter, before[name])


def test_hd_finite_guard_reports_nonfinite_fast_state() -> None:
    state = SimpleNamespace(tensors=lambda: (torch.tensor([[float("nan")]]),))

    with pytest.raises(RuntimeError, match="fast_state"):
        _hd_ttt_finite_guard(
            loss=torch.tensor(1.0),
            fast_states={3: state},
            accelerator=_SingleProcessAccelerator(),
            stage="unit test",
            check_gradients=False,
            check_parameters=False,
        )


def test_hd_finite_guard_rejects_recovered_inner_nonfinite_marker() -> None:
    """A finite candidate fallback must remain visible to the trainer."""

    with pytest.raises(RuntimeError, match="ttt_nonfinite_seen"):
        _hd_ttt_finite_guard(
            loss=torch.tensor(1.0),
            observations=(("ttt_nonfinite_seen", 1.0),),
            accelerator=_SingleProcessAccelerator(),
            stage="unit test",
            check_gradients=False,
            check_parameters=False,
        )
