"""Tests for variable-length exact-batch DDP flow weighting."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from lerobot.scripts.lerobot_train import (
    _ddp_frame_weighted_flow_scale,
    _ddp_reduce_gradients,
    _sequence_valid_action_slots,
)
from lerobot.policies.smolvla_ttt.sequence import EqualLengthBatchSampler, TailPreservingSequenceDataset


class _EpisodeDataset(Dataset):
    def __init__(self, lengths: list[int]) -> None:
        starts: list[int] = []
        ends: list[int] = []
        cursor = 0
        for length in lengths:
            starts.append(cursor)
            cursor += length
            ends.append(cursor)
        self.meta = type("Meta", (), {
            "episodes": {"dataset_from_index": starts, "dataset_to_index": ends}
        })()
        self.episodes = None

    def __len__(self) -> int:
        return int(self.meta.episodes["dataset_to_index"][-1])

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"frame_index": index}


class _ReduceAccelerator:
    def __init__(self, *, world_size: int, global_count: float) -> None:
        self.num_processes = world_size
        self.device = torch.device("cpu")
        self.global_count = float(global_count)
        self.reduce_calls = 0

    def reduce(self, value: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        assert reduction == "sum"
        self.reduce_calls += 1
        return torch.tensor(self.global_count, dtype=value.dtype, device=value.device)


class _GradientReduceAccelerator:
    """Small deterministic stand-in for Accelerate's distributed reduction."""

    def __init__(self, global_presence: list[int], global_unsupported: list[int] | None = None) -> None:
        self.num_processes = 2
        self.device = torch.device("cpu")
        self.global_presence = torch.tensor(global_presence, dtype=torch.int32)
        self.global_unsupported = torch.tensor(
            global_unsupported if global_unsupported is not None else [0] * len(global_presence),
            dtype=torch.int32,
        )
        self.calls: list[tuple[tuple[int, ...], torch.dtype, str]] = []

    def reduce(self, value: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        assert reduction in {"sum", "mean"}
        self.calls.append((tuple(value.shape), value.dtype, reduction))
        if len(self.calls) == 1:
            return self.global_presence.to(device=value.device)
        if len(self.calls) == 2:
            return self.global_unsupported.to(device=value.device)
        # Simulate another rank's contribution while keeping the expected
        # mean operation deterministic for the assertions below.
        return value + 1


def _module_with_parameters(*parameters: torch.nn.Parameter) -> torch.nn.Module:
    module = torch.nn.Module()
    for index, parameter in enumerate(parameters):
        module.register_parameter(f"p{index}", parameter)
    return module


def test_sequence_valid_action_slots_counts_chunk_slots() -> None:
    # B*T=3 rows and two action slots per row; one row has only one valid slot.
    batch = {
        "action_is_pad": torch.tensor(
            [[False, False], [False, True], [True, True]], dtype=torch.bool
        )
    }
    assert _sequence_valid_action_slots(batch, (1, 3)) == 3


def test_ddp_flow_scale_is_global_frame_weighted() -> None:
    # For P=4 and a local count of 12 out of N=50 slots, the local flow mean
    # must be multiplied by P*n/N.  The trainer then takes a mean over ranks.
    batch = {"action_is_pad": torch.zeros((2 * 6, 1), dtype=torch.bool)}
    accelerator = _ReduceAccelerator(world_size=4, global_count=50)
    scale, local_count, global_count = _ddp_frame_weighted_flow_scale(
        batch,
        (2, 6),
        accelerator,
        enabled=True,
    )
    assert local_count == 12
    assert global_count == 50
    assert scale == pytest.approx(48 / 50)
    assert accelerator.reduce_calls == 1


def test_ddp_flow_scale_disabled_does_not_collective_or_change_b1() -> None:
    batch = {"action_is_pad": torch.zeros((3, 2), dtype=torch.bool)}
    accelerator = _ReduceAccelerator(world_size=4, global_count=999)
    scale, local_count, global_count = _ddp_frame_weighted_flow_scale(
        batch,
        (1, 3),
        accelerator,
        enabled=False,
    )
    assert scale == 1.0
    assert local_count == 6
    assert global_count == 6
    assert accelerator.reduce_calls == 0


def test_ddp_flow_scale_rejects_empty_global_slot_population() -> None:
    batch = {"action_is_pad": torch.ones((2, 2), dtype=torch.bool)}
    accelerator = _ReduceAccelerator(world_size=2, global_count=0)
    with pytest.raises(ValueError, match="no valid action slots"):
        _ddp_frame_weighted_flow_scale(
            batch,
            (1, 2),
            accelerator,
            enabled=True,
        )


def test_ddp_gradient_reduction_flattens_dense_groups_and_fills_none() -> None:
    """Flattened all-reduce preserves values while collapsing collectives."""

    first = torch.nn.Parameter(torch.zeros(2, 3))
    second = torch.nn.Parameter(torch.zeros(4))
    third = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    first.grad = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second.grad = None  # Used on another rank; must receive an explicit zero.
    third.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    module = _module_with_parameters(first, second, third)
    accelerator = _GradientReduceAccelerator([2, 2, 2])

    _ddp_reduce_gradients(module, accelerator)

    torch.testing.assert_close(first.grad, torch.arange(6, dtype=torch.float32).reshape(2, 3) + 1)
    torch.testing.assert_close(second.grad, torch.ones(4))
    torch.testing.assert_close(third.grad, torch.tensor([4.0, 5.0], dtype=torch.float64))
    # One presence collective, one layout collective, and one flattened
    # collective per dtype (float32/float64) replace three per-parameter
    # gradient reductions.
    assert len(accelerator.calls) == 4
    assert accelerator.calls[0][0] == (3,)
    assert accelerator.calls[1][0] == (3,)
    assert {call[0] for call in accelerator.calls[2:]} == {(10,), (2,)}


def test_ddp_gradient_reduction_leaves_globally_unused_and_b1_unchanged() -> None:
    unused = torch.nn.Parameter(torch.zeros(3))
    unused.grad = None
    module = _module_with_parameters(unused)
    accelerator = _GradientReduceAccelerator([0])
    _ddp_reduce_gradients(module, accelerator)
    assert unused.grad is None
    # The single-process guard is the explicit B=1 compatibility path.
    used = torch.nn.Parameter(torch.zeros(3))
    used.grad = torch.tensor([1.0, 2.0, 3.0])
    single = _GradientReduceAccelerator([1])
    single.num_processes = 1
    _ddp_reduce_gradients(_module_with_parameters(used), single)
    torch.testing.assert_close(used.grad, torch.tensor([1.0, 2.0, 3.0]))
    assert single.calls == []


def test_ddp_gradient_reduction_uses_fallback_for_sparse_layout() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, 3))
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([2.0, 4.0]),
        size=(3, 3),
    )
    parameter.grad = sparse
    module = _module_with_parameters(parameter)
    accelerator = _GradientReduceAccelerator([2], [2])

    _ddp_reduce_gradients(module, accelerator)

    assert parameter.grad is not None
    assert parameter.grad.layout == torch.strided
    expected = sparse.to_dense() + 1
    torch.testing.assert_close(parameter.grad, expected)
    # Presence + layout + one compatibility per-parameter reduction.
    assert len(accelerator.calls) == 3


def test_equal_length_sampler_advances_seed_on_each_epoch() -> None:
    dataset = TailPreservingSequenceDataset(
        _EpisodeDataset([3, 3, 3, 4, 4, 4]),
        sequence_length=4,
        sequence_stride=4,
    )
    sampler = EqualLengthBatchSampler(
        dataset,
        batch_size=1,
        shuffle=True,
        num_replicas=1,
        seed=123,
    )
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert first != second
    # A fresh sampler reproduces epoch zero exactly, which is important for
    # rank-consistent Accelerate construction.
    fresh = EqualLengthBatchSampler(
        dataset,
        batch_size=1,
        shuffle=True,
        num_replicas=1,
        seed=123,
    )
    assert first == list(iter(fresh))
