"""Tests for variable-length exact-batch DDP flow weighting."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from lerobot.scripts.lerobot_train import (
    _ddp_frame_weighted_flow_scale,
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
