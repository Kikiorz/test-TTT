#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

SEQUENCE_SHAPE_KEY = "_lerobot_sequence_shape"
SEQUENCE_VALID_KEY = "_lerobot_sequence_valid"
SEQUENCE_WAVE_START_KEY = "_lerobot_sequence_wave_start"
SEQUENCE_WAVE_END_KEY = "_lerobot_sequence_wave_end"
SEQUENCE_EPISODE_INDEX_KEY = "_lerobot_sequence_episode_index"
SEQUENCE_WINDOW_ORDINAL_KEY = "_lerobot_sequence_window_ordinal"
SEQUENCE_ACTIVE_KEY = "_lerobot_sequence_active"


@dataclass(frozen=True)
class EpisodeWindowRef:
    """One independent training sequence assigned to a local batch lane."""

    window_index: int
    episode_index: int
    window_ordinal: int
    wave_start: bool
    wave_end: bool
    active: bool = True


@dataclass
class EpisodeWindow:
    """Materialized samples plus diagnostic metadata for one batch lane."""

    samples: list[dict[str, Any]]
    episode_index: int
    window_ordinal: int
    wave_start: bool
    wave_end: bool
    active: bool


def _selected_episode_lengths(dataset: Dataset) -> list[int]:
    if not hasattr(dataset, "meta") or dataset.meta.episodes is None:
        raise ValueError("Contiguous sequence training requires episode boundary metadata")

    episode_indices = dataset.episodes
    if episode_indices is None:
        num_episodes = (
            len(dataset.meta.episodes["dataset_from_index"])
            if isinstance(dataset.meta.episodes, dict)
            else len(dataset.meta.episodes)
        )
        episode_indices = range(num_episodes)
    episode_indices = sorted(int(episode_index) for episode_index in episode_indices)

    lengths = []
    for episode_index in episode_indices:
        if isinstance(dataset.meta.episodes, dict):
            start_index = dataset.meta.episodes["dataset_from_index"][episode_index]
            end_index = dataset.meta.episodes["dataset_to_index"][episode_index]
        else:
            episode = dataset.meta.episodes[episode_index]
            start_index = episode["dataset_from_index"]
            end_index = episode["dataset_to_index"]
        lengths.append(int(end_index) - int(start_index))
    return lengths


def _dataset_episode_lengths(dataset: Dataset) -> list[int]:
    child_datasets = getattr(dataset, "_datasets", None)
    if child_datasets is None:
        return _selected_episode_lengths(dataset)
    return [length for child_dataset in child_datasets for length in _selected_episode_lengths(child_dataset)]


class TailPreservingSequenceDataset(Dataset):
    """Return episode-local windows of at most ``sequence_length`` without dropping the tail.

    Starts advance by ``sequence_stride``. Short episodes and non-divisible
    tails remain training examples instead of being filtered out. Every window
    is an independent RoboTTT training sequence and never crosses an episode
    boundary.
    """

    def __init__(
        self,
        dataset: Dataset,
        sequence_length: int,
        sequence_stride: int,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive")
        if sequence_stride > sequence_length:
            raise ValueError("sequence_stride cannot exceed sequence_length")

        self.dataset = dataset
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.window_specs: list[tuple[int, int]] = []
        self.window_episode_indices: list[int] = []
        self.window_ordinals: list[int] = []
        self.episode_window_indices: list[list[int]] = []

        episode_start = 0
        for episode_index, episode_length in enumerate(_dataset_episode_lengths(dataset)):
            if episode_length <= 0:
                raise ValueError("Episode lengths must be positive")
            episode_windows: list[int] = []
            for window_ordinal, offset in enumerate(range(0, episode_length, sequence_stride)):
                window_length = min(sequence_length, episode_length - offset)
                window_index = len(self.window_specs)
                self.window_specs.append((episode_start + offset, window_length))
                self.window_episode_indices.append(episode_index)
                self.window_ordinals.append(window_ordinal)
                episode_windows.append(window_index)
            self.episode_window_indices.append(episode_windows)
            episode_start += episode_length

        if episode_start != len(dataset):
            raise ValueError(
                f"Episode metadata covers {episode_start} frames, but the dataset contains {len(dataset)}"
            )
        if not self.window_specs:
            raise ValueError("The dataset does not contain any non-empty episode")

    def __len__(self) -> int:
        return len(self.window_specs)

    def __getitem__(self, index: int | EpisodeWindowRef) -> list[dict[str, Any]] | EpisodeWindow:
        ref = index if isinstance(index, EpisodeWindowRef) else None
        window_index = ref.window_index if ref is not None else index
        start_index, window_length = self.window_specs[window_index]
        samples = [self.dataset[start_index + offset] for offset in range(window_length)]
        if ref is None:
            return samples

        if ref.active:
            expected_episode = self.window_episode_indices[window_index]
            expected_ordinal = self.window_ordinals[window_index]
            if (ref.episode_index, ref.window_ordinal) != (expected_episode, expected_ordinal):
                raise RuntimeError(
                    "Episode-window scheduler corrupted trajectory order: "
                    f"expected episode/window {(expected_episode, expected_ordinal)}, got "
                    f"{(ref.episode_index, ref.window_ordinal)}"
                )
        return EpisodeWindow(
            samples=samples,
            episode_index=ref.episode_index,
            window_ordinal=ref.window_ordinal,
            wave_start=ref.wave_start,
            wave_end=ref.wave_end,
            active=ref.active,
        )


class EpisodeSequenceBatchSampler:
    """Shuffle independent sequences and shard complete outer batches by rank.

    Every yielded real window is a complete RoboTTT training sequence and must
    start from the learned fast-weight initialization. The final incomplete
    global batch is padded with fully masked dummy lanes so every rank performs
    the same number of gradient collectives without duplicating real data.
    """

    def __init__(
        self,
        dataset: TailPreservingSequenceDataset,
        *,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        start_epoch: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas - 1}]")
        if start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self._next_epoch = start_epoch
        self.global_batch_size = batch_size * num_replicas

        self.steps_per_epoch = (len(dataset) + self.global_batch_size - 1) // self.global_batch_size
        if self.steps_per_epoch <= 0:
            raise ValueError("The sequence sampler has no optimizer steps")

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._next_epoch = epoch

    def __iter__(self) -> Iterator[list[EpisodeWindowRef]]:
        epoch = self._next_epoch
        self._next_epoch += 1
        rng = random.Random(self.seed + epoch)
        window_indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(window_indices)

        num_padding = (-len(window_indices)) % self.global_batch_size
        window_indices.extend([-1] * num_padding)

        local_start = self.rank * self.batch_size
        local_end = local_start + self.batch_size
        fallback_window_index = 0
        for batch_start in range(0, len(window_indices), self.global_batch_size):
            global_windows = window_indices[batch_start : batch_start + self.global_batch_size]
            local_windows = global_windows[local_start:local_end]
            batch: list[EpisodeWindowRef] = []
            for scheduled_window_index in local_windows:
                active = scheduled_window_index >= 0
                window_index = scheduled_window_index if active else fallback_window_index
                episode_index = self.dataset.window_episode_indices[window_index] if active else -1
                window_ordinal = self.dataset.window_ordinals[window_index] if active else -1
                batch.append(
                    EpisodeWindowRef(
                        window_index=window_index,
                        episode_index=episode_index,
                        window_ordinal=window_ordinal,
                        wave_start=True,
                        wave_end=True,
                        active=active,
                    )
                )
            yield batch


def _set_timestep_padding(sample: dict[str, Any], *, padded_timestep: bool) -> dict[str, Any]:
    """Copy one timestep and make its action mask explicit.

    Right-padded timesteps reuse the final real observation only to keep every
    tensor collatable. Their full action chunk is masked, and a separate
    sequence-valid mask prevents them from advancing the recurrent TTT state.
    """
    copied = dict(sample)
    action = sample.get(ACTION)
    if action is None or not hasattr(action, "shape") or len(action.shape) == 0:
        raise ValueError("Sequence samples need an action chunk to construct action_is_pad")
    action_chunk_length = int(action.shape[0])
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is None:
        action_is_pad = torch.zeros(action_chunk_length, dtype=torch.bool)
    else:
        action_is_pad = torch.as_tensor(action_is_pad, dtype=torch.bool).clone()
        if action_is_pad.ndim != 1 or action_is_pad.shape[0] != action_chunk_length:
            raise ValueError(
                "action_is_pad must be one-dimensional and match the action chunk length; "
                f"got {tuple(action_is_pad.shape)} for action shape {tuple(action.shape)}"
            )
    if padded_timestep:
        action_is_pad.fill_(True)
    copied["action_is_pad"] = action_is_pad
    return copied


def sequence_collate_fn(
    batch: list[list[dict[str, Any]] | EpisodeWindow],
) -> dict[str, Any]:
    """Right-pad and flatten episode-local sequences in batch-major order.

    TTT fast states retain a separate batch dimension, so trajectories never
    interact. Variable-length tails are aligned to the longest sequence in the
    batch. Padding is both action-masked and marked invalid for fast-state
    updates, preserving the result of processing each trajectory separately.
    """
    if not batch:
        raise ValueError("Cannot collate an empty sequence batch")

    episode_windows = [
        item
        if isinstance(item, EpisodeWindow)
        else EpisodeWindow(
            samples=item,
            episode_index=-1,
            window_ordinal=0,
            wave_start=True,
            wave_end=True,
            active=True,
        )
        for item in batch
    ]
    if len({item.wave_start for item in episode_windows}) != 1:
        raise ValueError("Every lane in a sequence batch must start the wave together")
    if len({item.wave_end for item in episode_windows}) != 1:
        raise ValueError("Every lane in a sequence batch must end the wave together")

    sequence_lengths = [len(item.samples) for item in episode_windows]
    if any(length == 0 for length in sequence_lengths):
        raise ValueError("A sequence must contain at least one timestep")
    if any(sample is None for item in episode_windows for sample in item.samples):
        raise ValueError("Sequence samples cannot be None because dropping a timestep corrupts TTT state")

    sequence_length = max(sequence_lengths)
    flattened: list[dict[str, Any]] = []
    valid_timesteps: list[bool] = []
    for item in episode_windows:
        sequence = item.samples
        flattened.extend(
            _set_timestep_padding(sample, padded_timestep=not item.active) for sample in sequence
        )
        valid_timesteps.extend([item.active] * len(sequence))

        num_padding_steps = sequence_length - len(sequence)
        flattened.extend(
            _set_timestep_padding(sequence[-1], padded_timestep=True) for _ in range(num_padding_steps)
        )
        valid_timesteps.extend([False] * num_padding_steps)

    collated = lerobot_collate_fn(flattened)
    if collated is None:
        raise ValueError("Sequence collation unexpectedly produced an empty batch")
    collated[SEQUENCE_SHAPE_KEY] = torch.tensor([len(episode_windows), sequence_length], dtype=torch.int64)
    collated[SEQUENCE_VALID_KEY] = torch.tensor(valid_timesteps, dtype=torch.bool)
    collated[SEQUENCE_WAVE_START_KEY] = torch.tensor(episode_windows[0].wave_start)
    collated[SEQUENCE_WAVE_END_KEY] = torch.tensor(episode_windows[0].wave_end)
    collated[SEQUENCE_EPISODE_INDEX_KEY] = torch.tensor(
        [item.episode_index for item in episode_windows], dtype=torch.int64
    )
    collated[SEQUENCE_WINDOW_ORDINAL_KEY] = torch.tensor(
        [item.window_ordinal for item in episode_windows], dtype=torch.int64
    )
    collated[SEQUENCE_ACTIVE_KEY] = torch.tensor([item.active for item in episode_windows], dtype=torch.bool)
    return collated
