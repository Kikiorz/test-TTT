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

from typing import Any

import torch
from torch.utils.data import Dataset

from lerobot.utils.collate import lerobot_collate_fn

SEQUENCE_SHAPE_KEY = "_lerobot_sequence_shape"


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

    Starts advance by ``sequence_stride``. Each final window ends exactly at the
    episode boundary, so short episodes and non-divisible tails remain training
    examples instead of being filtered out. The formal four-GPU recipe uses
    ``sequence_length == sequence_stride == 256``, which covers every frame once.
    """

    def __init__(self, dataset: Dataset, sequence_length: int, sequence_stride: int) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive")
        if sequence_stride > sequence_length:
            raise ValueError("sequence_stride cannot exceed sequence_length because that would drop frames")

        self.dataset = dataset
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.window_specs: list[tuple[int, int]] = []

        episode_start = 0
        for episode_length in _dataset_episode_lengths(dataset):
            if episode_length <= 0:
                raise ValueError("Episode lengths must be positive")
            for offset in range(0, episode_length, sequence_stride):
                window_length = min(sequence_length, episode_length - offset)
                self.window_specs.append((episode_start + offset, window_length))
            episode_start += episode_length

        if episode_start != len(dataset):
            raise ValueError(
                f"Episode metadata covers {episode_start} frames, but the dataset contains {len(dataset)}"
            )
        if not self.window_specs:
            raise ValueError("The dataset does not contain any non-empty episode")

    def __len__(self) -> int:
        return len(self.window_specs)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        start_index, window_length = self.window_specs[index]
        return [self.dataset[start_index + offset] for offset in range(window_length)]


def sequence_collate_fn(batch: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Flatten one variable-length sequence to ``T`` and attach its logical shape."""
    if not batch:
        raise ValueError("Cannot collate an empty sequence batch")
    if len(batch) != 1:
        raise ValueError("smolvla_ttt tail-preserving batches require per-device batch_size=1")

    sequence_length = len(batch[0])
    if sequence_length == 0:
        raise ValueError("A sequence must contain at least one timestep")
    if any(sample is None for sample in batch[0]):
        raise ValueError("Sequence samples cannot be None because dropping a timestep corrupts TTT state")

    collated = lerobot_collate_fn(batch[0])
    if collated is None:
        raise ValueError("Sequence collation unexpectedly produced an empty batch")
    collated[SEQUENCE_SHAPE_KEY] = torch.tensor([1, sequence_length], dtype=torch.int64)
    return collated
