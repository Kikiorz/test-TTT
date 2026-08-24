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

    return [
        length for child_dataset in child_datasets for length in _selected_episode_lengths(child_dataset)
    ]


class ContiguousSequenceDataset(Dataset):
    """Map-style view that returns fixed-length windows within episodes."""

    def __init__(self, dataset: Dataset, sequence_length: int, sequence_stride: int = 1) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive")

        self.dataset = dataset
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.window_start_indices: list[int] = []

        episode_start = 0
        for episode_length in _dataset_episode_lengths(dataset):
            final_offset = episode_length - sequence_length
            if final_offset >= 0:
                self.window_start_indices.extend(
                    episode_start + offset for offset in range(0, final_offset + 1, sequence_stride)
                )
            episode_start += episode_length

        if episode_start != len(dataset):
            raise ValueError(
                f"Episode metadata covers {episode_start} frames, but the dataset contains {len(dataset)}"
            )
        if not self.window_start_indices:
            raise ValueError(
                f"No episode is long enough for a sequence_length of {sequence_length}; "
                "use a shorter sequence or a dataset with longer episodes"
            )

    def __len__(self) -> int:
        return len(self.window_start_indices)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        start_index = self.window_start_indices[index]
        return [self.dataset[start_index + offset] for offset in range(self.sequence_length)]


def sequence_collate_fn(batch: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Flatten sequence windows to ``B*T`` and attach their logical shape."""
    if not batch:
        raise ValueError("Cannot collate an empty sequence batch")

    sequence_length = len(batch[0])
    if sequence_length == 0 or any(len(sequence) != sequence_length for sequence in batch):
        raise ValueError("Every sequence in a batch must have the same positive length")

    flattened_batch = [sample for sequence in batch for sample in sequence]
    if any(sample is None for sample in flattened_batch):
        raise ValueError("Sequence samples cannot be None because dropping a timestep would corrupt TBPTT state")

    collated = lerobot_collate_fn(flattened_batch)
    if collated is None:
        raise ValueError("Sequence collation unexpectedly produced an empty batch")
    collated[SEQUENCE_SHAPE_KEY] = torch.tensor([len(batch), sequence_length], dtype=torch.int64)
    return collated
