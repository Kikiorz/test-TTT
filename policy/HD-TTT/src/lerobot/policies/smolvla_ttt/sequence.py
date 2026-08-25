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
from lerobot.utils.constants import ACTION

SEQUENCE_SHAPE_KEY = "_lerobot_sequence_shape"
# Complementary frame mask for training-only HD writer objectives.  Unlike
# ``action_is_pad`` it remains true on replayed history frames: those frames
# have no imitation target in the current window, but their causal interaction
# still must train the local K/V objective.  Window-local counterfactual gate
# supervision is masked separately through ``hd_write_gate_observed``.
HD_WRITER_VALID_KEY = "hd_writer_valid"


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
    examples instead of being filtered out. The MIKASA recipes choose the
    sequence capacity explicitly: bounded-window runs use a short context,
    while the full-history Shuffle protocol sets it to the longest selected
    episode and consumes one complete window per episode.
    """

    def __init__(
        self,
        dataset: Dataset,
        sequence_length: int,
        sequence_stride: int,
        max_windows_per_episode: int | None = None,
        history_warmup_length: int | None = 0,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive")
        if sequence_stride > sequence_length:
            raise ValueError("sequence_stride cannot exceed sequence_length because that would drop frames")
        if max_windows_per_episode is not None and max_windows_per_episode <= 0:
            raise ValueError("max_windows_per_episode must be positive when provided")
        if history_warmup_length is not None and history_warmup_length < 0:
            raise ValueError("history_warmup_length must be non-negative")

        self.dataset = dataset
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.history_warmup_length = (
            None if history_warmup_length is None else int(history_warmup_length)
        )
        # Window-local hindsight labels are generated from exactly the
        # preceding context used by this sampler.  Their warm-up rows remain
        # valid for the instantaneous local K/V objective; only the
        # counterfactual/gate labels are reset below for legacy frame-local
        # artifacts.
        self.hd_window_local = bool(getattr(dataset, "hd_window_local", False))
        self.hd_window_keyed = bool(getattr(dataset, "hd_window_keyed", False))
        self.window_specs: list[tuple[int, int]] = []
        self.history_specs: list[tuple[int, int]] = []

        episode_start = 0
        for episode_length in _dataset_episode_lengths(dataset):
            if episode_length <= 0:
                raise ValueError("Episode lengths must be positive")
            if (
                max_windows_per_episode == 1
                and self.history_warmup_length is None
                and episode_length > sequence_length
            ):
                # A single full-history window is an explicit recurrent-state
                # contract: silently selecting the first prefix would discard
                # the suffix and make the learned memory/evaluation protocol
                # depend on an implementation detail.  The shell recipe also
                # performs this check before distributed launch; keep the
                # dataset-level guard for callers that construct it directly.
                raise ValueError(
                    "full-history replay with max_windows_per_episode=1 requires "
                    f"sequence_length >= every episode length; got episode_length={episode_length} "
                    f"and sequence_length={sequence_length}"
                )
            offsets = list(range(0, episode_length, sequence_stride))
            if max_windows_per_episode is not None and len(offsets) > max_windows_per_episode:
                # Deterministic, episode-local coverage: retain evenly spaced
                # *full* windows and always include the first and terminal
                # windows.  Sampling raw ``range(0, length, stride)`` offsets
                # can otherwise select a one-frame tail (e.g. length=513,
                # stride=64), leaving the terminal phase almost unobserved.
                # The default remains the full tail-preserving set, so this
                # only changes explicitly requested speed/episode-balanced
                # ablations.
                last_full_offset = max(episode_length - sequence_length, 0)
                full_offsets = list(range(0, last_full_offset + 1, sequence_stride))
                if not full_offsets or full_offsets[-1] != last_full_offset:
                    full_offsets.append(last_full_offset)
                offsets = full_offsets
                positions = torch.linspace(
                    0,
                    len(offsets) - 1,
                    max_windows_per_episode,
                    dtype=torch.float32,
                ).round().to(torch.long).tolist()
                offsets = [offsets[int(position)] for position in sorted(set(positions))]
            for offset in offsets:
                window_length = min(sequence_length, episode_length - offset)
                target_start = episode_start + offset
                self.window_specs.append((target_start, window_length))
                history_start = (
                    episode_start
                    if self.history_warmup_length is None
                    else max(episode_start, target_start - self.history_warmup_length)
                )
                self.history_specs.append((history_start, target_start))
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
        history_start, target_start = self.history_specs[index]
        if target_start != start_index:
            raise RuntimeError("Internal history/target window bookkeeping is inconsistent")
        window_labels = None
        get_window_labels = getattr(self.dataset, "get_window_labels", None)
        if callable(get_window_labels):
            window_labels = get_window_labels(
                start_index,
                history_start,
                # ``window_length`` is the target suffix length; the
                # window-keyed artifact stores the complete replay context,
                # including its history warm-up prefix.
                start_index + window_length - history_start,
            )
        samples: list[dict[str, Any]] = []
        for absolute_index in range(history_start, start_index + window_length):
            sample = dict(self.dataset[absolute_index])
            # Hindsight labels are complementary data and are present only in
            # an HD run.  Mark every physical frame carrying them as a valid
            # writer interaction, including the history prefix below.  The
            # target action remains masked independently via action_is_pad.
            if any(isinstance(key, str) and key.startswith("hd_") for key in sample):
                existing_writer_valid = sample.get(HD_WRITER_VALID_KEY)
                if isinstance(existing_writer_valid, torch.Tensor):
                    sample[HD_WRITER_VALID_KEY] = existing_writer_valid.bool()
                else:
                    sample[HD_WRITER_VALID_KEY] = torch.tensor(True, dtype=torch.bool)
            # The warm-up is intentionally part of the recurrent computation,
            # but it must not contribute an imitation/HCA/grounding target.
            # Reusing LeRobot's action padding convention masks those action
            # losses.  The separate local writer objective remains active on
            # warm-up rows; ``hd_write_gate_observed`` masks their unavailable
            # hindsight gate target.
            if absolute_index < target_start:
                if self.hd_window_local and not self.hd_window_keyed:
                    sample[HD_WRITER_VALID_KEY] = torch.tensor(True, dtype=torch.bool)
                    # A frame-local artifact cannot encode a full replay
                    # context.  Reset its warm-up intervention to the
                    # ordinary all-write branch; window-keyed artifacts carry
                    # an exact gate vector and bypass this path.
                    for gate_key in ("hd_write_gate", "hd_counterfactual_write_gate"):
                        gate_value = sample.get(gate_key)
                        if isinstance(gate_value, torch.Tensor):
                            sample[gate_key] = torch.ones_like(gate_value)
                    observed_value = sample.get("hd_write_gate_observed")
                    if isinstance(observed_value, torch.Tensor):
                        sample["hd_write_gate_observed"] = torch.zeros_like(observed_value)
                    for attribution_key in ("hd_attribution", "hd_rho"):
                        attribution_value = sample.get(attribution_key)
                        if isinstance(attribution_value, torch.Tensor):
                            sample[attribution_key] = torch.zeros_like(attribution_value)
                action_is_pad = sample.get("action_is_pad")
                if isinstance(action_is_pad, torch.Tensor):
                    sample["action_is_pad"] = torch.ones_like(action_is_pad, dtype=torch.bool)
                elif ACTION in sample and isinstance(sample[ACTION], torch.Tensor):
                    sample["action_is_pad"] = torch.ones(
                        sample[ACTION].shape[:-1], dtype=torch.bool
                    )
            # Window-keyed artifacts are authoritative.  Apply them after the
            # warm-up action mask so their full counterfactual gate vector is
            # preserved exactly for the replay branch, including history rows.
            if window_labels is not None:
                overlay = window_labels.get(absolute_index)
                if overlay is not None:
                    sample.update(overlay)
            samples.append(sample)
        return samples


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
