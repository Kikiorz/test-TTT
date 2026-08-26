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
from torch.utils.data import BatchSampler, Dataset

from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

SEQUENCE_SHAPE_KEY = "_lerobot_sequence_shape"
# Episode-local index of the first physical frame represented by a sampled
# sequence.  Unlike ``SEQUENCE_SHAPE_KEY`` this metadata must survive the
# policy preprocessor: CreditTTT pair labels are episode-local, while the
# dataloader flattens a selected episode view into one contiguous tensor.
# Keeping the key outside the ``hd_*`` namespace is intentional; its presence
# must not make an otherwise unlabeled/clean batch look like an HD-label batch.
SEQUENCE_OFFSET_KEY = "_lerobot_sequence_offset"
# Complementary frame mask for training-only HD writer objectives.  Unlike
# ``action_is_pad`` it remains true on replayed history frames: those frames
# have no imitation target in the current window, but their causal interaction
# still must train the local K/V objective.  Window-local counterfactual gate
# supervision is masked separately through ``hd_write_gate_observed``.
HD_WRITER_VALID_KEY = "hd_writer_valid"
# Preserve action-chunk validity before warm-up rows are masked with
# ``action_is_pad=True``.  The HD local writer objective still trains on
# history interactions, but should not learn from repeated terminal slots.
HD_ACTION_SLOT_VALID_KEY = "hd_action_slot_valid"


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
        # One entry per sampled target window.  Values are episode-local (not
        # global selected-view indices) and include any replayed warm-up rows.
        # For a window beginning at frame ``o`` with a warm-up of ``w``, the
        # value is ``max(0, o-w)``; this is exactly the coordinate origin used
        # by the offline event/future pair artifact.
        self.window_sequence_offsets: list[int] = []

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
                self.window_sequence_offsets.append(int(history_start - episode_start))
            episode_start += episode_length

        if episode_start != len(dataset):
            raise ValueError(
                f"Episode metadata covers {episode_start} frames, but the dataset contains {len(dataset)}"
            )
        if not self.window_specs:
            raise ValueError("The dataset does not contain any non-empty episode")

        # Cache physical lengths so an equal-length batch sampler can bucket
        # windows without materializing every sample (or decoding images).
        self.sequence_lengths = [
            (target_start + window_length) - history_start
            for (target_start, window_length), (history_start, _) in zip(
                self.window_specs, self.history_specs
            )
        ]

    def __len__(self) -> int:
        return len(self.window_specs)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        start_index, window_length = self.window_specs[index]
        history_start, target_start = self.history_specs[index]
        sequence_offset = self.window_sequence_offsets[index]
        if sequence_offset < 0:
            raise RuntimeError("Internal sequence offset bookkeeping produced a negative value")
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
            # Attach the same scalar to every row so ordinary/default collate
            # cannot accidentally infer a different origin from one timestep.
            # ``sequence_collate_fn`` validates consistency and stores a
            # single scalar in the flattened batch.
            sample[SEQUENCE_OFFSET_KEY] = torch.tensor(sequence_offset, dtype=torch.int64)
            # Save the physical action-slot mask before the warm-up convention
            # below replaces ``action_is_pad`` with all-true values.  This
            # auxiliary mask is consumed only by the HD local writer loss;
            # ordinary imitation losses continue to use ``action_is_pad``.
            original_action_slot_valid = None
            original_action_is_pad = sample.get("action_is_pad")
            if isinstance(original_action_is_pad, torch.Tensor):
                original_action_slot_valid = (~original_action_is_pad.bool()).clone()
            elif ACTION in sample and isinstance(sample[ACTION], torch.Tensor):
                original_action_slot_valid = torch.ones(
                    sample[ACTION].shape[:-1], dtype=torch.bool
                )
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
            # Apply the preserved mask after window overlays so frame-level and
            # window-keyed artifacts share the same writer contract.  Do not
            # add it to ordinary datasets: the model uses the presence of
            # ``hd_*`` fields to detect an offline label replay.
            has_hd_labels = any(
                isinstance(key, str)
                and key.startswith("hd_")
                and key not in {HD_WRITER_VALID_KEY, HD_ACTION_SLOT_VALID_KEY}
                for key in sample
            )
            if has_hd_labels and original_action_slot_valid is not None:
                sample[HD_ACTION_SLOT_VALID_KEY] = original_action_slot_valid
            samples.append(sample)
        return samples


def batched_sequence_collate_fn(
    batch: list[list[dict[str, Any]]], *, require_zero_offsets: bool = False
) -> dict[str, Any]:
    """Collate one or more equal-length trajectories without temporal padding.

    Each trajectory is flattened independently along time and then concatenated
    in batch order.  ``SEQUENCE_SHAPE_KEY`` records the original ``[B, T]``
    shape so the recurrent policy can restore per-trajectory state.  Mixed
    lengths are rejected rather than padded; callers should use
    :class:`EqualLengthBatchSampler` to form batches.
    """
    if not batch:
        raise ValueError("Cannot collate an empty sequence batch")

    sequence_lengths = [len(sequence) for sequence in batch]
    if any(length == 0 for length in sequence_lengths):
        raise ValueError("A sequence must contain at least one timestep")
    if len(set(sequence_lengths)) != 1:
        raise ValueError(
            "Equal-length trajectory batching requires every sequence to have the same T; "
            f"got {sequence_lengths}"
        )
    sequence_length = sequence_lengths[0]
    if any(sample is None for sequence in batch for sample in sequence):
        raise ValueError("Sequence samples cannot be None because dropping a timestep corrupts TTT state")

    # All rows in one sampled window share one episode-local origin.  Keep a
    # backward-compatible fallback for direct callers that construct legacy
    # sequences without metadata, but reject partial/mixed metadata rather
    # than silently training against misaligned pair labels.
    raw_offsets = [sample.get(SEQUENCE_OFFSET_KEY) for sequence in batch for sample in sequence]
    present = [value is not None for value in raw_offsets]
    if any(present) and not all(present):
        raise ValueError(
            f"Inconsistent sequence batch: {SEQUENCE_OFFSET_KEY!r} is present on only "
            f"{sum(present)}/{len(present)} timesteps"
        )
    if any(present):
        offsets: list[int] = []
        for value in raw_offsets:
            try:
                tensor = torch.as_tensor(value)
            except Exception as exc:
                raise ValueError(
                    f"{SEQUENCE_OFFSET_KEY!r} must be an integer scalar"
                ) from exc
            if tensor.numel() != 1:
                raise ValueError(
                    f"{SEQUENCE_OFFSET_KEY!r} must be scalar per timestep, "
                    f"got shape {tuple(tensor.shape)}"
                )
            scalar = tensor.reshape(()).item()
            if isinstance(scalar, float) and not scalar.is_integer():
                raise ValueError(
                    f"{SEQUENCE_OFFSET_KEY!r} must be integral, got {scalar!r}"
                )
            offset = int(scalar)
            if offset < 0:
                raise ValueError(
                    f"{SEQUENCE_OFFSET_KEY!r} must be non-negative, got {offset}"
                )
            offsets.append(offset)
        if len(set(offsets)) != 1:
            raise ValueError(
                f"All timesteps in a sequence must share {SEQUENCE_OFFSET_KEY!r}; "
                f"got {offsets}"
            )
        # A batched canonical CreditTTT V3 window is episode-local and starts
        # at frame zero.  Rejecting non-zero origins here prevents silently
        # applying one episode's pair labels at another episode's coordinate.
        sequence_offset = offsets[0]
        if require_zero_offsets and sequence_offset != 0:
            raise ValueError(
                f"Canonical V3 equal-length batches require {SEQUENCE_OFFSET_KEY!r}=0, "
                f"got {sequence_offset}"
            )
    else:
        sequence_offset = 0

    flattened_batch = [sample for sequence in batch for sample in sequence]
    collated = lerobot_collate_fn(flattened_batch)
    if collated is None:
        raise ValueError("Sequence collation unexpectedly produced an empty batch")
    collated[SEQUENCE_SHAPE_KEY] = torch.tensor(
        [len(batch), sequence_length], dtype=torch.int64
    )
    # Override default_collate's [T] representation with one scalar.  This
    # keeps the metadata invariant under TBPTT slicing and processor batching.
    collated[SEQUENCE_OFFSET_KEY] = torch.tensor(sequence_offset, dtype=torch.int64)
    return collated


def sequence_collate_fn(batch: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Legacy single-trajectory collate function.

    Keep the historical API strict (``B=1``) while exposing
    :func:`batched_sequence_collate_fn` for opt-in equal-length batching.
    """
    if len(batch) != 1:
        raise ValueError("smolvla_ttt tail-preserving batches require per-device batch_size=1")
    return batched_sequence_collate_fn(batch)


class EqualLengthBatchSampler(BatchSampler):
    """Group trajectory indices by exact physical length, with no padding.

    Every sequence in a group has identical ``T``.  A short final group is
    completed by repeating indices from that same length bucket (never by
    temporal padding).  In distributed training, complete groups are repeated
    as needed so the total batch count is divisible by the process count; this
    keeps all ranks synchronized while preserving recurrent semantics.
    """

    def __init__(
        self,
        dataset: TailPreservingSequenceDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        num_replicas: int = 1,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not hasattr(dataset, "sequence_lengths"):
            raise TypeError("EqualLengthBatchSampler requires TailPreservingSequenceDataset")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_replicas = int(num_replicas)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

    def _batches(self) -> list[list[int]]:
        # The collator stores one episode-local offset for the complete
        # flattened batch.  Bucket by ``(T, offset)`` rather than T alone so
        # bounded windows from different episode origins can never be mixed
        # into a batch whose pair-label coordinate system is ambiguous.
        buckets: dict[tuple[int, int], list[int]] = {}
        offsets = getattr(self.dataset, "window_sequence_offsets", None)
        if offsets is None or len(offsets) != len(self.dataset.sequence_lengths):
            raise ValueError(
                "EqualLengthBatchSampler requires one sequence offset per trajectory window"
            )
        for index, (length, offset) in enumerate(
            zip(self.dataset.sequence_lengths, offsets, strict=True)
        ):
            buckets.setdefault((int(length), int(offset)), []).append(index)
        groups: list[list[int]] = []
        generator = torch.Generator()
        # An explicit seed makes every rank construct the same global batch
        # order even when DataLoader workers or Accelerate initialize the
        # process-local torch RNGs differently.
        generator.manual_seed((self.seed + self._epoch) % (2**63 - 1))
        self._epoch += 1
        bucket_items = list(buckets.items())
        if self.shuffle:
            order = torch.randperm(len(bucket_items), generator=generator).tolist()
            bucket_items = [bucket_items[i] for i in order]
        for _, indices in bucket_items:
            if self.shuffle and len(indices) > 1:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in permutation]
            for start in range(0, len(indices), self.batch_size):
                group = indices[start : start + self.batch_size]
                if len(group) < self.batch_size:
                    if self.drop_last:
                        continue
                    # Repeat only from this exact-length bucket.  This is
                    # index-level balancing, not temporal sequence padding.
                    group.extend(indices[i % len(indices)] for i in range(self.batch_size - len(group)))
                groups.append(group)
        if self.shuffle and len(groups) > 1:
            permutation = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[i] for i in permutation]
        if self.num_replicas > 1:
            if not groups:
                raise ValueError("No trajectory batches available for distributed training")
            remainder = len(groups) % self.num_replicas
            if remainder:
                groups.extend(groups[i % len(groups)] for i in range(self.num_replicas - remainder))
        return groups

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        # Compute using deterministic bucket arithmetic (without shuffling).
        bucket_counts: dict[tuple[int, int], int] = {}
        offsets = getattr(self.dataset, "window_sequence_offsets", None)
        if offsets is None or len(offsets) != len(self.dataset.sequence_lengths):
            raise ValueError(
                "EqualLengthBatchSampler requires one sequence offset per trajectory window"
            )
        for length, offset in zip(self.dataset.sequence_lengths, offsets, strict=True):
            key = (int(length), int(offset))
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
        count = sum(
            n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
            for n in bucket_counts.values()
        )
        if self.num_replicas > 1 and count:
            count += (-count) % self.num_replicas
        return count
