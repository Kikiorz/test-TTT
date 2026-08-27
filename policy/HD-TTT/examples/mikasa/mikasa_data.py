#!/usr/bin/env python
"""Shared MIKASA episode decoding for the CreditTTT data pipeline.

This module contains preprocessing only.  It deliberately has no hindsight,
teacher, or TTT objective so the canonical CreditTTT scripts do not depend on
the removed legacy HD/V2 label builders.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _as_int(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.detach().cpu().reshape(()).item())
    return int(value)


def _task_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _task_text(value[0])
    if isinstance(value, Tensor) and value.numel() == 1:
        return str(value.item())
    return str(value)


def _strip_batch(value: Any) -> Any:
    """Remove the processor's leading singleton batch dimension."""

    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def _concat_processed(processed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor model inputs shared by every processed frame."""

    if not processed_rows:
        raise ValueError("Cannot concatenate an empty episode")
    keys = set(processed_rows[0])
    for row in processed_rows[1:]:
        keys.intersection_update(row)
    result: dict[str, Any] = {}
    for key in sorted(keys):
        values = [row[key] for row in processed_rows]
        if all(isinstance(value, Tensor) for value in values):
            result[key] = torch.stack(values, dim=0)
    return result


def prepare_episode(
    dataset: Any,
    local_start: int,
    length: int,
    policy: Any,
    preprocessor: Any,
    *,
    task_override: str | None,
    device: torch.device,
) -> tuple[dict[str, Tensor], list[int], list[int], list[int]]:
    """Decode and preprocess one complete episode into model-ready tensors."""

    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    rows: list[dict[str, Any]] = []
    global_indices: list[int] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    language_cache: dict[str, dict[str, Tensor]] = {}

    for offset in range(length):
        sample = dataset[local_start + offset]
        task = task_override or _task_text(sample.get("task", ""))
        raw: dict[str, Any] = {"task": task}
        for key in policy.config.image_features:
            if key not in sample:
                raise KeyError(f"Dataset frame is missing policy camera feature {key!r}")
            image = sample[key]
            if not isinstance(image, Tensor):
                image = torch.as_tensor(image)
            image = image.to(dtype=torch.float32)
            if image.numel() and float(image.max()) > 1.0 + 1e-5:
                image = image / 255.0
            raw[key] = image

        raw[OBS_STATE] = sample[OBS_STATE].to(dtype=torch.float32)
        raw[ACTION] = sample[ACTION].to(dtype=torch.float32)
        if "action_is_pad" in sample:
            raw["action_is_pad"] = sample["action_is_pad"].bool()

        processed = dict(preprocessor(raw))
        # MIKASA repeats one instruction through an episode.  Cache only the
        # tokenized language fields; image/state/action preprocessing remains
        # frame-specific.
        if task in language_cache:
            for language_key, language_value in language_cache[task].items():
                processed[language_key] = language_value.clone()
        else:
            language_cache[task] = {
                language_key: processed[language_key].detach().clone()
                for language_key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
                if language_key in processed and isinstance(processed[language_key], Tensor)
            }

        rows.append({key: _strip_batch(value) for key, value in processed.items()})
        global_indices.append(_as_int(sample["index"]))
        episode_indices.append(_as_int(sample["episode_index"]))
        frame_indices.append(_as_int(sample.get("frame_index", offset)))

    batch = {
        key: value.to(device=device, non_blocking=True)
        for key, value in _concat_processed(rows).items()
        if isinstance(value, Tensor)
    }
    return batch, global_indices, episode_indices, frame_indices


__all__ = ["prepare_episode"]
