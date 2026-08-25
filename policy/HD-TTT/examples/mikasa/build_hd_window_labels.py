#!/usr/bin/env python
"""Build phase-matched, window-keyed HD-TTT labels for long episodes.

The hindsight teacher must replay the same bounded recurrent context that the
student sees.  A frame-only artifact cannot express this when windows overlap:
the same frame has different noise/state/counterfactual-gate values in the two
contexts.  This collector therefore stores one complete label record per
training window and the sequence dataset selects the record by target frame.

The collector mirrors :class:`TailPreservingSequenceDataset`, including its
optional episode-balanced window cap.  Noise is sampled once per episode and
sliced into every context, so warm-up and target uses are deterministic and
consistent.  By default ``phase_mode=deployment`` makes the written action
interaction pure Gaussian noise at ``t=1``; future expert actions remain only
the offline flow target.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:  # Direct ``python examples/mikasa/...py`` invocation.
    from build_hd_labels import (
        _episode_labels,
        _episode_table,
        _fixed_noise_time,
        _prepare_episode,
        _selected_episodes,
    )
except ImportError:  # Package-style invocation.
    from .build_hd_labels import (
        _episode_labels,
        _episode_table,
        _fixed_noise_time,
        _prepare_episode,
        _selected_episodes,
    )


LOGGER = logging.getLogger("build_hd_window_labels")

_FRAME_LABEL_KEYS = (
    "hd_teacher_velocity",
    "hd_teacher_true_velocity",
    "hd_teacher_wrong_velocity",
    "hd_noise",
    "hd_time",
    "hd_attribution",
    "hd_rho",
    "hd_write_gate",
    "hd_write_gate_observed",
    "hd_counterfactual_write_gate",
)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _slice_prepared(prepared: Mapping[str, Tensor], start: int, end: int) -> dict[str, Tensor]:
    """Slice frame-aligned model inputs while preserving scalar metadata."""

    result: dict[str, Tensor] = {}
    for key, value in prepared.items():
        if not isinstance(value, Tensor):
            continue
        if value.ndim > 0 and value.shape[0] >= end:
            result[key] = value[start:end]
        else:
            result[key] = value
    return result


def _window_specs(
    length: int,
    sequence_length: int,
    sequence_stride: int,
    max_windows_per_episode: int | None,
) -> list[tuple[int, int]]:
    """Exactly reproduce ``TailPreservingSequenceDataset``'s offset logic."""

    if length <= 0:
        raise ValueError("episode length must be positive")
    if sequence_length <= 0 or sequence_stride <= 0:
        raise ValueError("sequence length/stride must be positive")
    if sequence_stride > sequence_length:
        raise ValueError("sequence stride cannot exceed sequence length")
    if max_windows_per_episode is not None and max_windows_per_episode <= 0:
        raise ValueError("max_windows_per_episode must be positive")

    offsets = list(range(0, length, sequence_stride))
    if max_windows_per_episode is not None and len(offsets) > max_windows_per_episode:
        last_full_offset = max(length - sequence_length, 0)
        full_offsets = list(range(0, last_full_offset + 1, sequence_stride))
        if not full_offsets or full_offsets[-1] != last_full_offset:
            full_offsets.append(last_full_offset)
        positions = (
            torch.linspace(0, len(full_offsets) - 1, max_windows_per_episode)
            .round()
            .to(torch.long)
            .tolist()
        )
        offsets = [full_offsets[int(position)] for position in sorted(set(positions))]
    return [(offset, min(offset + sequence_length, length)) for offset in offsets]


def _window_record(
    *,
    labels: Mapping[str, Any],
    source_indices: list[int],
    target_global_index: int,
    history_start_source: int,
    target_start: int,
    target_end: int,
    context_start: int,
    context_end: int,
    episode: int,
) -> dict[str, Any]:
    context_length = context_end - context_start
    record_labels: dict[str, Tensor] = {}
    for key in _FRAME_LABEL_KEYS:
        value = labels.get(key)
        if not isinstance(value, Tensor) or value.ndim == 0 or value.shape[0] != context_length:
            raise ValueError(
                f"Window label {key!r} must have {context_length} frame rows"
            )
        record_labels[key] = value.detach().cpu().float() if value.is_floating_point() else value.detach().cpu()
    # Every replayed interaction can train the instantaneous local writer.
    # Gate/attribution supervision itself is carried by the observed mask.
    record_labels["hd_writer_valid"] = torch.ones(context_length, dtype=torch.bool)
    audit: dict[str, Tensor] = {}
    for key in (
        "hd_C",
        "hd_event_u",
        "hd_event_starts",
        "hd_event_ends",
        "hd_selected_event",
        "hd_full_flow_loss",
        "hd_event_selection_scores",
    ):
        value = labels.get(key)
        if isinstance(value, Tensor):
            audit[key] = value.detach().cpu()
    return {
        "target_global_index": int(target_global_index),
        "history_start_source": int(history_start_source),
        "source_indices": torch.tensor(source_indices, dtype=torch.int64),
        "length": int(context_length),
        "episode_index": int(episode),
        "target_start_frame": int(target_start),
        "target_end_frame": int(target_end),
        "context_start_frame": int(context_start),
        "context_end_frame": int(context_end),
        "labels": record_labels,
        "audit": audit,
    }


def _build_shard(args: argparse.Namespace) -> None:
    # Delayed imports keep ``--help`` and syntax checks usable on a CPU-only
    # machine without importing SAPIEN/video backends.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla_ttt.configuration_smolvla_ttt import SmolVLATTTConfig
    from lerobot.policies.smolvla_ttt.processor_smolvla_ttt import (
        make_smolvla_ttt_pre_post_processors,
    )

    device = torch.device(args.device)
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    selected = _selected_episodes(metadata, args.episode_start, args.episode_end)
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]
    fps = int(metadata.fps)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=selected,
        delta_timestamps={"action": [index / fps for index in range(args.action_chunk_size)]},
        download_videos=args.download_videos,
        video_backend=args.video_backend,
    )
    table = _episode_table(dataset, selected)

    config = SmolVLATTTConfig(
        device=args.device,
        pretrained_path=Path(args.checkpoint),
        ttt_training_stage="ttt_only",
    )
    policy = make_policy(config, ds_meta=metadata)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    preprocessor, _ = make_smolvla_ttt_pre_post_processors(
        policy.config,
        dataset_stats=metadata.stats,
    )
    if int(policy.config.chunk_size) != args.action_chunk_size:
        raise ValueError(
            f"Checkpoint chunk_size={policy.config.chunk_size} but action_chunk_size="
            f"{args.action_chunk_size}"
        )
    action_dim = int(policy.config.max_action_dim)

    windows: list[dict[str, Any]] = []
    episode_details: dict[str, Any] = {}
    for episode, local_episode_start, source_length in table:
        length = source_length if args.max_frames is None else min(source_length, args.max_frames)
        if length <= 0:
            raise ValueError(f"Episode {episode} has no frames after --max-frames")
        LOGGER.info("Episode %d: %d frames (local start %d)", episode, length, local_episode_start)
        prepared, global_indices, _episode_indices, _frame_indices = _prepare_episode(
            dataset,
            local_episode_start,
            length,
            policy,
            preprocessor,
            task_override=args.task,
            device=device,
        )
        episode_seed = args.seed + episode * 1_000_003
        episode_noise, episode_time = _fixed_noise_time(
            length=length,
            chunk_size=int(policy.config.chunk_size),
            action_dim=action_dim,
            seed=episode_seed,
            phase_mode=args.phase_mode,
            device=device,
        )
        detail_windows: list[dict[str, Any]] = []
        for target_start, target_end in _window_specs(
            length,
            args.sequence_length,
            args.sequence_stride,
            args.max_windows_per_episode,
        ):
            context_start = max(0, target_start - args.context_length)
            context_end = target_end
            local_target_start = target_start - context_start
            local_target_end = target_end - context_start
            local_prepared = _slice_prepared(prepared, context_start, context_end)
            local_noise = episode_noise[context_start:context_end]
            local_time = episode_time[context_start:context_end]
            future_mask = torch.zeros(context_end - context_start, dtype=torch.bool, device=device)
            future_mask[local_target_start:local_target_end] = True
            labels = _episode_labels(
                policy,
                local_prepared,
                local_noise,
                local_time,
                event_block_size=args.event_block_size,
                max_events=args.max_events,
                attribution_threshold=args.attribution_threshold,
                frame_batch_size=args.frame_batch_size,
                future_mask=future_mask,
                global_offset=context_start,
            )
            record = _window_record(
                labels=labels,
                source_indices=global_indices[context_start:context_end],
                target_global_index=global_indices[target_start],
                history_start_source=global_indices[context_start],
                target_start=target_start,
                target_end=target_end,
                context_start=context_start,
                context_end=context_end,
                episode=episode,
            )
            windows.append(record)
            selected_event = labels["hd_selected_event"].tolist()
            detail_windows.append(
                {
                    "target_start": target_start,
                    "target_end": target_end,
                    "context_start": context_start,
                    "context_end": context_end,
                    "selected_event_local": selected_event,
                    "selected_event_global": (
                        [
                            selected_event[0] + context_start,
                            selected_event[1] + context_start,
                        ]
                        if selected_event[0] >= 0
                        else [-1, -1]
                    ),
                }
            )
            del local_prepared, labels, local_noise, local_time
            if device.type == "cuda":
                torch.cuda.empty_cache()
        episode_details[str(episode)] = {
            "length": length,
            "global_indices": [int(value) for value in global_indices],
            "noise_seed": int(episode_seed),
            "windows": detail_windows,
        }
        del prepared, episode_noise, episode_time

    payload = {
        "windows": windows,
        "metadata": {
            "format": "hd_ttt_labels_v1",
            "window_local": True,
            "window_keyed": True,
            "window_key": "target_global_index",
            "dataset_repo_id": args.dataset_repo_id,
            "dataset_root": str(args.dataset_root),
            "checkpoint": str(args.checkpoint),
            "fps": fps,
            "episodes": selected,
            "sequence_length": args.sequence_length,
            "sequence_stride": args.sequence_stride,
            "context_length": args.context_length,
            "max_windows_per_episode": args.max_windows_per_episode,
            "event_block_size": args.event_block_size,
            "max_events": args.max_events,
            "phase_mode": args.phase_mode,
            "writer_observation": (
                "pure_gaussian_action_noise_at_t1"
                if args.phase_mode == "deployment"
                else "random_flow_interpolation_with_expert_action_chunk"
            ),
            "attribution_aggregation": "all_event_max_for_hca_selected_event_for_grounding",
            "action_chunk_size": int(policy.config.chunk_size),
            "max_action_dim": action_dim,
            "episodes_detail": episode_details,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    LOGGER.info("Wrote %d window records -> %s", len(windows), args.output)


def _merge_shards(inputs: list[Path], output: Path) -> None:
    if not inputs:
        raise ValueError("--merge requires at least one input shard")
    payloads = [_load_torch(path) for path in inputs]
    all_windows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    seen_targets: set[int] = set()
    contract_keys = (
        "sequence_length",
        "sequence_stride",
        "context_length",
        "max_windows_per_episode",
        "event_block_size",
        "phase_mode",
        "checkpoint",
    )
    reference: dict[str, Any] | None = None
    for path, payload in zip(inputs, payloads, strict=True):
        if not isinstance(payload, Mapping) or not isinstance(payload.get("windows"), list):
            raise ValueError(f"Shard {path} is missing a window record list")
        shard_meta = payload.get("metadata")
        if not isinstance(shard_meta, Mapping):
            raise ValueError(f"Shard {path} is missing metadata")
        shard_meta = dict(shard_meta)
        if reference is None:
            reference = shard_meta
        else:
            mismatches = {
                key: (reference.get(key), shard_meta.get(key))
                for key in contract_keys
                if reference.get(key) != shard_meta.get(key)
            }
            if mismatches:
                raise ValueError(f"Cannot merge incompatible window shards: {mismatches}")
        for window in payload["windows"]:
            target = int(window["target_global_index"])
            if target in seen_targets:
                raise ValueError(f"Duplicate window target source frame {target}")
            seen_targets.add(target)
            all_windows.append(window)
        shard_meta["source_path"] = str(path)
        metadata.append(shard_meta)
    if reference is None:
        raise ValueError("No shard metadata")
    merged_meta = dict(reference)
    merged_meta.update(
        {
            "merged_from": [str(path) for path in inputs],
            "shard_metadata": metadata,
            "num_windows": len(all_windows),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"windows": all_windows, "metadata": merged_meta}, output)
    LOGGER.info("Merged %d window shards (%d windows) -> %s", len(inputs), len(all_windows), output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", nargs="*", type=Path, default=None)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--action-chunk-size", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--sequence-stride", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--max-windows-per-episode", type=int, default=None)
    parser.add_argument("--event-block-size", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--attribution-threshold", type=float, default=0.0)
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--phase-mode", choices=("random", "deployment"), default="deployment")
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec", "video_reader"))
    parser.add_argument("--download-videos", action="store_true")
    args = parser.parse_args()
    if args.merge is not None and len(args.merge) == 0:
        parser.error("--merge needs one or more input shards")
    if args.merge is None:
        missing = [
            name
            for name in ("dataset_repo_id", "dataset_root", "checkpoint")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("generation requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.context_length < 0:
        parser.error("--context-length must be non-negative")
    if args.frame_batch_size <= 0:
        parser.error("--frame-batch-size must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()
    if args.merge is not None:
        _merge_shards(args.merge, args.output)
    else:
        _build_shard(args)


if __name__ == "__main__":
    main()
