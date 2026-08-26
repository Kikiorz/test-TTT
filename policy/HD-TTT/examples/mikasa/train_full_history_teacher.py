#!/usr/bin/env python
"""Train the explicit causal Full-History Action Teacher used by CreditTTT.

The teacher is a training-time instrument.  It consumes one observation
prefix token and the action executed immediately before that observation, then
predicts the current executed slot-0 action.  It never receives the current
expert action, a denoising sample, or a future observation.  The resulting
checkpoint is used only by :mod:`build_credit_labels`; it is not loaded by the
deployed policy.

The script deliberately keeps feature extraction and teacher fitting in one
reproducible command.  ``--features-output`` can be supplied to cache the
causal event tokens, which is useful when generating several intervention
artifacts from the same base checkpoint.  The cache contains no images or
model parameters, only detached prefix summaries and normalized actions.

Example (MIKASA environment)::

    python examples/mikasa/train_full_history_teacher.py \
      --dataset-repo-id shell_game_shuffle_color_lamp_touch_long_vla_v0 \
      --dataset-root /workspace/data_mikasa_robo/data_lerobot/\
        shell_game_shuffle_color_lamp_touch_long_vla_v0 \
      --base-checkpoint /workspace/experiments/clean_ttt/checkpoints/last/pretrained_model \
      --output /workspace/credit_ttt/teacher.pt \
      --features-output /workspace/credit_ttt/features.pt \
      --episode-start 0 --episode-end 200 --epochs 20 --device cuda

This file only reads the LeRobot dataset and writes the requested artifacts;
it does not modify the source dataset or either protected policy directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.smolvla_ttt.history_teacher import (
    FULL_HISTORY_TEACHER_FORMAT,
    FullHistoryActionTeacher,
    history_teacher_state_sha256,
    summarize_prefix,
)


LOGGER = logging.getLogger("credit_ttt.full_history_teacher")
FEATURE_FORMAT = "credit_ttt_v3_prefix_features_v1"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - PyTorch < 2.0 compatibility
        return torch.load(path, map_location="cpu")


def _as_cpu_float(value: Any) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.detach().cpu().float()


def _selected_episodes(metadata: Any, start: int, end: int | None, max_episodes: int | None) -> list[int]:
    total = int(metadata.total_episodes)
    stop = total if end is None else int(end)
    if start < 0 or stop <= start or stop > total:
        raise ValueError(f"episode range must satisfy 0 <= start < end <= {total}; got {start}:{stop}")
    episodes = list(range(int(start), stop))
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        episodes = episodes[: int(max_episodes)]
    return episodes


def _episode_table(dataset: Any, episodes: Sequence[int]) -> list[tuple[int, int, int]]:
    metadata = getattr(dataset, "meta", dataset)
    rows = metadata.episodes
    if isinstance(rows, Mapping):
        starts, ends = rows["dataset_from_index"], rows["dataset_to_index"]
    else:
        starts = [row["dataset_from_index"] for row in rows]
        ends = [row["dataset_to_index"] for row in rows]
    table: list[tuple[int, int, int]] = []
    local = 0
    for episode in episodes:
        start = int(starts[int(episode)])
        end = int(ends[int(episode)])
        if end <= start:
            raise ValueError(f"episode {episode} has non-positive length")
        table.append((int(episode), local, end - start))
        local += end - start
    if local != len(dataset):
        raise ValueError(
            f"selected episode table covers {local} frames but dataset view has {len(dataset)}; "
            "construct LeRobotDataset with the same episode subset"
        )
    return table


def _checkpoint_writer_mode(checkpoint: str | Path) -> str:
    path = Path(str(checkpoint)).expanduser()
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.is_file():
        # Hub resolution is intentionally lazy; the normal policy loader will
        # produce a clearer error if a remote id cannot be resolved.
        return "suffix"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read base checkpoint config {config_path}: {exc}") from exc
    mode = str(raw.get("ttt_writer_mode") or "suffix")
    if mode not in {"suffix", "prefix_only"}:
        raise ValueError(f"Unsupported checkpoint ttt_writer_mode={mode!r}")
    return mode


def _load_base_policy(args: argparse.Namespace):
    """Construct the frozen SmolVLA-TTT encoder with dataset features injected."""

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla_ttt.configuration_smolvla_ttt import SmolVLATTTConfig
    from lerobot.policies.smolvla_ttt.processor_smolvla_ttt import (
        make_smolvla_ttt_pre_post_processors,
    )

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    writer_mode = _checkpoint_writer_mode(args.base_checkpoint)
    config = SmolVLATTTConfig(
        device=args.device,
        pretrained_path=Path(args.base_checkpoint),
        ttt_writer_mode=writer_mode,
        hd_ttt_enabled=False,
        hd_effect_weight=0.0,
        hd_attribution_protocol="legacy_raw_hinge_max",
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
    return metadata, policy, preprocessor


def _strip_singleton(value: Any) -> Any:
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def _extract_episode_features(
    dataset: Any,
    local_start: int,
    length: int,
    policy: Any,
    preprocessor: Any,
    *,
    task_override: str | None,
    device: torch.device,
) -> dict[str, Any]:
    """Extract one event token per physical frame from the frozen VLM prefix."""

    # Importing the helper keeps the exact camera/action preprocessing shared
    # with the existing MIKASA label builder, avoiding a second normalization
    # implementation in the teacher path.
    from build_hd_labels import _prepare_episode

    prepared, global_indices, episode_indices, frame_indices = _prepare_episode(
        dataset,
        local_start,
        length,
        policy,
        preprocessor,
        task_override=task_override,
        device=device,
    )
    images, image_masks = policy.prepare_images(prepared)
    state = policy.prepare_state(prepared)
    with torch.no_grad():
        prefix, prefix_mask, _ = policy.model.embed_prefix(
            images,
            image_masks,
            prepared["observation.language.tokens"],
            prepared["observation.language.attention_mask"],
            state=state,
        )
        events = summarize_prefix(prefix, prefix_mask)
        actions = policy.prepare_action(prepared)
    if actions.ndim < 3:
        raise ValueError(f"Expected action chunks [T,S,D], got {tuple(actions.shape)}")
    executed = actions[:, 0, : int(policy.config.action_feature.shape[0])].detach()
    previous = torch.zeros_like(executed)
    if executed.shape[0] > 1:
        previous[1:] = executed[:-1]
    if events.shape[0] != executed.shape[0]:
        raise ValueError(
            f"Prefix/event count {events.shape[0]} does not match actions {executed.shape[0]}"
        )
    return {
        "global_indices": torch.as_tensor(global_indices, dtype=torch.int64),
        "episode_indices": torch.as_tensor(episode_indices, dtype=torch.int64),
        "frame_indices": torch.as_tensor(frame_indices, dtype=torch.int64),
        "event_tokens": events.detach().cpu().float(),
        "previous_executed_actions": previous.cpu().float(),
        "target_actions": executed.cpu().float(),
    }


def _save_features(
    path: Path,
    episodes: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    event_dim: int,
    action_dim: int,
    source_config_sha256: str | None,
    fps: int | float | None = None,
) -> None:
    serializable = []
    for episode in episodes:
        serializable.append(
            {
                key: (value.detach().cpu() if isinstance(value, Tensor) else value)
                for key, value in episode.items()
            }
        )
    metadata = {
        "format": FEATURE_FORMAT,
        "version": 1,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "fps": None if fps is None else int(fps),
        "base_checkpoint": str(args.base_checkpoint),
        "base_config_sha256": source_config_sha256,
        "episode_start": int(args.episode_start),
        "episode_end": args.episode_end,
        "event_schema": "frozen_smolvla_prefix_mean",
        "event_dim": int(event_dim),
        "action_dim": int(action_dim),
        "target": "normalized_executed_slot0_action",
        "causal_previous_action": True,
    }
    metadata["features_sha256"] = _sha256_json(
        {
            "global_indices": [row["global_indices"].tolist() for row in serializable],
            "episode_indices": [row["episode_indices"].tolist() for row in serializable],
            "frame_indices": [row["frame_indices"].tolist() for row in serializable],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FEATURE_FORMAT, "metadata": metadata, "episodes": serializable}, path)
    LOGGER.info("Wrote prefix feature cache (%d episodes) to %s", len(serializable), path)


def _load_features(path: Path) -> tuple[dict[str, Any], list[dict[str, Tensor]]]:
    payload = _load_torch(path)
    if not isinstance(payload, Mapping) or payload.get("format") != FEATURE_FORMAT:
        raise ValueError(f"Unsupported feature artifact {path}; expected {FEATURE_FORMAT!r}")
    metadata = payload.get("metadata")
    episodes = payload.get("episodes")
    if not isinstance(metadata, Mapping) or not isinstance(episodes, Sequence):
        raise ValueError("Feature artifact needs metadata and episodes")
    rows: list[dict[str, Tensor]] = []
    required = {
        "event_tokens",
        "previous_executed_actions",
        "target_actions",
        "global_indices",
        "episode_indices",
        "frame_indices",
    }
    for index, raw in enumerate(episodes):
        if not isinstance(raw, Mapping) or not required.issubset(raw):
            raise ValueError(f"Feature episode {index} is missing required columns")
        row = {key: _as_cpu_float(value) if key not in {"global_indices", "episode_indices", "frame_indices"} else torch.as_tensor(value, dtype=torch.int64) for key, value in raw.items()}
        if row["event_tokens"].ndim != 2 or row["target_actions"].ndim != 2:
            raise ValueError(f"Feature episode {index} has invalid event/action ranks")
        if row["event_tokens"].shape[0] != row["target_actions"].shape[0]:
            raise ValueError(f"Feature episode {index} event/action lengths disagree")
        rows.append(row)
    return dict(metadata), rows


def _source_config_sha256(checkpoint: str | Path) -> str | None:
    path = Path(str(checkpoint)).expanduser()
    config = path / "config.json" if path.is_dir() else path
    if not config.is_file():
        return None
    return hashlib.sha256(config.read_bytes()).hexdigest()


def _episode_tensors(
    row: Mapping[str, Tensor],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return one cached episode in an unambiguous ``[1,T,*]`` layout.

    The feature cache intentionally stores compact episodes as ``[T,D]``.
    Keeping an explicit batch axis inside the training/audit helpers avoids a
    subtle class of bugs where ``shape[1]`` (the feature width) is mistaken
    for the number of physical frames.
    """

    events = row["event_tokens"].to(device=device)
    previous = row["previous_executed_actions"].to(device=device)
    targets = row["target_actions"].to(device=device)
    if events.ndim != 2 or previous.ndim != 2 or targets.ndim != 2:
        raise ValueError("cached episode tensors must have [T,D]/[T,A] shapes")
    if not (events.shape[0] == previous.shape[0] == targets.shape[0]):
        raise ValueError("cached episode event/action lengths disagree")
    return events.unsqueeze(0), previous.unsqueeze(0), targets.unsqueeze(0)


def _run_teacher_epoch(
    teacher: FullHistoryActionTeacher,
    rows: Sequence[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
) -> float:
    total = 0.0
    count = 0
    for row in rows:
        events, previous, targets = _episode_tensors(row, device=device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        output = teacher(events, previous)
        loss = teacher.action_loss(output.actions, targets)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
        # Feature caches store one episode as unbatched ``[T,D]``.  Weight
        # epoch statistics by physical frames (axis 0), not by event feature
        # width (axis 1); the latter silently distorted validation reports
        # while leaving gradients numerically unchanged.
        frame_count = int(events.shape[1])
        total += float(loss.detach().item()) * frame_count
        count += frame_count
    return total / max(count, 1)


@torch.no_grad()
def _evaluate_teacher(
    teacher: FullHistoryActionTeacher,
    rows: Sequence[Mapping[str, Tensor]],
    *,
    device: torch.device,
    short_context: int = 0,
) -> float:
    """Evaluate full history or a reset-at-every-frame short-context replay."""

    total = 0.0
    count = 0
    for row in rows:
        events, previous, targets = _episode_tensors(row, device=device)
        if short_context <= 0:
            predictions = teacher(events, previous).actions
        else:
            pieces = []
            for index in range(events.shape[1]):
                start = max(0, index - short_context + 1)
                pieces.append(teacher(events[:, start : index + 1], previous[:, start : index + 1]).actions[:, -1])
            predictions = torch.stack(pieces, dim=1)
        loss = teacher.action_loss(predictions, targets)
        frame_count = int(events.shape[1])
        total += float(loss.item()) * frame_count
        count += frame_count
    return total / max(count, 1)


@torch.no_grad()
def _history_swap_audit(
    teacher: FullHistoryActionTeacher,
    rows: Sequence[Mapping[str, Tensor]],
    *,
    device: torch.device,
) -> dict[str, float]:
    effects: list[Tensor] = []
    full_predictions: list[Tensor] = []
    for row in rows:
        events, previous, _targets = _episode_tensors(row, device=device)
        if events.shape[1] < 2:
            continue
        mask = torch.zeros(events.shape[1], dtype=torch.float32, device=device)
        mask[0] = 1.0
        full, deleted = teacher.replay_pair(events, previous, delete_mask=mask)
        effects.append((full.actions - deleted.actions).reshape(-1, full.actions.shape[-1]))
        full_predictions.append(full.actions.reshape(-1, full.actions.shape[-1]))
    if not effects:
        return {"history_swap_action_delta_rms": 0.0, "history_swap_action_delta_cosine": 0.0}
    effect = torch.cat(effects, dim=0)
    rms = float(effect.square().mean().sqrt().item())
    # Compare the intervention effect with the full prediction direction.  A
    # non-zero absolute effect is the primary audit; cosine is diagnostic and
    # remains defined for an all-zero teacher.
    full_direction = torch.cat(full_predictions, dim=0)
    denominator = effect.norm(dim=-1) * full_direction.norm(dim=-1)
    cosine = torch.where(
        denominator > 1e-8,
        (effect * full_direction).sum(dim=-1) / denominator.clamp_min(1e-8),
        torch.zeros_like(denominator),
    )
    return {
        "history_swap_action_delta_rms": rms,
        "history_swap_action_delta_cosine": float(cosine.mean().item()),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)

    if args.features_input:
        feature_metadata, rows = _load_features(Path(args.features_input))
        event_dim = int(feature_metadata["event_dim"])
        action_dim = int(feature_metadata["action_dim"])
    else:
        metadata, policy, preprocessor = _load_base_policy(args)
        episodes = _selected_episodes(metadata, args.episode_start, args.episode_end, args.max_episodes)
        dataset = __import__("lerobot.datasets.lerobot_dataset", fromlist=["LeRobotDataset"]).LeRobotDataset(
            args.dataset_repo_id,
            root=args.dataset_root,
            episodes=episodes,
            delta_timestamps={"action": [0.0]},
            download_videos=args.download_videos,
            video_backend=args.video_backend,
        )
        table = _episode_table(dataset, episodes)
        rows = []
        for episode, local_start, length in table:
            LOGGER.info("Extracting causal prefix events for episode %d (%d frames)", episode, length)
            rows.append(
                _extract_episode_features(
                    dataset,
                    local_start,
                    length,
                    policy,
                    preprocessor,
                    task_override=args.task,
                    device=device,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not rows:
            raise ValueError("No episodes selected")
        event_dim = int(rows[0]["event_tokens"].shape[-1])
        action_dim = int(rows[0]["target_actions"].shape[-1])
        for index, row in enumerate(rows):
            if row["event_tokens"].shape[-1] != event_dim or row["target_actions"].shape[-1] != action_dim:
                raise ValueError(f"Episode {index} has inconsistent feature dimensions")
        if args.features_output:
            _save_features(
                Path(args.features_output),
                rows,
                args=args,
                event_dim=event_dim,
                action_dim=action_dim,
                source_config_sha256=_source_config_sha256(args.base_checkpoint),
                fps=getattr(metadata, "fps", None),
            )

    if args.validation_episode_start is None:
        split = max(1, int(round(len(rows) * 0.8)))
        train_rows, validation_rows = rows[:split], rows[split:]
    else:
        threshold = int(args.validation_episode_start)
        train_rows = [row for row in rows if int(row["episode_indices"][0]) < threshold]
        validation_rows = [row for row in rows if int(row["episode_indices"][0]) >= threshold]
        if not train_rows or not validation_rows:
            raise ValueError("validation_episode_start must leave both train and validation episodes")

    teacher = FullHistoryActionTeacher(
        event_dim=event_dim,
        action_dim=action_dim,
        hidden_dim=int(args.hidden_dim),
        memory_dim=int(args.hidden_dim),
        previous_action_dim=action_dim,
        action_horizon=1,
        include_current=True,
        deletion_mode="skip",
        target_mode="normalized_executed_slot0_action",
    ).to(device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: list[dict[str, float]] = []
    epochs = int(args.epochs)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    for epoch in range(epochs):
        teacher.train()
        train_loss = _run_teacher_epoch(teacher, train_rows, optimizer, device=device)
        teacher.eval()
        validation_loss = _evaluate_teacher(teacher, validation_rows, device=device, short_context=0)
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "validation_loss": validation_loss})
        if epoch == 0 or (epoch + 1) % max(1, int(args.log_every)) == 0 or epoch + 1 == epochs:
            LOGGER.info("teacher epoch %d/%d: train=%.6f validation=%.6f", epoch + 1, epochs, train_loss, validation_loss)
        if args.max_steps and epoch + 1 >= int(args.max_steps):
            break

    teacher.eval()
    full_loss = _evaluate_teacher(teacher, validation_rows, device=device, short_context=0)
    short_loss = _evaluate_teacher(teacher, validation_rows, device=device, short_context=int(args.short_context))
    improvement = (short_loss - full_loss) / max(short_loss, 1e-8)
    audit = _history_swap_audit(teacher, validation_rows, device=device)
    output_path = Path(args.output)
    manifest = teacher.save_checkpoint(
        output_path,
        metadata={
            "dataset_id": args.dataset_repo_id,
            "base_checkpoint": str(args.base_checkpoint),
            "base_config_sha256": _source_config_sha256(args.base_checkpoint),
            "feature_format": FEATURE_FORMAT,
            "train_episode_count": len(train_rows),
            "validation_episode_count": len(validation_rows),
            "seed": int(args.seed),
            "epochs": len(history),
            "history": history,
            "full_history_validation_loss": full_loss,
            "short_context_validation_loss": short_loss,
            "full_vs_short_relative_loss_improvement": improvement,
            **audit,
        },
    )
    report = {
        "format": FULL_HISTORY_TEACHER_FORMAT,
        "teacher_checkpoint": str(output_path),
        "teacher_parameter_sha256": history_teacher_state_sha256(teacher),
        "manifest": manifest,
        "audit": {
            "full_history_validation_loss": full_loss,
            "short_context_validation_loss": short_loss,
            "full_vs_short_relative_loss_improvement": improvement,
            **audit,
        },
    }
    report_path = output_path.with_name(output_path.stem + ".audit.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Teacher checkpoint: %s", output_path)
    LOGGER.info("Audit: full=%.6f short=%.6f relative_improvement=%.3f", full_loss, short_loss, improvement)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-repo-id", required=False, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--base-checkpoint", required=False, default=None)
    parser.add_argument("--features-input", type=Path, default=None, help="Reuse a prefix feature cache")
    parser.add_argument("--features-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--validation-episode-start", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional epoch cap for a smoke run")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--short-context", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec", "video_reader"), default="pyav")
    parser.add_argument("--download-videos", action="store_true")
    args = parser.parse_args()
    if args.features_input is None:
        missing = [name for name in ("dataset_repo_id", "dataset_root", "base_checkpoint") if getattr(args, name) is None]
        if missing:
            parser.error("feature extraction requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.short_context < 0:
        parser.error("--short-context must be non-negative")
    if args.hidden_dim <= 0 or args.lr <= 0:
        parser.error("--hidden-dim and --lr must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    report = train(_parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
