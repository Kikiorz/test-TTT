#!/usr/bin/env python
"""Build offline Hindsight-Distilled TTT labels for a LeRobot episode shard.

The teacher is the frozen SmolVLA-TTT model.  A deterministic noise and time
sample is used for every frame, so a label shard can be regenerated or merged
without changing the intervention experiment.  For an event block ``E_i`` we
replay the *same* observation/action/noise/time sequence while setting the TTT
write gate to zero on ``E_i``.  Only futures after the block are attributed:

``C[i,j] = [L(v_reset(j), u(j)) - L(v_full(j), u(j))]_+`` for ``j >= end(E_i)``.

The training-facing columns are one row per source frame:

* ``hd_teacher_velocity`` / ``hd_teacher_true_velocity``: full-history flow
  velocity, padded to the model's ``max_action_dim``;
* ``hd_teacher_wrong_velocity``: velocity from the highest-credit event
  intervention in the episode;
* ``hd_noise`` / ``hd_time``: the fixed flow-matching noise and timestep used
  by every full/reset replay;
* ``hd_attribution`` and ``hd_rho``: normalized future dependency ``rho[j]``;
* ``hd_write_gate``: normalized event importance ``u[i]``;
* ``hd_write_gate_observed``: one where the event was actually replayed (used
  to avoid treating capped-event defaults as supervision);
* ``hd_counterfactual_write_gate``: the selected event's causal zero-write
  mask (one value per frame);
* ``global_index`` / ``episode_index`` / ``frame_index``: source indexing.

The complete ``C[i,j]`` matrices and event metadata are retained under the
top-level ``metadata`` key for auditing, but are intentionally not presented
as frame labels.  This keeps the artifact compatible with
``HindsightLabelDataset`` and the normal sequence collator.

Examples (run in the Python 3.11 MIKASA environment)::

    python examples/mikasa/build_hd_labels.py \
      --dataset-repo-id shell_game_color_lamp_touch_vla_v0 \
      --dataset-root /workspace/data_mikasa_robo/data_lerobot/\
        shell_game_color_lamp_touch_vla_v0 \
      --checkpoint /workspace/experiments/short_ttt150_clean/checkpoints/016375/pretrained_model \
      --output /workspace/labels/color-000.pt \
      --episode-start 0 --episode-end 50 --max-events 8

    # Merge independently generated shards.  Inputs are sorted by
    # global_index and duplicate indices are rejected.
    python examples/mikasa/build_hd_labels.py \
      --merge /workspace/labels/color-*.pt \
      --output /workspace/labels/color-all.pt

This script never writes to ``Method1_lerobot-pi0-ttt`` or to the source
dataset.  It only reads the dataset and writes the requested label artifact.
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


LOGGER = logging.getLogger("build_hd_labels")


def _validate_teacher_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Require a trained SmolVLA-TTT teacher rather than random TTT weights."""

    checkpoint_text = str(checkpoint)
    checkpoint_path = Path(checkpoint_text).expanduser()
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / "config.json"
    else:
        config_path = checkpoint_path if checkpoint_path.name == "config.json" else None
    if config_path is None or not config_path.is_file():
        try:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(repo_id=checkpoint_text, filename="config.json"))
        except Exception as error:
            raise ValueError(
                "Could not resolve teacher config.json for "
                f"{checkpoint_text!r}: {error}"
            ) from error
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read teacher config {config_path}: {error}") from error
    if raw.get("type") != "smolvla_ttt":
        raise ValueError(
            "HD hindsight labels require a trained clean/HD SmolVLA-TTT teacher "
            f"(config.type='smolvla_ttt'), got {raw.get('type')!r} from {checkpoint_text!r}. "
            "Train clean-TTT first; a standard SmolVLA checkpoint would leave "
            "the TTT/register weights randomly initialized."
        )
    if "ttt_layer_indices" not in raw or "ttt_num_register_tokens" not in raw:
        raise ValueError(
            f"Teacher config {config_path} lacks TTT architecture fields; refusing an untrained teacher"
        )
    config_bytes = config_path.read_bytes()
    layer_indices = raw.get("ttt_layer_indices")
    if layer_indices is None:
        # ``None`` is the valid config representation for the contiguous
        # default range; resolve it exactly as ``SmolVLATTTConfig`` does.
        try:
            layer_indices = list(
                range(int(raw.get("ttt_start_layer", 12)), int(raw.get("num_vlm_layers", 16)))
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Teacher config {config_path} cannot resolve default TTT layer range"
            ) from error
    if isinstance(layer_indices, tuple):
        layer_indices = list(layer_indices)
    if not isinstance(layer_indices, list):
        raise ValueError(
            f"Teacher config {config_path} has non-list ttt_layer_indices={layer_indices!r}"
        )
    try:
        layer_indices = [int(index) for index in layer_indices]
        register_tokens = int(raw["ttt_num_register_tokens"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Teacher config {config_path} has invalid TTT architecture fields"
        ) from error
    if not layer_indices or register_tokens < 0:
        raise ValueError(
            f"Teacher config {config_path} has empty/invalid TTT architecture fields"
        )
    return {
        "policy_type": raw.get("type"),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "ttt_layer_indices": layer_indices,
        "ttt_num_register_tokens": register_tokens,
        "config_path": str(config_path),
    }


def _load_torch(path: Path) -> Any:
    """Load a tensor-only artifact on both old and new PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def _as_int(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.detach().cpu().reshape(()).item())
    return int(value)


def _episode_table(dataset: Any, selected: Sequence[int]) -> list[tuple[int, int, int]]:
    """Return ``(episode_id, local_start, length)`` for a selected dataset."""

    # ``LeRobotDataset`` exposes metadata through ``dataset.meta`` while
    # ``LeRobotDatasetMetadata`` (used before constructing the selected view)
    # exposes the same table directly.  Accept both so shard selection does
    # not depend on which object the caller already has in memory.
    metadata = getattr(dataset, "meta", dataset)
    episodes = metadata.episodes
    if isinstance(episodes, Mapping):
        starts = episodes["dataset_from_index"]
        ends = episodes["dataset_to_index"]
    else:
        starts = [row["dataset_from_index"] for row in episodes]
        ends = [row["dataset_to_index"] for row in episodes]

    table: list[tuple[int, int, int]] = []
    local_start = 0
    for episode in selected:
        length = _as_int(ends[episode]) - _as_int(starts[episode])
        if length <= 0:
            raise ValueError(f"Episode {episode} has non-positive length {length}")
        table.append((episode, local_start, length))
        local_start += length
    if local_start != len(dataset):
        raise ValueError(
            f"Selected episode metadata covers {local_start} frames, dataset has {len(dataset)}"
        )
    return table


def _selected_episodes(dataset: Any, start: int, end: int | None) -> list[int]:
    metadata = getattr(dataset, "meta", dataset)
    total = int(metadata.total_episodes)
    if start < 0 or start >= total:
        raise ValueError(f"episode-start must be in [0, {total}), got {start}")
    stop = total if end is None else end
    if stop <= start or stop > total:
        raise ValueError(f"episode-end must satisfy {start} < end <= {total}, got {stop}")
    return list(range(start, stop))


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
    """Concatenate processor outputs while retaining only tensor model inputs."""

    if not processed_rows:
        raise ValueError("Cannot concatenate an empty episode")
    keys = set(processed_rows[0])
    for row in processed_rows[1:]:
        keys.intersection_update(row)
    result: dict[str, Any] = {}
    for key in sorted(keys):
        values = [row[key] for row in processed_rows]
        if not all(isinstance(value, Tensor) for value in values):
            # Language text/complementary fields are not consumed after
            # tokenization.  Ignore them rather than creating an object batch.
            continue
        result[key] = torch.stack(values, dim=0)
    return result


def _clone_state_dict(states: Mapping[int, Any] | None, *, detach: bool = True):
    if states is None:
        return None
    cloned = {}
    for layer_index, state in states.items():
        # ``TTTFastState.clone`` is intentionally used instead of a shallow
        # copy: intervention replays must never mutate the full-history branch.
        cloned[layer_index] = state.clone(detach=detach, requires_grad=False)
    return cloned


def _detach_states(states: Mapping[int, Any]) -> dict[int, Any]:
    return {layer_index: state.detach(requires_grad=True) for layer_index, state in states.items()}


def _prepare_episode(
    dataset: Any,
    local_start: int,
    length: int,
    policy: Any,
    preprocessor: Any,
    *,
    task_override: str | None,
    device: torch.device,
) -> tuple[dict[str, Tensor], list[int], list[int], list[int]]:
    """Decode and preprocess one episode into model-ready tensors."""

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

        from lerobot.utils.constants import ACTION, OBS_STATE

        raw[OBS_STATE] = sample[OBS_STATE].to(dtype=torch.float32)
        raw[ACTION] = sample[ACTION].to(dtype=torch.float32)
        if "action_is_pad" in sample:
            raw["action_is_pad"] = sample["action_is_pad"].bool()

        processed = dict(preprocessor(raw))
        # Tokenizing an identical MIKASA instruction hundreds of times is
        # needlessly expensive.  Reuse the already-tokenized tensors, while
        # leaving image/state/action normalization per-frame.
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

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

    batch = _concat_processed(rows)
    # ``_concat_processed`` creates CPU tensors.  Keep one contiguous transfer
    # per model input rather than moving every decoded frame independently.
    batch = {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }
    return batch, global_indices, episode_indices, frame_indices


def _fixed_noise_time(
    *,
    length: int,
    chunk_size: int,
    action_dim: int,
    seed: int,
    phase_mode: str = "random",
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Generate reproducible flow-matching noise/time on CPU then transfer.

    ``deployment`` is deliberately phase-matched to the first online denoise:
    the writer sees a pure Gaussian action chunk at ``t=1``.  The ordinary
    ``random`` mode is retained for base flow-matching diagnostics, but it
    mixes the future expert action chunk into the writer and must not be used
    for a strict deployment-causal HD claim.
    """

    if phase_mode not in {"random", "deployment"}:
        raise ValueError("phase_mode must be 'random' or 'deployment'")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    noise = torch.randn(
        (length, chunk_size, action_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    if phase_mode == "deployment":
        time = torch.ones((length,), dtype=torch.float32, device="cpu")
    else:
        # Keep away from both interpolation endpoints while remaining fixed.
        time = torch.rand((length,), generator=generator, dtype=torch.float32, device="cpu")
        time = time.mul(0.998).add(0.001)
    return noise.to(device=device), time.to(device=device)


def _run_replay(
    policy: Any,
    prepared: dict[str, Tensor],
    noise: Tensor,
    time: Tensor,
    *,
    frame_batch_size: int,
    write_gate: Tensor | None = None,
) -> Tensor:
    """Replay one episode in small contiguous chunks while carrying fast state."""

    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    model = policy.model
    images, img_masks = policy.prepare_images(prepared)
    state = policy.prepare_state(prepared)
    actions = policy.prepare_action(prepared)
    language_tokens = prepared[OBS_LANGUAGE_TOKENS]
    language_masks = prepared[OBS_LANGUAGE_ATTENTION_MASK]
    length = int(actions.shape[0])
    if noise.shape[:2] != actions.shape[:2]:
        raise ValueError(
            f"Fixed noise shape {tuple(noise.shape)} does not match actions {tuple(actions.shape)}"
        )
    if time.shape[0] != length:
        raise ValueError(f"Fixed time has {time.shape[0]} rows, expected {length}")

    states = None
    outputs: list[Tensor] = []
    for start in range(0, length, frame_batch_size):
        end = min(start + frame_batch_size, length)
        chunk_images = [image[start:end] for image in images]
        chunk_masks = [mask[start:end] for mask in img_masks]
        chunk_state = state[start:end]
        chunk_tokens = language_tokens[start:end]
        chunk_language_masks = language_masks[start:end]
        chunk_actions = actions[start:end]
        chunk_noise = noise[start:end]
        chunk_time = time[start:end]
        chunk_gate = None if write_gate is None else write_gate[start:end].unsqueeze(0)

        # TTTMLPLayer temporarily enables gradients internally only for its
        # local fast-weight update.  The frozen teacher replay itself must not
        # retain an outer graph across chunks/events.
        with torch.no_grad():
            velocity, states = model.forward_with_state(
                chunk_images,
                chunk_masks,
                chunk_tokens,
                chunk_language_masks,
                chunk_state,
                chunk_actions,
                chunk_noise,
                chunk_time,
                sequence_shape=(1, end - start),
                fast_states=states,
                create_graph=False,
                write_gate=chunk_gate,
                return_velocity=True,
            )
        expected_shape = (end - start, int(policy.config.chunk_size), int(policy.config.max_action_dim))
        if tuple(velocity.shape) != expected_shape:
            raise ValueError(
                "Teacher replay returned an unexpected velocity shape: "
                f"got {tuple(velocity.shape)}, expected {expected_shape}"
            )
        outputs.append(velocity.detach().cpu())
        states = _detach_states(states)
    return torch.cat(outputs, dim=0)


def _flow_losses(
    velocity: Tensor,
    noise: Tensor,
    actions: Tensor,
    action_is_pad: Tensor | None,
    active_dim: int,
) -> Tensor:
    """Per-frame flow loss used to form the positive HCA credit."""

    target = noise[..., :active_dim].cpu() - actions[..., :active_dim].cpu()
    # Keep the action-chunk axis until padding is applied.  Near an episode
    # boundary LeRobot repeats the final action values but marks those future
    # positions in ``action_is_pad``; counting them would create artificial
    # hindsight credit.
    error = (velocity[..., :active_dim] - target).square().mean(dim=-1)
    if action_is_pad is not None:
        valid = (~action_is_pad.bool()).to(dtype=error.dtype)
        error = (error * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
    else:
        error = error.mean(dim=-1)
    return error


def _event_candidates(
    length: int,
    block_size: int,
    max_events: int,
    *,
    global_offset: int = 0,
    global_start_min: int | None = None,
    global_end_max: int | None = None,
) -> list[tuple[int, int]]:
    if block_size <= 0:
        raise ValueError("event-block-size must be positive")
    if max_events < 0:
        raise ValueError("max-events must be non-negative")
    events = []
    # Start at the first local position whose *episode-global* index is on the
    # event grid.  Using ``range(0, ...)`` and then filtering would silently
    # drop every event when a replay context starts off-grid.
    first_start = (-global_offset) % block_size
    for start in range(first_start, length, block_size):
        global_start = global_offset + start
        global_end = global_start + block_size
        # Keep intervention blocks on the episode-global event grid.  A local
        # replay must not silently redefine an event merely because its
        # warm-up context starts in the middle of a block.
        if global_start % block_size != 0:
            continue
        end = min(start + block_size, length)
        if global_end > global_offset + length:
            continue
        if global_start_min is not None and global_start < global_start_min:
            continue
        if global_end_max is not None and global_end > global_end_max:
            continue
        events.append((start, end))
    # Events with no future cannot receive causal credit and need no replay.
    events = [event for event in events if event[1] < length]
    # ``max_events=0`` means no sampling cap: generate every causal block.
    # This is the safe default for a paper label pass.  A positive value is an
    # explicit compute-budget trade-off.
    if max_events == 0 or len(events) <= max_events:
        return events
    # Uniform coverage prevents a shard's first few events from monopolizing
    # long MIKASA tracking episodes while keeping the replay budget bounded.
    positions = torch.linspace(0, len(events) - 1, max_events).round().to(torch.long).tolist()
    return [events[int(position)] for position in sorted(set(positions))]


def _episode_labels(
    policy: Any,
    prepared: dict[str, Tensor],
    noise: Tensor,
    time: Tensor,
    *,
    event_block_size: int,
    max_events: int,
    attribution_threshold: float,
    frame_batch_size: int,
    future_mask: Tensor | None = None,
    global_offset: int = 0,
    event_global_start_min: int | None = None,
    event_global_end_max: int | None = None,
) -> dict[str, Any]:
    """Compute full teacher, causal interventions, and frame-aligned labels."""

    actions = policy.prepare_action(prepared).detach().cpu()
    action_is_pad = prepared.get("action_is_pad")
    if action_is_pad is not None:
        action_is_pad = action_is_pad.detach().cpu()
    active_dim = int(math.prod(policy.config.action_feature.shape))

    full_velocity = _run_replay(
        policy,
        prepared,
        noise,
        time,
        frame_batch_size=frame_batch_size,
    )
    full_loss = _flow_losses(full_velocity, noise, actions, action_is_pad, active_dim)
    length = int(full_velocity.shape[0])
    if future_mask is not None:
        future_mask = future_mask.detach().cpu().bool()
        if future_mask.shape != (length,):
            raise ValueError(f"future_mask must have shape [{length}], got {tuple(future_mask.shape)}")
    events = _event_candidates(
        length,
        event_block_size,
        max_events,
        global_offset=global_offset,
        global_start_min=event_global_start_min,
        global_end_max=event_global_end_max,
    )
    credit_rows: list[Tensor] = []
    best_wrong: Tensor | None = None
    best_gate = torch.ones(length, dtype=torch.float32)
    best_score = float("-inf")
    best_event: tuple[int, int] | None = None
    best_event_index = -1
    selection_scores: list[float] = []

    for event_index, (event_start, event_end) in enumerate(events):
        gate = torch.ones(length, dtype=torch.float32, device=noise.device)
        gate[event_start:event_end] = 0.0
        wrong_velocity = _run_replay(
            policy,
            prepared,
            noise,
            time,
            frame_batch_size=frame_batch_size,
            write_gate=gate,
        )
        wrong_loss = _flow_losses(wrong_velocity, noise, actions, action_is_pad, active_dim)
        row = (wrong_loss - full_loss).clamp_min(0.0)
        causal = torch.zeros(length, dtype=torch.bool)
        causal[event_end:] = True
        if future_mask is not None:
            causal &= future_mask
        row = torch.where(causal, row, torch.zeros_like(row))
        if attribution_threshold > 0:
            row = torch.where(row >= attribution_threshold, row, torch.zeros_like(row))
        credit_rows.append(row)

        # Normalize by the number of eligible future frames.  Otherwise an
        # early event wins simply because it has a longer future horizon,
        # even when its per-frame intervention effect is identical.
        eligible_count = int(causal.sum().item())
        score = float(row.sum().item()) / max(eligible_count, 1) if eligible_count else float("-inf")
        selection_scores.append(score)
        positive_credit = float(row.sum().item()) > 0.0
        if eligible_count > 0 and positive_credit and (best_wrong is None or score > best_score):
            best_score = score
            best_wrong = wrong_velocity
            best_gate = gate.detach().cpu()
            best_event = (event_start, event_end)
            best_event_index = event_index

    if credit_rows:
        credits = torch.stack(credit_rows, dim=0)
        # ``hd_attribution`` retains all-event max credit for HCA/write
        # utility, while ``hd_rho`` is restricted to the selected branch used
        # by the stored counterfactual velocity.  Keeping these distinct avoids
        # mixing a per-future event with an episode-global wrong branch in the
        # grounding direction target.
        rho_hca_raw = credits.amax(dim=0)
        rho_grounding_raw = (
            credits[best_event_index]
            if 0 <= best_event_index < credits.shape[0]
            else torch.zeros(length, dtype=credits.dtype)
        )
        event_u_raw = credits.amax(dim=1)
    else:
        credits = torch.zeros((0, length), dtype=torch.float32)
        rho_hca_raw = torch.zeros(length, dtype=torch.float32)
        rho_grounding_raw = torch.zeros(length, dtype=torch.float32)
        event_u_raw = torch.zeros(0, dtype=torch.float32)

    # Normalize independently per episode.  A no-dependency episode remains
    # exactly zero instead of producing NaNs or an arbitrary write signal.
    rho_hca_scale = rho_hca_raw.max().clamp_min(1e-8)
    rho_grounding_scale = rho_grounding_raw.max().clamp_min(1e-8)
    rho_hca = rho_hca_raw / rho_hca_scale if float(rho_hca_raw.max()) > 0 else rho_hca_raw
    rho_grounding = (
        rho_grounding_raw / rho_grounding_scale
        if float(rho_grounding_raw.max()) > 0
        else rho_grounding_raw
    )
    if event_u_raw.numel() and float(event_u_raw.max()) > 0:
        event_u = event_u_raw / event_u_raw.max().clamp_min(1e-8)
    else:
        event_u = event_u_raw
    # Unobserved event blocks (when ``max_events`` is a positive sampling cap)
    # retain the ordinary writer rather than being silently trained as
    # permanent skips.  Every block is therefore either assigned its measured
    # ``u_i`` or receives the safe default gate 1.0.  Keep a separate observed
    # mask so the learned gate is *not* trained on that default; otherwise a
    # capped long-horizon pass would turn missing credit into a spurious
    # all-positive target.
    write_gate = torch.ones(length, dtype=torch.float32)
    write_gate_observed = torch.zeros(length, dtype=torch.float32)
    for event_index, (event_start, event_end) in enumerate(events):
        write_gate[event_start:event_end] = event_u[event_index]
        write_gate_observed[event_start:event_end] = 1.0

    if best_wrong is None:
        best_wrong = full_velocity.clone()
        best_gate = torch.ones(length, dtype=torch.float32)

    return {
        "hd_teacher_velocity": full_velocity.float(),
        "hd_teacher_true_velocity": full_velocity.float().clone(),
        "hd_teacher_wrong_velocity": best_wrong.float(),
        "hd_noise": noise.detach().cpu().float(),
        "hd_time": time.detach().cpu().float(),
        "hd_attribution": rho_hca.float(),
        "hd_rho": rho_grounding.float(),
        "hd_write_gate": write_gate.float(),
        "hd_write_gate_observed": write_gate_observed.float(),
        "hd_counterfactual_write_gate": best_gate.float(),
        "hd_C": credits.float(),
        "hd_event_u": event_u.float(),
        "hd_attribution_aggregation": "all_event_max_for_hca_selected_event_for_grounding",
        "hd_event_selection_scores": torch.tensor(selection_scores, dtype=torch.float32),
        "hd_event_starts": torch.tensor([event[0] for event in events], dtype=torch.int64),
        "hd_event_ends": torch.tensor([event[1] for event in events], dtype=torch.int64),
        "hd_selected_event": torch.tensor(
            [-1 if best_event is None else best_event[0], -1 if best_event is None else best_event[1]],
            dtype=torch.int64,
        ),
        "hd_full_flow_loss": full_loss.float(),
    }


def _merge_shards(inputs: Sequence[Path], output: Path) -> None:
    if not inputs:
        raise ValueError("--merge requires at least one input shard")
    payloads = [_load_torch(path) for path in inputs]
    required = {
        "global_index",
        "episode_index",
        "frame_index",
        "hd_teacher_velocity",
        "hd_teacher_true_velocity",
        "hd_teacher_wrong_velocity",
        "hd_noise",
        "hd_time",
        "hd_attribution",
        "hd_rho",
        "hd_write_gate",
        "hd_counterfactual_write_gate",
    }
    for path, payload in zip(inputs, payloads, strict=True):
        if not isinstance(payload, Mapping) or not required.issubset(payload):
            raise ValueError(f"Shard {path} is missing one or more required columns")
    observed_available = [
        isinstance(payload, Mapping) and "hd_write_gate_observed" in payload for payload in payloads
    ]
    if any(observed_available) and not all(observed_available):
        raise ValueError("All merged shards must either contain hd_write_gate_observed or omit it")

    # Keep the full per-shard audit trail.  In particular, generation shards
    # store the causal ``C`` matrices and event scores under
    # ``metadata.episodes_detail``; dropping those fields during merge would
    # make the merged artifact impossible to inspect or reproduce.  Metadata
    # is copied as plain dictionaries so the output remains torch-loadable and
    # independent of the input payload objects.
    shard_metadata: list[dict[str, Any]] = []
    episodes_detail: dict[str, Any] = {}
    merged_episode_ids: set[int] = set()
    metadata_contract_keys = (
        "dataset_repo_id",
        "dataset_root",
        "checkpoint",
        "teacher_checkpoint",
        "teacher_policy_type",
        "teacher_config_sha256",
        "teacher_ttt_layer_indices",
        "teacher_ttt_num_register_tokens",
        "seed",
        "phase_mode",
        "history_mode",
        "event_block_size",
        "max_events",
        "attribution_threshold",
        "action_chunk_size",
        "max_action_dim",
    )
    reference_metadata: dict[str, Any] | None = None
    for path, payload in zip(inputs, payloads, strict=True):
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        copied = dict(metadata)
        copied["source_path"] = str(path)
        shard_metadata.append(copied)
        if reference_metadata is None:
            reference_metadata = dict(metadata)
        else:
            missing_contract_fields = {
                key
                for key in metadata_contract_keys
                if (key in reference_metadata) != (key in metadata)
            }
            if missing_contract_fields:
                raise ValueError(
                    "Cannot merge hindsight shards with incomplete metadata contract: "
                    f"{sorted(missing_contract_fields)}"
                )
            mismatches = {
                key: (reference_metadata.get(key), metadata.get(key))
                for key in metadata_contract_keys
                if key in reference_metadata and key in metadata and reference_metadata[key] != metadata[key]
            }
            if mismatches:
                raise ValueError(f"Cannot merge incompatible hindsight shards: {mismatches}")
        declared_episodes = metadata.get("episodes")
        if isinstance(declared_episodes, Sequence) and not isinstance(
            declared_episodes, (str, bytes, bytearray)
        ):
            merged_episode_ids.update(int(episode) for episode in declared_episodes)
        details = metadata.get("episodes_detail")
        if isinstance(details, Mapping):
            for episode_key, episode_detail in details.items():
                key = str(episode_key)
                if key in episodes_detail:
                    raise ValueError(
                        "Duplicate episode metadata while merging shards: "
                        f"episode {key!r} appears in more than one shard"
                    )
                episodes_detail[key] = episode_detail

    phase_modes = {
        str(metadata.get("phase_mode"))
        for metadata in shard_metadata
        if metadata.get("phase_mode") is not None
    }
    if len(phase_modes) > 1:
        raise ValueError(
            "Cannot merge hindsight shards generated with different phase modes: "
            f"{sorted(phase_modes)}"
        )
    merged_phase_mode = next(iter(phase_modes), "unknown")

    columns = {
        key: torch.cat([payload[key].detach().cpu() for payload in payloads], dim=0)
        for key in required
    }
    if all(observed_available):
        columns["hd_write_gate_observed"] = torch.cat(
            [payload["hd_write_gate_observed"].detach().cpu() for payload in payloads], dim=0
        )
    else:
        # Legacy artifacts predate the observed mask and were generated with
        # all event blocks measured.  Treat their available gate values as
        # observed rather than silently dropping gate supervision.
        columns["hd_write_gate_observed"] = torch.ones_like(columns["hd_write_gate"])
    global_index = columns["global_index"].to(torch.long)
    order = torch.argsort(global_index)
    sorted_indices = global_index[order]
    if sorted_indices.numel() > 1 and bool((sorted_indices[1:] == sorted_indices[:-1]).any()):
        duplicates = sorted_indices[1:][sorted_indices[1:] == sorted_indices[:-1]].tolist()
        raise ValueError(f"Duplicate global indices while merging shards: {duplicates[:8]}")
    merged = {key: value.index_select(0, order) for key, value in columns.items()}
    merged_metadata = {
        "format": "hd_ttt_labels_v1",
        "merged_from": [str(path) for path in inputs],
        "num_frames": int(sorted_indices.numel()),
        "shard_metadata": shard_metadata,
        "episodes_detail": episodes_detail,
        "phase_mode": merged_phase_mode,
        "fixed_phase": {
            "noise_column": "hd_noise",
            "time_column": "hd_time",
            "noise_shape_per_frame": list(columns["hd_noise"].shape[1:]),
            "time_shape_per_frame": list(columns["hd_time"].shape[1:]),
        },
    }
    if reference_metadata is not None:
        # Carry the common generation contract to the merged artifact.  Do
        # not copy per-shard episode ranges; those are retained in
        # ``shard_metadata`` and the union below.
        for key in metadata_contract_keys:
            if key in reference_metadata:
                merged_metadata[key] = reference_metadata[key]
        merged_metadata["episodes"] = sorted(merged_episode_ids | {int(key) for key in episodes_detail})
    merged["metadata"] = merged_metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    LOGGER.info("Merged %d shards (%d frames) -> %s", len(inputs), sorted_indices.numel(), output)


def _build_shard(args: argparse.Namespace) -> None:
    # Imports are delayed so ``--merge`` works on a lightweight CPU machine
    # without transformers/SAPIEN dependencies.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla_ttt.configuration_smolvla_ttt import SmolVLATTTConfig
    from lerobot.policies.smolvla_ttt.processor_smolvla_ttt import make_smolvla_ttt_pre_post_processors

    device = torch.device(args.device)
    teacher_info = _validate_teacher_checkpoint(args.checkpoint)
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    selected = _selected_episodes(metadata, args.episode_start, args.episode_end)
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]
    fps = int(metadata.fps)
    # SmolVLA's action expert predicts a 50-step chunk.  Loading future action
    # timestamps here also gives us a strict action-padding mask at episode ends.
    action_indices = list(range(args.action_chunk_size))
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=selected,
        delta_timestamps={"action": [index / fps for index in action_indices]},
        download_videos=args.download_videos,
        video_backend=args.video_backend,
    )
    table = _episode_table(dataset, selected)

    # ``make_policy(..., ds_meta=...)`` is important here: it injects the
    # dataset's top/wrist cameras and 7-D action feature into the config before
    # loading the generic SmolVLA checkpoint.  Direct ``from_pretrained`` would
    # otherwise retain an empty or unrelated checkpoint feature schema and can
    # silently omit one of the MIKASA views.
    config = SmolVLATTTConfig(
        device=args.device,
        pretrained_path=Path(args.checkpoint),
        ttt_training_stage="ttt_only",
    )
    policy = make_policy(config, ds_meta=metadata)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    preprocessor, _ = make_smolvla_ttt_pre_post_processors(policy.config, dataset_stats=metadata.stats)

    # The source checkpoint owns chunk/max-action dimensions.  Refuse a silent
    # mismatch: changing the action horizon would invalidate fixed labels.
    if int(policy.config.chunk_size) != args.action_chunk_size:
        raise ValueError(
            f"Checkpoint chunk_size={policy.config.chunk_size} but --action-chunk-size={args.action_chunk_size}; "
            "generate labels with the checkpoint's native horizon"
        )
    action_dim = int(policy.config.max_action_dim)
    columns: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "global_index",
            "episode_index",
            "frame_index",
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
    }
    episode_metadata: dict[str, Any] = {}

    for episode, local_start, length in table:
        if args.max_frames is not None:
            length = min(length, args.max_frames)
        LOGGER.info("Episode %d: %d frames (local start %d)", episode, length, local_start)
        prepared, global_indices, episode_indices, frame_indices = _prepare_episode(
            dataset,
            local_start,
            length,
            policy,
            preprocessor,
            task_override=args.task,
            device=device,
        )
        noise, time = _fixed_noise_time(
            length=length,
            chunk_size=int(policy.config.chunk_size),
            action_dim=action_dim,
            seed=args.seed + episode * 1_000_003,
            phase_mode=args.phase_mode,
            device=device,
        )
        labels = _episode_labels(
            policy,
            prepared,
            noise,
            time,
            event_block_size=args.event_block_size,
            max_events=args.max_events,
            attribution_threshold=args.attribution_threshold,
            frame_batch_size=args.frame_batch_size,
        )
        for key in columns:
            if key == "global_index":
                columns[key].append(torch.tensor(global_indices, dtype=torch.int64))
            elif key == "episode_index":
                columns[key].append(torch.tensor(episode_indices, dtype=torch.int64))
            elif key == "frame_index":
                columns[key].append(torch.tensor(frame_indices, dtype=torch.int64))
            else:
                columns[key].append(labels[key].detach().cpu())
        episode_metadata[str(episode)] = {
            "length": length,
            "global_indices": [int(value) for value in global_indices],
            "noise_seed": int(args.seed + episode * 1_000_003),
            "event_starts": labels["hd_event_starts"].tolist(),
            "event_ends": labels["hd_event_ends"].tolist(),
            "selected_event": labels["hd_selected_event"].tolist(),
            "C": labels["hd_C"].tolist(),
            "event_u": labels["hd_event_u"].tolist(),
            "full_flow_loss": labels["hd_full_flow_loss"].tolist(),
        }
        del prepared, labels, noise, time
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: torch.cat(values, dim=0) for key, values in columns.items()}
    payload["metadata"] = {
        "format": "hd_ttt_labels_v1",
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "teacher_checkpoint": str(args.checkpoint),
        "teacher_policy_type": teacher_info["policy_type"],
        "teacher_config_sha256": teacher_info["config_sha256"],
        "teacher_ttt_layer_indices": list(teacher_info["ttt_layer_indices"]),
        "teacher_ttt_num_register_tokens": int(teacher_info["ttt_num_register_tokens"]),
        "fps": fps,
        "episode_start": args.episode_start,
        "episode_end": args.episode_end,
        "episodes": selected,
        "seed": args.seed,
        "phase_mode": args.phase_mode,
        "history_mode": "full_episode_replay",
        "writer_observation": (
            "pure_gaussian_action_noise_at_t1"
            if args.phase_mode == "deployment"
            else "random_flow_interpolation_with_expert_action_chunk"
        ),
        "event_block_size": args.event_block_size,
        "max_events": args.max_events,
        "event_sampling": "all_causal_blocks" if args.max_events == 0 else "uniform",
        "unsampled_write_gate_default": 1.0,
        "write_gate_observed_column": "hd_write_gate_observed",
        "attribution_threshold": args.attribution_threshold,
        "frame_batch_size": args.frame_batch_size,
        "action_chunk_size": int(policy.config.chunk_size),
        "max_action_dim": action_dim,
        "fixed_phase": {
            "noise_column": "hd_noise",
            "time_column": "hd_time",
            "noise_shape_per_frame": [int(policy.config.chunk_size), action_dim],
            "time_shape_per_frame": [],
        },
        "episodes_detail": episode_metadata,
    }
    torch.save(payload, output)
    LOGGER.info("Wrote %d frame labels -> %s", int(payload["global_index"].numel()), output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merge", nargs="*", type=Path, default=None, help="Merge existing .pt shards")
    parser.add_argument("--output", type=Path, required=True, help="Output label artifact")
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--checkpoint", default=None, help="SmolVLA/SmolVLA-TTT checkpoint or Hub id")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=None, help="Exclusive episode end")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Debug cap per episode")
    parser.add_argument("--action-chunk-size", type=int, default=50)
    parser.add_argument("--event-block-size", type=int, default=4)
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Maximum causal event replays; 0 (default) evaluates every event block",
    )
    parser.add_argument("--attribution-threshold", type=float, default=0.0)
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--phase-mode",
        choices=("random", "deployment"),
        default="random",
        help=(
            "Flow phase used for hindsight replay. 'deployment' uses t=1 and "
            "pure Gaussian action noise so the writer cannot see future expert actions."
        ),
    )
    parser.add_argument("--task", default=None, help="Override dataset language instruction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec", "video_reader"))
    parser.add_argument("--download-videos", action="store_true")
    args = parser.parse_args()
    if args.merge is not None and len(args.merge) == 0:
        parser.error("--merge needs one or more input shards")
    if args.merge is None:
        missing = [name for name in ("dataset_repo_id", "dataset_root", "checkpoint") if getattr(args, name) is None]
        if missing:
            parser.error("generation requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
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
