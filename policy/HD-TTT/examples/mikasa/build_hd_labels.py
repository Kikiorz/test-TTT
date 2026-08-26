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

Grounding stores one selected wrong-memory replay, so its ``hd_rho`` remains
aligned with that same branch.  By default the selector only considers events
with at least 64 eligible future frames and compares their mean causal credit;
if a short episode/window has no such event it falls back to the positive event
with the largest total credit.  The rule and threshold are recorded in
metadata and can be changed with ``--grounding-min-future-frames``.

Examples (run in the Python 3.11 MIKASA environment)::

    python examples/mikasa/build_hd_labels.py \
      --dataset-repo-id shell_game_color_lamp_touch_vla_v0 \
      --dataset-root /workspace/data_mikasa_robo/data_lerobot/\
        shell_game_color_lamp_touch_vla_v0 \
      --checkpoint /workspace/experiments/short_ttt150_clean/checkpoints/016375/pretrained_model \
      --output /workspace/labels/color-000.pt \
      --episode-start 0 --episode-end 50 --max-events 0 \
      --grounding-min-future-frames 64

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

# ``legacy`` reproduces the original positive raw-loss/max aggregation.  The
# v2 protocol is the paper path: antithetic common-random-number replay,
# symmetric relative degradation, adaptive top-k aggregation, and robust
# percentile normalization.  Keeping the protocol string in every artifact
# prevents an ablation from being mistaken for the main method.
HD_ATTRIBUTION_PROTOCOL_LEGACY = "legacy_raw_hinge_max"
HD_ATTRIBUTION_PROTOCOL_V2 = "v2_relative_antithetic_robust"
HD_ATTRIBUTION_PROTOCOLS = {
    HD_ATTRIBUTION_PROTOCOL_LEGACY,
    HD_ATTRIBUTION_PROTOCOL_V2,
}
# The v2 paper path stores one selected event branch.  Antithetic replay is
# still used for attribution (the two signs above), but action-effect training
# currently consumes only the selected event.  Older artifacts may contain
# K>1 branches; readers intentionally consume branch zero for compatibility.
V2_ANTITHETIC_REPLAYS = 2
V2_EFFECT_BRANCHES = 1
# The content/effect target intentionally stays on the deployment-matched
# ``+noise`` replay; antithetic ``-noise`` is attribution-only.
V2_EFFECT_TARGET = "plus_noise_full_minus_wrong"

# Grounding keeps one counterfactual branch so the stored wrong velocity and
# ``hd_rho`` always describe the same intervention.  Prefer an event with a
# sufficiently long causal future and compare its mean credit; when an
# episode/window is shorter than that horizon, fall back to total credit so a
# one-frame terminal event cannot win solely because its denominator is one.
GROUNDING_EVENT_POLICY = "min_future_horizon_mean_else_total_credit"


def _validate_teacher_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Require a trained *clean* SmolVLA-TTT teacher.

    The offline replay below explicitly controls the ordinary TTT write gate
    and does not call the learned HD gate path.  An HD-enabled checkpoint would
    therefore be silently replayed as a clean/all-write teacher, which makes
    its provenance and the resulting hindsight labels misleading.  Keep this
    guard here (rather than relying on the caller's config overrides) and
    record both HD switches in the returned contract.
    """

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
            "HD hindsight labels require a trained clean SmolVLA-TTT teacher "
            f"(config.type='smolvla_ttt'), got {raw.get('type')!r} from {checkpoint_text!r}. "
            "Train clean-TTT first; a standard SmolVLA checkpoint would leave "
            "the TTT/register weights randomly initialized."
        )
    teacher_hd_ttt_enabled = raw.get("hd_ttt_enabled", False)
    teacher_hd_learned_write_gate = raw.get("hd_learned_write_gate", False)
    # Early SmolVLA-TTT checkpoints serialized the newly introduced learned
    # gate as JSON ``null``.  That value means "field absent" for those clean
    # checkpoints, not an enabled HD gate; normalize it to the explicit clean
    # value while retaining strict rejection of arbitrary malformed values.
    if teacher_hd_ttt_enabled is None:
        teacher_hd_ttt_enabled = False
    if teacher_hd_learned_write_gate is None:
        teacher_hd_learned_write_gate = False
    # Generated draccus configs use JSON booleans.  Reject malformed values
    # instead of allowing e.g. the string ``"false"`` to become truthy and
    # bypass the clean-teacher guard.
    if (
        type(teacher_hd_ttt_enabled) is not bool
        or type(teacher_hd_learned_write_gate) is not bool
    ):
        raise ValueError(
            f"Teacher config {config_path} has malformed HD switches; "
            "hd_ttt_enabled and hd_learned_write_gate must be JSON booleans"
        )
    if teacher_hd_ttt_enabled or teacher_hd_learned_write_gate:
        raise ValueError(
            "HD hindsight replay requires a clean SmolVLA-TTT teacher with "
            "hd_ttt_enabled=false and hd_learned_write_gate=false. The label "
            "replay currently uses the clean/all-write path and does not invoke "
            "the learned HD gate; regenerate with a clean-TTT checkpoint."
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
    # A few early configs serialized optional extension fields as JSON null;
    # null has the same semantics as the absent legacy suffix writer.
    writer_mode = raw.get("ttt_writer_mode") or "suffix"
    if writer_mode not in {"suffix", "prefix_only"}:
        raise ValueError(
            f"Teacher config {config_path} has invalid ttt_writer_mode={writer_mode!r}"
        )
    return {
        "policy_type": raw.get("type"),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "ttt_layer_indices": layer_indices,
        "ttt_num_register_tokens": register_tokens,
        "ttt_writer_mode": writer_mode,
        "hd_ttt_enabled": teacher_hd_ttt_enabled,
        "hd_learned_write_gate": teacher_hd_learned_write_gate,
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
    *,
    slot_mode: str = "all",
) -> Tensor:
    """Per-frame flow loss used to form hindsight control credit.

    ``slot_mode='all'`` is the historical chunk-average objective.  The v2
    protocol uses ``slot_mode='slot0'`` because MIKASA executes the first
    action and replans at the next observation; future chunk slots are useful
    for imitation but should not define whether a memory event was causally
    useful to the deployed controller.
    """

    if slot_mode not in {"all", "slot0"}:
        raise ValueError("slot_mode must be 'all' or 'slot0'")
    if velocity.ndim < 3 or noise.ndim < 3 or actions.ndim < 3:
        raise ValueError("velocity, noise, and actions must have [frame, slot, feature] dimensions")
    if slot_mode == "slot0":
        velocity = velocity[:, :1]
        noise = noise[:, :1]
        actions = actions[:, :1]
        if action_is_pad is not None:
            action_is_pad = (
                action_is_pad[:, :1]
                if action_is_pad.ndim >= 2
                else action_is_pad[:, None]
            )

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


def _select_grounding_event(
    eligible_counts: Sequence[int],
    total_credits: Sequence[float],
    mean_scores: Sequence[float],
    *,
    min_future_frames: int,
) -> tuple[int, str]:
    """Select the one event whose wrong branch is stored for grounding.

    ``hd_attribution`` is an all-event maximum and may safely combine rows
    from different interventions.  Grounding is different: the stored wrong
    velocity comes from exactly one event, so its ``hd_rho`` must select that
    same branch.  A positive minimum horizon avoids the terminal-event
    denominator pathology while retaining the old mean-credit rule whenever
    the threshold is zero.  If no positive event has the requested horizon,
    use the highest *total* credit as a short-episode fallback.

    Returns ``(-1, "none")`` when no event has positive causal credit.
    Ties are resolved by the earliest event index for reproducibility.
    """

    if min_future_frames < 0:
        raise ValueError("min_future_frames must be non-negative")
    if not (len(eligible_counts) == len(total_credits) == len(mean_scores)):
        raise ValueError("grounding event summaries must have equal lengths")

    positive = [
        index
        for index, (eligible, total) in enumerate(zip(eligible_counts, total_credits, strict=True))
        if int(eligible) > 0 and float(total) > 0.0
    ]
    if not positive:
        return -1, "none"

    preferred = [
        index for index in positive if int(eligible_counts[index]) >= min_future_frames
    ]
    if preferred:
        # ``max`` with ``-index`` keeps the earliest event on an exact tie.
        selected = max(preferred, key=lambda index: (float(mean_scores[index]), -index))
        return selected, "min_future_horizon_mean"

    selected = max(
        positive,
        key=lambda index: (float(total_credits[index]), float(mean_scores[index]), -index),
    )
    return selected, "total_credit_fallback"


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
    grounding_min_future_frames: int = 64,
    future_mask: Tensor | None = None,
    global_offset: int = 0,
    event_global_start_min: int | None = None,
    event_global_end_max: int | None = None,
    attribution_protocol: str = "legacy",
) -> dict[str, Any]:
    """Compute causal teacher interventions and frame-aligned HD labels.

    ``attribution_protocol='legacy'`` is bit-compatible with the original
    positive raw-loss/max collector.  ``'v2'`` (or the full protocol constant)
    is the paper path: each event is replayed with an antithetic ``z,-z``
    common-random-number pair; signed *relative* degradation is averaged across
    the pair; event/future credit uses the adaptive top-sqrt reducer; and gate
    targets use robust percentile normalization.  The selected-event fields
    remain present for old reader grounding, while v2 additionally emits a
    compact selected-event action-effect target (slot 0 only) for differentiable
    writer/content distillation.
    """

    if grounding_min_future_frames < 0:
        raise ValueError("grounding_min_future_frames must be non-negative")
    if attribution_protocol == "legacy":
        protocol = HD_ATTRIBUTION_PROTOCOL_LEGACY
    elif attribution_protocol == "v2":
        protocol = HD_ATTRIBUTION_PROTOCOL_V2
    else:
        protocol = str(attribution_protocol)
    if protocol not in HD_ATTRIBUTION_PROTOCOLS:
        raise ValueError(
            f"Unknown attribution_protocol={attribution_protocol!r}; expected 'legacy' or 'v2'"
        )
    robust_protocol = protocol == HD_ATTRIBUTION_PROTOCOL_V2
    # Delayed import keeps this script's ``--merge``/unit-test path lightweight.
    if robust_protocol:
        from lerobot.policies.smolvla_ttt.hd_ttt import (
            adaptive_topk_mean,
            robust_percentile_normalize,
            robust_signed_normalize,
            symmetric_relative_credit,
        )

    if robust_protocol and attribution_threshold < 0:
        raise ValueError("attribution_threshold must be non-negative")
    actions = policy.prepare_action(prepared).detach().cpu()
    action_is_pad = prepared.get("action_is_pad")
    if action_is_pad is not None:
        action_is_pad = action_is_pad.detach().cpu()
    active_dim = int(math.prod(policy.config.action_feature.shape))

    length_hint = int(actions.shape[0])
    replay_noises = [noise, -noise] if robust_protocol else [noise]
    slot_mode = "slot0" if robust_protocol else "all"
    full_velocities: list[Tensor] = []
    full_losses: list[Tensor] = []
    for replay_noise in replay_noises:
        velocity = _run_replay(
            policy,
            prepared,
            replay_noise,
            time,
            frame_batch_size=frame_batch_size,
        )
        full_velocities.append(velocity)
        full_losses.append(
            _flow_losses(
                velocity,
                replay_noise,
                actions,
                action_is_pad,
                active_dim,
                slot_mode=slot_mode,
            )
        )
    full_velocity = full_velocities[0]
    full_loss = torch.stack(full_losses, dim=0).mean(dim=0)
    length = int(full_velocity.shape[0])
    if length != length_hint:
        raise ValueError(f"Teacher replay returned {length} rows, expected {length_hint}")
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
    signed_rows: list[Tensor] = []
    harm_rows: list[Tensor] = []
    # Keep only a few branch tensors.  ``preferred``/``fallback`` reproduce the
    # historical selector; ``branch_pool`` supplies the compact selected-event
    # action-effect target without retaining every intervention.
    preferred_branch: dict[str, Any] | None = None
    fallback_branch: dict[str, Any] | None = None
    branch_pool: list[dict[str, Any]] = []
    preferred_score = float("-inf")
    fallback_total = float("-inf")
    eligible_counts: list[int] = []
    total_credits: list[float] = []
    selection_scores: list[float] = []

    for event_index, (event_start, event_end) in enumerate(events):
        gate = torch.ones(length, dtype=torch.float32, device=noise.device)
        gate[event_start:event_end] = 0.0
        wrong_velocities: list[Tensor] = []
        wrong_losses: list[Tensor] = []
        for replay_noise, full_variant_loss in zip(replay_noises, full_losses, strict=True):
            wrong_velocity_variant = _run_replay(
                policy,
                prepared,
                replay_noise,
                time,
                frame_batch_size=frame_batch_size,
                write_gate=gate,
            )
            wrong_velocities.append(wrong_velocity_variant)
            wrong_losses.append(
                _flow_losses(
                    wrong_velocity_variant,
                    replay_noise,
                    actions,
                    action_is_pad,
                    active_dim,
                    slot_mode=slot_mode,
                )
            )

        if robust_protocol:
            signed = torch.stack(
                [
                    symmetric_relative_credit(full_variant_loss, wrong_variant_loss)
                    for full_variant_loss, wrong_variant_loss in zip(
                        full_losses, wrong_losses, strict=True
                    )
                ],
                dim=0,
            ).mean(dim=0)
            # Keep the raw loss delta as an audit quantity, but make all
            # training-facing credit dimensionless and robust to action/task
            # scale.  The positive/negative split prevents harmful writes from
            # cancelling useful-memory evidence.
            raw_delta = torch.stack(
                [wrong_loss - full_variant_loss for wrong_loss, full_variant_loss in zip(wrong_losses, full_losses, strict=True)],
                dim=0,
            ).mean(dim=0)
            row = signed.clamp_min(0.0)
            harm = (-signed).clamp_min(0.0)
        else:
            raw_delta = wrong_losses[0] - full_losses[0]
            signed = raw_delta
            row = raw_delta.clamp_min(0.0)
            harm = (-raw_delta).clamp_min(0.0)
        causal = torch.zeros(length, dtype=torch.bool)
        causal[event_end:] = True
        if future_mask is not None:
            causal &= future_mask
        row = torch.where(causal, row, torch.zeros_like(row))
        signed = torch.where(causal, signed, torch.zeros_like(signed))
        harm = torch.where(causal, harm, torch.zeros_like(harm))
        if attribution_threshold > 0:
            # Threshold applies to positive evidence only.  Signed/harmful
            # diagnostics remain available even when a row is not selected.
            row = torch.where(row >= attribution_threshold, row, torch.zeros_like(row))
        credit_rows.append(row)
        signed_rows.append(signed)
        harm_rows.append(harm)

        eligible_count = int(causal.sum().item())
        if robust_protocol:
            # The adaptive reducer is meant to average the strongest *actual
            # positive* future effects.  Passing the whole causal horizon
            # would count zero-credit rows when choosing ``k`` and dilute a
            # sparse but decisive intervention, contrary to the protocol used
            # for ``event_u``/``rho`` below.
            score_tensor = adaptive_topk_mean(row, causal & (row > 0), dim=0)
            score = float(score_tensor.item()) if eligible_count else float("-inf")
        else:
            score = float(row.sum().item()) / max(eligible_count, 1) if eligible_count else float("-inf")
        total_credit = float(row.sum().item())
        eligible_counts.append(eligible_count)
        total_credits.append(total_credit)
        selection_scores.append(score)
        positive_credit = eligible_count > 0 and total_credit > 0.0
        if not positive_credit:
            continue
        branch = {
            "event_index": event_index,
            "wrong_velocity": wrong_velocities[0].detach().cpu(),
            # Content/effect supervision is phase-matched to the student's
            # single ``+noise`` replay.  The antithetic ``-noise`` branch is
            # used only for signed attribution/event ranking; averaging its
            # velocity effect here would give the student a target from a
            # distribution it never evaluates online.
            "effect_velocity": (full_velocities[0] - wrong_velocities[0]).detach().cpu(),
            "gate": gate.detach().cpu(),
            "row": row.detach().cpu(),
            "score": score,
            "total": total_credit,
        }
        fallback_key = (total_credit, score, -event_index)
        current_fallback_key = (
            fallback_total,
            float("-inf") if fallback_branch is None else float(fallback_branch["score"]),
            -1 if fallback_branch is None else -int(fallback_branch["event_index"]),
        )
        if fallback_branch is None or fallback_key > current_fallback_key:
            fallback_total = total_credit
            fallback_branch = branch
        preferred_key = (score, -event_index)
        current_preferred_key = (
            preferred_score,
            -1 if preferred_branch is None else -int(preferred_branch["event_index"]),
        )
        if eligible_count >= grounding_min_future_frames and (
            preferred_branch is None or preferred_key > current_preferred_key
        ):
            preferred_score = score
            preferred_branch = branch
        if robust_protocol:
            branch_pool.append(branch)
            branch_pool.sort(key=lambda item: (float(item["score"]), float(item["total"]), -int(item["event_index"])), reverse=True)
            del branch_pool[V2_EFFECT_BRANCHES:]

    selected_event_index, grounding_selection_mode = _select_grounding_event(
        eligible_counts,
        total_credits,
        selection_scores,
        min_future_frames=grounding_min_future_frames,
    )
    selected_branch: dict[str, Any] | None = None
    if preferred_branch is not None and selected_event_index == int(preferred_branch["event_index"]):
        selected_branch = preferred_branch
    elif fallback_branch is not None and selected_event_index == int(fallback_branch["event_index"]):
        selected_branch = fallback_branch
    best_wrong = selected_branch["wrong_velocity"] if selected_branch is not None else None
    best_gate = selected_branch["gate"] if selected_branch is not None else torch.ones(
        length, dtype=torch.float32
    )
    best_event_index = selected_event_index
    best_event: tuple[int, int] | None = (
        events[selected_event_index] if selected_event_index >= 0 else None
    )

    if credit_rows:
        credits = torch.stack(credit_rows, dim=0)
        signed_credits = torch.stack(signed_rows, dim=0)
        harmful_credits = torch.stack(harm_rows, dim=0)
        if robust_protocol:
            positive_mask = credits > 0
            harm_mask = harmful_credits > 0
            rho_hca_raw = adaptive_topk_mean(credits, positive_mask, dim=0)
            event_u_raw = adaptive_topk_mean(credits, positive_mask, dim=1)
            harm_rho_raw = adaptive_topk_mean(harmful_credits, harm_mask, dim=0)
            harm_u_raw = adaptive_topk_mean(harmful_credits, harm_mask, dim=1)
        else:
            rho_hca_raw = credits.amax(dim=0)
            event_u_raw = credits.amax(dim=1)
            harm_rho_raw = harmful_credits.amax(dim=0)
            harm_u_raw = harmful_credits.amax(dim=1)
        rho_grounding_raw = (
            credits[best_event_index]
            if 0 <= best_event_index < credits.shape[0]
            else torch.zeros(length, dtype=credits.dtype)
        )
        signed_rho_raw = rho_hca_raw - harm_rho_raw
    else:
        credits = torch.zeros((0, length), dtype=torch.float32)
        signed_credits = torch.zeros_like(credits)
        harmful_credits = torch.zeros_like(credits)
        rho_hca_raw = torch.zeros(length, dtype=torch.float32)
        rho_grounding_raw = torch.zeros(length, dtype=torch.float32)
        event_u_raw = torch.zeros(0, dtype=torch.float32)
        harm_rho_raw = torch.zeros(length, dtype=torch.float32)
        harm_u_raw = torch.zeros(0, dtype=torch.float32)
        signed_rho_raw = torch.zeros(length, dtype=torch.float32)

    if robust_protocol:
        rho_hca = robust_percentile_normalize(rho_hca_raw, rho_hca_raw > 0, dim=0)
        rho_grounding = robust_percentile_normalize(
            rho_grounding_raw, rho_grounding_raw > 0, dim=0
        )
        event_u = robust_percentile_normalize(event_u_raw, event_u_raw > 0, dim=0)
        harm_rho = robust_percentile_normalize(harm_rho_raw, harm_rho_raw > 0, dim=0)
        harm_u = robust_percentile_normalize(harm_u_raw, harm_u_raw > 0, dim=0)
        signed_rho = robust_signed_normalize(signed_rho_raw, dim=0)
    else:
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
        harm_rho = harm_rho_raw / harm_rho_raw.max().clamp_min(1e-8) if float(harm_rho_raw.max()) > 0 else harm_rho_raw
        harm_u = harm_u_raw / harm_u_raw.max().clamp_min(1e-8) if harm_u_raw.numel() and float(harm_u_raw.max()) > 0 else harm_u_raw
        signed_rho = signed_rho_raw

    # Unobserved event blocks (when ``max_events`` is a positive sampling cap)
    # retain the ordinary writer rather than being silently trained as skips.
    write_gate = torch.ones(length, dtype=torch.float32)
    write_gate_observed = torch.zeros(length, dtype=torch.float32)
    for event_index, (event_start, event_end) in enumerate(events):
        write_gate[event_start:event_end] = event_u[event_index]
        write_gate_observed[event_start:event_end] = 1.0

    if best_wrong is None:
        best_wrong = full_velocity.clone()
        best_gate = torch.ones(length, dtype=torch.float32)

    result: dict[str, Any] = {
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
        "hd_signed_C": signed_credits.float(),
        "hd_harm_C": harmful_credits.float(),
        "hd_harm_attribution": harm_rho.float(),
        "hd_harm_u": harm_u.float(),
        "hd_signed_attribution": signed_rho.float(),
        "hd_event_u": event_u.float(),
        "hd_attribution_protocol": protocol,
        "hd_attribution_slot_mode": slot_mode,
        "hd_attribution_replays": int(len(replay_noises)),
        "hd_effect_target": V2_EFFECT_TARGET if robust_protocol else "none",
        "hd_attribution_aggregation": (
            "antithetic_relative_adaptive_top_sqrt_percentile"
            if robust_protocol
            else "all_event_max_for_hca_selected_event_for_grounding"
        ),
        "hd_grounding_event_policy": GROUNDING_EVENT_POLICY,
        "hd_grounding_min_future_frames": int(grounding_min_future_frames),
        "hd_grounding_selection_mode": grounding_selection_mode,
        "hd_event_eligible_counts": torch.tensor(eligible_counts, dtype=torch.int64),
        "hd_event_total_credits": torch.tensor(total_credits, dtype=torch.float32),
        "hd_event_selection_scores": torch.tensor(selection_scores, dtype=torch.float32),
        "hd_event_starts": torch.tensor([event[0] for event in events], dtype=torch.int64),
        "hd_event_ends": torch.tensor([event[1] for event in events], dtype=torch.int64),
        "hd_selected_event": torch.tensor(
            [-1 if best_event is None else best_event[0], -1 if best_event is None else best_event[1]],
            dtype=torch.int64,
        ),
        "hd_full_flow_loss": full_loss.float(),
    }

    if robust_protocol:
        # Compact action-effect target for the selected event.  Only
        # executed slot 0 is stored: the action head replans every observation,
        # so this is the deployment-relevant causal effect and avoids a 50x
        # label-size multiplier.  The selected grounding event is always first
        # when available.  Readers remain compatible with older K>1 artifacts
        # and explicitly consume branch zero.
        ordered_branches: list[dict[str, Any]] = []
        if selected_branch is not None:
            ordered_branches.append(selected_branch)
        for branch in branch_pool:
            if not ordered_branches or int(branch["event_index"]) != int(ordered_branches[0]["event_index"]):
                ordered_branches.append(branch)
            if len(ordered_branches) >= V2_EFFECT_BRANCHES:
                break
        effect_dim = int(full_velocity.shape[-1])
        effect_velocity = torch.zeros(
            length, V2_EFFECT_BRANCHES, effect_dim, dtype=full_velocity.dtype
        )
        effect_gate = torch.ones(length, V2_EFFECT_BRANCHES, dtype=torch.float32)
        effect_rho = torch.zeros(length, V2_EFFECT_BRANCHES, dtype=torch.float32)
        effect_valid = torch.zeros(length, V2_EFFECT_BRANCHES, dtype=torch.float32)
        effect_events = torch.full((V2_EFFECT_BRANCHES, 2), -1, dtype=torch.int64)
        for branch_index, branch in enumerate(ordered_branches[:V2_EFFECT_BRANCHES]):
            event = events[int(branch["event_index"])]
            effect_velocity[:, branch_index] = branch["effect_velocity"][:, 0]
            effect_gate[:, branch_index] = branch["gate"]
            branch_row = branch["row"]
            if float(branch_row.max()) > 0:
                if robust_protocol:
                    branch_rho = robust_percentile_normalize(
                        branch_row, branch_row > 0, dim=0
                    )
                else:
                    branch_rho = branch_row / branch_row.max().clamp_min(1e-8)
                effect_rho[:, branch_index] = branch_rho
            # The effect target is a *future* consequence of removing this
            # event.  Marking every frame valid would apply an invariance
            # penalty to the event itself (and to its past), effectively
            # leaking the counterfactual into the write that caused it.  Keep
            # the exact half-open event boundary used by attribution and
            # additionally respect an episode/window future mask when one is
            # supplied.
            causal_effect = torch.zeros(length, dtype=torch.bool)
            causal_effect[event[1] :] = True
            if future_mask is not None:
                causal_effect &= future_mask
            effect_valid[:, branch_index] = causal_effect.to(dtype=torch.float32)
            effect_events[branch_index] = torch.tensor(event, dtype=torch.int64)
        result.update(
            {
                "hd_teacher_effect": effect_velocity.float(),
                "hd_effect_rho": effect_rho.float(),
                "hd_effect_write_gate": effect_gate.float(),
                "hd_effect_valid": effect_valid.float(),
                "hd_effect_events": effect_events,
                "hd_effect_slot": torch.tensor(0, dtype=torch.int64),
            }
        )
    return result


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
        "teacher_hd_ttt_enabled",
        "teacher_hd_learned_write_gate",
        "seed",
        "phase_mode",
        "history_mode",
        "event_block_size",
        "max_events",
        "grounding_event_policy",
        "grounding_min_future_frames",
        "attribution_threshold",
        "action_chunk_size",
        "max_action_dim",
        "fps",
        "frame_batch_size",
    )
    reference_metadata: dict[str, Any] | None = None
    shard_protocols: list[str] = []
    for path, payload in zip(inputs, payloads, strict=True):
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(
                f"Shard {path} is missing metadata; refusing to merge an unprovenanced HD artifact"
            )
        missing_metadata = sorted(set(metadata_contract_keys) - set(metadata))
        if missing_metadata:
            raise ValueError(
                f"Shard {path} is missing hindsight metadata contract fields: {missing_metadata}"
            )
        if metadata["grounding_event_policy"] != GROUNDING_EVENT_POLICY:
            raise ValueError(
                f"Shard {path} has unsupported grounding_event_policy="
                f"{metadata['grounding_event_policy']!r}"
            )
        if (
            type(metadata["grounding_min_future_frames"]) is not int
            or metadata["grounding_min_future_frames"] < 0
        ):
            raise ValueError(
                f"Shard {path} has malformed grounding_min_future_frames; "
                "expected a non-negative int"
            )
        for flag_name in ("teacher_hd_ttt_enabled", "teacher_hd_learned_write_gate"):
            flag_value = metadata[flag_name]
            if type(flag_value) is not bool:
                raise ValueError(
                    f"Shard {path} has malformed {flag_name}; expected a JSON boolean"
                )
            if flag_value:
                raise ValueError(
                    f"Shard {path} was generated with an HD teacher; "
                    "the clean/all-write replay contract must be used for hindsight labels"
                )
        copied = dict(metadata)
        # Artifacts generated before v2 did not carry an explicit protocol;
        # infer the legacy contract so they remain mergeable and auditable.
        copied.setdefault("attribution_protocol", HD_ATTRIBUTION_PROTOCOL_LEGACY)
        copied.setdefault("attribution_slot_mode", "all")
        copied.setdefault("attribution_replays", 1)
        # Resolve the default after normalizing the protocol: legacy artifacts
        # have no effect axis (K=0), while old v2 artifacts may omit the field
        # and need inference from their stored tensor.
        copied.setdefault("effect_branches", None)
        copied.setdefault("effect_target", "none")
        # Explicit JSON null is how older writers represented an omitted
        # optional field.  Normalize it before any contract comparison so it
        # is treated as the legacy suffix mode, never as ``"None"``.
        copied["teacher_ttt_writer_mode"] = str(
            copied.get("teacher_ttt_writer_mode") or "suffix"
        )
        if copied["attribution_protocol"] in {"legacy", "v1"}:
            copied["attribution_protocol"] = HD_ATTRIBUTION_PROTOCOL_LEGACY
        elif copied["attribution_protocol"] == "v2":
            copied["attribution_protocol"] = HD_ATTRIBUTION_PROTOCOL_V2
        if copied["attribution_protocol"] not in HD_ATTRIBUTION_PROTOCOLS:
            raise ValueError(
                f"Shard {path} has unsupported attribution_protocol="
                f"{copied['attribution_protocol']!r}"
            )
        # v2 readers consume branch zero, but preserve the declared K when
        # merging legacy K>1 artifacts so metadata cannot claim K=1 while the
        # concatenated tensors still carry additional event columns.  New
        # artifacts omit no branch and therefore use the formal K=1 constant.
        if copied["attribution_protocol"] == HD_ATTRIBUTION_PROTOCOL_V2:
            declared_effect_branches = copied.get("effect_branches")
            if declared_effect_branches is None:
                effect_tensor = payload.get("hd_teacher_effect")
                declared_effect_branches = (
                    int(effect_tensor.shape[1])
                    if isinstance(effect_tensor, Tensor) and effect_tensor.ndim >= 3
                    else 1
                )
            if type(declared_effect_branches) is not int or declared_effect_branches < 1:
                raise ValueError(
                    f"Shard {path} has malformed effect_branches={declared_effect_branches!r}; "
                    "expected a positive integer for v2"
                )
            effect_tensor = payload.get("hd_teacher_effect")
            if isinstance(effect_tensor, Tensor) and effect_tensor.ndim >= 3:
                actual_effect_branches = int(effect_tensor.shape[1])
                if actual_effect_branches != declared_effect_branches:
                    raise ValueError(
                        f"Shard {path} declares effect_branches={declared_effect_branches} "
                        f"but hd_teacher_effect has K={actual_effect_branches}"
                    )
            copied["effect_branches"] = declared_effect_branches
        else:
            copied["effect_branches"] = 0
        shard_protocols.append(str(copied["attribution_protocol"]))
        copied["source_path"] = str(path)
        shard_metadata.append(copied)
        if reference_metadata is None:
            reference_metadata = dict(metadata)
            reference_metadata.setdefault("attribution_protocol", copied["attribution_protocol"])
            reference_metadata.setdefault("attribution_slot_mode", copied["attribution_slot_mode"])
            reference_metadata.setdefault("attribution_replays", copied["attribution_replays"])
            if reference_metadata.get("effect_branches") is None:
                reference_metadata["effect_branches"] = copied["effect_branches"]
            reference_metadata.setdefault("effect_target", copied["effect_target"])
            reference_metadata["teacher_ttt_writer_mode"] = str(
                reference_metadata.get("teacher_ttt_writer_mode")
                or copied["teacher_ttt_writer_mode"]
            )
        else:
            mismatches = {
                key: (
                    reference_metadata.get(key),
                    copied["teacher_ttt_writer_mode"]
                    if key == "teacher_ttt_writer_mode"
                    else metadata.get(key),
                )
                for key in metadata_contract_keys
                if reference_metadata[key]
                != (
                    copied["teacher_ttt_writer_mode"]
                    if key == "teacher_ttt_writer_mode"
                    else metadata[key]
                )
            }
            if mismatches:
                raise ValueError(f"Cannot merge incompatible hindsight shards: {mismatches}")
            reference_protocol = reference_metadata.get(
                "attribution_protocol", HD_ATTRIBUTION_PROTOCOL_LEGACY
            )
            if reference_protocol != copied["attribution_protocol"]:
                raise ValueError(
                    "Cannot merge hindsight shards generated by different attribution protocols: "
                    f"{reference_protocol!r} vs {copied['attribution_protocol']!r}"
                )
            if reference_metadata.get("teacher_ttt_writer_mode", "suffix") != copied["teacher_ttt_writer_mode"]:
                raise ValueError(
                    "Cannot merge hindsight shards generated with different TTT writer modes: "
                    f"{reference_metadata.get('teacher_ttt_writer_mode', 'suffix')!r} vs "
                    f"{copied['teacher_ttt_writer_mode']!r}"
                )
            if reference_metadata.get("effect_target", "none") != copied["effect_target"]:
                raise ValueError(
                    "Cannot merge hindsight shards with different action-effect targets: "
                    f"{reference_metadata.get('effect_target', 'none')!r} vs "
                    f"{copied['effect_target']!r}"
                )
            if reference_metadata.get("effect_branches", 0) != copied["effect_branches"]:
                raise ValueError(
                    "Cannot merge hindsight shards with different effect branch counts: "
                    f"{reference_metadata.get('effect_branches', 0)!r} vs "
                    f"{copied['effect_branches']!r}"
                )
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
    merged_protocol = shard_protocols[0] if shard_protocols else HD_ATTRIBUTION_PROTOCOL_LEGACY
    if merged_protocol == HD_ATTRIBUTION_PROTOCOL_V2:
        optional_v2 = (
            "hd_signed_attribution",
            "hd_harm_attribution",
            "hd_teacher_effect",
            "hd_effect_rho",
            "hd_effect_write_gate",
            "hd_effect_valid",
        )
        for key in optional_v2:
            if not all(isinstance(payload, Mapping) and key in payload for payload in payloads):
                raise ValueError(
                    f"v2 hindsight shards must all contain {key!r}; refusing a partially merged artifact"
                )
            columns[key] = torch.cat(
                [payload[key].detach().cpu() for payload in payloads], dim=0
            )
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
    merged_effect_branches = (
        int(reference_metadata.get("effect_branches", V2_EFFECT_BRANCHES))
        if merged_protocol == HD_ATTRIBUTION_PROTOCOL_V2 and reference_metadata is not None
        else (V2_EFFECT_BRANCHES if merged_protocol == HD_ATTRIBUTION_PROTOCOL_V2 else 0)
    )
    merged_metadata = {
        "format": "hd_ttt_labels_v2" if merged_protocol == HD_ATTRIBUTION_PROTOCOL_V2 else "hd_ttt_labels_v1",
        "merged_from": [str(path) for path in inputs],
        "num_frames": int(sorted_indices.numel()),
        "shard_metadata": shard_metadata,
        "episodes_detail": episodes_detail,
        "phase_mode": merged_phase_mode,
        "attribution_protocol": merged_protocol,
        "attribution_slot_mode": "slot0" if merged_protocol == HD_ATTRIBUTION_PROTOCOL_V2 else "all",
        "attribution_replays": V2_ANTITHETIC_REPLAYS if merged_protocol == HD_ATTRIBUTION_PROTOCOL_V2 else 1,
        "effect_branches": merged_effect_branches,
        "effect_target": (
            reference_metadata.get("effect_target", "none")
            if reference_metadata is not None
            else "none"
        ),
        "teacher_ttt_writer_mode": (
            reference_metadata.get("teacher_ttt_writer_mode", "suffix")
            if reference_metadata is not None
            else "suffix"
        ),
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
        # Replay must instantiate the teacher's actual writer path.  In
        # particular, a prefix-only checkpoint cannot be silently converted
        # to the legacy suffix writer by the config/checkpoint merge.
        ttt_writer_mode=str(teacher_info.get("ttt_writer_mode", "suffix")),
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
    attribution_protocol = getattr(args, "attribution_protocol", "legacy")
    if attribution_protocol == "legacy":
        attribution_protocol = HD_ATTRIBUTION_PROTOCOL_LEGACY
    elif attribution_protocol == "v2":
        attribution_protocol = HD_ATTRIBUTION_PROTOCOL_V2
    if attribution_protocol not in HD_ATTRIBUTION_PROTOCOLS:
        raise ValueError(
            f"Unknown --attribution-protocol={attribution_protocol!r}; expected legacy or v2"
        )
    column_names = [
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
        "hd_signed_attribution",
        "hd_harm_attribution",
    ]
    if attribution_protocol == HD_ATTRIBUTION_PROTOCOL_V2:
        column_names.extend(
            [
                "hd_teacher_effect",
                "hd_effect_rho",
                "hd_effect_write_gate",
                "hd_effect_valid",
            ]
        )
    columns: dict[str, list[Tensor]] = {key: [] for key in column_names}
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
            grounding_min_future_frames=args.grounding_min_future_frames,
            attribution_protocol=attribution_protocol,
        )
        for key in columns:
            if key == "global_index":
                columns[key].append(torch.tensor(global_indices, dtype=torch.int64))
            elif key == "episode_index":
                columns[key].append(torch.tensor(episode_indices, dtype=torch.int64))
            elif key == "frame_index":
                columns[key].append(torch.tensor(frame_indices, dtype=torch.int64))
            else:
                value = labels.get(key)
                if value is None:
                    raise ValueError(
                        f"Attribution protocol {attribution_protocol!r} did not emit required label {key!r}"
                    )
                columns[key].append(value.detach().cpu())
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
            "event_eligible_counts": labels["hd_event_eligible_counts"].tolist(),
            "event_total_credits": labels["hd_event_total_credits"].tolist(),
            "grounding_event_policy": labels["hd_grounding_event_policy"],
            "grounding_min_future_frames": int(labels["hd_grounding_min_future_frames"]),
            "grounding_selection_mode": labels["hd_grounding_selection_mode"],
            "attribution_protocol": labels["hd_attribution_protocol"],
            "attribution_slot_mode": labels["hd_attribution_slot_mode"],
            "attribution_replays": int(labels["hd_attribution_replays"]),
            "effect_target": labels["hd_effect_target"],
        }
        del prepared, labels, noise, time
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: torch.cat(values, dim=0) for key, values in columns.items()}
    payload["metadata"] = {
        "format": "hd_ttt_labels_v2" if attribution_protocol == HD_ATTRIBUTION_PROTOCOL_V2 else "hd_ttt_labels_v1",
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "teacher_checkpoint": str(args.checkpoint),
        "teacher_policy_type": teacher_info["policy_type"],
        "teacher_config_sha256": teacher_info["config_sha256"],
        "teacher_ttt_layer_indices": list(teacher_info["ttt_layer_indices"]),
        "teacher_ttt_num_register_tokens": int(teacher_info["ttt_num_register_tokens"]),
        "teacher_ttt_writer_mode": str(teacher_info.get("ttt_writer_mode", "suffix")),
        "teacher_hd_ttt_enabled": bool(teacher_info["hd_ttt_enabled"]),
        "teacher_hd_learned_write_gate": bool(teacher_info["hd_learned_write_gate"]),
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
        "grounding_event_policy": GROUNDING_EVENT_POLICY,
        "grounding_min_future_frames": args.grounding_min_future_frames,
        "attribution_protocol": attribution_protocol,
        "attribution_slot_mode": "slot0" if attribution_protocol == HD_ATTRIBUTION_PROTOCOL_V2 else "all",
        "attribution_replays": V2_ANTITHETIC_REPLAYS if attribution_protocol == HD_ATTRIBUTION_PROTOCOL_V2 else 1,
        "effect_branches": V2_EFFECT_BRANCHES if attribution_protocol == HD_ATTRIBUTION_PROTOCOL_V2 else 0,
        "effect_target": V2_EFFECT_TARGET if attribution_protocol == HD_ATTRIBUTION_PROTOCOL_V2 else "none",
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
    parser.add_argument(
        "--grounding-min-future-frames",
        type=int,
        default=64,
        help=(
            "Minimum eligible future frames for the selected grounding event; "
            "short episodes fall back to highest total credit (default: 64)"
        ),
    )
    parser.add_argument("--attribution-threshold", type=float, default=0.0)
    parser.add_argument(
        "--attribution-protocol",
        choices=("legacy", "v2"),
        default="v2",
        help=(
            "Hindsight credit protocol. 'v2' (default) uses antithetic relative "
            "credit, robust aggregation, and slot-0 action effects; 'legacy' "
            "reproduces the original raw hinge/max labels."
        ),
    )
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
    if args.grounding_min_future_frames < 0:
        parser.error("--grounding-min-future-frames must be non-negative")
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
