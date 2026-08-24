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

"""Compare a trained PI0-TTT checkpoint with its source PI0 checkpoint."""

import argparse
import gc
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import draccus
import torch
from torch import Tensor

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0_ttt.configuration_pi0_ttt import PI0TTTConfig
from lerobot.policies.pi0_ttt.modeling_pi0_ttt import PI0TTTPolicy, TTTFastStates
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--ttt-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--flow-time", type=float, default=0.5)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def clone_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    return value


def load_samples(args: argparse.Namespace, config: PI0Config) -> list[dict[str, Any]]:
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        video_backend=args.video_backend,
        return_uint8=True,
    )
    end_index = args.start_index + args.num_frames
    if args.start_index < 0 or end_index > len(dataset):
        raise IndexError(f"Requested frame range [{args.start_index}, {end_index}) for {len(dataset)} frames")

    samples = [dataset[index] for index in range(args.start_index, end_index)]
    episode_indices = {int(sample["episode_index"]) for sample in samples}
    if len(episode_indices) != 1:
        raise ValueError(
            f"Requested frames cross episode boundaries (episodes={sorted(episode_indices)}); "
            "choose a different start index or fewer frames"
        )
    return samples


def make_fixed_noise(config: PI0Config, num_frames: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        num_frames,
        1,
        config.chunk_size,
        config.max_action_dim,
        generator=generator,
        dtype=torch.float32,
    )


def load_checkpoint_config(checkpoint: Path, config_class):
    raw_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    valid_fields = {field.name for field in fields(config_class) if field.init}
    return draccus.decode(
        config_class,
        {key: value for key, value in raw_config.items() if key in valid_fields},
    )


def fixed_flow_loss(
    policy: PI0Policy,
    batch: dict[str, Tensor],
    noise: Tensor,
    flow_time: float,
    fast_states: TTTFastStates | None,
) -> tuple[float, TTTFastStates | None]:
    images, image_masks = policy._preprocess_images(batch)
    language_tokens = batch[OBS_LANGUAGE_TOKENS]
    language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    time = torch.full((actions.shape[0],), flow_time, dtype=torch.float32, device=actions.device)

    if isinstance(policy, PI0TTTPolicy):
        losses, fast_states = policy.model.forward_with_state(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            actions,
            noise,
            time,
            sequence_shape=(actions.shape[0], 1),
            fast_states=fast_states,
            create_graph=False,
        )
    else:
        losses = policy.model.forward(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            actions,
            noise,
            time,
        )

    action_dim = policy.config.output_features[ACTION].shape[0]
    return float(losses[:, :, :action_dim].mean().cpu()), fast_states


def reset_processors(preprocessor, postprocessor) -> None:
    preprocessor.reset()
    postprocessor.reset()


def predict_once(policy, preprocessor, postprocessor, sample, noise) -> Tensor:
    policy.reset()
    reset_processors(preprocessor, postprocessor)
    batch = preprocessor(clone_value(sample))
    prediction = policy.predict_action_chunk(batch, noise=noise.to(policy.config.device))
    return postprocessor(prediction).squeeze(0).detach().float().cpu()


def summarize_actions(predictions: Tensor, targets: Tensor, flow_losses: list[float]) -> dict[str, Any]:
    errors = predictions - targets
    return {
        "finite": bool(torch.isfinite(predictions).all() and torch.isfinite(torch.tensor(flow_losses)).all()),
        "action_mean": float(predictions.mean()),
        "action_std": float(predictions.std(unbiased=False)),
        "action_min": float(predictions.min()),
        "action_max": float(predictions.max()),
        "target_mae": float(errors.abs().mean()),
        "target_mse": float(errors.square().mean()),
        "first_action_mae": float(errors[:, 0].abs().mean()),
        "fixed_flow_loss": float(torch.tensor(flow_losses).mean()),
        "per_dim_mae": errors.abs().mean(dim=(0, 1)).tolist(),
    }


def evaluate_policy(
    policy,
    checkpoint: Path,
    samples: list[dict[str, Any]],
    noises: Tensor,
    flow_time: float,
) -> tuple[dict[str, Any], Tensor]:
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    policy.eval()
    policy.reset()
    reset_processors(preprocessor, postprocessor)

    predictions = []
    targets = []
    flow_losses = []
    flow_states = None
    # TTT performs its inner update under torch.enable_grad(), so inference_mode is too strict here.
    with torch.no_grad():
        for frame_offset, (sample, noise) in enumerate(zip(samples, noises, strict=True)):
            batch = preprocessor(clone_value(sample))
            device_noise = noise.to(policy.config.device)
            prediction = policy.predict_action_chunk(batch, noise=device_noise)
            predictions.append(postprocessor(prediction).squeeze(0).detach().float().cpu())
            targets.append(sample[ACTION].detach().float().cpu())
            loss, flow_states = fixed_flow_loss(
                policy,
                batch,
                device_noise,
                flow_time,
                flow_states,
            )
            flow_losses.append(loss)
            print(f"{policy.name}: evaluated frame {frame_offset + 1}/{len(samples)}", flush=True)

        stacked_predictions = torch.stack(predictions)
        stacked_targets = torch.stack(targets)
        metrics = summarize_actions(stacked_predictions, stacked_targets, flow_losses)

        if isinstance(policy, PI0TTTPolicy):
            replayed_first = predict_once(
                policy, preprocessor, postprocessor, samples[0], noises[0]
            )
            cold_last = predict_once(
                policy, preprocessor, postprocessor, samples[-1], noises[-1]
            )
            metrics.update(
                {
                    "reset_replay_max_abs_diff": float(
                        (replayed_first - stacked_predictions[0]).abs().max()
                    ),
                    "carry_vs_cold_last_mae": float(
                        (stacked_predictions[-1] - cold_last).abs().mean()
                    ),
                    "carry_vs_cold_last_max_abs": float(
                        (stacked_predictions[-1] - cold_last).abs().max()
                    ),
                }
            )

    return metrics, stacked_predictions


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.flow_time <= 1.0:
        raise ValueError("--flow-time must be between 0 and 1")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")

    source_config = load_checkpoint_config(args.source_checkpoint, PI0Config)
    ttt_config = load_checkpoint_config(args.ttt_checkpoint, PI0TTTConfig)
    if source_config.chunk_size != ttt_config.chunk_size:
        raise ValueError("Source and TTT checkpoints use different action chunk sizes")
    if source_config.max_action_dim != ttt_config.max_action_dim:
        raise ValueError("Source and TTT checkpoints use different padded action dimensions")

    samples = load_samples(args, source_config)
    noises = make_fixed_noise(source_config, len(samples), args.seed)

    print("Loading source PI0 checkpoint", flush=True)
    source_policy = PI0Policy.from_pretrained(
        args.source_checkpoint,
        config=source_config,
        local_files_only=True,
    )
    source_metrics, source_predictions = evaluate_policy(
        source_policy,
        args.source_checkpoint,
        samples,
        noises,
        args.flow_time,
    )
    del source_policy
    release_cuda_memory()

    print("Loading PI0-TTT checkpoint", flush=True)
    ttt_policy = PI0TTTPolicy.from_pretrained(
        args.ttt_checkpoint,
        config=ttt_config,
        local_files_only=True,
    )
    ttt_metrics, ttt_predictions = evaluate_policy(
        ttt_policy,
        args.ttt_checkpoint,
        samples,
        noises,
        args.flow_time,
    )
    del ttt_policy
    release_cuda_memory()

    comparison = {
        "ttt_minus_source_target_mae": ttt_metrics["target_mae"] - source_metrics["target_mae"],
        "ttt_minus_source_fixed_flow_loss": (
            ttt_metrics["fixed_flow_loss"] - source_metrics["fixed_flow_loss"]
        ),
        "ttt_vs_source_action_mae": float((ttt_predictions - source_predictions).abs().mean()),
        "ttt_vs_source_action_max_abs": float((ttt_predictions - source_predictions).abs().max()),
    }
    results = {
        "dataset": {
            "repo_id": args.dataset_repo_id,
            "root": str(args.dataset_root),
            "start_index": args.start_index,
            "num_contiguous_frames": args.num_frames,
            "episode_index": int(samples[0]["episode_index"]),
        },
        "evaluation": {
            "seed": args.seed,
            "fixed_flow_time": args.flow_time,
            "source_checkpoint": str(args.source_checkpoint),
            "ttt_checkpoint": str(args.ttt_checkpoint),
        },
        "source_pi0": source_metrics,
        "pi0_ttt": ttt_metrics,
        "comparison": comparison,
    }

    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
