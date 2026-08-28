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

"""Trace and visualize PI0-TTT fast-weight changes during a closed-loop LIBERO rollout."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import draccus
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0_ttt.configuration_pi0_ttt import PI0TTTConfig
from lerobot.policies.pi0_ttt.modeling_pi0_ttt import PI0TTTPolicy
from lerobot.policies.pi0_ttt.trace import (
    clone_fast_state,
    sketch_fast_state_delta,
    summarize_fast_state,
)
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--max-env-steps", type=int, default=280)
    parser.add_argument("--observation-size", type=int, default=256)
    parser.add_argument("--sketch-samples-per-tensor", type=int, default=256)
    parser.add_argument("--animation-fps", type=int, default=4)
    return parser.parse_args()


def load_config(checkpoint: Path, n_action_steps: int) -> PI0TTTConfig:
    raw_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    valid_fields = {field.name for field in fields(PI0TTTConfig) if field.init}
    config = draccus.decode(
        PI0TTTConfig,
        {key: value for key, value in raw_config.items() if key in valid_fields},
    )
    config.n_action_steps = n_action_steps
    config.compile_model = False
    config.gradient_checkpointing = False
    return config


def reset_pipeline(pipeline: Any) -> None:
    if hasattr(pipeline, "reset"):
        pipeline.reset()


def extract_success(info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).any())
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            return bool(np.asarray(final_info["is_success"]).any())
    return False


def flatten_decision(decision: dict[str, Any]) -> dict[str, Any]:
    row = {
        "decision_index": decision["decision_index"],
        "env_step": decision["env_step"],
        "reward_sum": decision["reward_sum"],
        "env_steps_executed": decision["env_steps_executed"],
        "success": decision["success"],
        "action_norm": decision["action_norm"],
    }
    for layer_index, metrics in decision["layers"].items():
        for name, value in metrics.items():
            row[f"layer_{layer_index}_{name}"] = value
    return row


def compute_pca(sketches: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    coordinates = {}
    for layer_index, layer_sketches in sketches.items():
        matrix = np.asarray(layer_sketches, dtype=np.float64)
        if matrix.shape[0] < 2:
            coordinates[layer_index] = np.zeros((matrix.shape[0], 2), dtype=np.float64)
            continue
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        components = right_vectors[: min(2, right_vectors.shape[0])].T
        projected = centered @ components
        if projected.shape[1] == 1:
            projected = np.pad(projected, ((0, 0), (0, 1)))
        coordinates[layer_index] = projected
    return coordinates


def save_overview(decisions: list[dict[str, Any]], output_path: Path) -> None:
    layer_indices = sorted(int(index) for index in decisions[0]["layers"])
    decision_indices = np.asarray([decision["decision_index"] for decision in decisions])
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    for layer_index in layer_indices:
        metrics = [decision["layers"][str(layer_index)] for decision in decisions]
        axes[0, 0].plot(
            decision_indices,
            [metric["drift_relative"] for metric in metrics],
            label=f"layer {layer_index}",
        )
        axes[0, 1].plot(
            decision_indices,
            [metric["step_delta_l2"] for metric in metrics],
            label=f"layer {layer_index}",
        )

    axes[0, 0].set(
        title="Fast-weight drift from learned initialization", xlabel="Decision", ylabel="Relative L2"
    )
    axes[0, 1].set(title="Fast-weight update per observation", xlabel="Decision", ylabel="L2")
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(ncol=2)

    heatmap = np.asarray(
        [
            [decision["layers"][str(layer_index)]["drift_relative"] for layer_index in layer_indices]
            for decision in decisions
        ]
    ).T
    image = axes[1, 0].imshow(heatmap, aspect="auto", origin="lower", interpolation="nearest")
    axes[1, 0].set(
        title="Layer-by-time drift heatmap",
        xlabel="Decision",
        ylabel="TTT layer",
        yticks=np.arange(len(layer_indices)),
        yticklabels=layer_indices,
    )
    figure.colorbar(image, ax=axes[1, 0], label="Relative L2")

    axes[1, 1].plot(
        decision_indices,
        [decision["action_norm"] for decision in decisions],
        color="black",
        label="action norm",
    )
    success_decisions = [decision["decision_index"] for decision in decisions if decision["success"]]
    for success_decision in success_decisions:
        axes[1, 1].axvline(success_decision, color="green", alpha=0.6, linewidth=2)
    axes[1, 1].set(title="Executed action magnitude", xlabel="Decision", ylabel="L2")
    axes[1, 1].legend()

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_pca_plot(
    pca_coordinates: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    layer_indices = sorted(pca_coordinates)
    for axis, layer_index in zip(axes.flat, layer_indices, strict=False):
        coordinates = pca_coordinates[layer_index]
        colors = np.arange(len(coordinates))
        axis.plot(coordinates[:, 0], coordinates[:, 1], color="0.65", linewidth=1)
        scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, cmap="viridis", s=28)
        axis.scatter(coordinates[0, 0], coordinates[0, 1], marker="s", color="black", s=45)
        axis.set(title=f"Layer {layer_index}", xlabel="PC1", ylabel="PC2")
        figure.colorbar(scatter, ax=axis, label="Decision")
    for axis in list(axes.flat)[len(layer_indices) :]:
        axis.axis("off")
    figure.suptitle("Sampled fast-weight drift trajectory")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_animation(
    decisions: list[dict[str, Any]],
    camera_frames: list[np.ndarray],
    output_path: Path,
    fps: int,
) -> None:
    layer_indices = sorted(int(index) for index in decisions[0]["layers"])
    rendered_frames = []
    for end_index, (decision, camera_frame) in enumerate(zip(decisions, camera_frames, strict=True), start=1):
        figure, (image_axis, trace_axis) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        image_axis.imshow(camera_frame)
        image_axis.set_title(
            f"{decision['task_description']}\n"
            f"env step {decision['env_step']} | decision {decision['decision_index']}"
        )
        image_axis.axis("off")

        visible = decisions[:end_index]
        x_values = [item["decision_index"] for item in visible]
        for layer_index in layer_indices:
            trace_axis.plot(
                x_values,
                [item["layers"][str(layer_index)]["drift_relative"] for item in visible],
                label=f"layer {layer_index}",
            )
        trace_axis.axvline(decision["decision_index"], color="black", alpha=0.25)
        trace_axis.set(
            title="TTT fast-weight drift during inference",
            xlabel="Decision",
            ylabel="Relative L2 from initialization",
        )
        trace_axis.legend(loc="upper left", ncol=2)
        trace_axis.grid(alpha=0.2)
        if decision["success"]:
            trace_axis.text(
                0.98,
                0.04,
                "SUCCESS",
                color="green",
                fontsize=14,
                fontweight="bold",
                ha="right",
                transform=trace_axis.transAxes,
            )

        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        rendered_frames.append(frame)
        plt.close(figure)

    write_video(output_path, rendered_frames, fps=fps)


def main() -> None:
    args = parse_args()
    if args.n_action_steps <= 0 or args.max_env_steps <= 0:
        raise ValueError("n-action-steps and max-env-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    config = load_config(args.checkpoint, args.n_action_steps)
    policy = PI0TTTPolicy.from_pretrained(
        args.checkpoint,
        config=config,
        local_files_only=True,
    ).to(config.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(config.device)}},
    )
    env_config = LiberoEnv(
        task=args.suite,
        task_ids=[args.task_id],
        observation_height=args.observation_size,
        observation_width=args.observation_size,
        max_parallel_tasks=1,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_config, config)
    env_map = make_env(env_config, n_envs=1, use_async_envs=False)
    env = env_map[args.suite][args.task_id]

    policy.reset()
    for pipeline in (preprocessor, postprocessor, env_preprocessor, env_postprocessor):
        reset_pipeline(pipeline)
    observation, _ = env.reset(seed=[args.seed])

    previous_states = {}
    latest_metrics: dict[str, dict[str, float]] = {}
    sketches: dict[int, list[np.ndarray]] = {
        layer_index: [] for layer_index in config.resolved_ttt_layer_indices
    }
    decisions: list[dict[str, Any]] = []
    camera_frames: list[np.ndarray] = []
    env_steps: list[dict[str, Any]] = []
    active_decision: dict[str, Any] | None = None

    try:
        for env_step in range(args.max_env_steps):
            state_will_update = len(policy._action_queue) == 0
            camera_frame = np.asarray(env.call("render")[0]).copy() if state_will_update else None
            policy_observation = preprocess_observation(observation)
            policy_observation["task"] = list(env.call("task_description"))
            policy_observation = env_preprocessor(policy_observation)
            batch = preprocessor(policy_observation)
            action: PolicyAction = policy.select_action(batch)

            if state_will_update:
                latest_metrics = {}
                for layer_index, state in policy._ttt_fast_states.items():
                    layer = policy.model.ttt_layers[str(layer_index)]
                    latest_metrics[str(layer_index)] = summarize_fast_state(
                        layer,
                        state,
                        previous_states.get(layer_index),
                    )
                    sketches[layer_index].append(
                        sketch_fast_state_delta(
                            layer,
                            state,
                            samples_per_tensor=args.sketch_samples_per_tensor,
                        ).numpy()
                    )
                    previous_states[layer_index] = clone_fast_state(state)

                active_decision = {
                    "decision_index": len(decisions),
                    "env_step": env_step,
                    "task_description": policy_observation["task"][0],
                    "layers": latest_metrics,
                    "action_norm": float(action.detach().float().norm().cpu()),
                    "reward_sum": 0.0,
                    "env_steps_executed": 0,
                    "success": False,
                }
                decisions.append(active_decision)
                camera_frames.append(camera_frame)

            action = postprocessor(action)
            action_transition = env_postprocessor({ACTION: action})
            action_numpy = action_transition[ACTION].detach().cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            reward_value = float(np.asarray(reward).mean())
            success = extract_success(info)
            assert active_decision is not None
            active_decision["reward_sum"] += reward_value
            active_decision["env_steps_executed"] += 1
            active_decision["success"] = active_decision["success"] or success
            env_steps.append(
                {
                    "env_step": env_step,
                    "decision_index": active_decision["decision_index"],
                    "state_updated": state_will_update,
                    "reward": reward_value,
                    "success": success,
                    "action_norm": float(torch.from_numpy(action_numpy).float().norm()),
                }
            )
            print(
                f"env_step={env_step} decision={active_decision['decision_index']} "
                f"updated={state_will_update} reward={reward_value:.1f} success={success}",
                flush=True,
            )
            if bool(np.asarray(terminated).all() or np.asarray(truncated).all() or success):
                break
    finally:
        env.close()

    if not decisions:
        raise RuntimeError("The rollout ended before any TTT state update was recorded")

    pca_coordinates = compute_pca(sketches)
    for layer_index, coordinates in pca_coordinates.items():
        for decision, coordinate in zip(decisions, coordinates, strict=True):
            decision["layers"][str(layer_index)]["pca_x"] = float(coordinate[0])
            decision["layers"][str(layer_index)]["pca_y"] = float(coordinate[1])

    trace = {
        "checkpoint": str(args.checkpoint),
        "suite": args.suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "n_action_steps": args.n_action_steps,
        "completed_env_steps": len(env_steps),
        "completed_decisions": len(decisions),
        "success": any(decision["success"] for decision in decisions),
        "decisions": decisions,
        "env_steps": env_steps,
    }
    (args.output_dir / "ttt_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n",
        encoding="utf-8",
    )

    flattened = [flatten_decision(decision) for decision in decisions]
    with (args.output_dir / "ttt_trace.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)

    save_overview(decisions, args.output_dir / "ttt_state_overview.png")
    save_pca_plot(pca_coordinates, args.output_dir / "ttt_state_pca.png")
    save_animation(
        decisions,
        camera_frames,
        args.output_dir / "ttt_state_rollout.mp4",
        fps=args.animation_fps,
    )
    print(
        json.dumps(
            {key: value for key, value in trace.items() if key not in {"decisions", "env_steps"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
