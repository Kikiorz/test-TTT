#!/usr/bin/env python
"""Evaluate the original LeRobot SmolVLA with native 50-step chunking.

This adapter uses the official MIKASA-Robo-VLA environment and episode runner,
but deliberately loads :class:`lerobot.policies.smolvla.SmolVLAPolicy` rather
than the independent ``smolvla_ttt`` implementation.  The policy is asked for
one complete flow-matching action chunk (``predict_action_chunk``), and the
official runner executes that chunk for up to 50 simulator steps before asking
for the next observation.  Consequently ``action_chunk_size`` in the result
is 50, matching the original SmolVLA inference contract.

Run this file inside the MIKASA Python environment, for example::

    PYTHONPATH=/workspace/test-TTT/policy/HD-TTT/src:/workspace/MIKASA-Robo \
    python examples/mikasa/evaluate_smolvla_baseline.py \
      --checkpoint lerobot/smolvla_base \
      --dataset-root /workspace/data_mikasa_robo/data_lerobot/\
shell_game_color_lamp_touch_vla_v0 \
      --task ShellGameColorLampTouch-VLA-v0 --num-episodes 50 \
      --sim-backend cpu --device cuda \
      --output /workspace/experiments/smolvla_base_color/eval50.json

Use one task/dataset per invocation when comparing the Short color task with
the shuffle task: their normalization statistics are different.  A local
``smolvla_ttt`` checkpoint is intentionally rejected; evaluate those with
``evaluate_smolvla_ttt.py`` instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

# The observation conversion is the same bridge used by the TTT evaluator.
# Importing the class does not instantiate a TTT policy or touch the benchmark
# until ``evaluate`` is called.  Keeping this small inheritance relationship
# avoids silently diverging in image/channel handling between the two adapters.
try:  # Supports both ``python examples/...`` and package-style invocation.
    from evaluate_smolvla_ttt import SmolVLAMikasaPolicy
except ImportError:  # pragma: no cover - exercised only with ``-m`` invocation.
    from .evaluate_smolvla_ttt import SmolVLAMikasaPolicy


class SmolVLABaselineMikasaPolicy(SmolVLAMikasaPolicy):
    """MIKASA bridge for the standard SmolVLA action-chunk API.

    ``SmolVLAMikasaPolicy`` intentionally exposes ``select_action`` and a
    one-step chunk for TTT, because its recurrent state must advance at every
    environment decision.  The original SmolVLA baseline has no such state:
    this subclass calls ``predict_action_chunk`` directly and exposes the
    native 50 actions to ``benchmarking.run_episode``.
    """

    chunk_size = 50

    @torch.inference_mode()
    def forward(self, obs: Mapping[str, Any]) -> torch.Tensor:
        raw = self._make_policy_observation(obs)
        processed = self.preprocessor(raw)
        action_chunk = self.policy.predict_action_chunk(processed)
        action_chunk = self.postprocessor(action_chunk)

        # Standard LeRobot policies return [B, K, action_dim].  The official
        # MIKASA runner accepts [K, action_dim] for its single vectorized env.
        action_chunk = action_chunk if torch.is_tensor(action_chunk) else torch.as_tensor(action_chunk)
        if action_chunk.ndim == 3:
            if action_chunk.shape[0] != 1:
                raise ValueError(
                    "MIKASA baseline adapter expects batch size 1 from "
                    f"predict_action_chunk, got {tuple(action_chunk.shape)}"
                )
            action_chunk = action_chunk[0]
        if action_chunk.ndim != 2:
            raise ValueError(
                "SmolVLA predict_action_chunk must return [1,K,D] or [K,D], "
                f"got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[0] != self.chunk_size:
            raise ValueError(
                "The original SmolVLA baseline must expose its complete "
                f"50-step chunk, got K={action_chunk.shape[0]}"
            )

        # MIKASA's action space is [-1, 1].  Keep the environment contract at
        # the adapter boundary after unnormalization, just as the TTT adapter
        # does, while retaining all 50 actions in the returned queue.
        return action_chunk.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)


def _checkpoint_type(checkpoint: str) -> str | None:
    """Read a local checkpoint discriminator without importing model code."""

    config_path = Path(checkpoint) / "config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read checkpoint config {config_path}: {error}") from error
    value = payload.get("type")
    return str(value) if value is not None else None


def _make_smolvla_config(args: argparse.Namespace):
    """Load source architecture fields, then infer MIKASA feature fields.

    ``make_policy`` fills ``input_features``/``output_features`` from the
    dataset metadata.  Other architecture fields (VLM name, expert width,
    resize, etc.) are taken from the checkpoint config so a fine-tuned standard
    SmolVLA checkpoint is not accidentally reconstructed with defaults.
    """

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    source_type = _checkpoint_type(args.checkpoint)
    if source_type == "smolvla_ttt":
        raise ValueError(
            f"Checkpoint {args.checkpoint!r} is type smolvla_ttt. "
            "Use evaluate_smolvla_ttt.py for TTT/HD-TTT checkpoints; this "
            "script is reserved for the original smolvla policy."
        )
    if source_type not in (None, "smolvla"):
        raise ValueError(
            f"Checkpoint {args.checkpoint!r} declares unsupported policy type "
            f"{source_type!r}; expected 'smolvla'."
        )

    # Parse the source config for both local paths and Hub IDs.  The Hub form
    # also ensures that a non-default checkpoint architecture is preserved.
    config = SmolVLAConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = Path(args.checkpoint)
    config.device = args.device
    config.push_to_hub = False
    config.input_features = {}
    config.output_features = {}

    # Native original SmolVLA execution is one 50-action model invocation and
    # one 50-step runner queue.  RTC is not part of this baseline protocol.
    config.chunk_size = 50
    config.n_action_steps = 50
    config.rtc_config = None
    config.compile_model = False
    config.__post_init__()
    return config


def _load_policy(args: argparse.Namespace) -> SmolVLABaselineMikasaPolicy:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    dataset_repo_id = args.dataset_repo_id or args.dataset_root.name
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=args.dataset_root)
    config = _make_smolvla_config(args)
    policy = make_policy(config, ds_meta=metadata)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config,
        dataset_stats=metadata.stats,
    )
    policy.eval()
    return SmolVLABaselineMikasaPolicy(
        policy,
        preprocessor,
        postprocessor,
        device=torch.device(args.device),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from mikasa_robo_suite.vla import benchmarking

    tasks = benchmarking.select_benchmark_tasks(env_ids=args.tasks)
    benchmark_commit_fn = getattr(benchmarking, "benchmark_commit", None)
    benchmark_revision = benchmark_commit_fn() if callable(benchmark_commit_fn) else None
    if len(tasks) == 2:
        benchmark_subset = "two_task_subset"
    elif len(tasks) == 1:
        benchmark_subset = "single_task"
    else:
        benchmark_subset = "selected_task_set"
    policy = _load_policy(args)
    results: list[dict[str, Any]] = []

    for task in tasks:
        policy.set_task(task.language_instruction)
        env = benchmarking.make_benchmark_env(
            task.env_id,
            benchmarking.BenchmarkConfig(
                n_episodes=args.num_episodes,
                start_seed=args.start_seed,
                sim_backend=args.sim_backend,
                include_overlays=False,
            ),
        )
        episodes = []
        try:
            for episode_index in range(args.num_episodes):
                # The runner owns its action queue; reset only clears any
                # policy-local observation/action cache at episode boundaries.
                policy.reset()
                episode, _ = benchmarking.run_episode(
                    env,
                    policy,
                    args.start_seed + episode_index,
                    collect_video=False,
                )
                episodes.append(episode)
        finally:
            env.close()

        successes = [bool(ep.success_once) for ep in episodes]
        returns = [float(ep.episode_return) for ep in episodes]
        results.append(
            {
                "env_id": task.env_id,
                "split": task.split.title(),
                "memory_type": task.memory_type,
                "data_source": task.data_source,
                "start_seed": args.start_seed,
                "n_episodes": args.num_episodes,
                "successes": successes,
                "returns": returns,
                "sr": float(np.mean(successes)),
                "mean_return": float(np.mean(returns)),
                "control_mode": "pd_ee_delta_pose",
                "obs_mode": "rgb",
                "sim_backend": args.sim_backend,
                "reward_mode": "normalized_dense",
                "benchmark_protocol": "MIKASA-Robo-VLA official runner",
                "benchmark_subset": benchmark_subset,
                "requested_tasks": [task.env_id for task in tasks],
                "benchmark_commit": benchmark_revision,
                "wrapper_chain": "apply_mikasa_vla_wrappers(include_overlays=False)",
                # Unlike TTT, the original policy's native action horizon is
                # also the runner execution chunk.
                "action_chunk_size": policy.chunk_size,
                "model_action_horizon": int(policy.policy.config.chunk_size),
                "execution_action_steps": int(policy.policy.config.n_action_steps),
                "model": {
                    "checkpoint": str(args.checkpoint),
                    "method": "SmolVLA",
                    "policy_type": "smolvla",
                    "policy_api": "predict_action_chunk",
                    "ttt_enabled": False,
                    "action_chunk_size": policy.chunk_size,
                },
                "episode_lengths": [int(ep.n_steps) for ep in episodes],
                "episode_seeds": [int(ep.seed) for ep in episodes],
            }
        )

    summary = benchmarking.summarize_task_results(results)
    payload = {"results": results, "summary": summary}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local SmolVLA checkpoint directory or a Hugging Face repo ID (e.g. lerobot/smolvla_base)",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="LeRobot repo id; defaults to the local dataset directory name",
    )
    parser.add_argument("--task", dest="tasks", action="append", required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
