#!/usr/bin/env python
"""Evaluate Method1 SmolVLA-family checkpoints with the official MIKASA runner.

The canonical single-task protocol is fixed to 50 simulator episodes with
environment seeds 4242424242..4242424291.  The adapter exposes only the two
128x128 RGB cameras, 7D proprioception, and the natural-language instruction;
it never reads privileged simulator state.  TTT memory and all policy-local
queues are reset exactly once at every episode boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

CANONICAL_NUM_EPISODES = 50
CANONICAL_START_SEED = 4_242_424_242
CANONICAL_TORCH_SEED = 7_000


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_checkpoint_config(checkpoint: Path, expected_type: str) -> dict[str, Any]:
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint must contain config.json and model.safetensors: {checkpoint}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("type") != expected_type:
        raise ValueError(f"Checkpoint {checkpoint} has type={raw.get('type')!r}; expected {expected_type!r}")
    return raw


class MikasaPolicyAdapter:
    """Convert the official packed MIKASA observation to LeRobot fields."""

    chunk_size = 1

    def __init__(self, policy, preprocessor, postprocessor):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self._task = ""

    def reset(self) -> None:
        self.policy.reset()

    def set_task(self, instruction: str) -> None:
        self._task = instruction

    @staticmethod
    def _to_chw(image: Any) -> torch.Tensor:
        tensor = image if torch.is_tensor(image) else torch.as_tensor(image)
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected one HWC/CHW image, got {tuple(tensor.shape)}")
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] != 3:
            raise ValueError(f"Cannot identify RGB channel axis in {tuple(tensor.shape)}")
        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() and tensor.max() > 1.0 + 1e-5:
            tensor = tensor / 255.0
        return tensor

    def _make_policy_observation(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        if "rgb" not in obs or "proprio" not in obs:
            raise KeyError("MIKASA observation must contain rgb and proprio")
        rgb = obs["rgb"] if torch.is_tensor(obs["rgb"]) else torch.as_tensor(obs["rgb"])
        if tuple(rgb.shape) != (1, 128, 128, 6):
            raise ValueError(f"Expected official MIKASA rgb [1,128,128,6], got {tuple(rgb.shape)}")

        proprio = obs["proprio"] if torch.is_tensor(obs["proprio"]) else torch.as_tensor(obs["proprio"])
        if proprio.ndim == 2 and proprio.shape[0] == 1:
            proprio = proprio[0]
        proprio = proprio.to(dtype=torch.float32)
        if tuple(proprio.shape) != (7,) or not torch.isfinite(proprio).all():
            raise ValueError(f"Expected finite official MIKASA proprio [1,7], got {tuple(proprio.shape)}")

        return {
            "observation.images.top": self._to_chw(rgb[..., :3]),
            "observation.images.wrist": self._to_chw(rgb[..., 3:]),
            "observation.state": proprio,
            "task": self._task,
        }


class SmolVLATTTMikasaPolicy(MikasaPolicyAdapter):
    """One-action adapter; TTT state advances once per physical observation."""

    chunk_size = 1

    @torch.inference_mode()
    def forward(self, obs: Mapping[str, Any]) -> torch.Tensor:
        processed = self.preprocessor(self._make_policy_observation(obs))
        action = self.postprocessor(self.policy.select_action(processed))
        action = action if torch.is_tensor(action) else torch.as_tensor(action)
        if action.ndim == 2 and tuple(action.shape) == (1, 7):
            action = action[0]
        if tuple(action.shape) != (7,) or not torch.isfinite(action).all():
            raise ValueError(f"SmolVLA-TTT must return one finite 7D action, got {tuple(action.shape)}")
        return action.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)


class SmolVLABaselineMikasaPolicy(MikasaPolicyAdapter):
    """Native SmolVLA adapter exposing its intended action-execution cadence."""

    def __init__(self, *args, execution_action_steps: int, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= execution_action_steps <= 50:
            raise ValueError("execution_action_steps must be in [1, 50]")
        self.chunk_size = execution_action_steps
        self.execution_action_steps = execution_action_steps

    @torch.inference_mode()
    def forward(self, obs: Mapping[str, Any]) -> torch.Tensor:
        processed = self.preprocessor(self._make_policy_observation(obs))
        actions = self.postprocessor(self.policy.predict_action_chunk(processed))
        actions = actions if torch.is_tensor(actions) else torch.as_tensor(actions)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if tuple(actions.shape) != (50, 7) or not torch.isfinite(actions).all():
            raise ValueError(f"Native SmolVLA must return a finite [50,7] chunk, got {tuple(actions.shape)}")
        return (
            actions[: self.execution_action_steps]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clamp(-1.0, 1.0)
        )


def _load_policy(args: argparse.Namespace):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    if args.policy_kind == "baseline":
        import draccus

        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        raw = _read_checkpoint_config(args.checkpoint, "smolvla")
        valid_fields = {field.name for field in fields(SmolVLAConfig) if field.init}
        config = draccus.decode(
            SmolVLAConfig,
            {name: value for name, value in raw.items() if name in valid_fields},
        )
        config.pretrained_path = args.checkpoint
        config.device = args.device
        config.push_to_hub = False
        config.input_features = {}
        config.output_features = {}
        config.chunk_size = 50
        config.n_action_steps = 50
        config.rtc_config = None
        config.compile_model = False
        config.__post_init__()
        policy = make_policy(config, ds_meta=metadata)
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            policy.config, dataset_stats=metadata.stats
        )
        adapter = SmolVLABaselineMikasaPolicy(
            policy,
            preprocessor,
            postprocessor,
            execution_action_steps=args.execution_action_steps,
        )
    else:
        from lerobot.policies.smolvla_ttt.configuration_smolvla_ttt import SmolVLATTTConfig
        from lerobot.policies.smolvla_ttt.processor_smolvla_ttt import (
            make_smolvla_ttt_pre_post_processors,
        )

        raw = _read_checkpoint_config(args.checkpoint, "smolvla_ttt")
        config = SmolVLATTTConfig(
            device=args.device,
            pretrained_path=args.checkpoint,
            push_to_hub=False,
        )
        policy = make_policy(config, ds_meta=metadata)
        preprocessor, postprocessor = make_smolvla_ttt_pre_post_processors(
            policy.config, dataset_stats=metadata.stats
        )
        adapter = SmolVLATTTMikasaPolicy(policy, preprocessor, postprocessor)

    policy.eval()
    return adapter, raw


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_sha256 = getattr(args, "_checkpoint_sha256", None)
    if checkpoint_sha256 is None:
        digest = hashlib.sha256()
        with (args.checkpoint / "model.safetensors").open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
        args._checkpoint_sha256 = checkpoint_sha256
    return {
        "policy_kind": args.policy_kind,
        "checkpoint": str(args.checkpoint),
        "checkpoint_model_sha256": checkpoint_sha256,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "env_id": args.task,
        "n_episodes": args.num_episodes,
        "start_seed": args.start_seed,
        "torch_seed": args.torch_seed,
        "execution_action_steps": (args.execution_action_steps if args.policy_kind == "baseline" else 1),
    }


def _load_completed(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.output.is_file():
        return None
    payload = json.loads(args.output.read_text(encoding="utf-8"))
    if payload.get("evaluation_identity") != _identity(args):
        raise ValueError(f"Existing output has different evaluation identity: {args.output}")
    result = payload.get("results", [{}])[0]
    if len(result.get("successes", [])) != args.num_episodes:
        raise ValueError(f"Existing output is incomplete: {args.output}")
    _write_official_outputs(payload["results"], payload["summary"], args.official_output_dir)
    print(f"Canonical evaluation already complete; keeping {args.output}", flush=True)
    return payload


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from mikasa_robo_suite.vla import benchmarking

    completed = _load_completed(args)
    if completed is not None:
        return completed

    tasks = benchmarking.select_benchmark_tasks(env_ids=[args.task])
    if len(tasks) != 1:
        raise ValueError(f"Expected exactly one official MIKASA task, got {len(tasks)}")
    task = tasks[0]
    policy, raw_config = _load_policy(args)
    policy.set_task(task.language_instruction)

    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    identity = _identity(args)
    episode_records: list[dict[str, Any]] = []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("evaluation_identity") != identity:
            raise ValueError(f"Partial output has different evaluation identity: {partial_path}")
        episode_records = list(partial.get("episodes", []))
        if len(episode_records) > args.num_episodes:
            raise ValueError(f"Partial output contains too many episodes: {partial_path}")

    env = benchmarking.make_benchmark_env(
        task.env_id,
        benchmarking.BenchmarkConfig(
            n_episodes=args.num_episodes,
            start_seed=args.start_seed,
            sim_backend=args.sim_backend,
            include_overlays=False,
        ),
    )
    try:
        for episode_index in range(len(episode_records), args.num_episodes):
            env_seed = args.start_seed + episode_index
            flow_seed = args.torch_seed + episode_index
            _set_torch_seed(flow_seed)
            policy.reset()
            episode, _ = benchmarking.run_episode(
                env,
                policy,
                env_seed,
                collect_video=False,
            )
            record = {
                "index": episode_index,
                "seed": int(episode.seed),
                "torch_seed": flow_seed,
                "success_once": bool(episode.success_once),
                "return": float(episode.episode_return),
                "n_steps": int(episode.n_steps),
            }
            episode_records.append(record)
            _atomic_write_json(
                partial_path,
                {"evaluation_identity": identity, "episodes": episode_records},
            )
            running_sr = float(np.mean([item["success_once"] for item in episode_records]))
            print(
                f"episode={episode_index + 1}/{args.num_episodes} seed={env_seed} "
                f"success={int(record['success_once'])} running_sr={running_sr:.3f} "
                f"steps={record['n_steps']}",
                flush=True,
            )
    finally:
        env.close()

    successes = [bool(item["success_once"]) for item in episode_records]
    returns = [float(item["return"]) for item in episode_records]
    benchmark_commit_fn = getattr(benchmarking, "benchmark_commit", None)
    benchmark_revision = benchmark_commit_fn() if callable(benchmark_commit_fn) else None
    execution_steps = args.execution_action_steps if args.policy_kind == "baseline" else 1
    result = {
        "env_id": task.env_id,
        "split": task.split.title(),
        "memory_type": task.memory_type,
        "data_source": task.data_source,
        "start_seed": args.start_seed,
        "n_episodes": args.num_episodes,
        "torch_seed": args.torch_seed,
        "episode_seeds": [int(item["seed"]) for item in episode_records],
        "episode_torch_seeds": [int(item["torch_seed"]) for item in episode_records],
        "episode_lengths": [int(item["n_steps"]) for item in episode_records],
        "successes": successes,
        "returns": returns,
        "sr": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "control_mode": "pd_ee_delta_pose",
        "obs_mode": "rgb",
        "sim_backend": args.sim_backend,
        "reward_mode": "normalized_dense",
        "benchmark_protocol": "MIKASA-Robo-VLA official runner",
        "benchmark_subset": "single_task",
        "benchmark_commit": benchmark_revision,
        "wrapper_chain": "apply_mikasa_vla_wrappers(include_overlays=False)",
        "action_chunk_size": execution_steps,
        "model_action_horizon": 50,
        "execution_action_steps": execution_steps,
        "execution_cadence": "native_chunk" if execution_steps == 50 else "receding_horizon",
        "model": {
            "checkpoint": str(args.checkpoint),
            "method": "SmolVLA" if args.policy_kind == "baseline" else "SmolVLA-TTT",
            "policy_type": raw_config["type"],
            "ttt_training_stage": raw_config.get("ttt_training_stage"),
            "ttt_enabled": args.policy_kind == "ttt",
            "ttt_persistent_within_episode": args.policy_kind == "ttt",
            "execution_action_steps": execution_steps,
        },
    }
    summary = benchmarking.summarize_task_results([result])
    payload = {"evaluation_identity": identity, "results": [result], "summary": summary}
    _atomic_write_json(args.output, payload)
    _write_official_outputs([result], summary, args.official_output_dir)
    return payload


def _write_official_outputs(
    results: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        _atomic_write_json(output_dir / f"{result['env_id']}.json", dict(result))
    _atomic_write_json(output_dir / "summary.json", dict(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-kind", choices=("baseline", "ttt"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int, default=CANONICAL_NUM_EPISODES)
    parser.add_argument("--start-seed", type=int, default=CANONICAL_START_SEED)
    parser.add_argument("--torch-seed", type=int, default=CANONICAL_TORCH_SEED)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-action-steps", type=int, choices=range(1, 51), default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_episodes != CANONICAL_NUM_EPISODES:
        parser.error(f"official evaluation requires --num-episodes={CANONICAL_NUM_EPISODES}")
    if args.start_seed != CANONICAL_START_SEED:
        parser.error(f"official evaluation requires --start-seed={CANONICAL_START_SEED}")
    if args.torch_seed < 0:
        parser.error("--torch-seed must be non-negative")
    if args.policy_kind == "ttt" and args.execution_action_steps != 50:
        parser.error("--execution-action-steps applies only to the baseline")
    return args


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
