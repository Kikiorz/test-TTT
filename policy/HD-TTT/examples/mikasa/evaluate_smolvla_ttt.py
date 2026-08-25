#!/usr/bin/env python
"""Evaluate a SmolVLA-TTT checkpoint on the official MIKASA-Robo-VLA runner.

The official runner intentionally keeps a minimal ``ChunkPolicy`` interface and
does not know about recurrent policy state.  This adapter therefore resets the
policy immediately before every official ``run_episode`` call while retaining
the benchmark's environment construction, wrappers, seeds, success latch and
JSON schema.  It is a metric adapter, not a second simulator implementation.

Example (run inside the MIKASA Python 3.11 environment)::

    python examples/mikasa/evaluate_smolvla_ttt.py \
      --checkpoint outputs/shell_color/checkpoints/last/pretrained_model \
      --dataset-root /workspace/data_mikasa_robo/data_lerobot/shell_game_color_lamp_touch_vla_v0 \
      --task ShellGameColorLampTouch-VLA-v0 --num-episodes 1 --sim-backend gpu

For the canonical two-task report use ``--num-episodes 50``.  The shuffle
``_Long`` task is officially in the Medium horizon split; the script preserves
that metadata in the output JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


class SmolVLAMikasaPolicy:
    """Bridge MIKASA's packed RGB observation to LeRobot processors."""

    chunk_size = 1

    def __init__(self, policy, preprocessor, postprocessor, *, device: torch.device):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device

    def reset(self) -> None:
        self.policy.reset()

    @staticmethod
    def _to_chw(image: Any) -> torch.Tensor:
        tensor = image if torch.is_tensor(image) else torch.as_tensor(image)
        # Official MIKASA returns [1,H,W,3] for one vectorized environment.
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
            raise KeyError("MIKASA observation must contain 'rgb' and 'proprio'")
        rgb = obs["rgb"]
        rgb = rgb if torch.is_tensor(rgb) else torch.as_tensor(rgb)
        if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 6:
            raise ValueError(f"Expected MIKASA rgb shape [1,H,W,6], got {tuple(rgb.shape)}")
        top = self._to_chw(rgb[..., :3])
        wrist = self._to_chw(rgb[..., 3:])
        proprio = obs["proprio"]
        proprio = proprio if torch.is_tensor(proprio) else torch.as_tensor(proprio)
        if proprio.ndim == 2 and proprio.shape[0] == 1:
            proprio = proprio[0]
        proprio = proprio.to(dtype=torch.float32)
        return {
            "observation.images.top": top,
            "observation.images.wrist": wrist,
            "observation.state": proprio,
            "task": self._task,
        }

    @torch.inference_mode()
    def forward(self, obs: Mapping[str, Any]) -> torch.Tensor:
        # ``run_episode`` supplies the language instruction through the wrapped
        # env.  It is copied by ``set_task`` immediately before each episode.
        raw = self._make_policy_observation(obs)
        processed = self.preprocessor(raw)
        action = self.policy.select_action(processed)
        action = self.postprocessor(action)
        if action.ndim == 2:
            action = action[0]
        # MIKASA's canonical action space is bounded to [-1, 1].  A freshly
        # initialized/partially fine-tuned flow head can briefly leave that
        # range after unnormalization; the official runner expects the policy
        # adapter to enforce the environment contract.
        return action.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)

    def set_task(self, instruction: str) -> None:
        self._task = instruction


def _load_policy(args: argparse.Namespace):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla_ttt.configuration_smolvla_ttt import SmolVLATTTConfig
    from lerobot.policies.smolvla_ttt.processor_smolvla_ttt import make_smolvla_ttt_pre_post_processors

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)

    # ``_restore_checkpoint_model_fields`` deliberately treats a target
    # ``False`` as an explicit HD opt-out.  Evaluation therefore must carry
    # the source checkpoint's HD switches into the requested config; otherwise
    # an HD checkpoint would silently be evaluated as clean TTT.  CLI values
    # override the source flags and make clean-vs-HD ablations reproducible.
    source_config: dict[str, Any] = {}
    checkpoint_config = args.checkpoint / "config.json"
    if checkpoint_config.is_file():
        try:
            source_config = json.loads(checkpoint_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read checkpoint config {checkpoint_config}: {error}") from error
    # Preserve every serialized HD hyperparameter (not just the two switches)
    # so an evaluation checkpoint with non-default attribution/loss weights is
    # evaluated under the same objective it was trained with.
    valid_config_fields = set(SmolVLATTTConfig.__dataclass_fields__)
    hd_kwargs = {
        name: value
        for name, value in source_config.items()
        if name.startswith("hd_") and name in valid_config_fields
    }
    hd_enabled = bool(hd_kwargs.get("hd_ttt_enabled", False))
    if args.hd_ttt_enabled is not None:
        hd_enabled = bool(args.hd_ttt_enabled)
    learned_gate = bool(hd_kwargs.get("hd_learned_write_gate", False))
    if args.hd_learned_write_gate is not None:
        learned_gate = bool(args.hd_learned_write_gate)
    if learned_gate and not hd_enabled:
        raise ValueError("--hd-learned-write-gate requires --hd-ttt-enabled")
    hd_kwargs["hd_ttt_enabled"] = hd_enabled
    hd_kwargs["hd_learned_write_gate"] = learned_gate
    # ``make_policy`` first projects the dataset schema into policy features;
    # constructing ``from_pretrained`` directly would leave input_features
    # empty and silently drop the two MIKASA cameras.
    config = SmolVLATTTConfig(
        device=args.device,
        pretrained_path=Path(args.checkpoint),
        **hd_kwargs,
    )
    policy = make_policy(config, ds_meta=metadata)
    preprocessor, postprocessor = make_smolvla_ttt_pre_post_processors(
        policy.config,
        dataset_stats=metadata.stats,
    )
    policy.eval()
    return SmolVLAMikasaPolicy(
        policy,
        preprocessor,
        postprocessor,
        device=torch.device(args.device),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from mikasa_robo_suite.vla import benchmarking

    tasks = benchmarking.select_benchmark_tasks(env_ids=args.tasks)
    policy = _load_policy(args)
    results: list[dict[str, Any]] = []
    for task in tasks:
        policy.set_task(task.language_instruction)
        env = benchmarking.make_benchmark_env(task.env_id, benchmarking.BenchmarkConfig(
            n_episodes=args.num_episodes,
            start_seed=args.start_seed,
            sim_backend=args.sim_backend,
            include_overlays=False,
        ))
        episodes = []
        try:
            for episode_index in range(args.num_episodes):
                # Official ``run_episode`` resets only the environment.  This
                # explicit line is the causal-memory boundary of HD-TTT.
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
        results.append({
            "env_id": task.env_id,
            "split": task.split.title(),
            "memory_type": task.memory_type,
            "start_seed": args.start_seed,
            "n_episodes": args.num_episodes,
            "successes": successes,
            "returns": returns,
            "sr": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "control_mode": "pd_ee_delta_pose",
            "obs_mode": "rgb",
            "wrapper_chain": "apply_mikasa_vla_wrappers(include_overlays=False)",
            "action_chunk_size": policy.chunk_size,
            "model": {
                "checkpoint": str(args.checkpoint),
                "method": "HD-TTT" if bool(policy.policy.config.hd_ttt_enabled) else "clean-TTT",
                "hd_ttt_enabled": bool(policy.policy.config.hd_ttt_enabled),
                "hd_learned_write_gate": bool(policy.policy.config.hd_learned_write_gate),
            },
            "episode_lengths": [int(ep.n_steps) for ep in episodes],
            "episode_seeds": [int(ep.seed) for ep in episodes],
        })
    summary = benchmarking.summarize_task_results(results)
    payload = {"results": results, "summary": summary}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="mikasa/shell_game")
    parser.add_argument("--task", dest="tasks", action="append", required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--hd-ttt-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable HD-TTT explicitly; default auto-detects the checkpoint config",
    )
    parser.add_argument(
        "--hd-learned-write-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable the deployable learned write gate; default auto-detects the checkpoint config",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
