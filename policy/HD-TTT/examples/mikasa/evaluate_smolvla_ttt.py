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

Run each task separately with its matching ``--dataset-root``.  The color and
shuffle datasets have different normalization statistics; accepting both in a
single invocation would silently apply the first dataset's processor to the
second task.  This is not the complete 90-task benchmark: the color task is in
the Short/Spatial split, while the shuffle ``_Long`` task is officially in the
Medium/Tracking/MP split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _set_torch_seed(seed: int | None) -> None:
    """Optionally fix flow-noise sampling for paired memory comparisons."""

    if seed is None:
        return
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"torch seed must be non-negative, got {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _credit_ttt_protocol_identity(config: Any) -> dict[str, Any] | None:
    """Return the exact model-method identity required by the V3 coordinator.

    This is provenance only and does not alter policy execution.  Importing
    the protocol object lazily keeps this evaluator importable by the native
    SmolVLA adapter while making the policy module the single source of truth
    for canonical strings.
    """

    if not bool(getattr(config, "credit_ttt_enabled", False)):
        return None
    from lerobot.policies.smolvla_ttt.credit_ttt_v3 import DEFAULT_CREDIT_TTT_PROTOCOL

    protocol = DEFAULT_CREDIT_TTT_PROTOCOL.as_dict()
    identity_fields = (
        "format",
        "protocol",
        "version",
        "pair_schema",
        "intervention",
        "target",
        "state",
        "causal",
    )
    identity = {field: protocol[field] for field in identity_fields}
    # Concrete implementation choices are recorded as extra fields.  The
    # benchmark validates the immutable protocol identity separately and
    # therefore cannot confuse ``intervention_mode`` with the protocol-level
    # content-replacement schema.
    identity["attribution_protocol"] = getattr(config, "hd_attribution_protocol", None)
    identity["intervention_mode"] = getattr(config, "hd_v3_intervention", None)
    identity["writer_mode"] = getattr(config, "ttt_writer_mode", None)
    identity["second_order"] = bool(getattr(config, "ttt_second_order", False))
    return identity


class SmolVLAMikasaPolicy:
    """Bridge MIKASA's packed RGB observation to LeRobot processors."""

    chunk_size = 1

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        *,
        device: torch.device,
        reset_memory_every_step: bool = False,
    ):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.reset_memory_every_step = bool(reset_memory_every_step)

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
        # The official VLA wrapper fixes both cameras at 128x128.  Failing
        # here is preferable to silently resizing a malformed observation and
        # then reporting a score that is not comparable with the benchmark.
        if tuple(rgb.shape[1:3]) != (128, 128):
            raise ValueError(
                "Expected official MIKASA camera resolution [1,128,128,6], "
                f"got {tuple(rgb.shape)}"
            )
        top = self._to_chw(rgb[..., :3])
        wrist = self._to_chw(rgb[..., 3:])
        proprio = obs["proprio"]
        proprio = proprio if torch.is_tensor(proprio) else torch.as_tensor(proprio)
        if proprio.ndim == 2 and proprio.shape[0] == 1:
            proprio = proprio[0]
        proprio = proprio.to(dtype=torch.float32)
        if tuple(proprio.shape) != (7,):
            raise ValueError(
                "Expected official MIKASA proprio shape [1,7] (or [7] after "
                f"batch removal), got {tuple(proprio.shape)}"
            )
        if not torch.isfinite(proprio).all():
            raise ValueError("MIKASA proprio observation contains NaN or Inf")
        return {
            "observation.images.top": top,
            "observation.images.wrist": wrist,
            "observation.state": proprio,
            "task": self._task,
        }

    @torch.inference_mode()
    def forward(self, obs: Mapping[str, Any]) -> torch.Tensor:
        if self.reset_memory_every_step:
            # Diagnostic ablation: retain the same checkpoint, observation,
            # flow denoising and within-step update-then-apply computation,
            # but remove only the recurrent fast-weight state carried between
            # physical environment steps.  The canonical/main evaluation
            # leaves this disabled.
            self.policy.reset()
        # ``run_episode`` supplies the language instruction through the wrapped
        # env.  It is copied by ``set_task`` immediately before each episode.
        raw = self._make_policy_observation(obs)
        processed = self.preprocessor(raw)
        action = self.policy.select_action(processed)
        action = self.postprocessor(action)
        action = action if torch.is_tensor(action) else torch.as_tensor(action)
        if action.ndim not in (1, 2):
            raise ValueError(
                "SmolVLA-TTT select_action must return [7] or [1,7] for MIKASA, "
                f"got {tuple(action.shape)}"
            )
        if action.ndim == 2:
            if tuple(action.shape) != (1, 7):
                raise ValueError(
                    "SmolVLA-TTT select_action must return [1,7] for a single "
                    f"MIKASA environment, got {tuple(action.shape)}"
                )
            action = action[0]
        elif tuple(action.shape) != (7,):
            raise ValueError(
                "SmolVLA-TTT select_action must return a 7D MIKASA action, "
                f"got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise ValueError("SmolVLA-TTT produced a NaN/Inf action")
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

    dataset_repo_id = args.dataset_repo_id or args.dataset_root.name
    metadata = LeRobotDatasetMetadata(dataset_repo_id, root=args.dataset_root)

    # ``_restore_checkpoint_model_fields`` deliberately treats a target
    # ``False`` as an explicit HD opt-out.  Evaluation therefore must carry
    # the source checkpoint's HD switches into the requested config; otherwise
    # an HD checkpoint would silently be evaluated as clean TTT.  CLI values
    # override the source flags and make clean-vs-HD ablations reproducible.
    source_config: dict[str, Any] = {}
    checkpoint_path = args.checkpoint / "config.json"
    if checkpoint_path.is_file():
        config_file = checkpoint_path
    else:
        # A Hub ID is accepted by the policy loader as well as a local path.
        # Resolve its config explicitly so HD switches are not silently
        # replaced by clean-TTT defaults during evaluation.
        try:
            from huggingface_hub import hf_hub_download

            config_file = Path(
                hf_hub_download(repo_id=str(args.checkpoint), filename="config.json")
            )
        except Exception as error:
            raise FileNotFoundError(
                f"Could not resolve config.json for checkpoint {args.checkpoint!s}: {error}"
            ) from error
    try:
        source_config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read checkpoint config {config_file}: {error}") from error
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
    elif args.hd_ttt_enabled is False:
        # An explicit clean ablation should not inherit a deploy-time gate
        # from an HD checkpoint.  Keeping this implicit only for the paired
        # ``--no-hd-ttt-enabled`` case makes the common ablation invocation
        # safe while preserving source-config auto-detection otherwise.
        learned_gate = False
    if learned_gate and not hd_enabled:
        raise ValueError("--hd-learned-write-gate requires --hd-ttt-enabled")
    if args.hd_ttt_enabled is False:
        # A clean-memory evaluation is a structural opt-out, not an HD
        # checkpoint with one boolean flipped.  In particular, v2's effect
        # weight is validated as requiring ``hd_ttt_enabled``; leaving the
        # source value at 1.0 would make the paired clean ablation fail before
        # the policy is even constructed.
        hd_kwargs["hd_effect_weight"] = 0.0
    hd_kwargs["hd_ttt_enabled"] = hd_enabled
    hd_kwargs["hd_learned_write_gate"] = learned_gate
    # ``ttt_writer_mode`` is structural rather than an HD loss switch.  It
    # therefore is not included in ``hd_kwargs`` above, but it must follow the
    # checkpoint during evaluation: a prefix-only checkpoint evaluated with
    # the default suffix writer would silently discard its prefix adapter and
    # report a different policy.  JSON null denotes the legacy suffix mode.
    source_writer_mode = str(source_config.get("ttt_writer_mode") or "suffix")
    if source_writer_mode not in {"suffix", "prefix_only"}:
        raise ValueError(
            "Checkpoint has unsupported ttt_writer_mode="
            f"{source_writer_mode!r}; expected 'suffix' or 'prefix_only'"
        )
    # ``make_policy`` first projects the dataset schema into policy features;
    # constructing ``from_pretrained`` directly would leave input_features
    # empty and silently drop the two MIKASA cameras.
    config = SmolVLATTTConfig(
        device=args.device,
        pretrained_path=Path(args.checkpoint),
        ttt_writer_mode=source_writer_mode,
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
        reset_memory_every_step=args.reset_memory_every_step,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from mikasa_robo_suite.vla import benchmarking

    tasks = benchmarking.select_benchmark_tasks(env_ids=args.tasks)
    if len(tasks) != 1:
        raise ValueError(
            "Evaluate exactly one MIKASA task per invocation.  Each task needs "
            "its own dataset-root/statistics; run this script separately and "
            "merge the JSON summaries afterward."
        )
    benchmark_commit_fn = getattr(benchmarking, "benchmark_commit", None)
    benchmark_revision = benchmark_commit_fn() if callable(benchmark_commit_fn) else None
    benchmark_subset = "single_task"
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
                episode_torch_seed = (
                    None
                    if args.torch_seed is None
                    else int(args.torch_seed) + episode_index
                )
                _set_torch_seed(episode_torch_seed)
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
            "data_source": task.data_source,
            "start_seed": args.start_seed,
            "n_episodes": args.num_episodes,
            "torch_seed": (None if args.torch_seed is None else int(args.torch_seed)),
            "episode_torch_seeds": (
                None
                if args.torch_seed is None
                else [int(args.torch_seed) + i for i in range(args.num_episodes)]
            ),
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
            "reset_memory_every_step": bool(args.reset_memory_every_step),
            # The adapter executes one action at a time so recurrent TTT state
            # is updated at every environment step.  SmolVLA still predicts
            # its native 50-action flow chunk internally.
            "action_chunk_size": policy.chunk_size,
            "model_action_horizon": int(policy.policy.config.chunk_size),
            "execution_action_steps": int(policy.policy.config.n_action_steps),
            "model": {
                "checkpoint": str(args.checkpoint),
                # Keep legacy/clean labels stable, but give the final method
                # an explicit provenance marker.  This is metadata only: the
                # evaluator's action path and environment protocol are
                # unchanged.  The benchmark coordinator rejects an HD-TTT
                # result without this marker instead of relabeling it as
                # CreditTTT.
                "method": (
                    "CreditTTT"
                    if bool(getattr(policy.policy.config, "credit_ttt_enabled", False))
                    else (
                        "HD-TTT"
                        if bool(policy.policy.config.hd_ttt_enabled)
                        else "clean-TTT"
                    )
                ),
                "policy_type": "smolvla_ttt",
                "hd_ttt_enabled": bool(policy.policy.config.hd_ttt_enabled),
                "hd_learned_write_gate": bool(policy.policy.config.hd_learned_write_gate),
                "hd_attribution_protocol": getattr(
                    policy.policy.config, "hd_attribution_protocol", None
                ),
                "ttt_writer_mode": getattr(policy.policy.config, "ttt_writer_mode", None),
                "ttt_second_order": bool(getattr(policy.policy.config, "ttt_second_order", False)),
                "protocol_version": (
                    "creditttt_qh2l_v3"
                    if bool(getattr(policy.policy.config, "credit_ttt_enabled", False))
                    else None
                ),
                "protocol_id": (
                    "credit_ttt_v3"
                    if bool(getattr(policy.policy.config, "credit_ttt_enabled", False))
                    else None
                ),
                # Required by the paper benchmark: a scalar V3 marker is
                # human-readable, while this complete identity object makes
                # legacy/V2 relabeling fail closed during aggregation.
                "credit_ttt_protocol": _credit_ttt_protocol_identity(
                    policy.policy.config
                ),
            },
            "episode_lengths": [int(ep.n_steps) for ep in episodes],
            "episode_seeds": [int(ep.seed) for ep in episodes],
        })
    summary = benchmarking.summarize_task_results(results)
    payload = {"results": results, "summary": summary}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.official_output_dir:
        _write_official_outputs(results, summary, args.official_output_dir)
    return payload


def _write_official_outputs(
    results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Write native MIKASA files while retaining the adapter envelope."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        env_id = str(result["env_id"])
        (output_dir / f"{env_id}.json").write_text(
            json.dumps(dict(result), indent=2) + "\n", encoding="utf-8"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(dict(summary), indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="LeRobot repo id; defaults to the local dataset directory name",
    )
    parser.add_argument("--task", dest="tasks", action="append", required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument(
        "--torch-seed",
        type=int,
        default=None,
        help=(
            "Optional base seed for per-episode SmolVLA flow-noise sampling; "
            "episode i uses torch-seed+i. Default keeps native stochastic inference."
        ),
    )
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
    parser.add_argument(
        "--reset-memory-every-step",
        action="store_true",
        help=(
            "Diagnostic only: clear recurrent fast weights before every environment step. "
            "The default keeps canonical episode-persistent memory."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--official-output-dir",
        type=Path,
        default=None,
        help=(
            "Also write native MIKASA <ENV_ID>.json and summary.json files to "
            "this directory; --output keeps the adapter envelope."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
