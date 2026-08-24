"""Small-benchmark data utilities owned by the standalone RoboTTT package.

These utilities adapt cached LIBERO features to the RoboTTT action head. They
are not part of the paper architecture and are intentionally isolated here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from RoboTTT.backbone import LiberoRoboTTTBackbone, RoboTTTBackboneConfig


def load_baseline(path: str, device: torch.device) -> LiberoRoboTTTBackbone:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = LiberoRoboTTTBackbone(RoboTTTBackboneConfig(**payload["config"]))
    model.load_state_dict(payload["ema_model"], strict=True)
    return model.to(device).eval()


def episode_decisions(
    start: int,
    end: int,
    stride: int,
    obs_steps: int,
    horizon: int,
) -> list[tuple[list[int], list[int], list[bool]]]:
    decisions = []
    for current in range(start, end, stride):
        observation_indices = [
            min(end - 1, max(start, current - (obs_steps - 1) + offset))
            for offset in range(obs_steps)
        ]
        action_indices = [min(end - 1, current + offset) for offset in range(horizon)]
        action_mask = [current + offset < end for offset in range(horizon)]
        decisions.append((observation_indices, action_indices, action_mask))
    return decisions


def trainable(parameters):
    return [parameter for parameter in parameters if parameter.requires_grad]


def sync_gradients(model: torch.nn.Module, world_size: int) -> None:
    if world_size == 1:
        return
    for parameter in trainable(model.parameters()):
        if parameter.grad is None:
            raise RuntimeError(
                "missing gradient for a trainable parameter; the paper trainer "
                f"does not silently skip disconnected tensors: {tuple(parameter.shape)}"
            )
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)
