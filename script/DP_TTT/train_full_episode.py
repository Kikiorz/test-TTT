from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
import torch.distributed as dist

from DP_TTT.fast_memory import FastState
from DP_TTT.full_episode_policy import FullEpisodeDPTTTPolicy


def load_baseline(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=None)
    workspace.load_payload(payload)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    return policy.to(device).eval()


def episode_decisions(start, end, stride, obs_steps, horizon):
    decisions = []
    for current in range(start, end, stride):
        obs_indices = [min(end - 1, max(start, current - (obs_steps - 1) + offset))
                       for offset in range(obs_steps)]
        action_start = current - (obs_steps - 1)
        action_indices = [min(end - 1, max(start, action_start + offset))
                          for offset in range(horizon)]
        decisions.append((obs_indices, action_indices))
    return decisions


def trainable(parameters):
    return [parameter for parameter in parameters if parameter.requires_grad]


def sync_gradients(model, world_size):
    if world_size == 1:
        return
    for parameter in trainable(model.parameters()):
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {tuple(parameter.shape)}")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def configure_stage(model, stage):
    if stage == "fixed_gate_ttt":
        model.configure_stage("forced_gate_ttt")
        model.set_fixed_gate(0.5)
        model.ttt.gate_raw.requires_grad_(False)
        groups = [{"params": trainable(model.ttt.parameters()), "lr": 3e-4}]
    elif stage == "joint":
        model.configure_stage("joint")
        with torch.no_grad():
            model.ttt.gate_raw.fill_(math.atanh(0.5))
        model.set_fixed_gate(None)
        groups = [
            {"params": trainable(model.ttt.parameters()), "lr": 2e-4},
            {"params": trainable(model.base_policy.model.parameters()), "lr": 1e-5},
        ]
    elif stage == "ttt_calibration":
        model.configure_stage("ttt_gate_only")
        model.set_fixed_gate(None)
        groups = [{"params": trainable(model.ttt.parameters()), "lr": 1e-4}]
    else:
        raise ValueError(stage)
    return torch.optim.AdamW(groups, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)


def save(path, model, optimizer, stage, epoch, update, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stage": stage,
        "epoch": epoch,
        "update": update,
        "args": vars(args),
    }, path)


def run_epoch(model, optimizer, episode_ids, sequences, frame_features, normalized_actions,
              device, stage, epoch, update, log_path, tbptt, rank, world_size):
    random.shuffle(episode_ids)
    padded = math.ceil(len(episode_ids) / world_size) * world_size
    local_ids = (episode_ids + episode_ids[: padded - len(episode_ids)])[rank:padded:world_size]
    losses_epoch = []
    for position, episode_id in enumerate(local_ids, start=1):
        state: FastState | None = None
        sequence = sequences[episode_id]
        for segment_start in range(0, len(sequence), tbptt):
            started = time.time()
            optimizer.zero_grad(set_to_none=True)
            losses, inner_losses = [], []
            segment = sequence[segment_start:segment_start + tbptt]
            for obs_indices, action_indices in segment:
                obs = frame_features[obs_indices].to(device=device, dtype=torch.float32).unsqueeze(0)
                actions = normalized_actions[action_indices].to(device=device, dtype=torch.float32).unsqueeze(0)
                loss, state, metrics = model.compute_loss_from_cached_features(
                    obs, actions, state, create_graph=True
                )
                losses.append(loss)
                inner_losses.append(metrics["inner_loss"])
            segment_loss = torch.stack(losses).mean()
            segment_loss.backward()
            sync_gradients(model, world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable(model.parameters()), 1.0)
            optimizer.step()
            state = tuple(value.detach() for value in state)
            update += 1
            values = torch.tensor(
                [float(segment_loss.detach()), float(np.mean(inner_losses))],
                device=device,
                dtype=torch.float64,
            )
            if world_size > 1:
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values.div_(world_size)
            losses_epoch.append(float(values[0]))
            record = {
                "stage": stage,
                "epoch": epoch,
                "episode_position": position,
                "episode_id_rank0": episode_id,
                "segment": segment_start // tbptt + 1,
                "segment_decisions": len(segment),
                "episode_decisions": len(sequence),
                "update": update,
                "loss": float(values[0]),
                "inner_loss": float(values[1]),
                "gate": float(model.ttt.gate().detach()),
                "inner_lr": float(model.ttt.positive_inner_lr().detach()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach()),
                "seconds": time.time() - started,
            }
            if rank == 0:
                print(json.dumps(record), flush=True)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
    totals = torch.tensor([sum(losses_epoch), len(losses_epoch)], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return update, float(totals[0] / totals[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decision-stride", type=int, default=6)
    parser.add_argument("--tbptt", type=int, default=64)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage3-epochs", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    torch.cuda.set_device(device)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    ends = cache["episode_ends"].tolist()
    starts = [0] + ends[:-1]
    sequences = [episode_decisions(start, end, args.decision_stride,
                                  cache["n_obs_steps"], cache["horizon"])
                 for start, end in zip(starts, ends)]
    episode_ids = cache["train_episode_ids"].tolist()

    model = FullEpisodeDPTTTPolicy(load_baseline(args.checkpoint, device)).to(device)
    if world_size > 1:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)
    model.base_policy.obs_encoder.requires_grad_(False)
    model.base_policy.obs_encoder.eval()
    update = 0
    stages = (
        ("fixed_gate_ttt", args.stage1_epochs),
        ("joint", args.stage2_epochs),
        ("ttt_calibration", args.stage3_epochs),
    )
    for stage, epochs in stages:
        optimizer = configure_stage(model, stage)
        for epoch in range(1, epochs + 1):
            update, mean_loss = run_epoch(
                model, optimizer, episode_ids.copy(), sequences,
                cache["frame_features"], cache["normalized_actions"],
                device, stage, epoch, update, log_path, args.tbptt, rank, world_size,
            )
            if rank == 0:
                summary = {"stage": stage, "epoch_complete": epoch, "mean_loss": mean_loss,
                           "gate": float(model.ttt.gate().detach()), "update": update}
                print(json.dumps(summary), flush=True)
                if epoch % args.save_every == 0:
                    save(output / f"{stage}-epoch-{epoch:03d}.pt", model, optimizer,
                         stage, epoch, update, args)
        if rank == 0:
            save(output / f"{stage}-final.pt", model, optimizer, stage, epochs, update, args)
    if rank == 0:
        torch.save({"model": model.state_dict(), "args": vars(args), "update": update},
                   output / "checkpoint-final.pt")
        (output / "TRAINING_COMPLETE").touch()
        print(f"DP_TTT_TRAINING_COMPLETE={update}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

