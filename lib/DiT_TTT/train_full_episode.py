from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from DiT.model import DiTConfig, LiberoGR00TDiT
from DiT_TTT.policy import DiTTTPolicy, LayerFastStates


def load_baseline(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = LiberoGR00TDiT(DiTConfig(**payload["config"]))
    model.load_state_dict(payload["ema_model"], strict=True)
    return model.to(device).eval()


def episode_decisions(start, end, stride, obs_steps, horizon):
    decisions = []
    for current in range(start, end, stride):
        obs_indices = [min(end - 1, max(start, current - (obs_steps - 1) + offset))
                       for offset in range(obs_steps)]
        action_indices = [min(end - 1, current + offset) for offset in range(horizon)]
        action_mask = [current + offset < end for offset in range(horizon)]
        decisions.append((obs_indices, action_indices, action_mask))
    return decisions


def trainable(parameters):
    return [parameter for parameter in parameters if parameter.requires_grad]


def sync_gradients(model, world_size):
    if world_size == 1:
        return
    for parameter in trainable(model.parameters()):
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for trainable parameter {tuple(parameter.shape)}")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def configure_stage(model, stage):
    model.configure_stage(stage)
    if stage == "fixed_gate_ttt":
        groups = [{"params": trainable(model.memories.parameters()), "lr": 3e-4}]
    elif stage == "joint":
        with torch.no_grad():
            for memory in model.memories:
                memory.gate_raw.fill_(math.atanh(0.5))
        groups = [
            {"params": trainable(model.memories.parameters()), "lr": 2e-4},
            {"params": trainable(model.base_policy.action_encoder.parameters()), "lr": 1e-5},
            {"params": trainable(model.base_policy.action_decoder.parameters()), "lr": 1e-5},
            {"params": trainable(model.base_policy.dit.parameters()), "lr": 1e-5},
        ]
    elif stage == "ttt_calibration":
        groups = [{"params": trainable(model.memories.parameters()), "lr": 1e-4}]
    else:
        raise ValueError(stage)
    return torch.optim.AdamW(groups, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)


def save(path, model, optimizer, stage, epoch, update, args):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "stage": stage, "epoch": epoch, "update": update, "args": vars(args)}, temporary)
    temporary.replace(path)


def run_epoch(model, optimizer, episode_ids, sequences, frame_tokens, actions,
              device, stage, epoch, update, log_path, tbptt, rank, world_size):
    random.shuffle(episode_ids)
    padded = math.ceil(len(episode_ids) / world_size) * world_size
    local_ids = (episode_ids + episode_ids[:padded - len(episode_ids)])[rank:padded:world_size]
    epoch_losses = []
    for position, episode_id in enumerate(local_ids, start=1):
        fast_state: LayerFastStates | None = None
        sequence = sequences[episode_id]
        for segment_start in range(0, len(sequence), tbptt):
            started = time.time()
            optimizer.zero_grad(set_to_none=True)
            losses, inner_losses = [], []
            segment = sequence[segment_start:segment_start + tbptt]
            for obs_indices, action_indices, action_mask in segment:
                raw_tokens = frame_tokens[obs_indices].to(device=device, dtype=torch.float32).unsqueeze(0)
                context = model.base_policy.assemble_context(raw_tokens)
                target_actions = actions[action_indices].to(device=device, dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor(action_mask, device=device, dtype=torch.bool).unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, fast_state, metrics = model.flow_matching_loss_from_context(
                        context, target_actions, mask, fast_state, create_graph=True,
                        activation_checkpointing=True,
                    )
                losses.append(loss)
                inner_losses.append(metrics["inner_loss"])
            segment_loss = torch.stack(losses).mean()
            segment_loss.backward()
            sync_gradients(model, world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable(model.parameters()), 1.0)
            optimizer.step()
            fast_state = tuple(
                tuple(value.detach() for value in layer_state) for layer_state in fast_state
            )
            update += 1
            values = torch.tensor([float(segment_loss.detach()), float(np.mean(inner_losses))],
                                  device=device, dtype=torch.float64)
            if world_size > 1:
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values.div_(world_size)
            epoch_losses.append(float(values[0]))
            gates = model.gate_values().detach()
            record = {
                "stage": stage, "epoch": epoch, "episode_position": position,
                "episode_id_rank0": episode_id, "segment": segment_start // tbptt + 1,
                "segment_decisions": len(segment), "episode_decisions": len(sequence),
                "update": update, "loss": float(values[0]), "inner_loss": float(values[1]),
                "gate_mean": float(gates.mean()), "gate_min": float(gates.min()),
                "gate_max": float(gates.max()), "grad_norm": float(torch.as_tensor(grad_norm).detach()),
                "seconds": time.time() - started,
            }
            if rank == 0:
                print(json.dumps(record), flush=True)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
    totals = torch.tensor([sum(epoch_losses), len(epoch_losses)], device=device, dtype=torch.float64)
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
    parser.add_argument("--save-every", type=int, default=25)
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
    episodes = cache["episodes"]
    sequences = [episode_decisions(int(row["start"]), int(row["end"]), args.decision_stride,
                                  cache["config"]["obs_steps"], cache["config"]["action_horizon"])
                 for row in episodes]
    episode_ids = cache["train_episode_ids"].tolist()
    model = DiTTTPolicy(load_baseline(args.checkpoint, device)).to(device)
    if world_size > 1:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)

    update = 0
    stages = (("fixed_gate_ttt", args.stage1_epochs),
              ("joint", args.stage2_epochs),
              ("ttt_calibration", args.stage3_epochs))
    for stage, epochs in stages:
        optimizer = configure_stage(model, stage)
        for epoch in range(1, epochs + 1):
            update, mean_loss = run_epoch(
                model, optimizer, episode_ids.copy(), sequences, cache["frame_tokens"], cache["actions"],
                device, stage, epoch, update, log_path, args.tbptt, rank, world_size,
            )
            if rank == 0:
                gates = model.gate_values().detach()
                summary = {"stage": stage, "epoch_complete": epoch, "mean_loss": mean_loss,
                           "gate_mean": float(gates.mean()), "gate_min": float(gates.min()),
                           "gate_max": float(gates.max()), "update": update}
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
        print(f"DIT_TTT_TRAINING_COMPLETE={update}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

