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

from DiT_TTT.robottt_policy import LayerFastStates, PaperRoboTTTPolicy
from DiT_TTT.train_full_episode import episode_decisions, load_baseline, sync_gradients, trainable


def configure_stage(model: PaperRoboTTTPolicy, stage: str, epochs: int):
    model.configure_stage(stage)
    if stage == "paper_sequence_pretrain":
        optimizer = torch.optim.AdamW(
            trainable(model.parameters()), lr=2e-5, betas=(0.9, 0.95), weight_decay=1e-5
        )

        def wsd(epoch: int) -> float:
            progress = epoch / max(1, epochs)
            if progress < 0.05:
                return max(progress / 0.05, 1e-3)
            if progress < 0.9:
                return 1.0
            return max((1.0 - progress) / 0.1, 0.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd)
    elif stage == "paper_posttrain":
        optimizer = torch.optim.AdamW(
            trainable(model.parameters()), lr=5e-5, betas=(0.9, 0.95), weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    else:
        raise ValueError(stage)
    return optimizer, scheduler


def save(path, model, optimizer, scheduler, stage, epoch, update, args):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "update": update,
            "architecture": "paper_robottt_reconstruction_v1",
            "args": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def run_epoch(
    model,
    optimizer,
    episode_ids,
    sequences,
    frame_tokens,
    actions,
    device,
    stage,
    epoch,
    update,
    log_path,
    tbptt,
    rank,
    world_size,
):
    random.shuffle(episode_ids)
    padded = math.ceil(len(episode_ids) / world_size) * world_size
    local_ids = (episode_ids + episode_ids[: padded - len(episode_ids)])[rank:padded:world_size]
    epoch_losses = []
    for position, episode_id in enumerate(local_ids, start=1):
        fast_state: LayerFastStates | None = None
        sequence = sequences[episode_id]
        for segment_start in range(0, len(sequence), tbptt):
            started = time.time()
            optimizer.zero_grad(set_to_none=True)
            losses, inner_losses = [], []
            segment = sequence[segment_start : segment_start + tbptt]
            for obs_indices, action_indices, action_mask in segment:
                raw_tokens = frame_tokens[obs_indices].to(device=device, dtype=torch.float32).unsqueeze(0)
                context = model.base_policy.assemble_context(raw_tokens)
                targets = actions[action_indices].to(device=device, dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor(action_mask, device=device, dtype=torch.bool).unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, fast_state, metrics = model.flow_matching_loss_from_context(
                        context, targets, mask, fast_state, create_graph=True
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
            values = torch.tensor(
                [float(segment_loss.detach()), float(np.mean(inner_losses))],
                device=device,
                dtype=torch.float64,
            )
            if world_size > 1:
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values.div_(world_size)
            epoch_losses.append(float(values[0]))
            gates = model.gate_values().detach()
            if rank == 0:
                record = {
                    "stage": stage,
                    "epoch": epoch,
                    "episode_position": position,
                    "episode_id_rank0": episode_id,
                    "segment": segment_start // tbptt + 1,
                    "segment_decisions": len(segment),
                    "episode_decisions": len(sequence),
                    "update": update,
                    "flow_loss": float(values[0]),
                    "inner_loss": float(values[1]),
                    "gate_mean": float(gates.mean()),
                    "gate_min": float(gates.min()),
                    "gate_max": float(gates.max()),
                    "inner_lr_mean": float(model.inner_lr_values().detach().mean()),
                    "grad_norm": float(torch.as_tensor(grad_norm).detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds": time.time() - started,
                }
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
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--posttrain-epochs", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--register-tokens", type=int, default=16)
    parser.add_argument("--fast-hidden-dim", type=int, default=2048)
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
    sequences = [
        episode_decisions(
            int(row["start"]),
            int(row["end"]),
            args.decision_stride,
            cache["config"]["obs_steps"],
            cache["config"]["action_horizon"],
        )
        for row in episodes
    ]
    episode_ids = cache["train_episode_ids"].tolist()
    model = PaperRoboTTTPolicy(
        load_baseline(args.checkpoint, device),
        num_register_tokens=args.register_tokens,
        fast_hidden_dim=args.fast_hidden_dim,
    ).to(device)
    if world_size > 1:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)

    update = 0
    stages = (
        ("paper_sequence_pretrain", args.pretrain_epochs),
        ("paper_posttrain", args.posttrain_epochs),
    )
    for stage, epochs in stages:
        optimizer, scheduler = configure_stage(model, stage, epochs)
        for epoch in range(1, epochs + 1):
            update, mean_loss = run_epoch(
                model,
                optimizer,
                episode_ids.copy(),
                sequences,
                cache["frame_tokens"],
                cache["actions"],
                device,
                stage,
                epoch,
                update,
                log_path,
                args.tbptt,
                rank,
                world_size,
            )
            scheduler.step()
            if rank == 0:
                summary = {
                    "stage": stage,
                    "epoch_complete": epoch,
                    "mean_flow_loss": mean_loss,
                    "gate_mean": float(model.gate_values().detach().mean()),
                    "update": update,
                }
                print(json.dumps(summary), flush=True)
                if epoch % args.save_every == 0:
                    save(
                        output / f"{stage}-epoch-{epoch:03d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        stage,
                        epoch,
                        update,
                        args,
                    )
        if rank == 0:
            save(
                output / f"{stage}-final.pt",
                model,
                optimizer,
                scheduler,
                stage,
                epochs,
                update,
                args,
            )
    if rank == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "architecture": "paper_robottt_reconstruction_v1",
                "args": vars(args),
                "update": update,
            },
            output / "checkpoint-final.pt",
        )
        (output / "TRAINING_COMPLETE").touch()
        print(f"PAPER_ROBOTTT_TRAINING_COMPLETE={update}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
