"""RoboTTT sequence trainer for the repository's encoded-LIBERO adapter.

The optimizer-step budgets, context schedule, sequence action forcing and
TBPTT semantics follow the public RoboTTT paper. Because this repository does
not contain the unpublished GR00T N1.7/Eagle stack or NVIDIA's data mixture,
the adapter cannot reproduce those two system components.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from RoboTTT.policy import LayerFastStates, RoboTTTPolicy
from RoboTTT.training_utils import episode_decisions, load_baseline, sync_gradients, trainable


PAPER_PRETRAIN_STEPS = 30_000
PAPER_POSTTRAIN_STEPS = 20_000
PAPER_POSTTRAIN_CONTEXT = 1_000
PAPER_PRETRAIN_CONTEXT_SCHEDULE = (128, 256, 512, 1_024, 2_048, 4_096, 8_192)


def paper_per_device_batch(stage: str, context_length: int) -> int:
    if stage == "paper_sequence_pretrain":
        return 4 if context_length <= 4_096 else 1
    if stage == "paper_posttrain":
        return 1
    raise ValueError(stage)


def context_length_for_step(step: int, total_steps: int, schedule: tuple[int, ...]) -> int:
    index = min(len(schedule) - 1, step * len(schedule) // max(1, total_steps))
    return schedule[index]


def configure_stage(model: RoboTTTPolicy, stage: str, total_steps: int):
    model.configure_stage(stage, encoded_vl_adapter=True)
    if stage == "paper_sequence_pretrain":
        peak_lr = 2e-5

        def multiplier(step: int) -> float:
            progress = step / max(1, total_steps)
            if progress < 0.05:
                return max(progress / 0.05, 1e-3)
            if progress < 0.9:
                return 1.0
            return max((1.0 - progress) / 0.1, 0.0)

    elif stage == "paper_posttrain":
        peak_lr = 5e-5

        def multiplier(step: int) -> float:
            progress = min(1.0, step / max(1, total_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    else:
        raise ValueError(stage)
    optimizer = torch.optim.AdamW(
        trainable(model.parameters()),
        lr=peak_lr,
        betas=(0.9, 0.95),
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


class SegmentStream:
    """Yield contiguous TBPTT segments while preserving state within a window."""

    def __init__(self, sequences, episode_ids, tbptt, rank, world_size, seed):
        ids = list(episode_ids)
        random.Random(seed).shuffle(ids)
        self.local_ids = ids[rank::world_size]
        if not self.local_ids:
            raise ValueError("fewer training episodes than distributed ranks")
        self.sequences = sequences
        self.tbptt = int(tbptt)
        self.rng = random.Random(seed + 10_007 * rank)
        self.window = None
        self.offset = 0
        self.fast_state: LayerFastStates | None = None
        self.context_length = None

    def next(self, context_length: int):
        if (
            self.window is None
            or self.offset >= len(self.window)
            or self.context_length != context_length
        ):
            episode_id = self.rng.choice(self.local_ids)
            sequence = self.sequences[episode_id]
            window_size = min(len(sequence), context_length)
            start = self.rng.randint(0, len(sequence) - window_size)
            self.window = sequence[start : start + window_size]
            self.offset = 0
            self.fast_state = None
            self.context_length = context_length
        segment = self.window[self.offset : self.offset + self.tbptt]
        self.offset += len(segment)
        return segment, self.fast_state

    def commit(self, state: LayerFastStates) -> None:
        self.fast_state = tuple(
            tuple(value.detach() for value in layer_state) for layer_state in state
        )


def save(path, model, optimizer, scheduler, stage, stage_step, global_step, args, world_size):
    temporary = path.with_suffix(path.suffix + ".partial")
    reported_world_size = 16 if stage == "paper_sequence_pretrain" else 8
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "stage": stage,
            "stage_step": stage_step,
            "global_step": global_step,
            "architecture": "robottt_public_paper_reconstruction_v2",
            "paper_disclosed_components_implemented": {
                "sequence_action_forcing": True,
                "tbptt_fast_state_carry": True,
                "per_layer_ttt": True,
                "register_tokens": 16,
                "per_device_batch_schedule": True,
                "masked_context_without_outer_loss": True,
            },
            "parameter_report": model.parameter_report(),
            "architecture_approximation": bool(args.allow_architecture_approximation),
            "actual_world_size": world_size,
            "paper_reported_world_size": reported_world_size,
            "paper_reported_world_size_matched": world_size == reported_world_size,
            "unpublished_components_not_reproduced": [
                "gr00t_n1_7_eagle_weights",
                "nvidia_pretraining_data_mixture",
                "multi_denoise_fast_state_commit_rule",
                "fast_mlp_hidden_width",
                "optimizer_betas",
                "qkv_bias_convention",
                "inner_lr_parameterization",
                "wsd_phase_fractions",
                "intermediate_context_schedule",
            ],
            "args": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def train_stage(
    model,
    optimizer,
    scheduler,
    streams,
    frame_tokens,
    actions,
    device,
    stage,
    total_steps,
    global_step,
    log_path,
    save_every,
    output,
    args,
    rank,
    world_size,
    outer_loss_mask,
):
    for stage_step in range(1, total_steps + 1):
        context_length = (
            context_length_for_step(
                stage_step - 1, total_steps, PAPER_PRETRAIN_CONTEXT_SCHEDULE
            )
            if stage == "paper_sequence_pretrain"
            else PAPER_POSTTRAIN_CONTEXT
        )
        per_device_batch = paper_per_device_batch(stage, context_length)
        active_streams = streams[:per_device_batch]
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        batch_losses, inner_losses, segment_lengths = [], [], []
        next_states = []
        for stream in active_streams:
            segment, fast_state = stream.next(context_length)
            losses = []
            for current, observation_indices, action_indices, action_mask in segment:
                raw_tokens = frame_tokens[observation_indices].to(
                    device=device, dtype=torch.float32
                ).unsqueeze(0)
                context = model.base_policy.assemble_context(raw_tokens)
                targets = actions[action_indices].to(
                    device=device, dtype=torch.float32
                ).unsqueeze(0)
                mask = torch.tensor(action_mask, device=device, dtype=torch.bool).unsqueeze(0)
                if outer_loss_mask is not None and not bool(outer_loss_mask[current]):
                    # Paper Sec. 3.3: this timestep remains causal context and
                    # updates fast weights, but supplies no imitation target.
                    mask.zero_()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, fast_state, metrics = model.flow_matching_loss_from_context(
                        context, targets, mask, fast_state, create_graph=True
                    )
                losses.append(loss)
                inner_losses.append(metrics["inner_loss"])
            batch_losses.append(torch.stack(losses).mean())
            segment_lengths.append(len(segment))
            next_states.append(fast_state)
        segment_loss = torch.stack(batch_losses).mean()
        segment_loss.backward()
        sync_gradients(model, world_size)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable(model.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        for stream, fast_state in zip(active_streams, next_states):
            stream.commit(fast_state)
        global_step += 1

        values = torch.tensor(
            [float(segment_loss.detach()), float(np.mean(inner_losses))],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values.div_(world_size)
        if rank == 0:
            gates = model.gate_values().detach()
            record = {
                "stage": stage,
                "stage_step": stage_step,
                "global_step": global_step,
                "context_length": context_length,
                "tbptt_segment_length_mean": float(np.mean(segment_lengths)),
                "per_device_batch": per_device_batch,
                "global_batch": per_device_batch * world_size,
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
            if stage_step % save_every == 0 or stage_step == total_steps:
                save(
                    output / f"{stage}-step-{stage_step:06d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    stage,
                    stage_step,
                    global_step,
                    args,
                    world_size,
                )
    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decision-stride", type=int, default=6)
    parser.add_argument("--tbptt", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=PAPER_PRETRAIN_STEPS)
    parser.add_argument("--posttrain-steps", type=int, default=PAPER_POSTTRAIN_STEPS)
    parser.add_argument("--save-every-steps", type=int, default=1_000)
    parser.add_argument("--register-tokens", type=int, default=16)
    parser.add_argument("--fast-hidden-dim", type=int, default=None)
    parser.add_argument("--allow-architecture-approximation", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("both", "pretrain", "posttrain"),
        default="both",
        help="Use separate pretrain/posttrain jobs to reproduce the paper's 16/8-GPU split.",
    )
    parser.add_argument(
        "--resume-robottt",
        default=None,
        help="RoboTTT pretraining checkpoint required by a standalone posttrain job.",
    )
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
    sequences = [
        episode_decisions(
            int(row["start"]),
            int(row["end"]),
            args.decision_stride,
            cache["config"]["obs_steps"],
            cache["config"]["action_horizon"],
        )
        for row in cache["episodes"]
    ]
    episode_ids = cache["train_episode_ids"].tolist()
    model = RoboTTTPolicy(
        load_baseline(args.checkpoint, device),
        num_register_tokens=args.register_tokens,
        fast_hidden_dim=args.fast_hidden_dim,
        strict_paper_action_head=not args.allow_architecture_approximation,
    ).to(device)
    global_step = 0
    if args.resume_robottt is not None:
        resume = torch.load(args.resume_robottt, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=True)
        global_step = int(resume.get("global_step", 0))
    if args.stage == "posttrain" and args.resume_robottt is None:
        raise ValueError("standalone posttrain requires --resume-robottt")
    if world_size > 1:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)

    stages = {
        "both": (
            ("paper_sequence_pretrain", args.pretrain_steps),
            ("paper_posttrain", args.posttrain_steps),
        ),
        "pretrain": (("paper_sequence_pretrain", args.pretrain_steps),),
        "posttrain": (("paper_posttrain", args.posttrain_steps),),
    }[args.stage]
    for stage_index, (stage, total_steps) in enumerate(stages):
        optimizer, scheduler = configure_stage(model, stage, total_steps)
        streams = [
            SegmentStream(
                sequences,
                episode_ids,
                args.tbptt,
                rank,
                world_size,
                args.seed + stage_index * 1_000_003 + stream_index * 97_409,
            )
            for stream_index in range(4)
        ]
        global_step = train_stage(
            model,
            optimizer,
            scheduler,
            streams,
            cache["frame_tokens"],
            cache["actions"],
            device,
            stage,
            total_steps,
            global_step,
            log_path,
            args.save_every_steps,
            output,
            args,
            rank,
            world_size,
            cache.get("outer_loss_mask"),
        )
    if rank == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "architecture": "robottt_public_paper_reconstruction_v2",
                "parameter_report": model.parameter_report(),
                "architecture_approximation": bool(args.allow_architecture_approximation),
                "args": vars(args),
                "global_step": global_step,
            },
            output / "checkpoint-final.pt",
        )
        (output / "TRAINING_COMPLETE").touch()
        print(f"ROBOTTT_TRAINING_COMPLETE={global_step}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
