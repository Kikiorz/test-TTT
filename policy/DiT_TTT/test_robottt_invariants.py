from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gr00t.model.modules.dit import DiT

from DiT_TTT.robottt_policy import PaperRoboTTTPolicy


@dataclass(frozen=True)
class ToyConfig:
    obs_steps: int = 3
    action_horizon: int = 4
    action_dim: int = 3
    hidden_dim: int = 32
    num_layers: int = 2
    inference_steps: int = 3
    timestep_buckets: int = 1000


class ToyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = ToyConfig()
        c = self.config
        self.action_encoder = nn.Linear(c.action_dim, c.hidden_dim)
        self.action_decoder = nn.Linear(c.hidden_dim, c.action_dim)
        self.action_position_embedding = nn.Parameter(
            torch.randn(c.action_horizon, c.hidden_dim) * 0.02
        )
        self.dit = DiT(
            num_attention_heads=4,
            attention_head_dim=8,
            output_dim=c.hidden_dim,
            num_layers=c.num_layers,
            dropout=0.0,
            norm_type="ada_norm",
            norm_elementwise_affine=False,
            max_num_positional_embeddings=64,
            final_dropout=False,
            positional_embeddings="sinusoidal",
            interleave_self_attention=True,
            cross_attention_dim=c.hidden_dim,
        )
        self.register_buffer("action_mean", torch.zeros(c.action_dim))
        self.register_buffer("action_std", torch.ones(c.action_dim))


def main() -> None:
    torch.manual_seed(7)
    base = ToyBase().eval()
    model = PaperRoboTTTPolicy(
        base,
        num_register_tokens=4,
        fast_hidden_dim=64,
    ).eval()
    batch = 2
    context = torch.randn(batch, base.config.obs_steps * 3, base.config.hidden_dim)
    actions = torch.randn(batch, base.config.action_horizon, base.config.action_dim)
    timestep = torch.tensor([17, 503])

    with torch.no_grad():
        disabled, _, _ = model._dit_forward(
            actions,
            timestep,
            context,
            None,
            create_graph=False,
            update_fast=False,
            enable_ttt=False,
        )
    model.set_fixed_gate(0.0)
    with torch.no_grad():
        gate0, next_state, _ = model._dit_forward(
            actions,
            timestep,
            context,
            None,
            create_graph=False,
            update_fast=True,
            enable_ttt=True,
        )
    error = float((disabled - gate0).abs().max())
    if error != 0.0:
        raise AssertionError(f"gate0 changed the matched registered backbone: {error}")
    if len(next_state) != base.config.num_layers:
        raise AssertionError("there must be exactly one TTT state per DiT layer")
    tokens_per_step = model.num_register_tokens + 1 + base.config.action_horizon
    if any(
        not torch.equal(layer_state[4], torch.full((batch,), tokens_per_step))
        for layer_state in next_state
    ):
        raise AssertionError("each layer must advance one current-step token group")

    # All denoising evaluations reuse W_(t-1); committing the final candidate
    # advances recurrent time once, not inference_steps times.
    committed = None
    for denoise_step in range(base.config.inference_steps):
        _, candidate, _ = model._dit_forward(
            actions,
            torch.full((batch,), denoise_step),
            context,
            None,
            create_graph=False,
            update_fast=True,
        )
        committed = candidate
    assert committed is not None
    if any(
        not torch.equal(layer_state[4], torch.full((batch,), tokens_per_step))
        for layer_state in committed
    ):
        raise AssertionError("denoising advanced recurrent time more than once")

    # Smoke-test the gradients-of-gradients path used to meta-learn Q/K/V and W0.
    model.set_fixed_gate(None)
    mask = torch.ones(batch, base.config.action_horizon, dtype=torch.bool)
    loss, _, metrics = model.flow_matching_loss_from_context(
        context, actions, mask, None, create_graph=True
    )
    loss.backward()
    required_gradients = {
        "register_tokens": model.register_tokens.grad,
        "q_proj": model.ttt_layers[0].q_proj.weight.grad,
        "k_proj": model.ttt_layers[0].k_proj.weight.grad,
        "v_proj": model.ttt_layers[0].v_proj.weight.grad,
        "w0": model.ttt_layers[0].w1.grad,
        "gate_alpha": model.ttt_layers[0].gate_alpha.grad,
    }
    missing = [name for name, gradient in required_gradients.items() if gradient is None]
    nonfinite = [
        name for name, gradient in required_gradients.items()
        if gradient is not None and not torch.isfinite(gradient).all()
    ]
    if missing or nonfinite:
        raise AssertionError(f"bad outer gradients: missing={missing}, nonfinite={nonfinite}")

    print(
        {
            "gate0_max_abs_error": error,
            "ttt_layers": len(next_state),
            "register_tokens": model.num_register_tokens,
            "state_tokens": 1,
            "action_tokens": base.config.action_horizon,
            "recurrent_commits_per_decision": 1,
            "vl_tokens_enter_ttt_directly": False,
            "outer_flow_loss": float(loss.detach()),
            "inner_loss": metrics["inner_loss"],
            "second_order_gradients_finite": True,
        }
    )


if __name__ == "__main__":
    main()
