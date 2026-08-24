from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# The last tensor stores the next RoPE token position. It is recurrent state,
# but unlike the four fast-model tensors it is not differentiated.
FastState = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class RoboTTTKVBLayer(nn.Module):
    """Paper-aligned TTT-KVB layer for one DiT attention layer.

    The fast model is a two-layer GeLU MLP. One gradient step associates K with
    V, then the freshly updated fast model is applied to Q. Learned slow
    parameters are Q/K/V projections, W0, the inner-rate multiplier and a
    channel-wise tanh gate.
    """

    def __init__(
        self,
        dim: int,
        fast_hidden_dim: Optional[int] = None,
        *,
        base_inner_lr: float = 0.1,
        init_gate_alpha: float = 0.001,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("RoboTTT RoPE requires an even token dimension")
        self.dim = int(dim)
        self.fast_hidden_dim = int(fast_hidden_dim or 4 * dim)
        self.base_inner_lr = float(base_inner_lr)
        self.rope_theta = float(rope_theta)

        self.input_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.w1 = nn.Parameter(torch.empty(dim, self.fast_hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.fast_hidden_dim))
        self.w2 = nn.Parameter(torch.empty(self.fast_hidden_dim, dim))
        self.b2 = nn.Parameter(torch.zeros(dim))
        self.inner_lr_log_multiplier = nn.Parameter(torch.zeros(()))

        # The paper learns alpha in R^d and applies tanh(alpha) channel-wise.
        self.gate_alpha = nn.Parameter(torch.full((dim,), float(init_gate_alpha)))
        self.register_buffer("fixed_gate", torch.tensor(float("nan")), persistent=True)

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def set_fixed_gate(self, value: Optional[float]) -> None:
        with torch.no_grad():
            if value is None:
                self.fixed_gate.fill_(float("nan"))
            else:
                if not -1.0 < float(value) < 1.0:
                    raise ValueError("fixed gate must lie strictly between -1 and 1")
                self.fixed_gate.fill_(float(value))

    def gate(self) -> torch.Tensor:
        if torch.isnan(self.fixed_gate):
            return torch.tanh(self.gate_alpha)
        return self.fixed_gate.to(self.gate_alpha).expand_as(self.gate_alpha)

    def positive_inner_lr(self) -> torch.Tensor:
        # Appendix A.1: learned on top of a constant base rate of 0.1.
        return self.base_inner_lr * torch.exp(self.inner_lr_log_multiplier)

    def initial_state(self, batch_size: int, device: torch.device) -> FastState:
        return (
            self.w1.unsqueeze(0).expand(batch_size, -1, -1),
            self.b1.unsqueeze(0).expand(batch_size, -1),
            self.w2.unsqueeze(0).expand(batch_size, -1, -1),
            self.b2.unsqueeze(0).expand(batch_size, -1),
            torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    @staticmethod
    def detach_state(state: FastState) -> FastState:
        return tuple(value.detach() for value in state)  # type: ignore[return-value]

    @staticmethod
    def _apply_fast(x: torch.Tensor, weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
        w1, b1, w2, b2 = weights
        hidden = torch.einsum("bld,bdh->blh", x, w1) + b1[:, None]
        hidden = F.gelu(hidden)
        return torch.einsum("blh,bhd->bld", hidden, w2) + b2[:, None]

    def _rope(self, x: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        _, length, dim = x.shape
        positions = start[:, None] + torch.arange(length, device=x.device)[None]
        frequencies = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
        )
        angles = positions.to(torch.float32)[..., None] * frequencies[None, None]
        cos = angles.cos().to(x.dtype)
        sin = angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(
        self,
        attention_tokens: torch.Tensor,
        state: Optional[FastState],
        *,
        create_graph: bool,
        update_fast: bool = True,
    ) -> tuple[torch.Tensor, FastState, torch.Tensor]:
        """Return TTT output and recurrent state for the next robot step."""
        if attention_tokens.ndim != 3 or attention_tokens.shape[-1] != self.dim:
            raise ValueError(f"expected [B,L,{self.dim}], got {tuple(attention_tokens.shape)}")
        batch, token_count, _ = attention_tokens.shape
        if state is None:
            state = self.initial_state(batch, attention_tokens.device)
        weights = state[:4]
        position = state[4]
        if create_graph and not any(weight.requires_grad for weight in weights):
            bases = (self.w1, self.b1, self.w2, self.b2)
            weights = tuple(
                value + 0.0 * base.unsqueeze(0) for value, base in zip(weights, bases)
            )

        normalized = self.input_norm(attention_tokens)
        queries = self._rope(self.q_proj(normalized), position)
        keys = self._rope(self.k_proj(normalized), position)
        values = self.v_proj(normalized)

        if update_fast:
            with torch.enable_grad():
                if not create_graph:
                    weights = tuple(value.detach().requires_grad_(True) for value in weights)
                prediction = self._apply_fast(keys, weights)
                per_sample_loss = (prediction - values).square().mean(dim=(1, 2))
                gradients = torch.autograd.grad(
                    per_sample_loss.sum(),
                    weights,
                    create_graph=create_graph,
                    retain_graph=create_graph,
                )
                updated_weights = tuple(
                    weight - self.positive_inner_lr() * gradient
                    for weight, gradient in zip(weights, gradients)
                )
            next_position = position + token_count
        else:
            prediction = self._apply_fast(keys, weights)
            per_sample_loss = (prediction - values).square().mean(dim=(1, 2))
            updated_weights = weights
            next_position = position

        output = self._apply_fast(queries, updated_weights)
        next_state: FastState = (*updated_weights, next_position)
        if not create_graph:
            output = output.detach()
            next_state = self.detach_state(next_state)
        return output, next_state, per_sample_loss.detach()
