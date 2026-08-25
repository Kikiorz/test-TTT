from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


FastState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class CausalFastMLPMemory(nn.Module):
    """RoboTTT-style KVB fast MLP with strictly causal read-before-write semantics."""

    def __init__(
        self,
        obs_dim: int,
        condition_dim: int,
        memory_dim: int = 256,
        fast_hidden_dim: int = 256,
        init_inner_lr: float = 0.05,
        init_gate: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.condition_dim = int(condition_dim)
        self.memory_dim = int(memory_dim)
        self.fast_hidden_dim = int(fast_hidden_dim)
        self.inner_lr_scale = float(init_inner_lr)

        self.obs_proj = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, memory_dim),
            nn.GELU(),
            nn.LayerNorm(memory_dim),
        )
        self.q_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, memory_dim, bias=False)

        self.w1 = nn.Parameter(torch.empty(memory_dim, fast_hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(fast_hidden_dim))
        self.w2 = nn.Parameter(torch.empty(fast_hidden_dim, memory_dim))
        self.b2 = nn.Parameter(torch.zeros(memory_dim))
        self.inner_lr_raw = nn.Parameter(torch.zeros(()))

        self.out_proj = nn.Linear(memory_dim, condition_dim)
        self.gate_raw = nn.Parameter(torch.tensor(float(init_gate)).atanh())
        self.register_buffer("fixed_gate", torch.tensor(float("nan")), persistent=True)

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        # Exact baseline equivalence at initialization even when the gate is forced open.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def set_fixed_gate(self, value: Optional[float]) -> None:
        with torch.no_grad():
            if value is None:
                self.fixed_gate.fill_(float("nan"))
            else:
                if not 0.0 <= float(value) < 1.0:
                    raise ValueError("fixed gate must be in [0, 1)")
                self.fixed_gate.fill_(float(value))

    def gate(self) -> torch.Tensor:
        if torch.isnan(self.fixed_gate):
            return torch.tanh(self.gate_raw)
        return self.fixed_gate.to(dtype=self.gate_raw.dtype)

    def positive_inner_lr(self) -> torch.Tensor:
        return self.inner_lr_scale * F.softplus(self.inner_lr_raw) / math.log(2.0)

    def initial_state(self, batch_size: int) -> FastState:
        return (
            self.w1.unsqueeze(0).expand(batch_size, -1, -1),
            self.b1.unsqueeze(0).expand(batch_size, -1),
            self.w2.unsqueeze(0).expand(batch_size, -1, -1),
            self.b2.unsqueeze(0).expand(batch_size, -1),
        )

    @staticmethod
    def detach_state(state: FastState) -> FastState:
        return tuple(value.detach() for value in state)  # type: ignore[return-value]

    @staticmethod
    def apply_fast(x: torch.Tensor, state: FastState) -> torch.Tensor:
        w1, b1, w2, b2 = state
        hidden = torch.einsum("bnd,bdf->bnf", x, w1) + b1[:, None]
        hidden = F.gelu(hidden)
        return torch.einsum("bnf,bfd->bnd", hidden, w2) + b2[:, None]

    def forward(
        self,
        newest_observation_feature: torch.Tensor,
        global_condition: torch.Tensor,
        state: Optional[FastState] = None,
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, FastState, torch.Tensor]:
        """Read history for this decision, then write the current observation for the next one."""
        batch_size = global_condition.shape[0]
        if state is None:
            state = self.initial_state(batch_size)
        elif create_graph:
            # Keeps learned initial fast weights in every DDP graph after a TBPTT detach.
            bases = (self.w1, self.b1, self.w2, self.b2)
            state = tuple(value + 0.0 * base.unsqueeze(0) for value, base in zip(state, bases))  # type: ignore[assignment]

        token = self.obs_proj(newest_observation_feature).unsqueeze(1)

        # Causal read: the action at time t sees only W_{t-1}.
        read = self.apply_fast(self.q_proj(token), state).squeeze(1)
        residual = self.out_proj(read)
        adapted_condition = global_condition + self.gate() * residual

        # Self-supervised write uses observations only. Expert/clean actions never enter K, V or W.
        with torch.enable_grad():
            if not create_graph:
                state = tuple(value.detach().requires_grad_(True) for value in state)  # type: ignore[assignment]
            keys = self.k_proj(token)
            values = self.v_proj(token)
            prediction = self.apply_fast(keys, state)
            per_sample_inner_loss = (prediction - values).square().mean(dim=(1, 2))
            gradients = torch.autograd.grad(
                per_sample_inner_loss.sum(),
                state,
                create_graph=create_graph,
                retain_graph=create_graph,
            )
            next_state = tuple(
                weight - self.positive_inner_lr() * gradient
                for weight, gradient in zip(state, gradients)
            )
        if not create_graph:
            next_state = self.detach_state(next_state)  # type: ignore[arg-type]
        return adapted_condition, next_state, per_sample_inner_loss.detach()
