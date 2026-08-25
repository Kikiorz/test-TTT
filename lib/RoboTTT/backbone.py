"""Self-contained LIBERO backbone adapter for the RoboTTT action head.

This module intentionally lives inside :mod:`RoboTTT`.  It mirrors the tensor
layout of the repository's historical DiT baseline so those checkpoints remain
loadable, but it does not import that package.  The only model-level dependency
outside RoboTTT is NVIDIA Isaac-GR00T's public ``DiT`` implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from gr00t.model.modules.dit import DiT


@dataclass(frozen=True)
class RoboTTTBackboneConfig:
    """Configuration of the benchmark adapter surrounding RoboTTT.

    Paper-faithful construction uses ``num_layers=16``.  Other depths are
    accepted only so an older baseline checkpoint can be loaded for an
    explicitly labelled architecture ablation; :class:`RoboTTTPolicy` enforces
    the 16-layer contract by default.
    """

    obs_steps: int = 3
    action_horizon: int = 8
    action_dim: int = 7
    state_dim: int = 8
    hidden_dim: int = 512
    num_layers: int = 16
    num_heads: int = 8
    inference_steps: int = 4
    timestep_buckets: int = 1000
    dropout: float = 0.1


def _replace_batch_norm(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = min(32, child.num_features)
            while child.num_features % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _replace_batch_norm(child)
    return module


def _resnet18_encoder(*, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return _replace_batch_norm(model)


class LiberoRoboTTTBackbone(nn.Module):
    """LIBERO observation encoder plus the official Isaac-GR00T DiT module.

    This is a benchmark adapter, not the unpublished GR00T N1.7 Eagle VLM.
    Keeping the adapter here makes RoboTTT runnable after every legacy policy
    folder is removed, while preserving the state-dict key layout needed to
    initialize from the historical DiT baseline.
    """

    def __init__(
        self,
        config: RoboTTTBackboneConfig | None = None,
        *,
        pretrained_visual_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or RoboTTTBackboneConfig()
        c = self.config
        if c.hidden_dim % c.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.static_encoder = _resnet18_encoder(pretrained=pretrained_visual_encoder)
        self.wrist_encoder = _resnet18_encoder(pretrained=pretrained_visual_encoder)
        self.image_projector = nn.Linear(512, c.hidden_dim)
        self.state_projector = nn.Sequential(
            nn.Linear(c.state_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(c.action_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.action_dim),
        )

        self.history_embedding = nn.Parameter(torch.zeros(c.obs_steps, c.hidden_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(3, c.hidden_dim))
        self.action_position_embedding = nn.Parameter(torch.zeros(c.action_horizon, c.hidden_dim))
        nn.init.normal_(self.history_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.action_position_embedding, std=0.02)

        self.dit = DiT(
            num_attention_heads=c.num_heads,
            attention_head_dim=c.hidden_dim // c.num_heads,
            output_dim=c.hidden_dim,
            num_layers=c.num_layers,
            dropout=c.dropout,
            attention_bias=True,
            activation_fn="gelu-approximate",
            norm_type="ada_norm",
            norm_elementwise_affine=False,
            max_num_positional_embeddings=max(64, c.action_horizon),
            final_dropout=True,
            positional_embeddings="sinusoidal",
            interleave_self_attention=True,
            cross_attention_dim=c.hidden_dim,
        )

        self.register_buffer("state_mean", torch.zeros(c.state_dim))
        self.register_buffer("state_std", torch.ones(c.state_dim))
        self.register_buffer("action_mean", torch.zeros(c.action_dim))
        self.register_buffer("action_std", torch.ones(c.action_dim))
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        )

    def config_dict(self) -> dict:
        return asdict(self.config)

    @torch.no_grad()
    def set_statistics(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        self.state_mean.copy_(state_mean)
        self.state_std.copy_(state_std.clamp_min(1e-4))
        self.action_mean.copy_(action_mean)
        self.action_std.copy_(action_std.clamp_min(1e-4))

    def encode_frame_tokens(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        static = observations["static_cam"]
        wrist = observations["wrist_cam"]
        state = observations["agent_pos"]
        batch, steps = state.shape[:2]
        if steps < 1:
            raise ValueError("at least one observation step is required")

        static = (static - self.image_mean) / self.image_std
        wrist = (wrist - self.image_mean) / self.image_std
        static_features = self.static_encoder(static.flatten(0, 1)).view(batch, steps, -1)
        wrist_features = self.wrist_encoder(wrist.flatten(0, 1)).view(batch, steps, -1)
        static_features = self.image_projector(static_features)
        wrist_features = self.image_projector(wrist_features)
        state = (state - self.state_mean) / self.state_std
        state_features = self.state_projector(state)
        return torch.stack((static_features, wrist_features, state_features), dim=2)

    def assemble_context(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        expected = (self.config.obs_steps, 3)
        if frame_tokens.ndim != 4 or frame_tokens.shape[1:3] != expected:
            raise ValueError(
                f"expected [B,{expected[0]},3,D] frame tokens, got {tuple(frame_tokens.shape)}"
            )
        frame_tokens = frame_tokens + self.history_embedding[None, :, None, :]
        frame_tokens = frame_tokens + self.modality_embedding[None, None, :, :]
        return frame_tokens.flatten(1, 2)

    def encode_context(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.assemble_context(self.encode_frame_tokens(observations))

    def predict_velocity(
        self,
        normalized_actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        action_tokens = self.action_encoder(normalized_actions)
        action_tokens = action_tokens + self.action_position_embedding.unsqueeze(0)
        hidden = self.dit(
            hidden_states=action_tokens,
            encoder_hidden_states=context,
            timestep=timestep,
        )
        return self.action_decoder(hidden)

    def flow_matching_loss(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_context(observations)
        normalized_actions = (actions - self.action_mean) / self.action_std
        noise = torch.randn_like(normalized_actions)
        beta = torch.distributions.Beta(
            torch.tensor(1.5, device=actions.device),
            torch.tensor(1.0, device=actions.device),
        )
        t = beta.sample((actions.shape[0],)).to(actions.dtype).mul_(0.999)
        noisy_actions = (1.0 - t[:, None, None]) * noise + t[:, None, None] * normalized_actions
        target_velocity = normalized_actions - noise
        timestep = (t * self.config.timestep_buckets).long()
        predicted_velocity = self.predict_velocity(noisy_actions, timestep, context)
        per_element = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        mask = action_mask.unsqueeze(-1).to(per_element.dtype)
        return (per_element * mask).sum() / (mask.sum() * actions.shape[-1]).clamp_min(1.0)

    @torch.no_grad()
    def sample_action(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        context = self.encode_context(observations)
        batch = context.shape[0]
        actions = torch.randn(
            batch,
            self.config.action_horizon,
            self.config.action_dim,
            device=context.device,
            dtype=context.dtype,
        )
        dt = 1.0 / self.config.inference_steps
        for index in range(self.config.inference_steps):
            continuous_t = index / float(self.config.inference_steps)
            timestep = torch.full(
                (batch,),
                int(continuous_t * self.config.timestep_buckets),
                device=context.device,
                dtype=torch.long,
            )
            actions = actions + dt * self.predict_velocity(actions, timestep, context)
        return actions * self.action_std + self.action_mean
