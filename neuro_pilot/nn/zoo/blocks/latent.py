"""LatentDrive V-JEPA video tubelet and trajectory planning blocks."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.common import MLP

ViTState = Tuple[torch.Tensor, Tuple[int, int, int]]


@BLOCK_REGISTRY.register("TubeletEmbed")
class TubeletEmbed(nn.Module):
    """3D tubelet patch embedding returning `(tokens, grid)` for RoPEViTBlock."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 1024,
        tubelet_size: int = 2,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        kernel = (tubelet_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=kernel)

    def forward(self, video: torch.Tensor) -> ViTState:
        features = self.proj(video)
        _, _, frames, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        return tokens, (frames, height, width)


@BLOCK_REGISTRY.register("RoPEFinalNorm")
class RoPEFinalNorm(nn.Module):
    """Apply LayerNorm to RoPE ViT tokens while preserving `(tokens, grid)`."""

    def __init__(self, dim: int = 1024, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, state: ViTState) -> ViTState:
        tokens, grid = state
        return self.norm(tokens), grid


@BLOCK_REGISTRY.register("LatentPoolFuse")
class LatentPoolFuse(nn.Module):
    """Pool ViT tokens, project image and ego status, and add pos embedding."""

    unpack_inputs = True

    def __init__(
        self,
        embed_dim: int = 1024,
        d_model: int = 256,
        status_dim: int = 16,
        num_src_tokens: int = 257,
        pool_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.status_dim = status_dim
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        self.image_fc = nn.Linear(embed_dim, d_model)
        self.status_encoding = nn.Linear(status_dim, d_model)
        self.encoder_pos_emb = nn.Parameter(torch.zeros(1, num_src_tokens, d_model))

    def forward(self, vit_state: ViTState, status: torch.Tensor) -> torch.Tensor:
        tokens, (frames, height, width) = vit_state
        batch, _, dim = tokens.shape
        grid = tokens.reshape(batch, frames, height, width, dim).permute(0, 1, 4, 2, 3)
        pooled = self.avg_pool(grid.reshape(batch * frames, dim, height, width))
        _, _, out_h, out_w = pooled.shape
        pooled = pooled.reshape(batch, frames, dim, out_h, out_w).permute(0, 1, 3, 4, 2)
        img_tokens = self.image_fc(pooled.reshape(batch, frames * out_h * out_w, dim))
        status_token = self.status_encoding(status[:, : self.status_dim]).unsqueeze(1)
        return torch.cat((img_tokens, status_token), dim=1) + self.encoder_pos_emb


@BLOCK_REGISTRY.register("LatentPlanTransformer")
class LatentPlanTransformer(nn.Module):
    """Encoder-decoder transformer decoding learned waypoint queries."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_enc: int = 3,
        num_dec: int = 3,
        ff_dim: int = 1024,
        num_waypoints: int = 40,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.decoder_query = nn.Parameter(torch.zeros(1, num_waypoints, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        query = self.decoder_query.expand(src.size(0), -1, -1)
        return self.transformer(src, query)


@BLOCK_REGISTRY.register("TrajectoryMLP")
class TrajectoryMLP(nn.Module):
    """Project waypoint tokens to `(x, y, tanh(yaw) * pi)`."""

    def __init__(self, d_model: int = 256, hidden_dim: int = 1024, out_dim: int = 3) -> None:
        super().__init__()
        self.mlp = MLP(d_model, hidden_dim, out_dim=out_dim, act="relu", bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(x)
        heading = torch.tanh(raw[..., 2:3]) * math.pi
        return torch.cat((raw[..., :2], heading), dim=-1)
