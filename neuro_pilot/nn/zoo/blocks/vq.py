"""VQGAN building blocks for CommaVQ encoder and decoder graphs."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.common import GroupNormAct


def _valid_groups(channels: int, num_groups: int = 32) -> int:
    groups = min(num_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


@BLOCK_REGISTRY.register("VQResBlock")
class VQResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, two 3x3 convs, and optional 1x1 shortcut."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.0,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm1 = GroupNormAct(_valid_groups(in_ch, num_groups), in_ch, eps=1e-6, act="silu")
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = GroupNormAct(_valid_groups(out_ch, num_groups), out_ch, eps=1e-6, act="silu")
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
            if in_ch != out_ch
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(self.norm1(x))
        hidden = self.conv2(self.drop(self.norm2(hidden)))
        shortcut = self.nin_shortcut(x) if self.nin_shortcut is not None else x
        return shortcut + hidden

    @classmethod
    def build_stage(
        cls,
        repeats: int,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.0,
        num_groups: int = 32,
    ) -> nn.Module:
        """Build a VQ stage where block 0 maps in_ch->out_ch and remaining blocks repeat out_ch."""
        blocks = [
            cls(in_ch if i == 0 else out_ch, out_ch, dropout=dropout, num_groups=num_groups)
            for i in range(repeats)
        ]
        return blocks[0] if repeats == 1 else nn.Sequential(*blocks)


@BLOCK_REGISTRY.register("VQAttnBlock")
class VQAttnBlock(nn.Module):
    """Single-head spatial self-attention with GroupNorm and 1x1 convolutions."""

    def __init__(self, channels: int, num_groups: int = 32) -> None:
        super().__init__()
        self.norm = GroupNormAct(
            _valid_groups(channels, num_groups), channels, eps=1e-6, act="none"
        )
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        hidden = self.norm(x)
        query = self.q(hidden).reshape(batch, channels, height * width).transpose(1, 2)
        key = self.k(hidden).reshape(batch, channels, height * width)
        value = self.v(hidden).reshape(batch, channels, height * width)
        weights = F.softmax(torch.bmm(query, key) * (channels ** -0.5), dim=2)
        out = torch.bmm(value, weights.transpose(1, 2)).reshape(batch, channels, height, width)
        return x + self.proj_out(out)


@BLOCK_REGISTRY.register("VQDown")
class VQDown(nn.Module):
    """Right/bottom 1px zero-pad followed by a stride-2 3x3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
        return self.conv(padded)


@BLOCK_REGISTRY.register("VQUp")
class VQUp(nn.Module):
    """Nearest-neighbor 2x spatial upsample followed by a 3x3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(upsampled)


@BLOCK_REGISTRY.register("VectorQuantizer")
class VectorQuantizer(nn.Module):
    """L2 codebook quantizer supporting encode, decode embedding, and STE training."""

    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 256,
        mode: str = "quantize",
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)

    def quantize_indices(self, latents: torch.Tensor) -> torch.Tensor:
        """Map continuous feature map [B, C, H, W] to nearest codebook indices [B, H, W]."""
        batch, dim, height, width = latents.shape
        flat = latents.permute(0, 2, 3, 1).contiguous().reshape(-1, dim)
        weight = self._embedding.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + weight.pow(2).sum(dim=1)
            - 2.0 * torch.matmul(flat, weight.t())
        )
        return torch.argmin(distances, dim=1).reshape(batch, height, width)

    def embed_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors for discrete token grid [B, H, W] -> [B, C, H, W]."""
        embedded = F.embedding(indices, self._embedding.weight)
        return embedded.permute(0, 3, 1, 2).contiguous()

    def forward_ste(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return straight-through quantized latents and discrete token indices."""
        indices = self.quantize_indices(latents)
        quantized = self.embed_indices(indices)
        return latents + (quantized - latents).detach(), indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "quantize":
            return self.quantize_indices(x)
        if self.mode == "embed":
            return self.embed_indices(x)
        raise ValueError(f"Unsupported VectorQuantizer mode: {self.mode}")
