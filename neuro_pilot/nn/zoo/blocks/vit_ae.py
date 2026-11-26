"""Autoencoder ViT blocks for CommaViT-AE."""

from __future__ import annotations

import torch
import torch.nn as nn

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.pos_embed import build_2d_sincos_pe


@BLOCK_REGISTRY.register("ViTPatchEmbed")
class ViTPatchEmbed(nn.Module):
    """Patch embedding with 2D sin-cos PE supporting `[B, 6, H, W]` and `[B, 3, 2H, W]`."""

    def __init__(
        self,
        in_channels: int = 6,
        img_h: int = 128,
        img_w: int = 256,
        patch_size: int = 8,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.img_h = img_h
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        pos_embed = build_2d_sincos_pe(img_h // patch_size, img_w // patch_size, embed_dim)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 3 and x.size(2) == 2 * self.img_h and self.in_channels == 6:
            top, bottom = x.split(self.img_h, dim=2)
            x = torch.cat((top, bottom), dim=1)
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return tokens + self.pos_embed.to(dtype=tokens.dtype)


@BLOCK_REGISTRY.register("ViTToLatent")
class ViTToLatent(nn.Module):
    """Project ViT encoder tokens to spatial latent map `[B, C, H, W]`."""

    def __init__(
        self,
        embed_dim: int = 768,
        latent_ch: int = 32,
        grid_h: int = 16,
        grid_w: int = 32,
        eps: float = 1e-6,
        latent_mean: float = 0.001418,
        latent_std: float = 0.112629,
        clamp_range: float = 20.0,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.latent_mean = float(latent_mean)
        self.latent_std = float(latent_std)
        self.clamp_range = float(clamp_range)
        self.normalize = bool(normalize)
        self.norm = nn.LayerNorm(embed_dim, eps=eps)
        self.proj = nn.Linear(embed_dim, latent_ch)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        latents = self.proj(self.norm(tokens))
        out = latents.transpose(1, 2).reshape(tokens.size(0), -1, self.grid_h, self.grid_w)
        if self.normalize:
            out = ((out - self.latent_mean) / self.latent_std).clamp(
                -self.clamp_range, self.clamp_range
            )
        return out


@BLOCK_REGISTRY.register("ViTFromLatent")
class ViTFromLatent(nn.Module):
    """Project spatial latent grid `[B, C, H, W]` to decoder tokens with 2D sin-cos PE."""

    def __init__(
        self, latent_ch: int = 32, embed_dim: int = 768, grid_h: int = 16, grid_w: int = 32
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_ch, embed_dim)
        pos_embed = build_2d_sincos_pe(grid_h, grid_w, embed_dim)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(latents.flatten(2).transpose(1, 2))
        return tokens + self.pos_embed.to(dtype=tokens.dtype)


@BLOCK_REGISTRY.register("ViTUnpatch")
class ViTUnpatch(nn.Module):
    """Project decoder tokens back to paired camera image `[B, 6, H, W]`."""

    def __init__(
        self,
        embed_dim: int = 768,
        out_channels: int = 6,
        patch_size: int = 8,
        grid_h: int = 16,
        grid_w: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.norm = nn.LayerNorm(embed_dim, eps=eps)
        self.proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.size(0)
        patch = self.patch_size
        patches = self.proj(self.norm(tokens))
        patches = patches.reshape(batch, self.grid_h, self.grid_w, self.out_channels, patch, patch)
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, self.out_channels, self.grid_h * patch, self.grid_w * patch
        )
