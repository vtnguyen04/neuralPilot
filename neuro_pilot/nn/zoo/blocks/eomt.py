"""Encoder-only Mask Transformer (EoMT / DINOv2 ViT) semantic segmentation blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@BLOCK_REGISTRY.register("EomtEmbed")
class EomtEmbed(nn.Module):
    """ImageNet norm + patch projection + interpolated 2D pos embed + CLS/register tokens."""

    def __init__(
        self,
        embed_dim: int = 1024,
        patch_size: int = 16,
        base_grid: int = 64,
        num_registers: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.base_grid = int(base_grid)
        self.register_buffer(
            "mean", torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(base_grid * base_grid, embed_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - self.mean.to(image.dtype)) / self.std.to(image.dtype)
        feat = self.proj(x)
        gh, gw = feat.shape[-2], feat.shape[-1]
        tokens = feat.flatten(2).transpose(1, 2)

        pos = self.pos_embed.reshape(1, self.base_grid, self.base_grid, -1).permute(0, 3, 1, 2)
        if (gh, gw) != (self.base_grid, self.base_grid):
            pos = F.interpolate(
                pos.float(), size=(gh, gw), mode="bicubic", align_corners=False
            ).to(tokens.dtype)
        else:
            pos = pos.to(tokens.dtype)
        tokens = tokens + pos.flatten(2).transpose(1, 2)

        bsz = image.shape[0]
        cls = self.cls_token.to(tokens.dtype).expand(bsz, -1, -1)
        regs = self.register_tokens.to(tokens.dtype).expand(bsz, -1, -1)
        return torch.cat((cls, regs, tokens), dim=1)


@BLOCK_REGISTRY.register("EomtBlock")
class EomtBlock(nn.Module):
    """Pre-LN Vision Transformer block with Flash SDPA and dual LayerScale."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // self.num_heads
        hidden = embed_dim * int(mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim, eps=eps)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.ls1 = nn.Parameter(torch.ones(embed_dim))

        self.norm2 = nn.LayerNorm(embed_dim, eps=eps)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.ls2 = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bsz, seq, dim)
        x = x + self.out_proj(attn) * self.ls1

        h = self.norm2(x)
        x = x + self.fc2(F.gelu(self.fc1(h))) * self.ls2
        return x


@BLOCK_REGISTRY.register("EomtQueryConcat")
class EomtQueryConcat(nn.Module):
    """Prepend learned mask query tokens before the final joint ViT blocks."""

    def __init__(self, num_queries: int = 100, embed_dim: int = 1024) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(num_queries, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat((q, x), dim=1)


class EomtScaleLayer(nn.Module):
    """2x ConvTranspose + GELU + depthwise 3x3 Conv + channel-last LayerNorm."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2(F.gelu(self.conv1(x)))
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


@BLOCK_REGISTRY.register("EomtHead")
class EomtHead(nn.Module):
    """EoMT mask + class head outputting per-class semantic probabilities `[B, C, H/4, W/4]`."""

    unpack_inputs = True

    def __init__(
        self,
        embed_dim: int = 1024,
        num_queries: int = 100,
        num_classes: int = 19,
        patch_size: int = 16,
        num_registers: int = 4,
        num_upscale: int = 2,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.prefix_offset = int(num_queries) + 1 + int(num_registers)
        self.patch_size = int(patch_size)

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.class_predictor = nn.Linear(embed_dim, num_classes + 1)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.upscale = nn.ModuleList([EomtScaleLayer(embed_dim) for _ in range(num_upscale)])

    def forward(self, tokens: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        normed = self.norm(tokens)
        q = normed[:, : self.num_queries, :]
        cls_logits = self.class_predictor(q)

        gh = image.shape[-2] // self.patch_size
        gw = image.shape[-1] // self.patch_size
        spatial = normed[:, self.prefix_offset :, :].transpose(1, 2)
        spatial = spatial.reshape(tokens.shape[0], tokens.shape[-1], gh, gw)
        for up in self.upscale:
            spatial = up(spatial)

        q_mask = self.fc3(F.gelu(self.fc2(F.gelu(self.fc1(q)))))
        mask_logits = torch.einsum("bqc,bchw->bqhw", q_mask.float(), spatial.float())
        cls_probs = cls_logits.float().softmax(dim=-1)[..., :-1]
        return torch.einsum("bqc,bqhw->bchw", cls_probs, mask_logits.sigmoid())
