"""Generic training-time dense feature distillation (DINOv3 teacher, projector, REPA loss)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class _DinoV3RoPE(nn.Module):
    """2D Rotary Position Embedding matching Meta DINOv3 RopePositionEmbedding (base=100)."""

    def __init__(self, head_dim: int = 64, base: float = 100.0) -> None:
        super().__init__()
        periods = base ** (2.0 * torch.arange(head_dim // 4, dtype=torch.float32) / (head_dim // 2))
        self.register_buffer("periods", periods, persistent=False)

    def forward(self, gh: int, gw: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        cy = (torch.arange(0.5, gh, device=device, dtype=torch.float32) / float(gh)) * 2.0 - 1.0
        cx = (torch.arange(0.5, gw, device=device, dtype=torch.float32) / float(gw)) * 2.0 - 1.0
        gy, gx = torch.meshgrid(cy, cx, indexing="ij")
        coords = torch.stack((gy, gx), dim=-1).reshape(gh * gw, 2)
        angles = (2.0 * math.pi * coords[:, :, None] / self.periods[None, None, :]).flatten(1, 2)
        angles = angles.tile(1, 2)
        cos = torch.cos(angles).to(dtype).view(1, 1, gh * gw, -1)
        sin = torch.sin(angles).to(dtype).view(1, 1, gh * gw, -1)
        return cos, sin


class _DinoV3Block(nn.Module):
    """Pre-LN ViT block with 2D RoPE on patch tokens, no QKV bias, and LayerScale."""

    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn.proj = nn.Linear(dim, dim, bias=True)
        self.gamma_1 = nn.Parameter(torch.ones(dim))

        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * mlp_ratio, bias=True)
        self.mlp.fc2 = nn.Linear(dim * mlp_ratio, dim, bias=True)
        self.gamma_2 = nn.Parameter(torch.ones(dim))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_prefix: int = 5,
    ) -> torch.Tensor:
        bsz, seq, dim = x.shape
        qkv = self.attn.qkv(self.norm1(x)).reshape(bsz, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = torch.cat((q[:, :, :num_prefix, :], _apply_rope(q[:, :, num_prefix:, :], cos, sin)), 2)
        k = torch.cat((k[:, :, :num_prefix, :], _apply_rope(k[:, :, num_prefix:, :], cos, sin)), 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bsz, seq, dim)
        x = x + self.attn.proj(attn) * self.gamma_1
        return x + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x)))) * self.gamma_2


class DinoV3Teacher(nn.Module):
    """Frozen DINOv3 ViT dense patch feature extractor (loads safetensors directly)."""

    def __init__(
        self,
        weights_path: str | Path,
        dim: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
        patch_size: int = 16,
        num_registers: int | None = None,
    ) -> None:
        super().__init__()
        state = load_file(str(weights_path), device="cpu")
        dim = int(dim or state["cls_token"].shape[-1])
        depth = int(depth or len({k.split(".")[1] for k in state if k.startswith("blocks.")}))
        num_heads = int(num_heads or (dim // 64))
        num_registers = int(
            state["reg_token"].shape[1] if num_registers is None else num_registers
        )

        self.dim = dim
        self.patch_size = patch_size
        self.num_prefix = 1 + num_registers
        self.register_buffer(
            "mean", torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_registers, dim))
        self.rope = _DinoV3RoPE(head_dim=dim // num_heads, base=100.0)
        self.blocks = nn.ModuleList([_DinoV3Block(dim, num_heads, 4) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-5)

        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"DINOv3 weight mismatch: missing={missing}, unexpected={unexpected}"
            )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Extract dense patch features `[B, D, H/16, W/16]` from RGB images in `[0, 1]`."""
        dtype = self.patch_embed.proj.weight.dtype
        feat = self.patch_embed.proj((rgb_01.to(dtype) - self.mean.to(dtype)) / self.std.to(dtype))
        bsz, dim, gh, gw = feat.shape
        x = torch.cat(
            (
                self.cls_token.expand(bsz, -1, -1),
                self.reg_token.expand(bsz, -1, -1),
                feat.flatten(2).transpose(1, 2),
            ),
            dim=1,
        )
        cos, sin = self.rope(gh, gw, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin, self.num_prefix)
        return self.norm(x)[:, self.num_prefix :, :].transpose(1, 2).reshape(bsz, dim, gh, gw)


class DistillProj(nn.Module):
    """Training-only projector aligning multi-scale student features to teacher patch tokens."""

    def __init__(self, p4_ch: int = 128, p5_ch: int = 256, teacher_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(p4_ch + p5_ch, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, teacher_dim, kernel_size=1),
        )

    def forward(self, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat((p4, p5_up), dim=1))


def dense_distill_loss(pred_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """Dense patch cosine + normalized Smooth-L1 + spatial Gram affinity REPA loss."""
    s_n = F.normalize(pred_feat.float(), dim=1, eps=1e-6)
    t_n = F.normalize(teacher_feat.float(), dim=1, eps=1e-6)
    l_cos = (1.0 - (s_n * t_n).sum(dim=1)).mean()
    l_l1 = F.smooth_l1_loss(s_n, t_n, beta=0.1)
    s_tok = s_n.flatten(2).transpose(1, 2)
    t_tok = t_n.flatten(2).transpose(1, 2)
    l_gram = F.mse_loss(
        torch.bmm(s_tok, s_tok.transpose(1, 2)),
        torch.bmm(t_tok, t_tok.transpose(1, 2)),
    )
    return l_cos + 0.5 * l_l1 + 0.5 * l_gram
