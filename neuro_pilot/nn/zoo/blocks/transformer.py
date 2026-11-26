"""Transformer blocks: QK-Norm, 3D-RoPE ViT, Causal GPT, and AdaRMSNorm DiT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.common import MLP, RMSNorm, SwiGLU
from neuro_pilot.nn.zoo.blocks.pos_embed import RoPE3D, build_2d_sincos_pe


@BLOCK_REGISTRY.register("QKNormBlock")
class QKNormBlock(nn.Module):
    """Pre-LN Transformer block with per-head QK LayerNorm."""

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        causal: bool = True,
        seq_len: int = 9,
        mlp_act: str = "gelu_tanh",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=eps)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=eps)
        self.proj = nn.Linear(dim, dim, bias=True)

        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            self.register_buffer("causal_mask", mask.view(1, 1, seq_len, seq_len), persistent=False)
        else:
            self.causal_mask = None

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            dim,
            hidden_dim,
            act=mlp_act,
            bias=True,
            pre_norm=True,
            residual=True,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q = self.q_norm(qkv[0])
        k = self.k_norm(qkv[1])
        v = qkv[2]

        if self.causal:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
            attn = torch.matmul(F.softmax(scores, dim=-1), v)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)

        out = attn.transpose(1, 2).reshape(batch, seq_len, dim)
        x = x + self.proj(out)
        return self.mlp(x)


@BLOCK_REGISTRY.register("RoPEViTBlock")
class RoPEViTBlock(nn.Module):
    """Pre-LN ViT block with 3D-RoPE attention operating on `(tokens, grid_thw)`."""

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rope = RoPE3D(self.head_dim, theta=theta)
        self.mlp = MLP(
            dim,
            int(dim * mlp_ratio),
            act="gelu",
            bias=True,
            pre_norm=True,
            residual=True,
            eps=eps,
        )

    def forward(
        self,
        state: Tuple[torch.Tensor, Tuple[int, int, int]],
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x, grid_thw = state
        batch, num_tokens, dim = x.shape

        qkv = self.qkv(self.norm1(x))
        qkv = qkv.reshape(batch, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k = self.rope(qkv[0], qkv[1], grid_thw)
        attn = F.scaled_dot_product_attention(q, k, qkv[2])
        out = attn.transpose(1, 2).reshape(batch, num_tokens, dim)

        x = x + self.proj(out)
        return self.mlp(x), grid_thw


@BLOCK_REGISTRY.register("GPTBlock")
class GPTBlock(nn.Module):
    """Pre-LN causal GPT-2 block with optional KV-cache state."""

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        hidden_dim: int = 4096,
        eps: float = 1e-5,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.mlp = MLP(
            dim,
            hidden_dim,
            act="gelu_new",
            bias=True,
            pre_norm=True,
            residual=True,
            eps=eps,
            drop=drop,
        )

    def forward(self, state: Union[torch.Tensor, Dict[str, Any]]) -> Any:
        is_dict = isinstance(state, dict)
        x = state["x"] if is_dict else state
        past_list = state.get("past_kv") if is_dict else None
        layer_idx = state.get("layer_idx", 0) if is_dict else 0
        past_kv = past_list[layer_idx] if past_list is not None else None

        batch, seq_len, dim = x.shape
        q, k, v = self.c_attn(self.norm1(x)).split(dim, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)

        if past_kv is None:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            total_len = k.size(2)
            q_pos = torch.arange(total_len - seq_len, total_len, device=x.device).unsqueeze(1)
            k_pos = torch.arange(total_len, device=x.device).unsqueeze(0)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=(k_pos <= q_pos))

        out = attn.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        x = self.mlp(x + self.drop(self.c_proj(out)))

        if not is_dict:
            return x
        present = state.get("present_kv")
        if present is not None:
            present.append((k, v))
        return {**state, "x": x, "layer_idx": layer_idx + 1}


class AdaRMSNorm(nn.Module):
    """Low-rank Adaptive RMSNorm for DiT conditioning."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        rank: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        bottleneck = rank if rank is not None else max(32, dim // 8)
        self.norm = RMSNorm(dim, eps=eps)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 3 * dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, shift, gate = self.proj(cond).unsqueeze(2).chunk(3, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift, gate


@BLOCK_REGISTRY.register("DiTBlock")
class DiTBlock(nn.Module):
    """Block-causal video DiT block with AdaRMSNorm, QK-RMSNorm, and SwiGLU."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        cond_dim: int = 256,
        mlp_ratio: float = 4.0,
        ada_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ada_attn = AdaRMSNorm(dim, cond_dim, rank=ada_rank)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.ada_mlp = AdaRMSNorm(dim, cond_dim, rank=ada_rank)
        hidden = max(32, (int(dim * mlp_ratio * 2 / 3) + 31) // 32 * 32)
        self.ffn = SwiGLU(dim, hidden, bias=False)

    def forward(self, state: Dict[str, Any]) -> Dict[str, Any]:
        x = state["x"]
        cond = state["cond"]
        mask = state["mask"]
        batch, num_frames, num_tokens, dim = x.shape

        normed, gate_attn = self.ada_attn(x, cond)
        seq_len = num_frames * num_tokens
        flat = normed.reshape(batch, seq_len, dim)
        qkv = self.qkv(flat).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q = self.q_norm(qkv[0])
        k = self.k_norm(qkv[1])
        v = qkv[2]

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = attn.transpose(1, 2).reshape(batch, num_frames, num_tokens, dim)
        x = x + gate_attn * self.proj(out)

        mlp_in, gate_mlp = self.ada_mlp(x, cond)
        x = x + gate_mlp * self.ffn(mlp_in)
        return {**state, "x": x}


__all__ = [
    "AdaRMSNorm",
    "DiTBlock",
    "GPTBlock",
    "QKNormBlock",
    "RoPE3D",
    "RoPEViTBlock",
    "build_2d_sincos_pe",
]
