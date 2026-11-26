"""YOLO-style CNN building blocks: VPConv, CSPBlock, VPSPPF, and VPC2PSA."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.common import apply_act


@BLOCK_REGISTRY.register("VPConv")
class VPConv(nn.Module):
    """Conv2d with optional BatchNorm2d and SiLU activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: bool = True,
        with_bn: bool = False,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not with_bn,
        )
        self.norm: Optional[nn.Module] = (
            nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum) if with_bn else None
        )
        self.act_name = "silu" if act else "none"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        return apply_act(out, self.act_name)


class Bottleneck(nn.Module):
    """Two-layer 3x3 residual bottleneck."""

    def __init__(
        self,
        channels: int,
        hidden_ch: int,
        shortcut: bool = True,
        with_bn: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = VPConv(channels, hidden_ch, kernel_size=3, padding=1, with_bn=with_bn)
        self.conv2 = VPConv(hidden_ch, channels, kernel_size=3, padding=1, with_bn=with_bn)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.shortcut else out


class C3kInner(nn.Module):
    """Nested C3k block with two residual 3x3 bottlenecks."""

    def __init__(self, channels: int, hidden_ch: int, with_bn: bool = False) -> None:
        super().__init__()
        self.conv1 = VPConv(channels, hidden_ch, kernel_size=1, with_bn=with_bn)
        self.conv2 = VPConv(channels, hidden_ch, kernel_size=1, with_bn=with_bn)
        self.conv3 = VPConv(hidden_ch * 2, channels, kernel_size=1, with_bn=with_bn)
        self.res_m = nn.Sequential(
            Bottleneck(hidden_ch, hidden_ch, shortcut=True, with_bn=with_bn),
            Bottleneck(hidden_ch, hidden_ch, shortcut=True, with_bn=with_bn),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_a = self.res_m(self.conv1(x))
        branch_b = self.conv2(x)
        return self.conv3(torch.cat((branch_a, branch_b), dim=1))


@BLOCK_REGISTRY.register("CSPBlock")
class CSPBlock(nn.Module):
    """Cross-Stage Partial block with Bottleneck or C3k inner layers."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_ch: int,
        use_c3k: bool = False,
        with_bn: bool = False,
    ) -> None:
        super().__init__()
        half_ch = out_ch // 2
        self.conv1 = VPConv(in_ch, out_ch, kernel_size=1, with_bn=with_bn)
        inner: nn.Module = (
            C3kInner(half_ch, hidden_ch, with_bn=with_bn)
            if use_c3k
            else Bottleneck(half_ch, hidden_ch, shortcut=True, with_bn=with_bn)
        )
        self.res_m = nn.ModuleList([inner])
        self.conv2 = VPConv(half_ch * 3, out_ch, kernel_size=1, with_bn=with_bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        part_a, part_b = self.conv1(x).chunk(2, dim=1)
        part_c = self.res_m[0](part_b)
        return self.conv2(torch.cat((part_a, part_b, part_c), dim=1))


@BLOCK_REGISTRY.register("VPSPPF")
class VPSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast with three sequential 5x5 max-pools."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        with_bn: bool = False,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        mid_ch = in_ch // 2
        bn_kw = {"with_bn": with_bn, "bn_eps": bn_eps, "bn_momentum": bn_momentum}
        self.cv1 = VPConv(in_ch, mid_ch, kernel_size=1, **bn_kw)
        self.cv2 = VPConv(mid_ch * 4, out_ch, kernel_size=1, **bn_kw)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cv1(x)
        pool1 = self.pool(feat)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)
        return self.cv2(torch.cat((feat, pool1, pool2, pool3), dim=1))


class PSAAttention(nn.Module):
    """Multi-head spatial self-attention with 3x3 depthwise positional encoding."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 2,
        key_dim: int = 32,
        with_bn: bool = False,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.val_dim = dim // num_heads
        self.scale = float(key_dim**-0.5)
        qkv_dim = num_heads * (2 * key_dim + self.val_dim)
        bn_kw = {"act": False, "with_bn": with_bn, "bn_eps": bn_eps, "bn_momentum": bn_momentum}
        self.qkv = VPConv(dim, qkv_dim, kernel_size=1, **bn_kw)
        self.conv1 = VPConv(dim, dim, kernel_size=3, padding=1, groups=dim, **bn_kw)
        self.conv2 = VPConv(dim, dim, kernel_size=1, **bn_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width
        head_dim = 2 * self.key_dim + self.val_dim
        qkv = self.qkv(x).reshape(batch, self.num_heads, head_dim, tokens)
        query, key, value = qkv.split((self.key_dim, self.key_dim, self.val_dim), dim=2)
        attn = ((query.transpose(-2, -1) @ key) * self.scale).softmax(dim=-1)
        out = (value @ attn.transpose(-2, -1)).reshape(batch, channels, height, width)
        pos_enc = self.conv1(value.reshape(batch, channels, height, width))
        return self.conv2(out + pos_enc)


class PSABlock(nn.Module):
    """Residual spatial attention block with 1x1 feed-forward network."""

    def __init__(
        self,
        dim: int = 128,
        with_bn: bool = False,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        bn_kw = {"with_bn": with_bn, "bn_eps": bn_eps, "bn_momentum": bn_momentum}
        self.conv1 = PSAAttention(dim=dim, **bn_kw)
        self.conv2 = nn.Sequential(
            VPConv(dim, dim * 2, kernel_size=1, act=True, **bn_kw),
            VPConv(dim * 2, dim, kernel_size=1, act=False, **bn_kw),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv1(x)
        return x + self.conv2(x)


@BLOCK_REGISTRY.register("VPC2PSA")
class VPC2PSA(nn.Module):
    """Cross-Stage Partial Pyramid Slice Attention block."""

    def __init__(
        self,
        dim: int = 256,
        with_bn: bool = False,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.03,
    ) -> None:
        super().__init__()
        bn_kw = {"with_bn": with_bn, "bn_eps": bn_eps, "bn_momentum": bn_momentum}
        self.cv1 = VPConv(dim, dim, kernel_size=1, **bn_kw)
        self.middle_block = PSABlock(dim=dim // 2, **bn_kw)
        self.cv2 = VPConv(dim, dim, kernel_size=1, **bn_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        part_a, part_b = self.cv1(x).chunk(2, dim=1)
        part_b = self.middle_block(part_b)
        return self.cv2(torch.cat((part_a, part_b), dim=1))
