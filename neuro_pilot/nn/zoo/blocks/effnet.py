"""EfficientNet and DeepLabV3+ separable convolution blocks."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.common import Conv2dBlock, apply_act


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention with SiLU and Sigmoid gating."""

    def __init__(self, in_ch: int, se_ch: int) -> None:
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_ch, se_ch, kernel_size=1, bias=True)
        self.conv_expand = nn.Conv2d(se_ch, in_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3), keepdim=True)
        scale = apply_act(self.conv_reduce(scale), "silu")
        return x * torch.sigmoid(self.conv_expand(scale))


@BLOCK_REGISTRY.register("MBConv")
class MBConv(nn.Module):
    """EfficientNet mobile inverted bottleneck with Squeeze-and-Excitation."""

    def __init__(
        self,
        in_ch: int,
        exp_ch: int,
        out_ch: int,
        se_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        residual: bool = False,
        expand: Optional[bool] = None,
    ) -> None:
        super().__init__()
        use_expand = (exp_ch != in_ch) if expand is None else expand
        mid_ch = exp_ch if use_expand else in_ch
        self.residual = residual
        self.expand: Optional[nn.Conv2d] = (
            nn.Conv2d(in_ch, exp_ch, kernel_size=1, bias=True) if use_expand else None
        )
        self.dw = nn.Conv2d(
            mid_ch,
            mid_ch,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=mid_ch,
            bias=True,
        )
        self.se = SqueezeExcite(mid_ch, se_ch)
        self.proj = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = apply_act(self.expand(x), "silu") if self.expand is not None else x
        out = apply_act(self.dw(out), "silu")
        out = self.proj(self.se(out))
        return out + x if self.residual else out

    @classmethod
    def build_stage(
        cls,
        repeats: int,
        in_ch: int,
        out_ch: int,
        expand_ratio: int = 6,
        kernel: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Module:
        """Build an EfficientNet stage: block 0 changes ch/stride; blocks 1..n-1 repeat."""
        pad = (kernel // 2) * dilation
        blocks: List[nn.Module] = []
        for idx in range(repeats):
            c_in = in_ch if idx == 0 else out_ch
            s = stride if idx == 0 else 1
            exp_ch = c_in * expand_ratio
            se_ch = max(1, c_in // 4)
            use_expand = expand_ratio != 1
            residual = c_in == out_ch and s == 1
            blocks.append(
                cls(c_in, exp_ch, out_ch, se_ch, kernel, s, pad, dilation, residual, use_expand)
            )
        return blocks[0] if repeats == 1 else nn.Sequential(*blocks)


@BLOCK_REGISTRY.register("SepConv")
class SepConv(nn.Module):
    """Depthwise-separable Conv2d (bias-free depthwise + pointwise with activation)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        padding: int = 1,
        dilation: int = 1,
        act: str = "none",
    ) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel, padding=padding, dilation=dilation, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.act_name = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_act(self.pw(self.dw(x)), self.act_name)


class _ASPPPooling(nn.Module):
    """Global average pooling branch with bilinear upsampling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = apply_act(self.conv(self.pool(x)), "relu")
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


@BLOCK_REGISTRY.register("SeparableASPP")
class SeparableASPP(nn.Module):
    """Multi-branch separable Atrous Spatial Pyramid Pooling."""

    def __init__(
        self,
        in_ch: int = 320,
        out_ch: int = 256,
        dilations: Sequence[int] = (12, 24, 36),
    ) -> None:
        super().__init__()
        branches: List[nn.Module] = [Conv2dBlock(in_ch, out_ch, k=1, s=1, p=0, act="relu")]
        for rate in dilations:
            branch = SepConv(in_ch, out_ch, kernel=3, padding=rate, dilation=rate, act="relu")
            branches.append(branch)
        branches.append(_ASPPPooling(in_ch, out_ch))
        self.convs = nn.ModuleList(branches)
        self.project = Conv2dBlock(out_ch * len(branches), out_ch, k=1, s=1, p=0, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.convs]
        return self.project(torch.cat(feats, dim=1))
