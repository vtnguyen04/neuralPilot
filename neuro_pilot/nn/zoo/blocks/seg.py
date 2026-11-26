"""Dual-resolution semantic segmentation blocks (DDRNet23-Slim)."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY


@BLOCK_REGISTRY.register("ImgNorm")
class ImgNorm(nn.Module):
    """Per-channel `(x - mean) / std` normalization buffer block."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)


class AffineReLU(nn.Module):
    """Channel-wise fused pre-activation affine scale/bias followed by ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x * self.weight + self.bias)


@BLOCK_REGISTRY.register("ResBasic")
class ResBasic(nn.Module):
    """Fused-BN ResNet BasicBlock with optional strided projection and trailing ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        final_relu: bool = True,
    ) -> None:
        super().__init__()
        self.final_relu = bool(final_relu)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=True)
        self.downsample = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)
            if (stride != 1 or in_ch != out_ch)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(F.relu(self.conv1(x)))
        shortcut = self.downsample(x) if self.downsample is not None else x
        out = out + shortcut
        return F.relu(out) if self.final_relu else out


@BLOCK_REGISTRY.register("ResBottleneck")
class ResBottleneck(nn.Module):
    """Fused-BN ResNet Bottleneck with 1x1-3x3-1x1 path and optional projection."""

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        stride: int = 1,
        final_relu: bool = False,
    ) -> None:
        super().__init__()
        self.final_relu = bool(final_relu)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=True)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, stride=1, bias=True)
        self.downsample = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)
            if (stride != 1 or in_ch != out_ch)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))
        shortcut = self.downsample(x) if self.downsample is not None else x
        out = out + shortcut
        return F.relu(out) if self.final_relu else out


@BLOCK_REGISTRY.register("BilateralFuse")
class BilateralFuse(nn.Module):
    """Bidirectional high/low-resolution fusion module used in DDRNet."""

    unpack_inputs = True

    def __init__(self, low_ch: int, high_ch: int, num_down: int = 1) -> None:
        super().__init__()
        self.comp = nn.Conv2d(low_ch, high_ch, 1, bias=True)
        if num_down == 1:
            self.down = nn.Sequential(nn.Conv2d(high_ch, low_ch, 3, stride=2, padding=1, bias=True))
        else:
            mid = high_ch * 2
            self.down = nn.Sequential(
                nn.Conv2d(high_ch, mid, 3, stride=2, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(mid, low_ch, 3, stride=2, padding=1, bias=True),
            )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> Dict[str, torch.Tensor]:
        low_out = F.relu(low + self.down(F.relu(high)))
        up = F.interpolate(
            self.comp(F.relu(low)),
            size=high.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_out = F.relu(high + up)
        return {"low": low_out, "high": high_out}


@BLOCK_REGISTRY.register("DAPPM")
class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module (5-scale hierarchical context)."""

    def __init__(self, in_ch: int = 512, branch_ch: int = 128, out_ch: int = 128) -> None:
        super().__init__()
        self.s0_act = AffineReLU(in_ch)
        self.s0_conv = nn.Conv2d(in_ch, branch_ch, 1, bias=False)

        self.pools = nn.ModuleList(
            [
                nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            ]
        )
        self.scale_acts = nn.ModuleList([AffineReLU(in_ch) for _ in range(4)])
        self.scale_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, branch_ch, 1, bias=False) for _ in range(4)]
        )
        self.proc_acts = nn.ModuleList([AffineReLU(branch_ch) for _ in range(4)])
        self.proc_convs = nn.ModuleList(
            [nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False) for _ in range(4)]
        )
        self.comp_act = AffineReLU(branch_ch * 5)
        self.comp_conv = nn.Conv2d(branch_ch * 5, out_ch, 1, bias=False)
        self.short_act = AffineReLU(in_ch)
        self.short_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = [self.s0_conv(self.s0_act(x))]
        pooled = [pool(x) for pool in self.pools] + [x.mean(dim=(-2, -1), keepdim=True)]
        for idx, p in enumerate(pooled):
            s = self.scale_convs[idx](self.scale_acts[idx](p))
            s_up = F.interpolate(s, size=size, mode="bilinear", align_corners=False)
            feats.append(self.proc_convs[idx](self.proc_acts[idx](s_up + feats[-1])))
        comp = self.comp_conv(self.comp_act(torch.cat(feats, dim=1)))
        return comp + self.short_conv(self.short_act(x))


@BLOCK_REGISTRY.register("SegHead")
class SegHead(nn.Module):
    """Upsample context onto 1/8 high-res features and predict per-class logits."""

    unpack_inputs = True

    def __init__(self, in_ch: int = 128, mid_ch: int = 64, num_classes: int = 19) -> None:
        super().__init__()
        self.act = AffineReLU(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(mid_ch, num_classes, 1, bias=True)

    def forward(self, low_ctx: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(low_ctx, size=high_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv2(F.relu(self.conv1(self.act(up + high_feat))))
