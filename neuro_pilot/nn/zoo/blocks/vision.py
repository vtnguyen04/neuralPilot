"""Atomic vision backbone blocks: FastViT, ConvNeXt, and MobileNetV4."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.common import MLP, LayerNorm2d, apply_act


@BLOCK_REGISTRY.register("FastViTStem")
class FastViTStem(nn.Module):
    """Reparameterized FastViT three-conv stem (stride 4)."""

    def __init__(self, in_ch: int = 24, out_ch: int = 64) -> None:
        super().__init__()
        self.stem_0 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=True)
        self.stem_1 = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, groups=out_ch, bias=True)
        self.stem_2 = nn.Conv2d(out_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.stem_0(x), approximate="none")
        x = F.gelu(self.stem_1(x), approximate="none")
        return F.gelu(self.stem_2(x), approximate="none")


@BLOCK_REGISTRY.register("RepMixerBlock")
class RepMixerBlock(nn.Module):
    """Reparameterized FastViT block with DW3x3 mixer, DW7x7 ConvFFN, and LayerScale."""

    def __init__(self, dim: int, mlp_ratio: int = 3) -> None:
        super().__init__()
        hidden = dim * mlp_ratio
        self.token_mixer = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.mlp_dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=True)
        self.mlp_fc1 = nn.Conv2d(dim, hidden, 1, bias=True)
        self.mlp_fc2 = nn.Conv2d(hidden, dim, 1, bias=True)
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mixer(x)
        hidden = self.mlp_dw(mixed)
        hidden = F.gelu(self.mlp_fc1(hidden), approximate="none")
        hidden = self.mlp_fc2(hidden)
        return mixed + hidden * self.layer_scale


@BLOCK_REGISTRY.register("PatchEmbed2d")
class PatchEmbed2d(nn.Module):
    """FastViT stride-2 patch downsampling (DW7x7 + PW1x1 + GELU)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj_dw = nn.Conv2d(in_ch, out_ch, 7, stride=2, padding=3, groups=in_ch, bias=True)
        self.proj_pw = nn.Conv2d(out_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj_pw(self.proj_dw(x)), approximate="none")


@BLOCK_REGISTRY.register("FastViTHead")
class FastViTHead(nn.Module):
    """FastViT final DW3x3 + Squeeze-Excite + 1x1 projection + global average pool."""

    def __init__(self, in_ch: int = 512, out_dim: int = 512) -> None:
        super().__init__()
        exp_ch = in_ch * 2
        self.final_dw = nn.Conv2d(in_ch, exp_ch, 3, padding=1, groups=in_ch, bias=True)
        self.se_fc1 = nn.Conv2d(exp_ch, exp_ch // 16, 1, bias=True)
        self.se_fc2 = nn.Conv2d(exp_ch // 16, exp_ch, 1, bias=True)
        self.head_fc = nn.Conv2d(exp_ch, out_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.final_dw(x)
        scale = feat.mean(dim=(2, 3), keepdim=True)
        scale = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(scale))))
        feat = F.gelu(feat * scale, approximate="none")
        feat = self.head_fc(feat).mean(dim=(2, 3), keepdim=True)
        return feat.permute(0, 2, 3, 1).reshape(feat.shape[0], 1, -1)


@BLOCK_REGISTRY.register("ConvNeXtStem")
class ConvNeXtStem(nn.Module):
    """ConvNeXt patchify 4x4 stride-4 stem with channels-first LayerNorm."""

    def __init__(
        self, in_ch: int = 24, out_ch: int = 256, eps: float = 1e-5, input_div: float = 1.0
    ) -> None:
        super().__init__()
        self.input_div = float(input_div)
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=4, bias=True)
        self.norm = LayerNorm2d(out_ch, eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x.to(self.conv.weight.dtype)
        if self.input_div != 1.0:
            feat = feat / self.input_div
        return self.norm(self.conv(feat))


@BLOCK_REGISTRY.register("ConvNeXtBlock")
class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: DW7x7 -> channels-last Pre-LN GELU MLP -> LayerScale residual."""

    def __init__(self, dim: int, mlp_ratio: int = 4, eps: float = 1e-5) -> None:
        super().__init__()
        self.conv_dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=True)
        self.mlp = MLP(
            dim,
            dim * mlp_ratio,
            act="gelu",
            bias=True,
            pre_norm=True,
            residual=False,
            eps=float(eps),
        )
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_dw(x).permute(0, 2, 3, 1)
        hidden = self.mlp(hidden).permute(0, 3, 1, 2)
        return x + hidden * self.gamma


@BLOCK_REGISTRY.register("ConvNeXtDown")
class ConvNeXtDown(nn.Module):
    """ConvNeXt stage transition: channels-first LayerNorm + 2x2 stride-2 Conv2d."""

    def __init__(self, in_ch: int, out_ch: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_ch, eps=float(eps))
        self.conv = nn.Conv2d(in_ch, out_ch, 2, stride=2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


@BLOCK_REGISTRY.register("ConvNeXtHead")
class ConvNeXtHead(nn.Module):
    """ConvNeXt output head: LayerNorm2d + optional 1x1 projection to spatial tokens."""

    def __init__(self, in_ch: int, out_ch: Optional[int] = 512, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_ch, eps=float(eps))
        self.head_fc: Optional[nn.Conv2d] = (
            nn.Conv2d(in_ch, out_ch, 1, bias=True) if out_ch is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.norm(x)
        if self.head_fc is None:
            return feat
        feat = self.head_fc(feat)
        batch, channels, height, width = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(batch, height * width, channels)


@BLOCK_REGISTRY.register("ConvBNAct")
class ConvBNAct(nn.Module):
    """Conv2d with optional BatchNorm2d and configurable activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        act: str = "gelu_tanh",
        groups: int = 1,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=pad, groups=groups, bias=not use_bn
        )
        self.bn: nn.Module = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act_name = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_act(self.bn(self.conv(x)), self.act_name)


@BLOCK_REGISTRY.register("FusedIB")
class FusedIB(nn.Module):
    """MobileNetV4 Fused Inverted Bottleneck (fused Conv+BN weights)."""

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 2,
        act: str = "gelu_tanh",
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.act_name = act
        self.residual = residual
        self.conv_exp = nn.Conv2d(
            in_ch, mid_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=True
        )
        self.conv_proj = nn.Conv2d(mid_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = apply_act(self.conv_exp(x), self.act_name)
        out = apply_act(self.conv_proj(hidden), self.act_name)
        return out + x if self.residual else out


@BLOCK_REGISTRY.register("UIB")
class UIB(nn.Module):
    """MobileNetV4 Universal Inverted Bottleneck (fused Conv+BN weights)."""

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        dw_start_k: Optional[int] = None,
        dw_mid_k: Optional[int] = None,
        stride: int = 1,
        residual: bool = False,
        act: str = "gelu_tanh",
    ) -> None:
        super().__init__()
        self.residual = residual
        self.act_name = act
        self.dw_start: Optional[nn.Conv2d] = (
            nn.Conv2d(in_ch, in_ch, dw_start_k, padding=dw_start_k // 2, groups=in_ch, bias=True)
            if dw_start_k is not None
            else None
        )
        self.pw_exp = nn.Conv2d(in_ch, mid_ch, 1, bias=True)
        self.dw_mid: Optional[nn.Conv2d] = (
            nn.Conv2d(
                mid_ch,
                mid_ch,
                dw_mid_k,
                stride=stride,
                padding=dw_mid_k // 2,
                groups=mid_ch,
                bias=True,
            )
            if dw_mid_k is not None
            else None
        )
        self.pw_proj = nn.Conv2d(mid_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dw_start(x) if self.dw_start is not None else x
        hidden = apply_act(self.pw_exp(hidden), self.act_name)
        if self.dw_mid is not None:
            hidden = apply_act(self.dw_mid(hidden), self.act_name)
        out = self.pw_proj(hidden)
        return out + x if self.residual else out
