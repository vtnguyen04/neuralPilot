"""Basic reusable layers shared across YAML model configs."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY


def apply_act(x: torch.Tensor, act: str) -> torch.Tensor:
    """Apply activation by name."""
    if act == "silu":
        return F.silu(x)
    if act == "relu":
        return F.relu(x)
    if act == "gelu":
        return F.gelu(x, approximate="none")
    if act == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    if act == "gelu_new":
        coeff = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(coeff * (x + 0.044715 * x.pow(3))))
    if act == "tanh":
        return torch.tanh(x)
    if act == "none":
        return x
    raise ValueError(f"Unsupported activation: {act}")


@BLOCK_REGISTRY.register("RMSNorm")
class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


@BLOCK_REGISTRY.register("LayerNorm2d")
class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


@BLOCK_REGISTRY.register("MLP")
class MLP(nn.Module):
    """Two-layer feed-forward block with optional Pre-LN and residual skip."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        act: str = "gelu",
        bias: bool = True,
        pre_norm: bool = False,
        residual: bool = False,
        eps: float = 1e-5,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = in_dim if out_dim is None else out_dim
        self.act_name = act
        self.residual = residual
        self.norm: Optional[nn.LayerNorm] = nn.LayerNorm(in_dim, eps=eps) if pre_norm else None
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x) if self.norm is not None else x
        h = self.drop(apply_act(self.fc1(h), self.act_name))
        h = self.drop(self.fc2(h))
        return x + h if self.residual else h


@BLOCK_REGISTRY.register("SwiGLU")
class SwiGLU(nn.Module):
    """Gated SiLU feed-forward layer."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


@BLOCK_REGISTRY.register("SiLUProj")
class SiLUProj(nn.Module):
    """Two-layer conditioning projection: Linear -> SiLU -> Linear(bias=False)."""

    def __init__(self, in_dim: int, out_dim: int = 512) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


@BLOCK_REGISTRY.register("ScaleLayer")
class ScaleLayer(nn.Module):
    """Learnable per-channel multiplier."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


@BLOCK_REGISTRY.register("GroupNormAct")
class GroupNormAct(nn.Module):
    """GroupNorm followed by an activation."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=True)
        self.act_name = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_act(self.norm(x), self.act_name)


@BLOCK_REGISTRY.register("Upsample")
class Upsample(nn.Module):
    """Spatial interpolation layer."""

    def __init__(
        self,
        scale_factor: float = 2.0,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.scale_factor = float(scale_factor)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


@BLOCK_REGISTRY.register("Concat")
class Concat(nn.Module):
    """Concatenate a list of tensors along `dim`."""

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=self.dim)


@BLOCK_REGISTRY.register("SplitCat")
class SplitCat(nn.Module):
    """Split along `split_dim` and concatenate chunks along `cat_dim`."""

    def __init__(self, chunks: int = 2, split_dim: int = 0, cat_dim: int = 1) -> None:
        super().__init__()
        self.chunks = chunks
        self.split_dim = split_dim
        self.cat_dim = cat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(x.chunk(self.chunks, dim=self.split_dim), dim=self.cat_dim)


@BLOCK_REGISTRY.register("Conv2d")
class Conv2dBlock(nn.Module):
    """Conv2d with optional activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        act: str = "none",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.act_name = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_act(self.conv(x), self.act_name)


@BLOCK_REGISTRY.register("Linear")
class LinearBlock(nn.Module):
    """Linear layer with optional activation."""

    def __init__(self, in_dim: int, out_dim: int, act: str = "none", bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.act_name = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_act(self.linear(x), self.act_name)


@BLOCK_REGISTRY.register("LayerNorm")
class LayerNormBlock(nn.LayerNorm):
    """Standard LayerNorm for YAML graphs."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__(dim, eps=eps)


@BLOCK_REGISTRY.register("Flatten")
class FlattenBlock(nn.Module):
    """Flatten tensor starting at `start_dim`."""

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[: self.start_dim], -1)


@BLOCK_REGISTRY.register("LastToken")
class LastToken(nn.Module):
    """Select the last token `x[:, -1, :]`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, :]


@BLOCK_REGISTRY.register("MeanPool")
class MeanPool(nn.Module):
    """Mean-pool along `dim`."""

    def __init__(self, dim: int = 1, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)


@BLOCK_REGISTRY.register("Permute")
class Permute(nn.Module):
    """Permute tensor axes."""

    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        self.dims = tuple(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)
