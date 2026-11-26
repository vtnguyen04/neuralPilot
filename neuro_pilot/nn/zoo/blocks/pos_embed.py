"""Positional embeddings shared by ViT and 3D video backbones."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def build_2d_sincos_pe(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """Build fixed 2D sinusoidal positional embedding `[1, grid_h * grid_w, dim]`."""
    quarter = dim // 4
    freqs = torch.arange(quarter, dtype=torch.float32) / float(quarter)
    omega = torch.exp(-math.log(10000.0) * freqs)

    ys = torch.arange(grid_h, dtype=torch.float32)
    xs = torch.arange(grid_w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    emb_y = grid_y.reshape(-1, 1) * omega.unsqueeze(0)
    emb_x = grid_x.reshape(-1, 1) * omega.unsqueeze(0)
    pos = torch.cat((emb_y.sin(), emb_y.cos(), emb_x.sin(), emb_x.cos()), dim=-1)
    return pos.unsqueeze(0)


class RoPE3D(nn.Module):
    """Factorized 3D Rotary Positional Embedding matching V-JEPA 2 layout."""

    def __init__(self, head_dim: int = 64, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = float(theta)
        self.axis_dim = 2 * (head_dim // 6)

    def _apply_axis(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        dim = self.axis_dim
        exponents = torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / float(dim)
        inv_freq = 1.0 / (self.theta ** exponents)
        angles = torch.outer(positions.to(device=x.device, dtype=torch.float32), inv_freq)

        cos = angles.cos().view(1, 1, -1, dim // 2).repeat(1, 1, 1, 2).to(dtype=x.dtype)
        sin = angles.sin().view(1, 1, -1, dim // 2).repeat(1, 1, 1, 2).to(dtype=x.dtype)

        even, odd = x.reshape(*x.shape[:-1], -1, 2).unbind(dim=-1)
        rotated = torch.stack((-odd, even), dim=-1).flatten(start_dim=-2)
        return x * cos + rotated * sin

    def _rotate_all(
        self,
        tensor: torch.Tensor,
        coords: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        dim = self.axis_dim
        parts = [
            self._apply_axis(tensor[..., axis * dim : (axis + 1) * dim], coords[axis])
            for axis in range(3)
        ]
        if 3 * dim < self.head_dim:
            parts.append(tensor[..., 3 * dim :])
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        grid_thw: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frames, height, width = grid_thw
        area = height * width
        device = query.device

        t_pos = torch.arange(frames, device=device, dtype=torch.float32).repeat_interleave(area)
        frame_idx = torch.arange(area, device=device, dtype=torch.long).repeat(frames)
        h_pos = torch.div(frame_idx, width, rounding_mode="trunc").to(torch.float32)
        w_pos = (frame_idx - h_pos.to(torch.long) * width).to(torch.float32)
        coords = (t_pos, h_pos, w_pos)
        return self._rotate_all(query, coords), self._rotate_all(key, coords)
