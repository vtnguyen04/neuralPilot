"""VisionPilot blocks and task heads for AutoSteer2, AutoDrive, and AutoSpeed2."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.cnn import VPConv
from neuro_pilot.nn.zoo.blocks.common import LinearBlock


@BLOCK_REGISTRY.register("SpatialContext")
class SpatialContext(nn.Module):
    """Spatial-context block projecting pooled channels into a 2D context map."""

    def __init__(self, in_ch: int, out_ch: int, feat_h: int, feat_w: int) -> None:
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.exp0 = nn.Conv1d(in_ch, feat_h * feat_w, kernel_size=3, padding=1)
        self.ctx0 = nn.Conv2d(1, in_ch // 2, kernel_size=3, padding=1)
        self.ctx1 = nn.Conv2d(in_ch // 2, in_ch, kernel_size=3, padding=1)
        self.ctx2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        ctx = x.mean(dim=(2, 3), keepdim=True).squeeze(-1)
        ctx = F.silu(self.exp0(ctx)).reshape(batch, 1, self.feat_h, self.feat_w)
        # VisionPilot applies a second SiLU immediately after reshaping the 1D spatial map.
        ctx = F.silu(ctx)
        ctx = F.silu(self.ctx0(ctx))
        ctx = F.silu(self.ctx1(ctx))
        fused = F.silu(ctx * x + x)
        return self.ctx2(fused)


@BLOCK_REGISTRY.register("AutoSteerHead")
class AutoSteerHead(nn.Module):
    """Row-anchor soft-argmax ego-lane and horizon-height prediction head."""

    unpack_inputs: bool = True

    def __init__(self, num_cols: int = 256) -> None:
        super().__init__()
        self.num_cols = num_cols
        self.v1 = nn.Conv2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.v2 = nn.Conv2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.c1 = VPConv(64, 1, kernel_size=3, padding=1)
        self.c2 = VPConv(64, 1, kernel_size=3, padding=1)
        self.h1 = nn.Conv2d(1, 1, kernel_size=(1, 16), stride=(1, 16))
        self.h2 = nn.Conv2d(1, 1, kernel_size=(1, 16), stride=(1, 16))

    def forward(self, p2: torch.Tensor, fpn_p3: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_p2 = F.silu(self.v1(p2))
        feat_p3 = F.interpolate(F.silu(self.v2(fpn_p3)), scale_factor=2.0, mode="nearest")
        fused = torch.cat((feat_p3, feat_p2), dim=1)

        # Both c1 and c2 apply a second SiLU after VPConv's internal SiLU.
        lane_logits = F.silu(self.c1(fused))
        lane_probs = torch.softmax(lane_logits, dim=-1)
        col_grid = torch.arange(self.num_cols, device=fused.device, dtype=fused.dtype)
        col_grid = col_grid.view(1, 1, 1, self.num_cols)
        lane_value = (lane_probs * col_grid).sum(dim=-1, keepdim=True) / float(self.num_cols)

        height = F.silu(self.c2(fused))
        height = F.silu(self.h1(height))
        height = F.silu(self.h2(height))
        return {"lane_value": lane_value, "height": height}


@BLOCK_REGISTRY.register("AutoDriveMLP")
class AutoDriveMLP(nn.Module):
    """MLP head predicting CIPO distance, road curvature, and in-lane flag logit."""

    def __init__(
        self,
        in_dim: int = 1024,
        mid_dim: int = 768,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.fc1 = LinearBlock(in_dim, mid_dim, act="silu")
        self.fc2 = LinearBlock(mid_dim, hidden_dim, act="silu")
        self.distance_head = LinearBlock(hidden_dim, 1, act="relu")
        self.curvature_head = LinearBlock(hidden_dim, 1, act="tanh")
        self.flag_head = LinearBlock(hidden_dim, 1, act="none")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.fc2(self.fc1(x))
        return {
            "distance": self.distance_head(hidden),
            "curvature": self.curvature_head(hidden),
            "flag_logit": self.flag_head(hidden),
        }


class DFL(nn.Module):
    """Distribution Focal Loss integral projection over discrete distance bins."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, kernel_size=1, bias=False)
        bins = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1)
        self.conv.weight.data.copy_(bins)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, num_anchors = x.shape
        probs = x.reshape(batch, 4, self.reg_max, num_anchors).transpose(1, 2).softmax(dim=1)
        return self.conv(probs).reshape(batch, 4, num_anchors)


@BLOCK_REGISTRY.register("DFLDetectHead")
class DFLDetectHead(nn.Module):
    """Anchor-free decoupled DFL detection head across multi-scale FPN features."""

    def __init__(
        self,
        in_channels: Sequence[int] = (64, 128, 256),
        num_classes: int = 4,
        reg_max: int = 16,
        strides: Sequence[float] = (8.0, 16.0, 32.0),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_ch = reg_max * 4
        self.strides = tuple(float(stride) for stride in strides)
        self.dfl = DFL(reg_max=reg_max)

        self.box = nn.ModuleList(
            nn.Sequential(
                VPConv(channels, 64, kernel_size=3, padding=1),
                VPConv(64, 64, kernel_size=3, padding=1),
                nn.Conv2d(64, self.box_ch, kernel_size=1),
            )
            for channels in in_channels
        )
        self.cls = nn.ModuleList(
            nn.Sequential(
                VPConv(channels, channels, kernel_size=3, padding=1, groups=channels),
                VPConv(channels, 80, kernel_size=1),
                VPConv(80, 80, kernel_size=3, padding=1, groups=80),
                VPConv(80, 80, kernel_size=1),
                nn.Conv2d(80, num_classes, kernel_size=1),
            )
            for channels in in_channels
        )

    def _make_anchors(
        self,
        feats: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor_points: List[torch.Tensor] = []
        stride_tensors: List[torch.Tensor] = []
        dtype = feats[0].dtype
        device = feats[0].device

        for feat, stride in zip(feats, self.strides):
            _, _, height, width = feat.shape
            shift_y = torch.arange(height, device=device, dtype=dtype) + 0.5
            shift_x = torch.arange(width, device=device, dtype=dtype) + 0.5
            grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor_points.append(torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2))
            stride_fill = torch.full((height * width, 1), stride, device=device, dtype=dtype)
            stride_tensors.append(stride_fill)

        anchors = torch.cat(anchor_points, dim=0).transpose(0, 1).unsqueeze(0)
        strides = torch.cat(stride_tensors, dim=0).transpose(0, 1)
        return anchors, strides

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        batch = feats[0].shape[0]
        pred_ch = self.box_ch + self.num_classes
        raw_preds = [
            torch.cat((box_head(feat), cls_head(feat)), dim=1).reshape(batch, pred_ch, -1)
            for feat, box_head, cls_head in zip(feats, self.box, self.cls)
        ]
        preds = torch.cat(raw_preds, dim=2)
        box_logits, cls_logits = preds.split((self.box_ch, self.num_classes), dim=1)

        anchors, strides = self._make_anchors(feats)
        left_top, right_bottom = self.dfl(box_logits).chunk(2, dim=1)
        x1y1 = anchors - left_top
        x2y2 = anchors + right_bottom
        center_xy = (x1y1 + x2y2) / 2.0
        size_wh = x2y2 - x1y1
        boxes = torch.cat((center_xy, size_wh), dim=1) * strides
        cls_probs = torch.sigmoid(cls_logits)
        return torch.cat((boxes, cls_probs), dim=1)
