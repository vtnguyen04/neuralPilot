"""VisionPilot blocks and task heads for AutoSteer2, AutoDrive, and AutoSpeed2."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, Union

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
        with_bn: bool = False,
        return_dict: bool = False,
        dense_cls: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.nc = self.num_classes
        self.reg_max = int(reg_max)
        self.box_ch = self.reg_max * 4
        self.strides = tuple(float(stride) for stride in strides)
        self.return_dict = bool(return_dict)
        self.register_buffer(
            "stride",
            torch.tensor(self.strides, dtype=torch.float32),
            persistent=self.return_dict,
        )
        self.dfl = DFL(reg_max=self.reg_max)

        c2 = max(64, self.box_ch)
        c3 = max(80, self.num_classes)
        self.box = nn.ModuleList(
            nn.Sequential(
                VPConv(ch, c2, kernel_size=3, padding=1, with_bn=with_bn),
                VPConv(c2, c2, kernel_size=3, padding=1, with_bn=with_bn),
                nn.Conv2d(c2, self.box_ch, kernel_size=1),
            )
            for ch in in_channels
        )
        if dense_cls:
            self.cls = nn.ModuleList(
                nn.Sequential(
                    VPConv(ch, c3, kernel_size=3, padding=1, with_bn=with_bn),
                    VPConv(c3, c3, kernel_size=3, padding=1, with_bn=with_bn),
                    nn.Conv2d(c3, self.num_classes, kernel_size=1),
                )
                for ch in in_channels
            )
        else:
            self.cls = nn.ModuleList(
                nn.Sequential(
                    VPConv(ch, ch, kernel_size=3, padding=1, groups=ch, with_bn=with_bn),
                    VPConv(ch, c3, kernel_size=1, with_bn=with_bn),
                    VPConv(c3, c3, kernel_size=3, padding=1, groups=c3, with_bn=with_bn),
                    VPConv(c3, c3, kernel_size=1, with_bn=with_bn),
                    nn.Conv2d(c3, self.num_classes, kernel_size=1),
                )
                for ch in in_channels
            )
        if with_bn:
            self._init_biases()

    def _init_biases(self) -> None:
        prior_bias = -math.log((1.0 - 0.01) / 0.01)
        for box_seq, cls_seq in zip(self.box, self.cls):
            nn.init.constant_(box_seq[-1].bias, 1.0)
            nn.init.constant_(cls_seq[-1].bias, prior_bias)

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

    def forward(
        self,
        feats: Sequence[torch.Tensor],
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        batch = feats[0].shape[0]
        box_logits = torch.cat(
            [b_head(feat).reshape(batch, self.box_ch, -1) for feat, b_head in zip(feats, self.box)],
            dim=2,
        )
        cls_logits = torch.cat(
            [
                c_head(feat).reshape(batch, self.num_classes, -1)
                for feat, c_head in zip(feats, self.cls)
            ],
            dim=2,
        )

        anchors, strides = self._make_anchors(feats)
        left_top, right_bottom = self.dfl(box_logits).chunk(2, dim=1)
        x1y1 = anchors - left_top
        x2y2 = anchors + right_bottom
        center_xy = (x1y1 + x2y2) * 0.5
        size_wh = x2y2 - x1y1
        decoded = torch.cat((center_xy * strides, size_wh * strides, torch.sigmoid(cls_logits)), 1)
        if self.return_dict:
            return {
                "decoded": decoded,
                "boxes": box_logits,
                "scores": cls_logits,
                "feats": list(feats),
            }
        return decoded
