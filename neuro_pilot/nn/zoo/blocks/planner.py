"""Openpilot policy, rolling-queue IO, and driver-monitoring blocks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.common import MLP, ScaleLayer, SiLUProj

_POINT_HEAD_DIMS = (
    ("lane_lines", 528), ("lane_lines_prob", 8), ("road_edges", 264), ("meta", 55),
    ("desire_pred", 32), ("road_transform", 12), ("wide_from_device_euler", 6), ("pose", 12),
)
_POINT_LEAD_KEYS = ("lane_lines", "lane_lines_prob", "road_edges", "meta", "desire_pred")
_DMON_VIS_KEYS = ("pose", "faceprob", "eyeprob", "eyeclosed")
_DMON_HEAD_SPECS = (
    ("pose", (512, 64, 48, 32, 12)), ("faceprob", (512, 32, 24, 16, 1)),
    ("eyeprob", (512, 32, 24, 16, 3)), ("eyeclosed", (512, 32, 24, 16, 2)),
    ("phone", (515, 128, 64, 32, 1)), ("sleep", (515, 96, 64, 32, 1)),
)


def _res_mlp(dim: int = 512) -> MLP:
    return MLP(dim, dim * 2, act="gelu_tanh", bias=False, pre_norm=True, residual=True)


def _shift_queue(queue: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
    return torch.cat([queue[1:], item.unsqueeze(0)], dim=0)


def _slice_hist(
    queue: torch.Tensor, current: torch.Tensor, num_frames: int, dtype: torch.dtype
) -> torch.Tensor:
    token = (current.unsqueeze(1) if current.ndim == 2 else current).to(dtype)
    history = queue[::4].permute(1, 0, 2).to(dtype)
    return torch.cat([history, token], dim=1)[:, -num_frames:, :]


def _mlp4(dims: Tuple[int, ...]) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2]), nn.ReLU(),
        nn.Linear(dims[2], dims[3]), nn.ReLU(), nn.Linear(dims[3], dims[4]),
    )


@BLOCK_REGISTRY.register("QueuePreproc")
class QueuePreproc(nn.Module):
    """Update rolling image/desire queues, normalize dual cameras, and slice history."""
    unpack_inputs = True

    def __init__(self, num_spatial_tokens: int = 1) -> None:
        super().__init__()
        self.num_spatial_tokens = int(num_spatial_tokens)
        self.register_buffer("vision_mean", torch.full((1, 24, 1, 1), 127.5))
        self.register_buffer("vision_std", torch.full((1, 24, 1, 1), 63.75))
        window_idx = (torch.arange(9)[:, None] + torch.arange(25)[None, :]).reshape(-1)
        self.register_buffer("desire_window_idx", window_idx, persistent=False)

    def forward(
        self, new_img: torch.Tensor, desire: torch.Tensor, state_img_q: torch.Tensor,
        state_desire_q: torch.Tensor, state_feat_q: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dtype = self.vision_mean.dtype
        next_img_q = torch.cat([state_img_q[:, 1:], new_img.unsqueeze(1)], dim=1)
        cam0 = next_img_q[0, ::4].reshape(1, 12, 128, 256)
        cam1 = next_img_q[1, ::4].reshape(1, 12, 128, 256)
        img_norm = (torch.cat([cam0, cam1], dim=1).to(dtype) - self.vision_mean) / self.vision_std
        next_desire_q = torch.cat([state_desire_q[1:], desire.reshape(1, 1, 8)], dim=0)
        desire_pooled = next_desire_q.reshape(33, 4, 1, 8).amax(dim=1).permute(1, 0, 2)
        desire_win = desire_pooled[:, self.desire_window_idx, :].reshape(1, 9, 200).to(dtype)
        hist_feats = state_feat_q[::4].permute(1, 0, 2)
        hist_feats = hist_feats.reshape(1, 32, self.num_spatial_tokens, 512).to(dtype)
        return {
            "img_norm": img_norm, "desire_win": desire_win, "hist_feats": hist_feats,
            "next_img_q": next_img_q, "next_desire_q": next_desire_q,
        }


@BLOCK_REGISTRY.register("VisionSeqCat")
class VisionSeqCat(nn.Module):
    """Append current vision tokens to history and keep the last `num_frames`."""
    unpack_inputs = True

    def __init__(self, num_frames: int = 9) -> None:
        super().__init__()
        self.num_frames = int(num_frames)

    def forward(self, hist_feats: torch.Tensor, cur_feat: torch.Tensor) -> torch.Tensor:
        seq = torch.cat([hist_feats.to(cur_feat.dtype), cur_feat.unsqueeze(1)], dim=1)
        return seq[:, -self.num_frames :, :, :]


@BLOCK_REGISTRY.register("NavContextFuse")
class NavContextFuse(nn.Module):
    """Fuse desire window with rolling map and turn-by-turn navigation features."""
    unpack_inputs = True

    def __init__(self, num_frames: int = 9) -> None:
        super().__init__()
        self.num_frames = int(num_frames)

    def forward(
        self, desire_win: torch.Tensor, cur_map: torch.Tensor, cur_nav: torch.Tensor,
        state_map_q: torch.Tensor, state_nav_q: torch.Tensor,
    ) -> torch.Tensor:
        dtype = desire_win.dtype
        map_seq = _slice_hist(state_map_q, cur_map, self.num_frames, dtype)
        nav_seq = _slice_hist(state_nav_q, cur_nav, self.num_frames, dtype)
        return torch.cat([desire_win, map_seq, nav_seq], dim=-1)


@BLOCK_REGISTRY.register("PointPolicy")
class PointPolicy(nn.Module):
    """Single-frame spatial head predicting lanes, road edges, meta, desire, and pose."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.mlp1, self.mlp2 = _res_mlp(dim), _res_mlp(dim)
        self.heads = nn.ModuleDict({k: nn.Linear(dim, out_d) for k, out_d in _POINT_HEAD_DIMS})
        self.pose_mlp = _res_mlp(dim)
        self.pose_scale = ScaleLayer(12)

    def forward(self, feat: torch.Tensor) -> List[torch.Tensor]:
        hidden = self.mlp2(self.mlp1(feat.mean(dim=1) if feat.ndim == 3 else feat))
        pose_out = self.pose_scale(self.heads["pose"](self.pose_mlp(hidden)))
        lead = [self.heads[name](hidden) for name in _POINT_LEAD_KEYS]
        tail = [self.heads["wide_from_device_euler"](hidden), self.heads["road_transform"](hidden)]
        return [*lead, pose_out, *tail]


@BLOCK_REGISTRY.register("TemporalEmbed")
class TemporalEmbed(nn.Module):
    """Embed vision sequence, desire/nav context, traffic convention, and action_t."""
    unpack_inputs = True

    def __init__(
        self, num_spatial_tokens: int = 1, desire_in_dim: int = 200,
        dim: int = 512, num_frames: int = 9,
    ) -> None:
        super().__init__()
        self.num_spatial_tokens = int(num_spatial_tokens)
        self.num_frames = int(num_frames)
        self.mlp1, self.mlp2 = _res_mlp(dim), _res_mlp(dim)
        self.desire_encoder = SiLUProj(desire_in_dim, dim)
        self.traffic_encoder = SiLUProj(2, dim)
        self.action_t_encoder = SiLUProj(2, dim)
        if self.num_spatial_tokens > 1:
            self.temporal_pos_embedding = nn.Parameter(torch.zeros(self.num_frames, 1, dim))
            self.spatial_pos_embedding = nn.Embedding(self.num_spatial_tokens, dim)
        else:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_frames, dim))

    def forward(
        self, vision_seq: torch.Tensor, desire_in: torch.Tensor,
        traffic_convention: torch.Tensor, action_t: torch.Tensor,
    ) -> torch.Tensor:
        batch, dtype = vision_seq.shape[0], vision_seq.dtype
        seq_len = self.num_frames * self.num_spatial_tokens
        tokens = self.mlp2(self.mlp1(vision_seq)).reshape(batch, seq_len, -1)
        desire_emb = self.desire_encoder(desire_in.to(dtype))
        if self.num_spatial_tokens > 1:
            desire_emb = desire_emb.unsqueeze(2).expand(-1, -1, self.num_spatial_tokens, -1)
            desire_emb = desire_emb.reshape(batch, -1, desire_emb.shape[-1])
            spatial_idx = torch.arange(self.num_spatial_tokens, device=tokens.device)
            pos = self.temporal_pos_embedding + self.spatial_pos_embedding(spatial_idx).unsqueeze(0)
            pos = pos.reshape(1, seq_len, -1)
        else:
            pos = self.pos_embedding
        traffic_emb = self.traffic_encoder(traffic_convention.to(dtype)).unsqueeze(1)
        action_emb = self.action_t_encoder(action_t.to(dtype)).unsqueeze(1)
        return tokens + pos + desire_emb + traffic_emb + action_emb


@BLOCK_REGISTRY.register("TemporalHydra")
class TemporalHydra(nn.Module):
    """Multi-head Pre-LN residual MLP with linear projection and optional output scaling."""

    def __init__(self, head_specs: Sequence[Sequence[Any]], dim: int = 512) -> None:
        super().__init__()
        specs = [(str(name), int(out_dim), bool(scaled)) for name, out_dim, scaled in head_specs]
        self.head_names = [name for name, _, _ in specs]
        self.head_mlps = nn.ModuleDict({name: _res_mlp(dim) for name, _, _ in specs})
        self.final_layers = nn.ModuleDict({name: nn.Linear(dim, out) for name, out, _ in specs})
        self.scale_layers = nn.ModuleDict({name: ScaleLayer(out) for name, out, s in specs if s})

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for name in self.head_names:
            pred = self.final_layers[name](self.head_mlps[name](feat))
            outputs[name] = self.scale_layers[name](pred) if name in self.scale_layers else pred
        return outputs


@BLOCK_REGISTRY.register("ResidualActionHead")
class ResidualActionHead(nn.Module):
    """Refine off-policy action mean with a linear or tanh on-policy residual."""
    unpack_inputs = True

    def __init__(self, dim: int = 512, use_tanh: bool = False) -> None:
        super().__init__()
        self.use_tanh = bool(use_tanh)
        self.action_encoder = SiLUProj(2, dim)
        self.head_mlp = _res_mlp(dim)
        self.final_layer = nn.Linear(dim, 2)

    def forward(
        self, feat: torch.Tensor, base_source: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        base_action = base_source["action"] if isinstance(base_source, dict) else base_source
        base_mean, base_std = base_action[:, :2], base_action[:, 2:]
        delta = self.final_layer(self.head_mlp(feat + self.action_encoder(base_mean)))
        delta = torch.tanh(delta) if self.use_tanh else delta
        return torch.cat([base_mean + delta, base_std], dim=-1)


@BLOCK_REGISTRY.register("SupercomboPack")
class SupercomboPack(nn.Module):
    """Pack supercombo policy outputs into float32 tensor and shift feature queues."""
    unpack_inputs = True

    def __init__(self, num_spatial_tokens: int = 1, use_nav: bool = False) -> None:
        super().__init__()
        self.feat_flat_dim = int(num_spatial_tokens) * 512
        self.use_nav = bool(use_nav)
        self.register_buffer("pad", torch.zeros(1, 2))

    def forward(
        self, point_parts: List[torch.Tensor], off_outs: Dict[str, torch.Tensor],
        action_source: Union[torch.Tensor, Dict[str, torch.Tensor]],
        cur_feat: torch.Tensor, state_feat_q: torch.Tensor,
        cur_map: Optional[torch.Tensor] = None, cur_nav: Optional[torch.Tensor] = None,
        state_map_q: Optional[torch.Tensor] = None, state_nav_q: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        action = action_source["action"] if isinstance(action_source, dict) else action_source
        parts: List[torch.Tensor] = [
            *point_parts, off_outs["plan"], off_outs["lead"], off_outs["lead_prob"],
            off_outs["desire_state"], action, cur_feat.reshape(1, self.feat_flat_dim),
        ]
        if self.use_nav and cur_map is not None and cur_nav is not None:
            parts.append(cur_map[:, 0, :] if cur_map.ndim == 3 else cur_map)
            parts.append(cur_nav[:, 0, :] if cur_nav.ndim == 3 else cur_nav)
        parts.append(self.pad.to(action.dtype))
        outputs = torch.cat(parts, dim=1).to(torch.float32)
        feat_end = 2066 + self.feat_flat_dim
        result = {
            "outputs": outputs,
            "next_state_feat_q": _shift_queue(state_feat_q, outputs[:, 2066:feat_end]),
        }
        if self.use_nav and state_map_q is not None and state_nav_q is not None:
            map_end = feat_end + 64
            next_map = _shift_queue(state_map_q, outputs[:, feat_end:map_end])
            next_nav = _shift_queue(state_nav_q, outputs[:, map_end : map_end + 64])
            result["next_state_map_features_q"] = next_map
            result["next_state_navigation_features_q"] = next_nav
        return result


@BLOCK_REGISTRY.register("DMonPreproc")
class DMonPreproc(nn.Module):
    """Unshuffle 4x4 spatial driver camera input and normalize calibration vector."""
    unpack_inputs = True

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("calib_mean", torch.zeros(1, 3))
        self.register_buffer("calib_rsqrt", torch.ones(1, 3))

    def forward(self, input_img: torch.Tensor, calib: torch.Tensor) -> Dict[str, torch.Tensor]:
        dtype, batch = self.calib_mean.dtype, input_img.shape[0]
        calib_norm = (calib.to(dtype) - self.calib_mean) * self.calib_rsqrt
        img = (input_img.to(dtype) / 255.0).reshape(batch, 960, 360, 4).permute(0, 2, 1, 3)
        img = img.reshape(batch, 360, 240, 16).permute(0, 3, 2, 1).contiguous()
        return {"img": img, "calib_norm": calib_norm}


@BLOCK_REGISTRY.register("DMonHeads")
class DMonHeads(nn.Module):
    """Dual LHD/RHD driver state heads, summary head, and feature concatenation."""
    unpack_inputs = True

    def __init__(self) -> None:
        super().__init__()
        self.driver_lhd = nn.ModuleDict({k: _mlp4(dims) for k, dims in _DMON_HEAD_SPECS})
        self.driver_rhd = nn.ModuleDict({k: _mlp4(dims) for k, dims in _DMON_HEAD_SPECS})
        self.summary_predictor = _mlp4((512, 48, 32, 16, 1))

    def forward(self, feat: torch.Tensor, calib_norm: torch.Tensor) -> torch.Tensor:
        feat_calib = torch.cat([feat, calib_norm.to(feat.dtype)], dim=1)
        parts: List[torch.Tensor] = []
        for branch in (self.driver_lhd, self.driver_rhd):
            parts.extend(branch[name](feat) for name in _DMON_VIS_KEYS)
            parts.extend([branch["phone"](feat_calib), branch["sleep"](feat_calib)])
        return torch.cat([*parts, self.summary_predictor(feat), feat], dim=1)
