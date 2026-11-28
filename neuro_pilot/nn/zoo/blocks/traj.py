"""Single-camera YUV prep and command-conditioned trajectory decoders (single-mode & MHP)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.nn.zoo.base import BLOCK_REGISTRY
from neuro_pilot.nn.zoo.blocks.cnn import VPConv


def _coord_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create FP32-safe normalized `[-1, 1]` `(1, 2, h, w)` CoordConv channels for ONNX."""
    ys = torch.arange(h, device=device, dtype=torch.float32) * (2.0 / max(1, h - 1)) - 1.0
    xs = torch.arange(w, device=device, dtype=torch.float32) * (2.0 / max(1, w - 1)) - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=0).to(dtype).unsqueeze(0)


@BLOCK_REGISTRY.register("YuvSinglePrep")
class YuvSinglePrep(nn.Module):
    """Normalize single-camera 6ch YUV420 and tile to 24ch for FastViTStem weight transfer."""

    def __init__(self, repeats: int = 4) -> None:
        super().__init__()
        self.repeats = int(repeats)
        self.register_buffer("mean", torch.tensor(127.5, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(63.75, dtype=torch.float32))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return ((img.float() - self.mean) / self.std).repeat(1, self.repeats, 1, 1)


class _TrajAttnBlock(nn.Module):
    """Shared Pre-LN waypoint self-attention + spatial cross-attention + FFN block."""

    def __init__(self, dim: int = 256, num_heads: int = 4, drop: float = 0.05) -> None:
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.sa = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.ca = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(dim * 2, dim)
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.n1(q)
        q = q + self.sa(qn, qn, qn, need_weights=False)[0]
        q = q + self.ca(self.n2(q), kv, kv, need_weights=False)[0]
        return q + self.ffn(self.n3(q))


class _RefineStage(nn.Module):
    """FP32 bilinear grid_sample refinement locking waypoints onto fine P3/P4 lane features."""

    def __init__(self, dim: int = 256, fine_ch: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(dim + fine_ch + 2, dim), nn.LayerNorm(dim), nn.GELU())
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.sa = nn.MultiheadAttention(dim, num_heads, dropout=0.05, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.delta = nn.Linear(dim, 2)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self, q: torch.Tensor, mu: torch.Tensor, fine_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, km, tw, dim = q.shape
        sampled = F.grid_sample(
            fine_map.float(), mu.float().clamp(0.0, 1.0) * 2.0 - 1.0,
            mode="bilinear", padding_mode="border", align_corners=True,
        ).permute(0, 2, 3, 1).to(q.dtype)
        h = self.fuse(torch.cat((q, sampled, mu.to(q.dtype)), dim=-1)).reshape(bsz * km, tw, dim)
        hn = self.n1(h)
        h = h + self.sa(hn, hn, hn, need_weights=False)[0]
        h = (h + self.ffn(self.n2(h))).reshape(bsz, km, tw, dim)
        return h, mu + 0.20 * torch.tanh(self.delta(h))


@BLOCK_REGISTRY.register("TrajHead")
class TrajHead(nn.Module):
    """Single-mode FPN-fused command-FiLM cross-attention decoder predicting 2D waypoints."""

    unpack_inputs: bool = True

    def __init__(
        self, fpn_ch: int = 256, spatial_ch: int = 512, global_ch: int = 512,
        num_waypoints: int = 10, num_commands: int = 4, dim: int = 256, num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.fpn_down = VPConv(fpn_ch, dim // 2, kernel_size=3, stride=2, padding=1, with_bn=True)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(spatial_ch + dim // 2 + 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim), nn.GELU(),
        )
        self.global_proj = nn.Sequential(nn.Linear(global_ch, dim), nn.LayerNorm(dim), nn.GELU())
        self.cmd_embed = nn.Embedding(num_commands, dim)
        self.film_kv, self.film_q = nn.Linear(dim * 2, dim * 2), nn.Linear(dim * 2, dim * 2)
        self.wp_queries = nn.Parameter(torch.randn(1, num_waypoints, dim) * 0.02)
        self.layers = nn.ModuleList([_TrajAttnBlock(dim, 4, 0.1) for _ in range(num_layers)])
        self.out_norm, self.out_head = nn.LayerNorm(dim), nn.Linear(dim, 2)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        y_init = torch.linspace(0.84, 0.36, num_waypoints)
        x_init = torch.full((num_waypoints,), 0.50)
        self.register_buffer("anchor_wp", torch.stack((x_init, y_init), dim=-1).unsqueeze(0))

    def forward(
        self, fpn_feat: torch.Tensor, spatial_feat: torch.Tensor,
        global_feat: torch.Tensor, command: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, h, w = spatial_feat.shape
        coords = _coord_grid(h, w, spatial_feat.device, spatial_feat.dtype).expand(bsz, -1, -1, -1)
        fpn_d = self.fpn_down(fpn_feat)
        if fpn_d.shape[-2:] != (h, w):
            fpn_d = F.interpolate(fpn_d, size=(h, w), mode="bilinear", align_corners=False)
        kv = self.spatial_proj(torch.cat((spatial_feat, fpn_d, coords), dim=1)).flatten(2)
        kv = kv.transpose(1, 2)
        g_vec = global_feat.mean(dim=1) if global_feat.ndim == 3 else global_feat
        ctx = torch.cat((self.global_proj(g_vec), self.cmd_embed(command.long().view(-1))), dim=-1)
        g_kv, b_kv = self.film_kv(ctx).unsqueeze(1).chunk(2, dim=-1)
        kv = kv * (1.0 + g_kv) + b_kv
        g_q, b_q = self.film_q(ctx).unsqueeze(1).chunk(2, dim=-1)
        q = self.wp_queries.expand(bsz, -1, -1) * (1.0 + g_q) + b_q
        for layer in self.layers:
            q = layer(q, kv)
        return self.anchor_wp.to(spatial_feat.dtype) + self.out_head(self.out_norm(q))


@BLOCK_REGISTRY.register("TrajMHPHead")
class TrajMHPHead(nn.Module):
    """Multi-hypothesis (K-mode) Laplace trajectory decoder with P3/P4 grid_sample & heatmap."""

    unpack_inputs: bool = True

    def __init__(
        self, p3_ch: int = 96, p4_ch: int = 160, p5_ch: int = 256, s3_ch: int = 512,
        global_ch: int = 512, num_modes: int = 5, num_waypoints: int = 10,
        num_commands: int = 4, dim: int = 256, fine_ch: int = 128,
        cmd_score: bool = False, hi_res: bool = False,
    ) -> None:
        super().__init__()
        self.num_modes, self.num_waypoints, self.dim = int(num_modes), int(num_waypoints), int(dim)
        self.cmd_score, self.hi_res = bool(cmd_score), bool(hi_res)
        self.p3_down = VPConv(p3_ch, 64, 3, 2, 1, with_bn=True)
        self.fine_proj = VPConv(p4_ch + 64, fine_ch, 3, 1, 1, with_bn=True)
        if self.hi_res:
            self.p3_fine_proj = VPConv(p3_ch + fine_ch, fine_ch, 3, 1, 1, with_bn=True)
            self.film_p3 = nn.Linear(dim * 2, fine_ch * 2)
            self.score_vis, self.mode_cmd_drop = nn.Conv2d(fine_ch, 32, 1), nn.Dropout(0.15)
        self.heatmap_head = nn.Conv2d(fine_ch, 1, kernel_size=1)
        self.p4_down = VPConv(p4_ch, 128, 3, 2, 1, with_bn=True)
        self.s3_proj = VPConv(s3_ch, 128, 1, 1, 0, with_bn=True)
        self.ctx_proj = nn.Sequential(
            nn.Conv2d(p5_ch + 258, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.GELU()
        )
        self.global_proj = nn.Sequential(nn.Linear(global_ch, dim), nn.LayerNorm(dim), nn.GELU())
        self.cmd_embed = nn.Embedding(num_commands, dim)
        self.film_fine = nn.Linear(dim * 2, fine_ch * 2)
        self.film_kv, self.film_q = nn.Linear(dim * 2, dim * 2), nn.Linear(dim * 2, dim * 2)
        self.mode_embed = nn.Parameter(torch.randn(1, num_modes, 1, dim) * 0.02)
        self.wp_embed = nn.Parameter(torch.randn(1, 1, num_waypoints, dim) * 0.02)
        n_c = 3 if self.hi_res else 2
        self.coarse_layers = nn.ModuleList([_TrajAttnBlock(dim, 4, 0.05) for _ in range(n_c)])
        self.coarse_delta = nn.Linear(dim, 2)
        nn.init.zeros_(self.coarse_delta.weight)
        nn.init.zeros_(self.coarse_delta.bias)
        self.refine1, self.refine2 = _RefineStage(dim, fine_ch, 4), _RefineStage(dim, fine_ch, 4)
        self.out_norm, self.log_b_head = nn.LayerNorm(dim), nn.Linear(dim, 2)
        nn.init.zeros_(self.log_b_head.weight)
        nn.init.constant_(self.log_b_head.bias, -1.8)
        ex = (dim if self.cmd_score else 0) + (dim * 2 + 96 if self.hi_res else 0)
        self.mode_fuse = nn.Sequential(
            nn.Linear(dim + ex + num_waypoints * 3, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.mode_attn = nn.MultiheadAttention(dim, 4, dropout=0.05, batch_first=True)
        self.mode_norm = nn.LayerNorm(dim)
        if self.hi_res:
            self.mode_block = _TrajAttnBlock(dim, 4, 0.05)
            self.refine_out = _RefineStage(dim, fine_ch, 4)
        self.mode_score = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self._init_anchors(num_modes, num_waypoints)

    def _init_anchors(self, num_modes: int, num_waypoints: int) -> None:
        t = torch.linspace(0.0, 1.0, num_waypoints)
        y_base = 0.84 - 0.44 * t
        x_curves = [
            0.50 + 0.00 * t, 0.50 - 0.16 * (t**1.4), 0.50 - 0.34 * (t**1.4),
            0.50 + 0.16 * (t**1.4), 0.50 + 0.34 * (t**1.4),
        ][:num_modes]
        anchors = torch.stack([torch.stack((xc, y_base), dim=-1) for xc in x_curves], dim=0)
        self.register_buffer("anchors", anchors.unsqueeze(0))

    def _score_modes(
        self, qn: torch.Tensor, heatmap: torch.Tensor, mu: torch.Tensor,
        cmd_emb: torch.Tensor, g_feat: torch.Tensor, fine_hi: torch.Tensor, kv: torch.Tensor,
    ) -> torch.Tensor:
        hm_prob = torch.sigmoid(heatmap.clamp(-12.0, 12.0))
        mu_s = mu.detach() if self.hi_res else mu
        grid_s = mu_s.clamp(0.0, 1.0) * 2.0 - 1.0
        hm_vals = F.grid_sample(
            hm_prob, grid_s, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1)
        parts = [qn.mean(dim=2)]
        if self.hi_res:
            tk = max(2, self.num_waypoints // 4)
            vm = F.grid_sample(
                self.score_vis(fine_hi), grid_s, mode="bilinear",
                padding_mode="border", align_corners=True,
            ).permute(0, 2, 3, 1)
            parts.extend([
                qn[:, :, -tk:].mean(dim=2),
                g_feat.unsqueeze(1).expand(-1, self.num_modes, -1),
                vm.mean(dim=2), vm[:, :, tk:-tk].mean(dim=2), vm[:, :, -tk:].mean(dim=2),
            ])
        if self.cmd_score:
            ce = self.mode_cmd_drop(cmd_emb) if self.hi_res else cmd_emb
            parts.append(ce.unsqueeze(1).expand(-1, self.num_modes, -1))
        parts.extend([hm_vals, mu_s.flatten(2)])
        m_tok = self.mode_fuse(torch.cat(parts, dim=-1))
        m_n = self.mode_norm(m_tok)
        m_tok = m_tok + self.mode_attn(m_n, m_n, m_n, need_weights=False)[0]
        if self.hi_res:
            m_tok = self.mode_block(m_tok, kv)
        return self.mode_score(m_tok).squeeze(-1)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor,
        s3: torch.Tensor, global_feat: torch.Tensor, command: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.amp.autocast("cuda", enabled=False):
            return self._forward_fp32(
                p3.float(), p4.float(), p5.float(), s3.float(), global_feat.float(), command
            )

    def _forward_fp32(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor,
        s3: torch.Tensor, global_feat: torch.Tensor, command: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz, h5, w5 = p5.shape[0], p5.shape[-2], p5.shape[-1]
        g_vec = global_feat.mean(dim=1) if global_feat.ndim == 3 else global_feat
        g_feat, cmd_emb = self.global_proj(g_vec), self.cmd_embed(command.long().view(-1))
        ctx = torch.cat((g_feat, cmd_emb), dim=-1)
        p3_d = self.p3_down(p3)
        if p3_d.shape[-2:] != p4.shape[-2:]:
            p3_d = F.interpolate(p3_d, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        fine = self.fine_proj(torch.cat((p4, p3_d), dim=1))
        gf, bf = self.film_fine(ctx).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        fine = fine * (1.0 + torch.tanh(gf)) + bf
        heatmap = self.heatmap_head(fine)
        if self.hi_res:
            fine_up = F.interpolate(fine, size=p3.shape[-2:], mode="bilinear", align_corners=False)
            fine_hi = self.p3_fine_proj(torch.cat((p3, fine_up), dim=1))
            gp3, bp3 = self.film_p3(ctx).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            fine_hi = fine_hi * (1.0 + torch.tanh(gp3)) + bp3
        else:
            fine_hi = fine
        p4_d = self.p4_down(p4)
        if p4_d.shape[-2:] != (h5, w5):
            p4_d = F.interpolate(p4_d, size=(h5, w5), mode="bilinear", align_corners=False)
        s3_u = F.interpolate(
            self.s3_proj(F.group_norm(s3, 1, eps=1e-5)), size=(h5, w5),
            mode="bilinear", align_corners=False,
        )
        coords = _coord_grid(h5, w5, p5.device, p5.dtype).expand(bsz, -1, -1, -1)
        kv = self.ctx_proj(torch.cat((p5, p4_d, s3_u, coords), dim=1)).flatten(2).transpose(1, 2)
        g_kv, b_kv = self.film_kv(ctx).unsqueeze(1).chunk(2, dim=-1)
        kv = kv * (1.0 + torch.tanh(g_kv)) + b_kv
        q = (self.mode_embed + self.wp_embed).expand(bsz, -1, -1, -1)
        g_q, b_q = self.film_q(ctx).view(bsz, 1, 1, self.dim * 2).chunk(2, dim=-1)
        q = (q * (1.0 + torch.tanh(g_q)) + b_q).reshape(bsz * self.num_modes, -1, self.dim)
        kv_rep = kv.unsqueeze(1).expand(-1, self.num_modes, -1, -1).reshape(
            bsz * self.num_modes, h5 * w5, self.dim
        )
        for blk in self.coarse_layers:
            q = blk(q, kv_rep)
        q = q.reshape(bsz, self.num_modes, self.num_waypoints, self.dim)
        mu0 = self.anchors.to(p5.dtype) + 0.55 * torch.tanh(self.coarse_delta(q))
        q, mu1 = self.refine1(q, mu0, fine)
        q, mu = self.refine2(q, mu1, fine_hi)
        qn = self.out_norm(q)
        log_b = self.log_b_head(qn).clamp(-3.5, 1.0)
        mode_logits = self._score_modes(qn, heatmap, mu, cmd_emb, g_feat, fine_hi, kv)
        if self.hi_res:
            p = F.softmax(mode_logits, dim=-1).view(bsz, self.num_modes, 1, 1)
            qm, mm = (p * qn).sum(1, keepdim=True), (p * mu).sum(1, keepdim=True)
            best_wp = self.refine_out(qm, mm, fine_hi)[1].squeeze(1)
        else:
            idx = mode_logits.argmax(dim=1).view(bsz, 1, 1, 1).expand(-1, 1, self.num_waypoints, 2)
            best_wp = torch.gather(mu, 1, idx).squeeze(1)
        return {
            "waypoints": best_wp, "mu": mu, "mu0": mu0,
            "log_b": log_b, "mode_logits": mode_logits, "heatmap": heatmap,
        }
