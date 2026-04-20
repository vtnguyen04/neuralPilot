"""Temporal-aware trajectory prediction head.

This head extends the existing trajectory prediction with optional
temporal context, enabling video-based trajectory training.

Key principles:
    - ONLY trajectory prediction benefits from temporal context
    - Detection, heatmap, classification heads are NOT changed
    - Falls back to standard single-frame behavior when no temporal
      context is provided (Liskov Substitution Principle)
    - Can be used independently — train trajectory-only with video data

The temporal context (from TemporalAggregator) is fused via an adaptive
gate that learns when temporal information is useful vs. spatial-only.

Reference:
    - Hu et al., "ST-P3", ECCV 2022
    - Wu et al., "TCP: Trajectory-Conditioned Policy", CoRL 2022
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseHead


class TemporalTrajectoryHead(BaseHead):
    """Trajectory head with optional temporal context for video training.

    When ``temporal_context`` is absent in kwargs: pure spatial (same as TrajectoryHead).
    When ``temporal_context`` is present: fuses spatial + temporal via adaptive gate.

    This head can be used in two modes:
        1. Single-frame: behaves exactly like TrajectoryHead
        2. Video/temporal: receives temporal context from TemporalAggregator

    Args:
        ch_in: Input channel dims from backbone.
        num_commands: Number of navigation commands.
        num_waypoints: Number of output waypoints.
        temporal_dim: Dimension of temporal context vector (must match aggregator output).
        predict_velocity: Whether to predict per-waypoint velocity.
    """

    forward_with_kwargs = True

    def __init__(
        self,
        ch_in,
        num_commands: int = 4,
        num_waypoints: int = 10,
        temporal_dim: int = 512,
        predict_velocity: bool = False,
    ):
        super().__init__()
        self.c5_dim = ch_in[0] if isinstance(ch_in, (list, tuple)) else ch_in
        self.num_commands = num_commands
        self.num_waypoints = num_waypoints
        self.predict_velocity = predict_velocity

        # --- Command embedding + FiLM (same as TrajectoryHead) ---
        self.cmd_embed = nn.Embedding(num_commands, 64)
        self.film_gen = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024),
        )

        # --- Spatial stem ---
        flatten_dim = self.c5_dim * 4 * 4
        self.spatial_stem = nn.Sequential(
            nn.Linear(flatten_dim + 64, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
        )

        # --- Temporal fusion (only created, only used when temporal_context exists) ---
        self.temporal_gate = nn.Sequential(
            nn.Linear(512 + temporal_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Sigmoid(),
        )
        self.temporal_proj = nn.Linear(temporal_dim, 512)

        # --- Trajectory head (Bézier) ---
        self.traj_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4 * 2),
        )
        self.exist_head = nn.Linear(512, 1)

        # --- Optional velocity prediction ---
        if predict_velocity:
            self.velocity_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_waypoints),
            )

        # Bézier basis
        t = torch.linspace(0, 1, num_waypoints)
        self.register_buffer("bernstein_m", self._bernstein(t))

    @staticmethod
    def _bernstein(t: torch.Tensor) -> torch.Tensor:
        b0 = (1 - t) ** 3
        b1 = 3 * (1 - t) ** 2 * t
        b2 = 3 * (1 - t) * t**2
        b3 = t**3
        return torch.stack([b0, b1, b2, b3], dim=1)

    def _apply_film(self, h, cmd_emb):
        film = self.film_gen(cmd_emb)
        gamma, beta = film.chunk(2, dim=1)
        return h * (1 + gamma) + beta

    def forward(self, x, **kwargs):
        """Forward pass.

        Args:
            x: Spatial features from backbone/neck.
            **kwargs:
                cmd / cmd_idx: Command indices.
                temporal_context: Dict from TemporalAggregator with 'context' key [B, D].
                    When absent, behaves as single-frame trajectory head.
                heatmap: Optional heatmap for attention guidance.
        """
        if isinstance(x, list):
            p5 = x[0]
        else:
            p5 = x
        B = p5.shape[0]

        # --- Command ---
        cmd_idx = kwargs.get("cmd", kwargs.get("cmd_idx"))
        if cmd_idx is None:
            cmd_idx = torch.zeros(B, dtype=torch.long, device=p5.device)
        if cmd_idx.dim() > 1:
            cmd_idx = cmd_idx.argmax(dim=-1) if cmd_idx.shape[-1] == self.num_commands else cmd_idx.view(-1)

        dtype = self.spatial_stem[0].weight.dtype

        # --- Heatmap guidance (optional, same as TrajectoryHead) ---
        heatmap = kwargs.get("heatmap")
        if heatmap is not None:
            if isinstance(heatmap, dict):
                heatmap = heatmap.get("heatmap")
            if heatmap is not None and heatmap.ndim == 3:
                heatmap = heatmap.unsqueeze(1)
            if heatmap is not None:
                mask = F.interpolate(torch.sigmoid(heatmap), size=p5.shape[2:], mode="bilinear", align_corners=False)
                p5 = p5 * (1.0 + mask)

        # --- Spatial pathway ---
        cmd_emb = self.cmd_embed(cmd_idx.long()).to(dtype)
        pooled = F.interpolate(p5, size=(4, 4), mode="bilinear", align_corners=False).flatten(1).to(dtype)
        h = self.spatial_stem(torch.cat([pooled, cmd_emb], dim=1))

        # --- Temporal fusion (only when temporal context is available) ---
        temporal_ctx = kwargs.get("temporal_context")
        if temporal_ctx is not None and isinstance(temporal_ctx, dict):
            ctx = temporal_ctx["context"].to(dtype)  # [B, temporal_dim]
            motion = temporal_ctx.get("motion")

            # Add motion if available
            if motion is not None:
                ctx = ctx + motion.to(dtype)

            # Adaptive gate: learn how much to trust temporal vs spatial
            gate = self.temporal_gate(torch.cat([h, ctx], dim=1))  # [B, 512] values in [0,1]
            ctx_proj = self.temporal_proj(ctx)  # [B, 512]
            h = gate * h + (1 - gate) * ctx_proj

        # --- FiLM modulation ---
        h = self._apply_film(h, cmd_emb.to(self.film_gen[0].weight.dtype))

        # --- Predict ---
        cp = torch.tanh(self.traj_head(h)).view(B, 4, 2)
        has_traj_logit = self.exist_head(h)
        waypoints = torch.einsum("nk,bkd->bnd", self.bernstein_m.to(cp.dtype), cp)
        waypoints = torch.nan_to_num(waypoints, 0.0)

        res = {"waypoints": waypoints, "control_points": cp, "has_traj_logit": has_traj_logit}

        if self.predict_velocity and temporal_ctx is not None:
            res["velocity"] = F.softplus(self.velocity_head(h))

        return {"trajectory": res, **res}
