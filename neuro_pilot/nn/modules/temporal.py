"""Temporal feature aggregation for trajectory prediction.

This module operates at the **feature level**, not the backbone level.
The backbone processes each frame independently (unchanged).
The aggregator takes per-frame features and produces temporal context
that is fed ONLY to the trajectory head.

Design decisions:
    - Backbone stays structurally unmodified during runtime forwarding.
    - Detection, heatmap, classification heads are NOT affected
    - Each aggregator strategy is registered via AGGREGATOR_REGISTRY (Strategy pattern)
    - When only 1 frame is provided, aggregation is a no-op (backward compat)

Usage in trainer:
    backbone_feats_per_frame = [backbone(frame_t) for t in range(T)]
    temporal_context = aggregator(backbone_feats_per_frame)
    trajectory_head(current_frame_feat, temporal_context=temporal_context)

Reference:
    - Hu et al., "ST-P3", ECCV 2022 (temporal feature learning for driving)
    - Chitta et al., "TransFuser", CVPR 2022 (multi-frame fusion)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_pilot.utils.logger import logger

# ---------------------------------------------------------------------------
# Aggregator Registry (GoF Strategy + Registry)
# ---------------------------------------------------------------------------
AGGREGATOR_REGISTRY: dict[str, type["BaseAggregator"]] = {}


def register_aggregator(name: str):
    """Class decorator that registers a temporal aggregation strategy."""
    def _decorator(cls: type[BaseAggregator]) -> type[BaseAggregator]:
        if name in AGGREGATOR_REGISTRY:
            logger.warning(f"Aggregator '{name}' already registered. Overwriting.")
        AGGREGATOR_REGISTRY[name] = cls
        return cls
    return _decorator


def build_aggregator(name: str, feature_dim: int, clip_length: int, **kwargs) -> "BaseAggregator":
    """Factory: instantiate an aggregator by config name."""
    if name not in AGGREGATOR_REGISTRY:
        available = list(AGGREGATOR_REGISTRY.keys())
        raise ValueError(f"Unknown aggregator '{name}'. Available: {available}")
    return AGGREGATOR_REGISTRY[name](feature_dim=feature_dim, clip_length=clip_length, **kwargs)


# ---------------------------------------------------------------------------
# Abstract Base (ISP: minimal interface)
# ---------------------------------------------------------------------------
class BaseAggregator(nn.Module, ABC):
    """Interface for temporal feature aggregation.

    Input:  list of per-frame pooled features ``[B, D]`` × T
    Output: temporal context tensor ``[B, D]``
    """

    def __init__(self, feature_dim: int, clip_length: int, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.clip_length = clip_length

    @abstractmethod
    def forward(
        self,
        frame_features: list[torch.Tensor],
        temporal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aggregate per-frame features into temporal context.

        Args:
            frame_features: List of T tensors, each ``[B, D]``.
            temporal_mask: Optional ``[B, T]`` (True = valid frame).

        Returns:
            Dict with at least 'context' key ``[B, D]``.
            May also return 'motion' ``[B, D]``, 'velocity' ``[B, 1]``, etc.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete Strategies
# ---------------------------------------------------------------------------
@register_aggregator("concat")
class ConcatAggregator(BaseAggregator):
    """Concat + MLP reduction. Simple and fast baseline."""

    def __init__(self, feature_dim: int, clip_length: int, **kwargs):
        super().__init__(feature_dim, clip_length)
        self.reduce = nn.Sequential(
            nn.Linear(feature_dim * clip_length, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, frame_features, temporal_mask=None):
        # Pad / truncate to clip_length
        T = len(frame_features)
        if T < self.clip_length:
            pad = [frame_features[-1]] * (self.clip_length - T)
            frame_features = list(frame_features) + pad
        feats = torch.cat(frame_features[:self.clip_length], dim=-1)  # [B, D*T]
        context = self.reduce(feats)  # [B, D]
        return {"context": context}


@register_aggregator("temporal_attention")
class TemporalAttentionAggregator(BaseAggregator):
    """Cross-attention: current frame queries, all frames as context.

    Lightweight and effective — the default strategy.
    """

    def __init__(self, feature_dim: int, clip_length: int, num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(feature_dim, clip_length)

        self.temporal_pe = nn.Parameter(torch.randn(1, clip_length, feature_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

        # Motion extractor from temporal differences
        self.motion_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, frame_features, temporal_mask=None):
        B = frame_features[0].shape[0]
        T = len(frame_features)
        D = self.feature_dim

        # Stack: [B, T, D]
        stacked = torch.stack(frame_features, dim=1)

        # Positional encoding
        pe = self.temporal_pe[:, :T]
        context_seq = stacked + pe

        # Current frame as query (last frame = most recent observation)
        query = stacked[:, -1:, :]  # [B, 1, D]

        # Key padding mask
        if temporal_mask is not None:
            key_mask = ~temporal_mask[:, :T]  # True = ignored
        else:
            key_mask = None

        # Cross-attention
        attn_out, _ = self.cross_attn(query, context_seq, context_seq, key_padding_mask=key_mask)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ffn(query))

        context = query.squeeze(1)  # [B, D]

        # Motion from temporal differences
        if T > 1:
            diffs = stacked[:, 1:] - stacked[:, :-1]
            motion = self.motion_proj(diffs.mean(dim=1))
        else:
            motion = torch.zeros(B, D, device=context.device, dtype=context.dtype)

        return {"context": context, "motion": motion}


@register_aggregator("gru")
class GRUAggregator(BaseAggregator):
    """GRU-based temporal encoding. Lightweight, good for edge deployment."""

    def __init__(self, feature_dim: int, clip_length: int, **kwargs):
        super().__init__(feature_dim, clip_length)
        self.gru = nn.GRU(
            input_size=feature_dim, hidden_size=feature_dim,
            num_layers=1, batch_first=True,
        )

    def forward(self, frame_features, temporal_mask=None):
        stacked = torch.stack(frame_features, dim=1)  # [B, T, D]
        output, _ = self.gru(stacked)
        context = output[:, -1]  # Last hidden state [B, D]
        return {"context": context}
