"""Standard Multi-Scale Deformable Attention for trajectory prediction.

Reference implementation from:
- Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
- HuggingFace Transformers: modeling_deformable_detr.py

Adapted for NeuroPilot waypoint prediction with single/multi-scale spatial features.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ms_deform_attn_core(
    value: torch.Tensor,
    spatial_shapes_list: list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Standard multi-scale deformable attention core using F.grid_sample.

    Reference: Zhu et al., Deformable DETR (ICLR 2021), Eq. (1).
    Implementation follows HuggingFace Transformers MultiScaleDeformableAttention.

    Args:
        value: [B, S, num_heads, head_dim] — Projected values (S = sum of H_l * W_l).
        spatial_shapes_list: List of (H, W) per feature level.
        sampling_locations: [B, Q, num_heads, n_levels, n_points, 2] — Normalized [0, 1].
        attention_weights: [B, Q, num_heads, n_levels, n_points] — Softmax weights.

    Returns:
        [B, Q, embed_dim] — Aggregated attention output.
    """
    batch_size, _, num_heads, head_dim = value.shape
    _, num_queries, _, _, num_points, _ = sampling_locations.shape

    # Split values by spatial level
    value_list = value.split([h * w for h, w in spatial_shapes_list], dim=1)

    # Convert sampling locations from [0, 1] to grid_sample range [-1, 1]
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level_id, (height, width) in enumerate(spatial_shapes_list):
        # [B, H*W, num_heads, head_dim] → [B*num_heads, head_dim, H, W]
        value_l = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, head_dim, height, width)
        )

        # [B, Q, num_heads, n_points, 2] → [B*num_heads, Q, n_points, 2]
        sampling_grid_l = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)

        # Bilinear interpolation at sampling points
        # Output: [B*num_heads, head_dim, Q, n_points]
        sampling_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)

    # Reshape attention weights: [B, Q, M, L, P] → [B*M, 1, Q, L*P]
    attention_weights = attention_weights.transpose(1, 2).reshape(batch_size * num_heads, 1, num_queries, -1)

    # Stack sampled values across levels and weight them
    # [B*M, head_dim, Q, L*P] * [B*M, 1, Q, L*P] → sum → [B*M, head_dim, Q]
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1)

    # [B*M, head_dim, Q] → [B, Q, M*head_dim] = [B, Q, embed_dim]
    output = output.view(batch_size, num_heads * head_dim, num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Standard Multi-Scale Deformable Attention (Zhu et al., ICLR 2021).

    Each query predicts K sampling offsets and attention weights per head per level.
    Features are sampled via bilinear interpolation (F.grid_sample) at learned positions.

    Args:
        embed_dim: Total embedding dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        n_levels: Number of feature map levels (1 for single-scale, 3-4 for multi-scale).
        n_points: Number of sampling points per head per level.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        n_levels: int = 1,
        n_points: int = 4,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.n_levels = n_levels
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following Deformable DETR paper."""
        nn.init.constant_(self.sampling_offsets.weight, 0.0)

        # Initialize sampling offsets bias as a radial grid pattern
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: [B, Q, D] — Query embeddings (waypoint queries).
            reference_points: [B, Q, n_levels, 2] — Normalized reference points.
            value: [B, S, D] — Flattened spatial features (S = sum(H_l * W_l)).
            spatial_shapes: [n_levels, 2] — (H, W) per level as tensor.
            spatial_shapes_list: List of (H, W) tuples (for grid_sample).
            level_start_index: [n_levels] — Start index per level in flattened value.
            position_embeddings: [B, Q, D] — Optional positional embeddings.

        Returns:
            [B, Q, D] — Attention output.
        """
        if position_embeddings is not None:
            query = query + position_embeddings

        batch_size, num_queries, _ = query.shape
        batch_size, seq_len, _ = value.shape

        # Project values: [B, S, D] → [B, S, num_heads, head_dim]
        value = self.value_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Predict sampling offsets: [B, Q, num_heads * n_levels * n_points * 2]
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_queries, self.num_heads, self.n_levels, self.n_points, 2
        )

        # Predict attention weights: [B, Q, num_heads * n_levels * n_points]
        attention_weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            batch_size, num_queries, self.num_heads, self.n_levels, self.n_points
        )

        # Compute sampling locations = reference_points + normalized offsets
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        # Core deformable attention
        output = ms_deform_attn_core(value, spatial_shapes_list, sampling_locations, attention_weights)

        return self.output_proj(output)


class DeformableDecoderLayer(nn.Module):
    """Standard DETR decoder layer: SelfAttn → Deformable CrossAttn → FFN.

    Each layer refines waypoint queries by:
    1. Self-attention among queries (temporal reasoning).
    2. Deformable cross-attention to spatial features (where to look).
    3. Feed-forward network (per-query feature transformation).

    Args:
        embed_dim: Hidden dimension.
        num_heads: Number of attention heads.
        n_levels: Number of feature levels.
        n_points: Number of sampling points per head per level.
        ffn_dim: FFN intermediate dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        n_levels: int = 1,
        n_points: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-Attention (standard MHA for inter-query reasoning)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-Attention (deformable for spatial feature sampling)
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, num_heads, n_levels, n_points)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: [B, T, D] — Waypoint query embeddings.
            query_pos: [B, T, D] — Positional embeddings for queries.
            reference_points: [B, T, n_levels, 2] — Normalized reference points.
            memory: [B, S, D] — Flattened spatial features from backbone.
            spatial_shapes: [n_levels, 2] — Feature map dimensions.
            spatial_shapes_list: List of (H, W) tuples.
            level_start_index: [n_levels] — Start indices.

        Returns:
            [B, T, D] — Refined query embeddings.
        """
        # 1. Self-Attention (queries attend to each other)
        q_sa = query + query_pos
        residual = query
        attn_out, _ = self.self_attn(q_sa, q_sa, query)
        query = self.norm1(residual + self.dropout1(attn_out))

        # 2. Deformable Cross-Attention (queries attend to spatial features)
        residual = query
        cross_out = self.cross_attn(
            query=query,
            reference_points=reference_points,
            value=memory,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            position_embeddings=query_pos,
        )
        query = self.norm2(residual + self.dropout2(cross_out))

        # 3. FFN
        residual = query
        query = self.norm3(residual + self.ffn(query))

        return query


class WaypointQueryDecoder(nn.Module):
    """DETR-style decoder stack for waypoint prediction.

    Stacks L DeformableDecoderLayers that iteratively refine
    waypoint queries through self-attention and cross-attention.

    Args:
        embed_dim: Hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of decoder layers.
        n_levels: Number of feature levels.
        n_points: Number of sampling points.
        ffn_dim: FFN intermediate dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        n_levels: int = 1,
        n_points: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DeformableDecoderLayer(embed_dim, num_heads, n_levels, n_points, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through all decoder layers.

        Args:
            queries: [B, T, D] — Initial waypoint queries.
            query_pos: [B, T, D] — Temporal positional embeddings.
            reference_points: [B, T, n_levels, 2] — Reference positions.
            memory: [B, S, D] — Spatial features.
            spatial_shapes: [n_levels, 2].
            spatial_shapes_list: List of (H, W).
            level_start_index: [n_levels].

        Returns:
            [B, T, D] — Refined query embeddings.
        """
        output = queries
        for layer in self.layers:
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                memory=memory,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
            )
        return self.norm(output)


def sinusoidal_positional_encoding(num_positions: int, embed_dim: int) -> torch.Tensor:
    """Generate sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        num_positions: Number of positions (T = num_waypoints).
        embed_dim: Embedding dimension.

    Returns:
        [num_positions, embed_dim] — Positional encoding matrix.
    """
    pe = torch.zeros(num_positions, embed_dim)
    position = torch.arange(0, num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
