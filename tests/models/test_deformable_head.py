"""Tests for Deformable Trajectory Head and SOTA losses."""

import pytest
import torch

from neuro_pilot.nn.modules.deformable import (
    MultiScaleDeformableAttention,
    DeformableDecoderLayer,
    WaypointQueryDecoder,
    sinusoidal_positional_encoding,
)
from neuro_pilot.nn.modules.head import DeformableTrajectoryHead
from neuro_pilot.utils.losses import CollisionLoss, ProgressLoss


class TestMultiScaleDeformableAttention:
    """Tests for the standard MSDA module."""

    def test_output_shape(self):
        B, Q, D, H, W = 2, 60, 256, 10, 10
        msda = MultiScaleDeformableAttention(embed_dim=D, num_heads=4, n_levels=1, n_points=4)
        query = torch.randn(B, Q, D)
        reference_points = torch.rand(B, Q, 1, 2)
        value = torch.randn(B, H * W, D)
        spatial_shapes = torch.tensor([[H, W]])
        level_start_index = torch.tensor([0])

        out = msda(query, reference_points, value, spatial_shapes, [(H, W)], level_start_index)
        assert out.shape == (B, Q, D)

    def test_multi_level(self):
        B, Q, D = 2, 30, 128
        H1, W1 = 10, 10
        H2, W2 = 5, 5
        S = H1 * W1 + H2 * W2
        msda = MultiScaleDeformableAttention(embed_dim=D, num_heads=4, n_levels=2, n_points=4)
        query = torch.randn(B, Q, D)
        reference_points = torch.rand(B, Q, 2, 2)
        value = torch.randn(B, S, D)
        spatial_shapes = torch.tensor([[H1, W1], [H2, W2]])
        level_start_index = torch.tensor([0, H1 * W1])

        out = msda(query, reference_points, value, spatial_shapes, [(H1, W1), (H2, W2)], level_start_index)
        assert out.shape == (B, Q, D)


class TestDeformableDecoderLayer:
    def test_forward(self):
        B, T, D, H, W = 2, 60, 256, 10, 10
        layer = DeformableDecoderLayer(embed_dim=D, num_heads=4, n_levels=1, n_points=4, ffn_dim=512)
        query = torch.randn(B, T, D)
        query_pos = torch.randn(B, T, D)
        ref_pts = torch.rand(B, T, 1, 2)
        memory = torch.randn(B, H * W, D)
        spatial_shapes = torch.tensor([[H, W]])
        level_start_index = torch.tensor([0])

        out = layer(query, query_pos, ref_pts, memory, spatial_shapes, [(H, W)], level_start_index)
        assert out.shape == (B, T, D)


class TestDeformableTrajectoryHead:
    def test_output_shape(self):
        B, C, H, W = 2, 256, 10, 10
        head = DeformableTrajectoryHead(
            ch=[C], num_commands=4, num_waypoints=60, embed_dim=256, num_heads=4, num_layers=2, n_points=4
        )
        x = torch.randn(B, C, H, W)
        out = head(x, cmd=torch.zeros(B, dtype=torch.long), vEgo=torch.ones(B))

        assert "waypoints" in out
        assert out["waypoints"].shape == (B, 60, 2)
        assert "has_traj_logit" in out

    def test_without_conditioning(self):
        B, C, H, W = 1, 128, 8, 8
        head = DeformableTrajectoryHead(ch=[C], num_commands=4, num_waypoints=10, embed_dim=128, num_heads=4)
        x = torch.randn(B, C, H, W)
        out = head(x)
        assert out["waypoints"].shape == (B, 10, 2)

    def test_with_list_input(self):
        B, C, H, W = 2, 256, 10, 10
        head = DeformableTrajectoryHead(ch=[C], num_commands=4, num_waypoints=60, embed_dim=256)
        p5 = torch.randn(B, C, H, W)
        hm = {"heatmap": torch.randn(B, 1, 40, 40)}
        out = head([p5, hm], cmd=torch.tensor([0, 1]))
        assert out["waypoints"].shape == (B, 60, 2)

    def test_gradient_flow(self):
        B, C, H, W = 1, 128, 8, 8
        head = DeformableTrajectoryHead(ch=[C], num_commands=4, num_waypoints=10, embed_dim=128, num_heads=4)
        x = torch.randn(B, C, H, W, requires_grad=True)
        out = head(x)
        loss = out["waypoints"].sum()
        loss.backward()
        assert x.grad is not None


class TestSinusoidalPE:
    def test_shape(self):
        pe = sinusoidal_positional_encoding(60, 256)
        assert pe.shape == (60, 256)

    def test_bounded(self):
        pe = sinusoidal_positional_encoding(100, 128)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0


class TestCollisionLoss:
    def test_with_heatmap(self):
        loss_fn = CollisionLoss()
        pred_wp = torch.zeros(2, 10, 2)
        heatmap = torch.ones(2, 1, 20, 20) * 10  # High = road = safe
        loss = loss_fn(pred_wp, heatmap=heatmap)
        assert loss.item() < 0.1  # Should be near 0 (safe)

    def test_with_obstacle_heatmap(self):
        loss_fn = CollisionLoss()
        pred_wp = torch.zeros(2, 10, 2)
        heatmap = torch.ones(2, 1, 20, 20) * -10  # Low = obstacle
        loss = loss_fn(pred_wp, heatmap=heatmap)
        assert loss.item() > 0.5  # Should be high (collision)

    def test_no_inputs_returns_zero(self):
        loss_fn = CollisionLoss()
        pred_wp = torch.randn(2, 10, 2)
        loss = loss_fn(pred_wp)
        assert loss.item() == 0.0


class TestProgressLoss:
    def test_forward_trajectory(self):
        loss_fn = ProgressLoss()
        # Monotonically increasing y → should be 0
        pred_wp = torch.zeros(1, 10, 2)
        pred_wp[0, :, 1] = torch.linspace(0, 1, 10)
        loss = loss_fn(pred_wp)
        assert loss.item() == 0.0

    def test_stationary_trajectory(self):
        loss_fn = ProgressLoss()
        # All same position → dy=0 → relu(-0)=0
        pred_wp = torch.zeros(1, 10, 2)
        loss = loss_fn(pred_wp)
        assert loss.item() == 0.0

    def test_backward_trajectory(self):
        loss_fn = ProgressLoss()
        # Decreasing y → should penalize
        pred_wp = torch.zeros(1, 10, 2)
        pred_wp[0, :, 1] = torch.linspace(1, 0, 10)
        loss = loss_fn(pred_wp)
        assert loss.item() > 0.0


class TestBuildDeformableModel:
    def test_build_from_yaml(self):
        from neuro_pilot.nn.factory import build_model

        model = build_model("neuro_pilot/cfg/models/neuralPilot_deformable.yaml", ch=3, nc=14, scale="s")
        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        print(f"Deformable model params: {total_params / 1e6:.2f}M")
