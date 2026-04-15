import pytest
import torch
import torch.nn as nn
from neuro_pilot.nn.modules.temporal import build_aggregator, BaseAggregator, AGGREGATOR_REGISTRY
from neuro_pilot.nn.modules.temporal_heads import TemporalTrajectoryHead

@pytest.fixture
def mock_frames():
    # Batch size 2, feature dim 256
    return [torch.randn(2, 256) for _ in range(4)]

def test_aggregator_registry_coverage():
    """Verify all defined temporal strategies can be instantiated correctly."""
    assert 'gru' in AGGREGATOR_REGISTRY
    assert 'concat' in AGGREGATOR_REGISTRY
    assert 'temporal_attention' in AGGREGATOR_REGISTRY

@pytest.mark.parametrize("agg_type", ['gru', 'concat', 'temporal_attention'])
def test_temporal_aggregator_forward(agg_type, mock_frames):
    """Test the base TemporalAggregator with dynamic strategy injection."""
    agg = build_aggregator(name=agg_type, feature_dim=256, clip_length=4)
    out = agg(mock_frames)

    # Must output a dict containing context
    assert isinstance(out, dict)
    assert 'context' in out

    # Context must match input batch size and sequence dim representation
    assert out['context'].shape == (2, 256)

def test_temporal_trajectory_head_fallback():
    """Ensure the Temporal Head correctly falls back to single-view spatial-only when no temporal context is provided."""
    head = TemporalTrajectoryHead(ch_in=[256], num_commands=4, num_waypoints=10)

    B, C, H, W = 2, 256, 8, 8
    # Using the last channel feature map to simulate backbone output
    spatial_input = torch.randn(B, C, H, W)

    cmd = torch.randint(0, 4, (B,))

    out = head(spatial_input, cmd=cmd, temporal_context=None)
    assert 'waypoints' in out
    assert out['waypoints'].shape == (B, 10, 2)

def test_temporal_trajectory_head_fusion():
    """Test adaptive fusion inside the Trajectory Head when Temporal Context is provided."""
    head = TemporalTrajectoryHead(ch_in=[256], num_commands=4, num_waypoints=10)

    B, C, H, W = 2, 256, 8, 8
    spatial_input = torch.randn(B, C, H, W)
    cmd = torch.randint(0, 4, (B,))

    # Provide temporal context explicitly (Context + Motion prior)
    temporal_ctx = {
        'context': torch.randn(B, 512),
        'motion': torch.randn(B, 512)
    }

    out = head(spatial_input, cmd=cmd, temporal_context=temporal_ctx)

    assert 'waypoints' in out
    assert out['waypoints'].shape == (B, 10, 2)

    # Simple check to see if the fused linear layers processed the logic
    assert not torch.isnan(out['waypoints']).any()

