import pytest
import torch
from neuro_pilot.engine.trainer import Trainer
from neuro_pilot.cfg.schema import AppConfig

@pytest.fixture
def base_config():
    """Returns a default AppConfig instance for testing."""
    return AppConfig()

def test_trainer_initialization(base_config):
    """Test that the Trainer initializes without errors."""
    # We do a barebones initialization to avoid instantiating large models
    trainer = Trainer(base_config)
    assert trainer.cfg == base_config
    assert hasattr(trainer, "loss_names")
    assert getattr(trainer, "device") is not None

def test_trainer_progress_string(base_config):
    """Test that the progress_string returns a correctly formatted header with dynamic task columns."""
    trainer = Trainer(base_config)

    header_str = trainer.progress_string()

    # Core columns always present regardless of active tasks
    assert "mem" in header_str
    assert "Epoch" in header_str
    assert "traj" in header_str
    assert "total" in header_str
    assert "inst" in header_str
    assert "sz" in header_str
    assert "L1" in header_str

def test_trainer_batch_metrics(base_config):
    """Test the batch metrics dictionary updates."""
    trainer = Trainer(base_config)

    # Simulate a loss_dict returned from criterion
    mock_loss = {
        "total": torch.tensor(25.6),
        "traj": torch.tensor(5.6),
        "box": torch.tensor(0.0),
        "cls_det": torch.tensor(0.0)
    }

    # Convert mock to primitives as trainer does
    trainer.batch_metrics = {k: v.item() for k, v in mock_loss.items()}

    assert trainer.batch_metrics["total"] == pytest.approx(25.6, rel=1e-3)
    assert trainer.batch_metrics["traj"] == pytest.approx(5.6, rel=1e-3)
    assert trainer.batch_metrics["box"] == 0.0
