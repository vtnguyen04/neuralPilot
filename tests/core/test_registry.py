import torch.nn as nn
from neuro_pilot.core.registry import Registry, register_backbone, register_head, register_loss, register_neck


def test_registry_decorators():
    # 1. Test Backbone Registration
    @register_backbone("TestBackbone")
    class TestBackbone(nn.Module):
        pass

    assert "TestBackbone" in Registry._BACKBONES
    assert Registry.get_backbone("TestBackbone") == TestBackbone
    assert Registry.get("TestBackbone") == TestBackbone

    # 2. Test Head Registration
    @register_head("TestHead")
    class TestHead(nn.Module):
        pass

    assert "TestHead" in Registry._HEADS
    assert Registry.get_head("TestHead") == TestHead
    assert Registry.get("TestHead") == TestHead


def test_registry_lookup():
    # Test fallback
    assert Registry.get("NonExistent") is None


def test_duplicate_registration_warning(caplog):
    # 1. Backbone duplicate warning
    @register_backbone("DuplicateBackbone")
    class B1(nn.Module):
        pass

    @register_backbone("DuplicateBackbone")
    class B2(nn.Module):
        pass

    assert "Backbone DuplicateBackbone already registered" in caplog.text
    assert Registry.get("DuplicateBackbone") == B2

    # 2. Head duplicate warning
    caplog.clear()
    @register_head("DuplicateHead")
    class H1(nn.Module):
        pass

    @register_head("DuplicateHead")
    class H2(nn.Module):
        pass

    assert "Head DuplicateHead already registered" in caplog.text
    assert Registry.get("DuplicateHead") == H2

    # 3. Loss duplicate warning
    caplog.clear()
    @register_loss("DuplicateLoss")
    class L1(nn.Module):
        pass

    @register_loss("DuplicateLoss")
    class L2(nn.Module):
        pass

    assert "Loss DuplicateLoss already registered" in caplog.text
    assert Registry.get("DuplicateLoss") == L2

    # 4. Neck duplicate warning
    caplog.clear()
    @register_neck("DuplicateNeck")
    class N1(nn.Module):
        pass

    @register_neck("DuplicateNeck")
    class N2(nn.Module):
        pass

    assert "Neck DuplicateNeck already registered" in caplog.text
    assert Registry.get("DuplicateNeck") == N2
