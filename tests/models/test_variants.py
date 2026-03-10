
import sys
import torch
import unittest
from unittest.mock import MagicMock, patch
from neuro_pilot.nn.tasks import DetectionModel

# We need to ensure timm is mocked ONLY for these tests
class TestModelVariants(unittest.TestCase):
    def setUp(self):
        self.cfg_path = "neuro_pilot/cfg/models/neuralPilot.yaml"

        # Create a mock timm module
        self.mock_timm = MagicMock()

        # Configure mock timm model to return real tensors
        class MockTimmModel(torch.nn.Module):
            def __init__(self, model_name=""):
                super().__init__()
                self.feature_info = MagicMock()
                # Mocking channels for MobileNetV4 variants to match TimmBackbone.get_channels fallback
                if 'small' in model_name:
                    self.chs = [32, 32, 64, 96, 960]
                elif 'medium' in model_name:
                    self.chs = [32, 48, 80, 160, 960]
                elif 'large' in model_name:
                    self.chs = [24, 48, 96, 192, 960]
                else:
                    self.chs = [32, 32, 64, 96, 960]

                self.feature_info.channels.return_value = self.chs

            def forward(self, x):
                return [torch.zeros(x.shape[0], self.chs[i], x.shape[2]//(2**i), x.shape[3]//(2**i))
                        for i in range(len(self.chs))]

        def create_mock_model(model_name, **kwargs):
            return MockTimmModel(model_name)

        self.mock_timm.create_model.side_effect = create_mock_model

        # Patch timm in the backbone module where it is used
        self.patcher = patch('neuro_pilot.nn.modules.backbone.timm', self.mock_timm)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_nano_variant(self):
        print("\n--- Testing Nano Variant (Scale='n') ---")
        model = DetectionModel(cfg=self.cfg_path, nc=14, scale='n', verbose=False)
        model.train()
        img = torch.randn(2, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)

    def test_small_variant(self):
        print("\n--- Testing Small Variant (Scale='s') ---")
        model = DetectionModel(cfg=self.cfg_path, nc=14, scale='s', verbose=False)
        model.train()
        img = torch.randn(2, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)

    def test_large_variant(self):
        print("\n--- Testing Large Variant (Scale='l') ---")
        model = DetectionModel(cfg=self.cfg_path, nc=14, scale='l', verbose=False)
        model.train()
        img = torch.randn(2, 3, 224, 224)
        out = model(img)
        self.assertIn('one2many', out)

    def test_all_tasks_outputs(self):
        print("\n--- Testing All Tasks Outputs ---")
        model = DetectionModel(cfg=self.cfg_path, nc=14, scale='n', verbose=False)
        model.train()
        img = torch.randn(2, 3, 224, 224)
        cmd = torch.zeros(2, dtype=torch.long)
        out = model(img, cmd_idx=cmd)

        self.assertIn('one2many', out) # Detect
        self.assertIn('heatmap', out)
        self.assertIn('waypoints', out)
        self.assertIn('classes', out)

if __name__ == "__main__":
    unittest.main()
