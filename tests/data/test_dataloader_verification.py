import unittest
import torch
import shutil
from pathlib import Path
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.data import prepare_dataloaders


class TestDataloaderVerification(unittest.TestCase):
    def setUp(self):
        # Create dummy data directory
        self.tmp_data = Path("tests/tmp_dataloader_data")
        self.tmp_data.mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dummy sample with waypoints (class, x, y, w, h, wp0_x, wp0_y, ..., wp9_x, wp9_y)
        # Total 5 + 20 = 25 columns
        waypoints = " ".join(["0.1"] * 20)
        label_content = f"0 0.5 0.5 1.0 1.0 {waypoints}\n"

        (self.tmp_data / "train" / "images" / "dummy.jpg").touch()
        with open(self.tmp_data / "train" / "labels" / "dummy.txt", "w") as f:
            f.write(label_content)
        (self.tmp_data / "val" / "images" / "dummy.jpg").touch()
        with open(self.tmp_data / "val" / "labels" / "dummy.txt", "w") as f:
            f.write(label_content)

        # Create data.yaml
        self.yaml_path = self.tmp_data / "data.yaml"
        with open(self.yaml_path, "w") as f:
            f.write(f"path: {self.tmp_data.absolute()}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("names: {0: class0}\n")

        model_cfg_path = "neuro_pilot/cfg/models/neuralPilot.yaml"
        self.model = NeuroPilot(
            model_cfg_path,
            data={
                "dataset_yaml": str(self.yaml_path),
                "batch_size": 1,
                "image_size": 32,
                "num_workers": 0,
                "augment": {},
            },
        )
        self.cfg = self.model.cfg_obj

    def tearDown(self):
        if self.tmp_data.exists():
            shutil.rmtree(self.tmp_data)

    def test_dataloader_batch_loading(self):
        print("\n--- Running Dataloader Batch Verification Test ---")
        train_loader, val_loader = prepare_dataloaders(self.cfg)

        print(f"Train loader size: {len(train_loader)} batches")
        print(f"Val loader size: {len(val_loader)} batches")

        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)

        batch = next(iter(train_loader))

        print(f"✅ Image Batch Shape: {batch['image'].shape}")
        print(f"✅ BBoxes Batch Indices: {batch['batch_idx']}")
        print(f"✅ Waypoints Batch Shape: {batch['waypoints'].shape}")
        print(f"✅ Waypoints Batch Indices Shape: {batch['batch_idx_waypoints'].shape}")

        # Verify 10 points per sample in batch_idx_waypoints
        expected_wp_indices = 1 * 10  # batch_size (1) * num_waypoints (10)
        self.assertEqual(batch["batch_idx_waypoints"].numel(), expected_wp_indices)

        print("\n--- Conclusion ---")
        print("Dataloader logic is confirmed and standardized.")


if __name__ == "__main__":
    unittest.main()
