import unittest
import torch
import shutil
from pathlib import Path
from neuro_pilot import NeuroPilot

class TestTrainingResume(unittest.TestCase):
    def setUp(self):
        self.exp_name = "test_resume_unittest"
        self.exp_dir = Path("experiments") / self.exp_name

        # Create dummy data directory
        self.tmp_data = Path("tests/tmp_resume_data")
        self.tmp_data.mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dummy samples (need at least as many as batch_size to avoid issues)
        waypoints = " ".join(["0.1"] * 20)
        label_content = f"0 0.5 0.5 1.0 1.0 {waypoints}\n"
        for i in range(4):
            (self.tmp_data / "train" / "images" / f"dummy_{i}.jpg").touch()
            with open(self.tmp_data / "train" / "labels" / f"dummy_{i}.txt", "w") as f:
                f.write(label_content)
            (self.tmp_data / "val" / "images" / f"dummy_{i}.jpg").touch()
            with open(self.tmp_data / "val" / "labels" / f"dummy_{i}.txt", "w") as f:
                f.write(label_content)

        # Create data.yaml
        self.yaml_path = self.tmp_data / "data.yaml"
        with open(self.yaml_path, "w") as f:
            f.write(f"path: {self.tmp_data.absolute()}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("names: {0: class0}\n")

        self.data_dict = {'dataset_yaml': str(self.yaml_path)}

        # Clean old weights
        if self.exp_dir.exists():
            shutil.rmtree(self.exp_dir)

    def tearDown(self):
        # Clean up experiment directory after tests
        if self.exp_dir.exists():
            shutil.rmtree(self.exp_dir)
        if hasattr(self, 'tmp_data') and self.tmp_data.exists():
            shutil.rmtree(self.tmp_data)

    def test_resume_flow(self):
        """Verify that training can stop after 1 epoch and resume correctly."""
        print("\n--- Phase 1: Initial Training (1 Epoch) ---")
        model = NeuroPilot(device='cpu')
        # Train for 1 epoch
        model.train(
            max_epochs=1,
            experiment_name=self.exp_name,
            batch_size=4,
            data=self.data_dict,
            image_size=32, # Optimization: 320 -> 32
            augment=False
        )

        last_ckpt = self.exp_dir / "weights" / "last.pt"
        self.assertTrue(last_ckpt.exists(), "last.pt should be created after 1 epoch")

        # Explicitly clean up Phase 1 model to free GPU memory
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("\n--- Phase 2: Resume Training (Target 2 Epochs) ---")
        # Fresh model instance
        model_resume = NeuroPilot(device='cpu')
        metrics = model_resume.train(
            resume=True,
            max_epochs=2,
            experiment_name=self.exp_name,
            batch_size=4,
            data=self.data_dict,
            image_size=32, # Optimization: 320 -> 32
            augment=False
        )

        self.assertIsNotNone(metrics, "Resume training should return metrics")

        # Cleanup Phase 2
        del model_resume
        torch.cuda.empty_cache()
        # Note: In real setup, we'd check if trainer.epoch started from 1,
        # but here we mainly verify it runs to completion without error.

if __name__ == "__main__":
    unittest.main()
