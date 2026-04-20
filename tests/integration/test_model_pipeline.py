import unittest
import torch
import shutil
from pathlib import Path
from neuro_pilot.engine.model import NeuroPilot
from neuro_pilot.data import prepare_dataloaders


class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        # Create dummy data directory
        self.tmp_data = Path("tests/tmp_pipeline_data")
        self.tmp_data.mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.tmp_data / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dummy samples
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

        self.model_cfg_path = "neuro_pilot/cfg/models/neuralPilot.yaml"
        self.overrides = {
            "data": {
                "dataset_yaml": str(self.yaml_path),
                "batch_size": 4,
                "image_size": 32,
                "num_workers": 0,
                "augment": {"mosaic": 0.0},
            },
            "trainer": {"max_epochs": 1, "use_amp": False},
        }
        self.model = NeuroPilot(self.model_cfg_path, **self.overrides)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tearDown(self):
        if hasattr(self, "tmp_data") and self.tmp_data.exists():
            shutil.rmtree(self.tmp_data)

    def test_forward_backward_pass(self):
        """
        Kiểm tra toàn bộ luồng từ Dataloader -> Model -> Loss -> Backward.
        Đảm bảo dữ liệu chuẩn bị trong data_v1 khớp hoàn toàn với kiến trúc mô hình.
        """
        print("\n--- Testing Model Pipeline with Real Data ---")

        # 1. Load Data
        train_loader, _ = prepare_dataloaders(self.model.cfg_obj)
        batch = next(iter(train_loader))

        # Chuyển batch sang device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        # 2. Forward Pass
        print(f"Executing forward pass on {self.device}...")
        # Lấy raw predictions từ model underlying
        preds = self.model.model(batch["image"], cmd=batch["command"])

        self.assertIsNotNone(preds, "Model returned None for predictions")

        # 3. Loss Calculation
        print("Building criterion and calculating Loss components...")
        self.model.task_wrapper.build_criterion()  # Ensure criterion is built
        loss_dict = self.model.task_wrapper.criterion(preds, batch)
        total_loss = loss_dict["total"]

        # Monitor the Gate!
        if "gate_score" in preds:
            gate_val = preds["gate_score"].mean().item()
            print(f"🔍 Current Gate Score (Importance of Command): {gate_val:.4f}")
            if gate_val > 0.8:
                print("   -> Model is heavily RELYING on the Command.")
            elif gate_val < 0.2:
                print("   -> Model is ignoring the Command (Vision dominant).")
            else:
                print("   -> Model is balancing Vision and Command.")

        print(f"✅ Total Loss: {total_loss.item():.4f}")
        print(f"✅ Loss components: {loss_dict}")

        self.assertFalse(torch.isnan(total_loss), "Loss is NaN!")
        self.assertGreater(total_loss.item(), 0, "Loss should be greater than 0")

        # 4. Backward Pass
        print("Executing backward pass...")
        total_loss.backward()
        print("✅ Backward pass successful.")


if __name__ == "__main__":
    unittest.main()
