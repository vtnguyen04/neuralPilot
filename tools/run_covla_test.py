import torch
from neuro_pilot.engine.model import NeuroPilot
import neuro_pilot.tasks.atomic
import neuro_pilot.tasks.detection
from neuro_pilot.data.datasets.covla_local import CoVLALocalDataset
from torch.utils.data import DataLoader
from neuro_pilot.utils.losses import MultiTaskLossManager

class TestConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 60 Waypoints config!
        self.model = "neuro_pilot/cfg/models/neuralPilot_covla.yaml"

        # Override loss components for 'waypoint only' mode
        self.lambda_traj = 1.0
        self.lambda_det = 0.0
        self.lambda_heatmap = 0.0
        self.lambda_cls = 0.0
        self.lambda_smooth = 0.1
        self.lambda_gate = 0.0
        self.box = 0.0
        self.cls_det = 0.0
        self.dfl = 0.0

        # Override FDAT usage internally if necessary to test raw L1, let's use standard FDAT to see
        self.use_fdat = True

def run_test():
    # 1. Config
    cfg = TestConfig()

    # 2. Model Initialization
    # Since we set nw: 60 in the yaml, TrajectoryHead should output 60 points!
    print("Initializing NeuroPilot for 60-Waypoint Only Training...")
    model = NeuroPilot(model=cfg.model, device=cfg.device)

    # 3. Create Dataset (Using our newly created adapter)
    train_dataset = CoVLALocalDataset(cfg=cfg, split='train', data_dir='/home/quynhthu/Documents/AI-project/e2e/data/covla', seq_name='2022-07-14--14-32-55--10_first', img_size=(320, 320), requested_waypoints=60)
    # (No need for val dataset during pipeline test)

    # We create a dummy collate_fn handling tensors since Dataloader needs it,
    # but the base protocol has one. It's actually safer to just pull batch manually to test the pipeline directly
    # and confirm 60 waypoints prediction without waiting hours!

    from neuro_pilot.data.datasets.base import BaseDrivingDataset
    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=BaseDrivingDataset.collate_fn)

    # 4. Fetch a single batch and do a Forward Pass and Loss pass!
    print("Testing Pipeline execution... Fetching batch")
    batch = next(iter(dataloader))

    # Pre-configure model targets (mimicking Trainer loop)
    from neuro_pilot.utils.torch_utils import prepare_batch
    batch = prepare_batch(batch, model.device)

    # 5. Model Inference (Generic Dict-in, Dict-out integration verify!)
    model_kwargs = {k: v for k, v in batch.items() if k not in ('image', 'targets', 'image_path')}
    print(f"Feeding model with arbitrary inputs: {list(model_kwargs.keys())}")

    preds = model.model(batch['image'], **model_kwargs)

    print("Model Ouputs mapping:")
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            print(f"- {k}: {v.shape}")
        elif isinstance(v, list):
            print(f"- {k}: {[x.shape for x in v if isinstance(x, torch.Tensor)]}")

    # Trajectory should have shape [B, 60, 2] !
    wp = preds.get('waypoints')
    assert wp is not None, "Predictions missing waypoints!"
    assert wp.shape[1] == 60, f"Expected 60 waypoints, got {wp.shape[1]}"

    print(f"\nSUCCESS! Model successfully predicted {wp.shape[1]} waypoints from {model_kwargs.get('command')}!")

    print("Running Loss calculation...")

    # Setup dict-style targets
    targets = {
        'waypoints': batch['waypoints'],
        'waypoints_mask': batch['waypoints_mask'],
        'bboxes': batch['bboxes'],
        'cls': batch['cls'],
        'batch_idx': torch.zeros(0, device=model.device),
        'curvature': torch.zeros(batch['image'].size(0), device=model.device),
        'command_idx': torch.zeros(batch['image'].size(0), dtype=torch.long, device=model.device),
    }

    loss_manager = MultiTaskLossManager(cfg, model.model, device=model.device)
    loss_dict = loss_manager(preds, targets)

    print("\nLoss components computed:")
    print("----------------------------")
    for k, loss_val in loss_dict.items():
        print(f"{k}: {loss_val.item():.4f}")

    print("\nALL tests passed! Pipeline supports CoVLA dataset and 60-waypoint prediction flawlessly.")

if __name__ == "__main__":
    run_test()
