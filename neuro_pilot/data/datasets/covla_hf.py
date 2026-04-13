import torch
import torchvision.transforms.functional as F
from PIL import Image
from datasets import load_dataset
from .base import BaseDrivingDataset, register_dataset


@register_dataset("covla_hf")
class CoVLADataset(BaseDrivingDataset):
    """
    Adapter for Hugging Face CoVLA-Dataset-Mini to work with NeuroPilot's dict-in dict-out generic trainer.
    Retrieves images and interpolates trajectories to match requested waypoints.
    """
    def __init__(self, cfg, split='train', img_size=(320, 320), requested_waypoints=60):
        self.cfg = cfg
        self.split = split
        self.img_size = tuple(img_size) if isinstance(img_size, (list, tuple)) else (img_size, img_size)
        self.requested_waypoints = requested_waypoints

        # Load from the official repo (set HF_TOKEN env var for gated datasets)
        import os
        print(f"Loading CoVLA Mini dataset ({split}) via HF Streaming...")
        raw_ds = load_dataset('turing-motors/CoVLA-Dataset-Mini', split=split, streaming=True, token=os.environ.get('HF_TOKEN'))
        # We will cast it to list because it's a small dataset (or use caching if it's large)
        self.data_stream = iter(raw_ds)
        self.cached_data = []

    def _get_item_from_stream(self, idx):
        while len(self.cached_data) <= idx:
            try:
                self.cached_data.append(next(self.data_stream))
            except StopIteration:
                break
        return self.cached_data[idx]

    def __len__(self):
        # We don't know the exact length in streaming mode until consumed, return an arbitrary large max or cached count
        return max(len(self.cached_data), 1000)

    def __getitem__(self, idx):
        sample = self._get_item_from_stream(idx)

        # Image
        image = sample['image'] # PIL Image
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        image = F.resize(image, self.img_size)
        image = F.to_tensor(image)  # [3, H, W]

        # Waypoints
        # CoVLA state is 80 floats = 40 (x, y) coordinates
        state = sample['state']
        if isinstance(state, str):
            import json
            import ast
            try:
                state = json.loads(state)
            except:
                state = ast.literal_eval(state)
        state_tensor = torch.tensor(state, dtype=torch.float32).view(-1, 2) # [40, 2]

        # Interpolate to requested waypoints
        H, W = self.img_size
        if state_tensor.shape[0] != self.requested_waypoints:
            # For linear interpolation we need [1, 2, 40]
            # state_tensor.permute(1, 0).unsqueeze(0) -> [1, 2, 40]
            interp = torch.nn.functional.interpolate(
                state_tensor.permute(1, 0).unsqueeze(0),
                size=self.requested_waypoints,
                mode='linear',
                align_corners=True
            )
            state_tensor = interp.squeeze(0).permute(1, 0) # [60, 2]

        # Commands parsing from caption (dummy for now since we do waypoint-only)
        # 0: Right, 1: Left, 2: Straight, 3: Follow
        command = 2

        # Return dict mapping for dict-in/dict-out generic framework
        return {
            'image': image,
            'waypoints': state_tensor,
            'command': command,
            'bboxes': torch.zeros((0, 5)), # Empty bboxes for detection
            'cls': torch.zeros((0,)),
            'waypoints_mask': torch.ones(self.requested_waypoints)
        }

    @classmethod
    def from_config(cls, config, split, yaml_dict):
        """Factory required by the Dataset Registry."""
        return cls(
            cfg=config,
            split=split,
            img_size=config.data.image_size,
            requested_waypoints=getattr(config.head, 'num_waypoints', 60),
        )

