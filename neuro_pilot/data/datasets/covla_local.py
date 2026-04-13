import torch
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path
import json
from .base import BaseDrivingDataset, register_dataset


@register_dataset("covla_local")
class CoVLALocalDataset(BaseDrivingDataset):
    """
    Adapter for local CoVLA dataset.
    Reads from JSONL states and loads corresponding images.
    Interpolates trajectory points to reach 'requested_waypoints'.
    """
    def __init__(self, cfg, split='train', data_dir='data/covla', seq_name='2022-07-14--14-32-55--10_first', img_size=(320, 320), requested_waypoints=60):
        self.cfg = cfg
        self.img_size = tuple(img_size) if isinstance(img_size, (list, tuple)) else (img_size, img_size)
        self.requested_waypoints = requested_waypoints
        self.data_dir = Path(data_dir)
        self.seq_name = seq_name

        import random

        # Load all states into memory (it's small)
        all_states = []
        with open(self.data_dir / 'state.jsonl', 'r') as f:
            for line in f:
                all_states.append(json.loads(line))

        # Train/val split
        random.seed(42)
        random.shuffle(all_states)
        split_idx = int(len(all_states) * 0.85)
        self.states = all_states[:split_idx] if split == 'train' else all_states[split_idx:]

        print(f"Loaded {len(self.states)} state frames from CoVLA Local ({split} split).")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        sample = self.states[idx]

        # Image path extraction logic
        img_filename = sample.get('image_path', '')
        # Usually looking like "images/2022-07-14--14-32-55--10_first/0000.png"
        # We need to drop "images/" because self.data_dir usually has it, or just use absolute
        img_path = self.data_dir / img_filename

        # Sometimes user configures data_dir="data/covla", so path is data/covla/images/...
        # Let's ensure it's absolute so .exists() works correctly during DataLoader multiprocessing
        img_path = img_path.resolve()

        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new('RGB', (1928, 1208), color='black') # Covla default orig size
            print(f"WARNING: CoVLALocalDataset failed to find image: {img_path}")

        orig_w, orig_h = image.size
        image = F.resize(image, self.img_size)
        image = F.to_tensor(image)  # [3, H, W]
        _, h_final, w_final = image.shape

        # Normalize image strictly like custom dataloader expects
        from neuro_pilot.utils.torch_utils import IMAGENET_MEAN, IMAGENET_STD
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image = (image - mean) / std

        # Waypoints: Project 3D -> 2D -> Normalized [-1, 1]
        traj_3d = sample.get('trajectory', [])
        wp_mask_val = 1.0
        if len(traj_3d) > 0:
            import numpy as np
            # Gather intrinsic and extrinsic matrices
            extrinsic = np.array(sample.get('extrinsic_matrix', np.eye(4)))
            intrinsic = np.array(sample.get('intrinsic_matrix', np.eye(3)))

            # Pad to 4D to multiply with extrinsic
            t_3d = np.array(traj_3d)
            if t_3d.shape[0] != self.requested_waypoints:
                # Need to interpolate to exactly 60 points first
                t_tensor = torch.tensor(t_3d).unsqueeze(0).permute(0, 2, 1) # [1, 3, N]
                t_tensor = torch.nn.functional.interpolate(t_tensor, size=self.requested_waypoints, mode='linear', align_corners=True)
                t_3d = t_tensor.squeeze(0).permute(1, 0).numpy() # [60, 3]

            num_points = t_3d.shape[0]
            t_4d = np.concatenate([t_3d, np.ones((num_points, 1))], axis=1) # [60, 4]

            # Project to Camera
            p_cam = (extrinsic @ t_4d.T).T[:, :3]

            # Project to Image Plane
            p_img = (intrinsic @ p_cam.T).T

            # Get U, V (safe division)
            z = np.maximum(p_img[:, 2], 1e-5)
            u = p_img[:, 0] / z
            v = p_img[:, 1] / z

            u_norm = (u / orig_w) * 2.0 - 1.0
            v_norm = (v / orig_h) * 2.0 - 1.0

            state_tensor = torch.tensor(np.stack([u_norm, v_norm], axis=1), dtype=torch.float32)
            state_tensor.clamp_(-2.0, 2.0) # Clamp extreme off-screen outliers to prevent L1 Explosion
        else:
            state_tensor = torch.zeros((self.requested_waypoints, 2), dtype=torch.float32)
            wp_mask_val = 0.0

        # Command (0: Right, 1: Left, 2: Straight, 3: Follow)
        ego = sample.get('ego_state', {})
        if ego.get('leftBlinker', False): command = 1
        elif ego.get('rightBlinker', False): command = 0
        else: command = 2

        from neuro_pilot.cfg.schema import HeadConfig
        _head = HeadConfig()
        cmd_onehot = torch.zeros(_head.num_commands)
        cmd_onehot[command] = 1.0

        hm_h, hm_w = h_final // 4, w_final // 4
        heatmap = torch.zeros((hm_h, hm_w))

        v_ego = torch.tensor(ego.get('vEgo', 0.0), dtype=torch.float32)

        return {
            'image': image,
            'image_path': str(img_path),
            'waypoints': state_tensor,
            'command': cmd_onehot,
            'command_idx': command,
            'bboxes': torch.zeros((0, 4), dtype=torch.float32),
            'cls': torch.zeros((0,), dtype=torch.long),
            'categories': torch.zeros((0,), dtype=torch.long),
            'waypoints_mask': torch.tensor(wp_mask_val),
            'heatmap': heatmap,
            'curvature': torch.tensor(0.0),
            'vEgo': v_ego
        }

    @classmethod
    def from_config(cls, config, split, yaml_dict):
        """Factory required by the Dataset Registry."""
        return cls(
            cfg=config,
            split=split,
            data_dir=yaml_dict.get('path', 'data/covla'),
            img_size=config.data.image_size,
            requested_waypoints=60,
        )

    @staticmethod
    def collate_fn(batch):
        from neuro_pilot.data.neuro_pilot_dataset import custom_collate_fn
        return custom_collate_fn(batch)
