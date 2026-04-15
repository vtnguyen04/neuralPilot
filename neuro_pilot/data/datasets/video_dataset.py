"""Abstract base class for video/clip-based driving datasets.

Follows the same pattern as CoVLA datasets: JSONL/JSON metadata + image files.
Temporal context is ONLY for trajectory prediction — detection, heatmap,
classification all operate on the single current frame.

Design:
    - Inherits from BaseDrivingDataset to preserve pipeline compatibility.
    - Returns same dict format as image datasets + extra temporal fields
    - The 'image' key contains the CURRENT frame [C, H, W] (not a clip!)
    - Extra key 'clip_images' has T frames [T, C, H, W] for temporal feature extraction
    - Standard collate works because 'image' is still a single frame tensor
    - When T=1, output is identical to single-frame dataset (backward compat)

Usage in trainer:
    batch['image']       → fed to backbone → detection, heatmap, classification
    batch['clip_images'] → each frame through backbone → temporal aggregator → trajectory head

This separation ensures detection/heatmap/classification are completely unaffected.
"""

from __future__ import annotations

import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch

from neuro_pilot.data.datasets.base import BaseDrivingDataset


@dataclass
class ClipSample:
    """Metadata for a single clip sample.

    Attributes:
        frame_paths: Ordered list of frame image paths (chronological).
        label_data: Annotation data for the LAST frame (prediction target).
        video_id: Source video identifier.
        timestamps: Optional per-frame timestamps (seconds).
        ego_states: Optional per-frame ego vehicle states.
    """
    frame_paths: list[str] = field(default_factory=list)
    video_path: str = ""
    frame_indices: list[int] | None = None
    label_data: dict = field(default_factory=dict)
    video_id: str = ""
    timestamps: list[float] | None = None
    ego_states: list[dict] | None = None


class BaseVideoDataset(BaseDrivingDataset):
    """Universal protocol for video/clip-based trajectory datasets.

    Subclasses must implement:
        - ``_build_samples()``: Parse data source, return list[ClipSample]
        - ``_load_frame(path)``: Load frame as BGR numpy array
        - ``_process_label(label_data)``: Convert raw label to standard dict

    The base class handles:
        - Temporal sampling with jitter and dropout
        - Consistent image processing across clip frames
        - Standard output dict format compatible with existing collate
    """

    def __init__(
        self,
        clip_length: int = 4,
        frame_stride: int = 1,
        temporal_jitter: float = 0.0,
        temporal_dropout: float = 0.0,
        imgsz: int | tuple = 320,
        split: str = "train",
    ):
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.temporal_jitter = temporal_jitter if split == "train" else 0.0
        self.temporal_dropout = temporal_dropout if split == "train" else 0.0
        self.imgsz = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        self.split = split

        self.samples: list[ClipSample] = self._build_samples()

    # ---- Abstract (subclass implements) -----------------------------------

    @abstractmethod
    def _build_samples(self) -> list[ClipSample]:
        """Scan data source and return clip samples."""
        ...

    @abstractmethod
    def _load_frame(self, frame_path: str) -> np.ndarray | None:
        """Load a single frame as BGR numpy array."""
        ...

    @abstractmethod
    def _process_label(self, label_data: dict) -> dict:
        """Convert raw label data to standard output dict.

        Must return at minimum:
            waypoints: list[list[float]] in normalized coords
            command: int
        Optionally:
            bboxes, categories, ego_state, etc.
        """
        ...

    # ---- Temporal sampling ------------------------------------------------

    def _sample_indices(self, total_frames: int) -> list[int]:
        """Sample T frame indices with jitter and dropout."""
        T = self.clip_length
        stride = self.frame_stride
        span = (T - 1) * stride + 1

        if span > total_frames:
            indices = list(range(total_frames))
        else:
            max_start = total_frames - span
            if self.temporal_jitter > 0 and max_start > 0:
                start = int(max_start * self.temporal_jitter * random.random())
            else:
                start = max(0, total_frames - span)  # Default: latest T frames
            indices = list(range(start, start + span, stride))

        # Temporal dropout: randomly drop interior frames
        if self.temporal_dropout > 0 and len(indices) > 2:
            keep = [indices[0]]
            for idx in indices[1:-1]:
                if random.random() > self.temporal_dropout:
                    keep.append(idx)
            keep.append(indices[-1])
            indices = keep

        # Pad to T
        while len(indices) < T:
            indices.append(indices[-1])

        return indices[:T]

    # ---- Image processing -------------------------------------------------

    def _process_frame(self, img: np.ndarray) -> torch.Tensor:
        """Resize, normalize a single frame to tensor [C, H, W]."""
        from neuro_pilot.utils.torch_utils import IMAGENET_MEAN, IMAGENET_STD

        # Resize
        img = cv2.resize(img, (self.imgsz[1], self.imgsz[0]))

        # BGR → RGB, normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (img_t - mean) / std

    # ---- Core __getitem__ -------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        clip_tensors = []
        temporal_mask = []

        if sample.video_path and sample.frame_indices is not None:
            # MP4 Fast Random Access Path (Decord)
            import decord
            indices = self._sample_indices(len(sample.frame_indices))
            vr = decord.VideoReader(sample.video_path, ctx=decord.cpu(0))

            # Extract absolute frame indices directly in C++
            abs_indices = [sample.frame_indices[fi] for fi in indices]
            try:
                frames_num = vr.get_batch(abs_indices).asnumpy()
                for img in frames_num: # Output is RGB directly from decord!
                    # Process directly (we convert to BGR here because _process_frame expects BGR)
                    import cv2
                    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    clip_tensors.append(self._process_frame(bgr_img))
                    temporal_mask.append(True)
            except Exception as e:
                import logging
                logging.getLogger("neuro_pilot.data").error(f"Decord Error '{sample.video_path}': {e}")
                for _ in indices:
                    clip_tensors.append(torch.zeros(3, *self.imgsz))
                    temporal_mask.append(False)
        else:
            # Classic image path parsing loop
            indices = self._sample_indices(len(sample.frame_paths))
            for fi in indices:
                img = self._load_frame(sample.frame_paths[fi]) if fi < len(sample.frame_paths) else None
                if img is not None:
                    clip_tensors.append(self._process_frame(img))
                    temporal_mask.append(True)
                else:
                    clip_tensors.append(torch.zeros(3, *self.imgsz))
                    temporal_mask.append(False)

        # Current frame = last frame in clip (most recent)
        current_image = clip_tensors[-1]  # [C, H, W]
        clip_images = torch.stack(clip_tensors)  # [T, C, H, W]

        # Process label (from last frame = prediction target)
        label = self._process_label(sample.label_data)

        # Build standard output dict
        wp = label.get("waypoints", [])
        command = label.get("command", 0)
        bboxes = label.get("bboxes", [])
        categories = label.get("categories", [])
        v_ego = label.get("vEgo", 0.0)

        _, h_final, w_final = current_image.shape

        # Waypoints
        if len(wp) > 0:
            wp_t = torch.tensor(wp, dtype=torch.float32)
            if wp_t.ndim == 1:
                wp_t = wp_t.unsqueeze(0)
            wp_mask = 1.0
        else:
            wp_t = torch.zeros((0, 2), dtype=torch.float32)
            wp_mask = 0.0

        # Bboxes
        bboxes_t = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros(0, 4)
        cls_t = torch.tensor(categories, dtype=torch.long) if categories else torch.zeros(0, dtype=torch.long)

        # Command
        from neuro_pilot.cfg.schema import HeadConfig
        num_commands = HeadConfig().num_commands
        cmd_onehot = torch.zeros(num_commands)
        cmd_onehot[min(command, num_commands - 1)] = 1.0

        # Heatmap placeholder
        hm_h, hm_w = h_final // 4, w_final // 4
        heatmap = torch.zeros((hm_h, hm_w))

        return {
            # Standard keys (same as image datasets — detection/heatmap/cls use this)
            "image": current_image,           # [C, H, W] — single frame
            "image_path": sample.frame_paths[-1] if sample.frame_paths else "",
            "command": cmd_onehot,
            "command_idx": command,
            "waypoints": wp_t,
            "waypoints_mask": torch.tensor(wp_mask),
            "bboxes": bboxes_t,
            "categories": cls_t,
            "heatmap": heatmap,
            "curvature": torch.tensor(0.0),
            "vEgo": torch.tensor(v_ego, dtype=torch.float32),
            # Temporal keys (ONLY used by trajectory head + temporal aggregator)
            "clip_images": clip_images,       # [T, C, H, W] — full clip
            "temporal_mask": torch.tensor(temporal_mask, dtype=torch.bool),
        }

    def __len__(self) -> int:
        return len(self.samples)

    # ---- Collate ----------------------------------------------------------

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate that handles both standard and temporal fields.

        - Standard fields: same logic as custom_collate_fn (detection compat)
        - clip_images: stacked to [B, T, C, H, W]
        - temporal_mask: stacked to [B, T]
        """
        from neuro_pilot.data.neuro_pilot_dataset import custom_collate_fn

        # First, separate temporal fields
        clip_images = torch.stack([b.pop("clip_images") for b in batch])  # [B, T, C, H, W]
        temporal_mask = torch.stack([b.pop("temporal_mask") for b in batch])  # [B, T]

        # Collate non-temporal fields using existing logic
        collated = custom_collate_fn(batch)

        # Add temporal fields back
        collated["clip_images"] = clip_images
        collated["temporal_mask"] = temporal_mask

        return collated
