"""Flexible video dataset adapter for driving sequences.

Follows CoVLA-style conventions:
    - JSONL or JSON metadata files
    - Frame images in organized directories
    - Flexible schema: supports 3D trajectory + camera projection,
      2D waypoints, bounding boxes, ego state, etc.

Supports multiple data layouts::

    # Layout 1: Single JSONL with all frames (CoVLA-style)
    dataset_root/
    ├── data.yaml          # type: video_driving
    ├── state.jsonl         # One JSON per frame with image_path + annotations
    └── images/
        └── sequence_001/
            ├── 0000.png
            ├── 0001.png
            └── ...

    # Layout 2: Per-sequence directories
    dataset_root/
    ├── data.yaml
    ├── sequence_001/
    │   ├── metadata.json  # Sequence-level metadata
    │   ├── frames/
    │   │   ├── 0000.jpg
    │   │   └── ...
    │   └── annotations.jsonl
    └── sequence_002/
        └── ...

data.yaml example::

    type: video_driving
    path: /path/to/dataset_root
    train: state_train.jsonl     # or a directory
    val: state_val.jsonl
    format: jsonl                # jsonl | json_dir
    nc: 14
    names: [...]
    # Optional camera params (can also be per-frame in JSONL)
    intrinsic_matrix: [...]
    extrinsic_matrix: [...]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from neuro_pilot.data.datasets.base import register_dataset
from neuro_pilot.data.datasets.video_dataset import BaseVideoDataset, ClipSample
from neuro_pilot.utils.logger import logger


@register_dataset("video_driving")
class VideoDrivingDataset(BaseVideoDataset):
    """Flexible video dataset for autonomous driving trajectory data.

    Reads frame metadata from JSONL files (CoVLA-style) and groups
    consecutive frames into temporal clips for trajectory prediction.
    """

    def __init__(
        self,
        data_path: str | Path,
        data_yaml: dict | None = None,
        clip_length: int = 4,
        frame_stride: int = 1,
        temporal_jitter: float = 0.2,
        temporal_dropout: float = 0.1,
        imgsz: int | tuple = 320,
        split: str = "train",
        requested_waypoints: int = 10,
    ):
        self.data_path = Path(data_path)
        self.data_yaml = data_yaml or {}
        self.requested_waypoints = requested_waypoints
        self.names = self.data_yaml.get("names", {})
        if isinstance(self.names, list):
            self.names = {i: n for i, n in enumerate(self.names)}

        # Camera defaults (can be overridden per-frame)
        self._default_intrinsic = np.array(
            self.data_yaml.get("intrinsic_matrix", np.eye(3).tolist())
        )
        self._default_extrinsic = np.array(
            self.data_yaml.get("extrinsic_matrix", np.eye(4).tolist())
        )

        super().__init__(
            clip_length=clip_length,
            frame_stride=frame_stride,
            temporal_jitter=temporal_jitter,
            temporal_dropout=temporal_dropout,
            imgsz=imgsz,
            split=split,
        )

        logger.info(
            f"VideoDrivingDataset: {len(self.samples)} clips, "
            f"T={clip_length}, stride={frame_stride}, split={split}"
        )

    def _build_samples(self) -> list[ClipSample]:
        """Scan data source and group frames into clips."""
        data_format = self.data_yaml.get("format", "jsonl")

        if data_format == "jsonl":
            return self._build_from_jsonl()
        elif data_format == "json_dir":
            return self._build_from_json_dir()
        elif data_format == "frame_dir":
            return self._build_from_frame_dir()
        else:
            logger.warning(f"Unknown format '{data_format}', trying JSONL")
            return self._build_from_jsonl()

    def _build_from_jsonl(self) -> list[ClipSample]:
        """Build clips from a JSONL file (CoVLA-style)."""
        jsonl_path = self.data_path
        if jsonl_path.is_dir():
            # Try to find JSONL in directory
            for candidate in ["state.jsonl", "annotations.jsonl", f"{self.split}.jsonl"]:
                if (jsonl_path / candidate).exists():
                    jsonl_path = jsonl_path / candidate
                    break

        if not jsonl_path.exists():
            return []

        # Load all frames
        all_frames: list[dict] = []

        # Support either a single .jsonl file or an entire directory of .jsonl files
        jsonl_files = [jsonl_path] if jsonl_path.is_file() else list(jsonl_path.glob("**/*.jsonl"))

        for j_path in jsonl_files:
            with open(j_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_data = json.loads(line)
                        # Handle CoVLA nested dict {"0": {"vEgo":...}} format
                        if len(raw_data) == 1 and list(raw_data.keys())[0].isdigit():
                            frame_idx_str = list(raw_data.keys())[0]
                            frame_data = raw_data[frame_idx_str]
                            frame_data["frame_idx"] = int(frame_idx_str)
                            # Implicit sequence mapping to corresponding .mp4
                            frame_data["sequence_id"] = f"{j_path.stem}.mp4"
                            frame_data["json_path"] = str(j_path)
                            all_frames.append(frame_data)
                        else:
                            # Inherit sequence_id from file name if missing
                            if "sequence_id" not in raw_data and "video_path" not in raw_data:
                                raw_data["sequence_id"] = f"{j_path.stem}.mp4"
                            raw_data["json_path"] = str(j_path)
                            all_frames.append(raw_data)

        if not all_frames:
            return []

        # Group by video/sequence
        sequences: dict[str, list[dict]] = {}
        for frame in all_frames:
            # Try to extract sequence ID from explicit video path or generic sequence_id
            seq_id = frame.get("video_path", frame.get("sequence_id", frame.get("video_id", "")))
            if not seq_id:
                img_path = frame.get("image_path", "")
                parts = Path(img_path).parts
                seq_id = parts[-2] if len(parts) >= 2 else "default"
            sequences.setdefault(seq_id, []).append(frame)

        # Sort frames within each sequence
        for seq_id in sequences:
            sequences[seq_id].sort(
                key=lambda f: f.get("timestamp", f.get("frame_idx", 0))
            )

        # Create clips with sliding window
        clips = []
        span = (self.clip_length - 1) * self.frame_stride + 1

        for seq_id, frames in sequences.items():
            # Robust mapping: Assume state jsonl is in .../states/, video is in .../videos/
            # For each sequence, base it dynamically off the specific json_path it came from
            if "json_path" in frames[0]:
                curr_jsonl_path = Path(frames[0]["json_path"])
                base_dir = curr_jsonl_path.parent
                videos_dir = Path(self.data_yaml.get("video_dir", base_dir.parent / "videos"))
            else:
                base_dir = jsonl_path.parent
                videos_dir = Path(self.data_yaml.get("video_dir", base_dir.parent / "videos"))

            # Check if MP4 file
            is_mp4 = str(seq_id).endswith(".mp4")
            video_full_str = ""
            if is_mp4:
                video_full = (videos_dir / seq_id).resolve()
                if not video_full.exists():
                    video_full = Path(seq_id)
                video_full_str = str(video_full)

            frame_paths = []
            if not is_mp4:
                for f in frames:
                    img_p = f.get("image_path", "")
                    full_p = (base_dir / img_p).resolve()
                    frame_paths.append(str(full_p))

            if len(frames) < span:
                clips.append(ClipSample(
                    frame_paths=frame_paths,
                    video_path=video_full_str,
                    frame_indices=[f.get("frame_idx", 0) for f in frames] if is_mp4 else None,
                    label_data=frames[-1],
                    video_id=seq_id,
                    ego_states=[f.get("ego_state", {}) for f in frames],
                ))
            else:
                step = max(1, self.frame_stride)
                for start in range(0, len(frames) - span + 1, step):
                    end = start + span
                    slice_f = frames[start:end]
                    clips.append(ClipSample(
                        frame_paths=frame_paths[start:end] if not is_mp4 else [],
                        video_path=video_full_str,
                        frame_indices=[f.get("frame_idx", i) for i, f in enumerate(slice_f)] if is_mp4 else None,
                        label_data=slice_f[-1],
                        video_id=seq_id,
                        ego_states=[f.get("ego_state", {}) for f in slice_f],
                    ))

        return clips

    def _build_from_json_dir(self) -> list[ClipSample]:
        """Build clips from per-sequence directories with JSON annotations."""
        clips = []
        if not self.data_path.is_dir():
            return clips

        for seq_dir in sorted(self.data_path.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name.startswith("."):
                continue

            # Find annotation file
            annot_path = None
            for candidate in ["annotations.jsonl", "metadata.jsonl", "labels.json"]:
                if (seq_dir / candidate).exists():
                    annot_path = seq_dir / candidate
                    break

            if annot_path is None:
                continue

            # Find frame directory
            frame_dir = seq_dir
            for candidate in ["frames", "images", "rgb"]:
                if (seq_dir / candidate).is_dir():
                    frame_dir = seq_dir / candidate
                    break

            # Load annotations
            frames = []
            if annot_path.suffix == ".jsonl":
                with open(annot_path, "r") as f:
                    for line in f:
                        if line.strip():
                            frames.append(json.loads(line))
            else:
                with open(annot_path, "r") as f:
                    data = json.load(f)
                    frames = data if isinstance(data, list) else data.get("frames", [])

            if not frames:
                continue

            # Get frame image paths
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            frame_files = sorted([
                f for f in frame_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ])

            frame_paths = [str(f) for f in frame_files]

            # Build clips
            span = (self.clip_length - 1) * self.frame_stride + 1
            n = min(len(frames), len(frame_paths))

            if n < span:
                clips.append(ClipSample(
                    frame_paths=frame_paths[:n],
                    label_data=frames[n - 1] if frames else {},
                    video_id=seq_dir.name,
                ))
            else:
                step = max(1, self.frame_stride)
                for start in range(0, n - span + 1, step):
                    end = start + span
                    clips.append(ClipSample(
                        frame_paths=frame_paths[start:end],
                        label_data=frames[end - 1],
                        video_id=seq_dir.name,
                    ))

        return clips

    def _build_from_frame_dir(self) -> list[ClipSample]:
        """Build clips from a flat directory of frames (simplest format)."""
        clips = []
        if not self.data_path.is_dir():
            return clips

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        frame_files = sorted([
            f for f in self.data_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if not frame_files:
            return clips

        frame_paths = [str(f) for f in frame_files]
        span = (self.clip_length - 1) * self.frame_stride + 1

        step = max(1, self.frame_stride)
        for start in range(0, len(frame_paths) - span + 1, step):
            end = start + span
            clips.append(ClipSample(
                frame_paths=frame_paths[start:end],
                label_data={},
                video_id="default",
            ))

        return clips

    def _load_frame(self, frame_path: str) -> np.ndarray | None:
        """Load a single frame as BGR numpy array."""
        img = cv2.imread(frame_path)
        if img is None:
            logger.warning(f"Failed to load frame: {frame_path}")
        return img

    def _process_label(self, label_data: dict) -> dict:
        """Convert raw label data to standard format.

        Handles CoVLA-style 3D trajectory projection and direct 2D waypoints.
        """
        result: dict[str, Any] = {"command": 0, "waypoints": [], "bboxes": [], "categories": []}

        if not label_data:
            return result

        # --- Waypoints ---
        # Priority 1: Direct 2D waypoints (already normalized)
        if "waypoints_2d" in label_data:
            result["waypoints"] = label_data["waypoints_2d"]

        # Priority 2: State vector (CoVLA HF format: flat list of x,y pairs)
        elif "state" in label_data:
            state = label_data["state"]
            if isinstance(state, str):
                import ast
                state = ast.literal_eval(state)
            wp = torch.tensor(state, dtype=torch.float32).view(-1, 2)
            if wp.shape[0] != self.requested_waypoints:
                wp = torch.nn.functional.interpolate(
                    wp.permute(1, 0).unsqueeze(0),
                    size=self.requested_waypoints, mode="linear", align_corners=True
                ).squeeze(0).permute(1, 0)
            result["waypoints"] = wp.tolist()

        # Priority 3: 3D trajectory + camera projection (CoVLA local format)
        elif "trajectory" in label_data:
            result["waypoints"] = self._project_3d_trajectory(label_data)

        # --- Command ---
        ego = label_data.get("ego_state", {})
        left_blinker = ego.get("leftBlinker", label_data.get("leftBlinker", False))
        right_blinker = ego.get("rightBlinker", label_data.get("rightBlinker", False))

        if left_blinker:
            result["command"] = 1
        elif right_blinker:
            result["command"] = 0
        elif "command" in label_data:
            result["command"] = label_data["command"]
        else:
            result["command"] = 2  # Default: straight

        # --- vEgo ---
        result["vEgo"] = ego.get("vEgo", label_data.get("vEgo", 0.0))

        # --- Bboxes (optional) ---
        if "bboxes" in label_data:
            result["bboxes"] = label_data["bboxes"]
            result["categories"] = label_data.get("categories", [0] * len(label_data["bboxes"]))

        return result

    def _project_3d_trajectory(self, label_data: dict) -> list[list[float]]:
        """Project 3D world trajectory to normalized 2D image coordinates."""
        traj_3d = np.array(label_data["trajectory"])
        if traj_3d.shape[1] == 2:
            traj_3d = np.concatenate([traj_3d, np.zeros((traj_3d.shape[0], 1))], axis=1)

        extrinsic = np.array(label_data.get("extrinsic_matrix", self._default_extrinsic))
        intrinsic = np.array(label_data.get("intrinsic_matrix", self._default_intrinsic))

        # Interpolate to requested waypoints
        if traj_3d.shape[0] != self.requested_waypoints:
            t_tensor = torch.tensor(traj_3d, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
            t_tensor = torch.nn.functional.interpolate(
                t_tensor, size=self.requested_waypoints, mode="linear", align_corners=True
            )
            traj_3d = t_tensor.squeeze(0).permute(1, 0).numpy()

        # Project: world → camera → image
        num_points = traj_3d.shape[0]
        t_4d = np.concatenate([traj_3d, np.ones((num_points, 1))], axis=1)
        p_cam = (extrinsic @ t_4d.T).T[:, :3]
        p_img = (intrinsic @ p_cam.T).T

        z = np.maximum(p_img[:, 2], 1e-5)
        u = p_img[:, 0] / z
        v = p_img[:, 1] / z

        # Normalize to [-1, 1] (assuming known image size from metadata)
        orig_w = label_data.get("image_width", 1928)
        orig_h = label_data.get("image_height", 1208)
        u_norm = (u / orig_w) * 2.0 - 1.0
        v_norm = (v / orig_h) * 2.0 - 1.0

        wp = np.stack([u_norm, v_norm], axis=1)
        wp = np.clip(wp, -2.0, 2.0)
        return wp.tolist()

    # ---- Factory ----------------------------------------------------------

    @classmethod
    def from_config(cls, config, split: str, yaml_dict: dict) -> "VideoDrivingDataset":
        """Factory for dataset registry dispatch."""
        path_str = yaml_dict.get("path", "")
        yaml_parent = Path(config.data.dataset_yaml).parent if config.data.dataset_yaml else Path(".")

        base_path = Path(path_str)
        if not base_path.is_absolute():
            base_path = (yaml_parent / base_path).resolve()

        # Data path: can be JSONL file or directory
        split_rel = yaml_dict.get(split, split)
        if isinstance(split_rel, list):
            split_rel = split_rel[0]

        data_path = base_path / split_rel
        if not data_path.exists():
            data_path = base_path  # Fallback to base

        # Temporal config
        temporal_cfg = getattr(config, "temporal", None)
        clip_length = temporal_cfg.clip_length if temporal_cfg and temporal_cfg.enabled else 1
        frame_stride = temporal_cfg.frame_stride if temporal_cfg else 1
        temporal_jitter = temporal_cfg.temporal_jitter if temporal_cfg else 0.0
        temporal_dropout = temporal_cfg.temporal_dropout if temporal_cfg else 0.0

        return cls(
            data_path=data_path,
            data_yaml=yaml_dict,
            clip_length=clip_length,
            frame_stride=frame_stride,
            temporal_jitter=temporal_jitter,
            temporal_dropout=temporal_dropout,
            imgsz=config.data.image_size,
            split=split,
            requested_waypoints=getattr(config.head, "num_waypoints", 10),
        )
