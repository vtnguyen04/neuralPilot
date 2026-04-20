import pytest
import os
import tempfile
import json
import torch
from pathlib import Path
from PIL import Image
from neuro_pilot.data.datasets.video_driving import VideoDrivingDataset


@pytest.fixture
def dummy_video_data_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Create a tiny jsonl dataset
        jsonl_path = root / "train.jsonl"
        with open(jsonl_path, "w") as f:
            for seq_id in range(3):  # 3 distinct video clips
                for frame_idx in range(5):  # 5 frames per clip
                    img_path = root / f"clip_{seq_id}_frame_{frame_idx}.jpg"
                    Image.new("RGB", (100, 100), color="blue").save(img_path)

                    # CoVLA-like format (1 frame = 1 line)
                    sample = {
                        "sequence_id": str(seq_id),
                        "frame_idx": frame_idx,
                        "image_path": str(img_path.name),  # relative path
                        "command": 0,
                        "target_point": [10.0, 5.0],
                        "waypoints": [[1.0, 1.0], [2.0, 2.0]],
                        "vEgo": 10.0,
                    }
                    f.write(json.dumps(sample) + "\n")

        yield str(root)


def test_video_driving_dataset_loading(dummy_video_data_dir):
    """Test the CoVLA adapter loads data cleanly, parsing sequences."""
    dataset = VideoDrivingDataset(data_path=dummy_video_data_dir, split="train", clip_length=4, imgsz=(128, 128))
    assert len(dataset) == 6  # 3 sequences of 5 frames, sliding window 4 -> 6 clips total

    sample = dataset[0]

    # Must yield standard current-frame elements to preserve backwards compatibility
    assert "image" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].ndim == 3  # C, H, W (single current frame for spatial detection pipeline)

    # Must yield specific clip sequence for the Temporal Aggregator
    assert "clip_images" in sample
    assert isinstance(sample["clip_images"], torch.Tensor)
    assert sample["clip_images"].shape[0] == 4  # Exactly the clip_length defined
    assert sample["clip_images"].ndim == 4  # T, C, H, W

    # Check targets mappings
    assert "waypoints" in sample
    assert isinstance(sample["waypoints"], torch.Tensor)
    assert sample["command_idx"] == 0


def test_video_dataset_undersample(dummy_video_data_dir):
    """Test that dataset successfully handles when the clip required is longer than the actual sequence lengths."""
    dataset = VideoDrivingDataset(
        data_path=dummy_video_data_dir,
        split="train",
        clip_length=10,  # Requiring 10 frames, but our mock data only has 5
        imgsz=(64, 64),
    )

    sample = dataset[0]

    # It must pad/interpolate the clip lengths correctly back to 10
    assert len(sample["clip_images"]) == 10
