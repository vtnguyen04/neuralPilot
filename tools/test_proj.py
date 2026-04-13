import json
import torch
import numpy as np
import cv2
from pathlib import Path
import os
import random

data_dir = Path("/home/quynhthu/Documents/AI-project/e2e/data/covla")
all_states = []
with open(data_dir / 'state.jsonl', 'r') as f:
    for line in f:
        all_states.append(json.loads(line))

# Pick 10 random samples
random.seed(42)
num_samples = len(all_states)
indices = sorted(random.sample(range(num_samples), 10))

for idx_count, idx in enumerate(indices):
    sample = all_states[idx]

    # Load image
    img_path = sample['image_path']
    if img_path.startswith("images/"):
        img_path = img_path[7:]

    full_img_path = data_dir / 'images' / img_path
    img = cv2.imread(str(full_img_path))
    if img is None:
        continue

    h, w, _ = img.shape

    # Get matrices
    extrinsic = np.array(sample['extrinsic_matrix'])
    intrinsic = np.array(sample['intrinsic_matrix'])
    traj_3d = np.array(sample['trajectory'])

    # Create 4D homogeneous coordinates
    num_points = traj_3d.shape[0]
    traj_4d = np.concatenate([traj_3d, np.ones((num_points, 1))], axis=1)

    # Project to camera coordinates
    p_cam = (extrinsic @ traj_4d.T).T[:, :3]

    # Project to image plane
    p_img = (intrinsic @ p_cam.T).T

    # Divide by Z, safe division
    z = np.maximum(p_img[:, 2], 1e-5)
    u = p_img[:, 0] / z
    v = p_img[:, 1] / z

    # Draw waypoints
    for i in range(num_points):
        x, y = int(u[i]), int(v[i])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            # Only put text for a few points so it doesn't clutter
            if i % 15 == 0:
                cv2.putText(img, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Extract Command & Speed
    ego_state = sample.get('ego_state', {})
    left_blinker = ego_state.get('leftBlinker', False)
    right_blinker = ego_state.get('rightBlinker', False)
    speed_ms = ego_state.get('vEgo', 0.0)
    speed_kmh = speed_ms * 3.6 # Convert m/s to km/h

    if left_blinker and not right_blinker:
        cmd_text = "TURN LEFT"
        color = (0, 255, 255) # Yellow
    elif right_blinker and not left_blinker:
        cmd_text = "TURN RIGHT"
        color = (0, 255, 255) # Yellow
    elif left_blinker and right_blinker:
        cmd_text = "HAZARD"
        color = (0, 0, 255) # Red
    else:
        cmd_text = "FORWARD"
        color = (0, 255, 0) # Green

    full_text = f"CMD: {cmd_text} | SPD: {speed_kmh:.1f} km/h"

    # Draw command background pill
    (text_w, text_h), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(img, (30, 30), (30 + text_w + 20, 30 + text_h + 30), (0, 0, 0), -1)
    cv2.putText(img, full_text, (40, 40 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    output_path = f"/home/quynhthu/.gemini/antigravity/artifacts/test_projection_v3_{idx_count+1}.png"
    cv2.imwrite(output_path, img)
    print(f"[{speed_kmh:.1f} km/h] Saved visualization to {output_path}")
