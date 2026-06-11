---
sidebar_position: 3
---
# Task Heads

NeuroPilot distributes predictions into dedicated head modules, which can be selectively enabled or disabled during training using loss weight multiplier configs (lambdas).

---

## 1. Bézier Trajectory Head

Standard regression of individual coordinates often leads to jagged, unsmooth trajectories. NeuroPilot models the future route `tau(t) = [x(t), y(t)]` for `t` in `[0, 1]` as a **Cubic Bézier Curve**:

```
tau(t) = (1-t)^3 * P0 + 3 * (1-t)^2 * t * P1 + 3 * (1-t) * t^2 * P2 + t^3 * P3
```

* **Control Points**: The model predicts control points `P0, P1, P2, P3` in 2D space.
* **Smoothness**: These control points are evaluated at discrete time intervals to reconstruct smooth waypoints. This guarantees continuous curvature and velocity bounds.
* **FiLM Modulation**: Fuses driving command embeddings directly into the control point generator, allowing navigation intents to steer the predicted path.

---

## 2. Deformable Attention Trajectory Head

For complex planning scenarios, NeuroPilot offers an alternative Deformable Trajectory Head:

* **Deformable Attention**: Utilizes deformable cross-attention layers over multi-scale feature maps.
* **Focused Receptive Field**: Restricts the cross-attention lookup key locations to dynamically generated offsets. This allows the head to focus on relevant road features (e.g., lane markings or road boundaries) rather than the entire image canvas, reducing computational complexity.

---

## 3. YOLO11 Anchor-Free Detection Head

The perception branch utilizes an anchor-free detection layout:

* **Distribution Focal Loss (DFL)**: Predicts box boundaries as probability distributions rather than absolute coordinates, handling occlusion and blurry edges.
* **CIoU Loss**: Measures box regression accuracy by accounting for overlap area, aspect ratio, and center distance.
* **Multi-Scale Prediction**: Detections are predicted across P3, P4, and P5 resolution stages to handle varying object scales.

---

## 4. Attention Heatmap Head

Estimates where a human driver would look. 
* **Feature Fusion**: It fuses high-level semantics (P3) with fine-grained textures (P2 skip connection) through a series of upsampling convolutions to reconstruct a pixel-level gaze heatmap.
* **Target Supervision**: Optimized using a combination of Dice Loss and Mean Squared Error (MSE) against human gaze annotations.

---

## 5. JEPA World Model Head

An experimental head for latent representation learning:
* **Joint Embedding Predictive Architecture (JEPA)**: Projects current frame states and actions into a latent space to predict the latent representation of the next frame.
* **Feature Regularization**: Prevents representation collapse and guides the model to learn task-relevant representations (such as lane structures and vehicle dynamics) without needing pixel-level reconstruction.
