---
sidebar_position: 4
---
# Loss Functions & Regularizers

NeuroPilot optimizes multiple tasks using a joint loss function. This guide documents the mathematical foundations of the specialized loss functions and regularizers.

---

## 1. Joint Loss Formulation

The joint loss `L_total` is a weighted combination of task-specific losses:

```
L_total = lambda_traj * L_traj + lambda_det * L_det + lambda_heatmap * L_heatmap + lambda_cls * L_cls + lambda_jepa * L_jepa
```

Where `L_det` is further decomposed into:

```
L_det = alpha_box * L_box + alpha_cls_det * L_cls_det + alpha_dfl * L_dfl
```

By adjusting the `lambda` hyperparameters in your training commands, the task engine dynamically handles backpropagation and metrics logging.

---

## 2. Frenet-Decomposed Anisotropic Trajectory (FDAT) Loss

Standard path regression uses L2 Euclidean distance. However, in driving, lateral error (drifting off-road) is far more dangerous than longitudinal error (arriving slightly early or late). FDAT loss projects coordinate errors into the road's **Frenet frame** (longitudinal axis `s` and lateral axis `d`):

```
L_fdat = w_lat * |d_pred - d_gt|^2 + w_long * |s_pred - s_gt|^2
```

* **Anisotropic Weighting**: By enforcing `w_lat >> w_long`, the model is heavily penalized for deviating sideways from the lane center.
* **Lane-Keeping Stability**: This formulation drastically improves lane-keeping and track compliance during closed-loop control evaluations.

---

## 3. Epps-Pulley SIGReg (Sketch Isotropic Gaussian Regularizer)

To prevent overfitting in small-batch scenarios, NeuroPilot uses SIGReg to regularize the latent representation space.

* **Statistical Preservation**: It computes the statistical divergence of latent features against an isotropic Gaussian distribution.
* **Overfitting Mitigation**: Preserves representative statistical properties during backpropagation, helping the model generalize to unseen environments.

---

## 4. Wing Loss for Heatmap Regression

For spatial heatmaps, standard L2 loss can be dominated by small errors. NeuroPilot integrates **Wing Loss` to improve boundary accuracy:

* **Logarithmic Scaling**: Wing Loss applies a logarithmic penalty to small errors while applying a constant linear penalty to larger errors.
* **Small-Error Prioritization**: This behavior prioritizes high-precision tracking of local features (e.g. lane corners and road borders) without getting derailed by large outliers.
