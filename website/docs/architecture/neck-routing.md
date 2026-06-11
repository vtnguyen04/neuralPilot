---
sidebar_position: 2
---
# Neck & Feature Routing Topologies

The Neck layer coordinates the aggregation and routing of features before they reach task-specific heads. Multi-task learning environments often suffer from "gradient conflict"—where gradients from the detection task degrade the representation quality needed for path planning. NeuroPilot resolves this using three specialized routing topologies.

---

## 1. Command State Modulation (CSM)

Path planning is conditioned on navigation directives (e.g. "turn left at the next intersection"). The CSM block routes the target command index `c` (where `c` is an integer in `[0, 1, 2, 3]`) into the driving feature maps.

```
Visual Features (P5) ──► [ CSM Layer ] ──► Modulated Features ──► Trajectory Head
                              ▲
Command Index (c)    ─────────┘
```

### A. FiLM (Feature-wise Linear Modulation)
FiLM applies scale and shift parameter overrides to feature channel activations:
```
FiLM(F) = gamma(c) * F + beta(c)
```
* **Mechanism**: The command index `c` is mapped to an embedding vector. A Multi-Layer Perceptron (MLP) projects this embedding into channel-wise scale parameters `gamma(c)` and bias offsets `beta(c)`.
* **Execution**: Activations `F` from the driving features are multiplied by `gamma(c)` and offset by `beta(c)`. This dynamically modulates the network's spatial awareness based on the driving direction.

### B. Cross-Attention State Modulation
For ViT-based configurations, CSM routes spatial image features from the backbone as Queries (`Q`) and the command embedding vectors as Keys (`K`) and Values (`V`). This aligns path-planning attention maps directly onto relevant spatial regions.

---

## 2. Causal Feature Routing (CFR)

Traditional multi-task models process task heads in parallel without interaction. In NeuroPilot CFR (`neuralPilot_cfr.yaml`), we enforce a causal dependency:

```
Backbone Features (P3-P5) ──► Perception Neck (PAN) ──► Object Detection Head
                                   │
                                   ▼ (CFRBridge)
Backbone Features (P3-P5) ──► Driving Neck (PAN)    ──► Trajectory Planning Head
```

1. **Perception Neck (PAN)** extracts features for boundary and object localization.
2. The intermediate representations are routed through a **CFRBridge** to inject spatial knowledge (obstacle locations) directly into the **Driving Neck**.
3. The driving neck combines global context with the obstacle mask.
4. This ensures that the trajectory planner is explicitly aware of detected objects, improving safety in critical driving scenarios.

---

## 3. Dual-Branch Separation

When training on highly disparate datasets, joint backpropagation can lead to task interference (e.g., object detection gradients corrupting path-planning representations). The Dual-Branch neck configuration (`neuralPilot_dual.yaml`) separates representation learning into two non-interfering pathways:

* **Driving Branch**: Fuses P2 and P5 contexts for Heatmap + Trajectory planning.
* **Perception Branch**: Focuses on object classification and boundary localization.
* **Benefits**: Prevents gradient conflict and allows independent fine-tuning of either branch without degrading the other's accuracy.
