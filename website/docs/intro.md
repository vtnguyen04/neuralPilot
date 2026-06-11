---
sidebar_position: 1
---
# Introduction to NeuroPilot

Welcome to the documentation for **NeuroPilot** — a unified, high-performance end-to-end (E2E) autonomous driving framework designed for multi-task perception and real-time edge deployment.

import useBaseUrl from '@docusaurus/useBaseUrl';

<p align="center">
  <img src={useBaseUrl('/img/logo.png')} alt="NeuroPilot Logo" width="200" />
</p>

## Overview

NeuroPilot enables self-driving vehicles and robots to perform multiple crucial perception and planning tasks simultaneously within a **single forward pass**. By sharing a unified feature backbone, the network drastically reduces computational overhead, making it ideal for power-constrained hardware like the **NVIDIA Jetson Orin Nano**.

---

## High-Level Pipeline Architecture

Below is the workflow showing how inputs are processed through the shared backbone to separate perception and planning heads:

```mermaid
graph TD
    subgraph Inputs["Inputs"]
        IMG["RGB Image 320x320"]
        CMD["Navigation Command"]
    end

    subgraph Backbone["Shared Stem"]
        STEM["Conv → Conv → C3k2"]
    end

    subgraph Driving["Driving Head Group"]
        D_ENC["YOLO11 Encoder"]
        CSM["Command State Modulation"]
        D_NECK["PAN Neck"]
        HM["Heatmap Head"]
        TJ["Trajectory Head"]
        CLS["Classification Head"]
    end

    subgraph Perception["Perception Head Group"]
        P_ENC["YOLO11 Encoder"]
        P_NECK["PAN Neck"]
        DET["Detect Head"]
    end

    IMG --> STEM
    STEM -->|P2| D_ENC
    STEM -->|P2| P_ENC
    CMD --> CSM
    D_ENC --> CSM --> D_NECK
    D_NECK --> HM & TJ & CLS
    STEM -.->|P2 skip| HM
    HM -.->|spatial mask| TJ
    CMD -.-> TJ
    P_ENC --> P_NECK --> DET
```

---

## Core Capabilities

- **Joint Multi-Task Learning**: Shares a single convolutional or attention-based backbone to solve path planning, bounding box detection, saliency heatmap prediction, and command classification simultaneously.
- **Dynamic Task Toggling**: Toggle task heads on/off during training using loss weight multiplier configs (lambdas).
- **Edge-Native Optimization**: Provides built-in TensorRT export engines achieving sub-30ms execution on NVIDIA Jetson Orin Nano boards.
- **Pluggable Backbones**: Swap backbones using a single configuration line, supporting CNNs and ViTs from the `timm` library.
- **Advanced Spatial Guidance**: Features Causal Feature Routing (CFR) to route obstacle-awareness representations into the trajectory planner path.
