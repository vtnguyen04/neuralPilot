"""Perception tasks (semantic segmentation, monocular depth, ego-lane, vehicle/CIPO)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from neuro_pilot.core.registry import Registry
from neuro_pilot.tasks.base import GraphDrivingTask


class _MetricValidator:
    """Lightweight task validator aggregating batch metrics into a summary dict."""

    def __init__(self, model: Any, eval_fn: Any) -> None:
        self.model = model
        self.eval_fn = eval_fn

    @torch.no_grad()
    def __call__(self, dataloader: Any = None, **kwargs: Any) -> Dict[str, float]:
        if dataloader is None:
            raise ValueError("Pass `dataloader` to validate this perception task.")
        return self.eval_fn(self.model, dataloader, **kwargs)


def _eval_seg_miou(
    model: Any, dataloader: Any, num_classes: int = 19, **_: Any
) -> Dict[str, float]:
    dev = next(model.parameters()).device
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    model.eval()
    for batch in dataloader:
        img = batch["image"].to(dev)
        gt = batch["mask"].cpu().numpy()
        pred = model(image=img)["mask"].argmax(dim=1).cpu().numpy()
        for c in range(num_classes):
            p_m, g_m = pred == c, gt == c
            inter[c] += float(np.logical_and(p_m, g_m).sum())
            union[c] += float(np.logical_or(p_m, g_m).sum())
    valid = union > 0
    miou = float(np.mean(inter[valid] / union[valid])) if np.any(valid) else 0.0
    return {"miou": round(miou, 4)}


def _eval_depth_err(model: Any, dataloader: Any, **_: Any) -> Dict[str, float]:
    dev = next(model.parameters()).device
    abs_rels, rmses = [], []
    model.eval()
    for batch in dataloader:
        img = batch["input"].to(dev)
        gt = batch["depth"].float().cpu().numpy()
        pred = model(input=img)["output"].float().cpu().numpy().reshape(gt.shape)
        valid = gt > 1e-3
        if np.any(valid):
            diff = pred[valid] - gt[valid]
            abs_rels.append(float(np.mean(np.abs(diff) / gt[valid])))
            rmses.append(float(np.sqrt(np.mean(diff ** 2))))
    return {
        "abs_rel": round(float(np.mean(abs_rels)), 4) if abs_rels else 0.0,
        "rmse": round(float(np.mean(rmses)), 4) if rmses else 0.0,
    }


@Registry.register_task("seg")
class SegTask(GraphDrivingTask):
    """Semantic road and obstacle segmentation (`seg`, `seg_l`)."""

    task_name: str = "seg"
    default_cfg: str = "seg"
    default_loss: str = "laplace_nll"

    def get_validator(self) -> Any:
        if self.model is None:
            self.build_model()
        return _MetricValidator(self.model, _eval_seg_miou)

    def visualize(
        self, frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        from neuro_pilot.utils.visualize import render_seg_frame

        return render_seg_frame(frame_bgr, output, **kwargs)


@Registry.register_task("depth")
class DepthTask(GraphDrivingTask):
    """Monocular 3D depth estimation (`depth`)."""

    task_name: str = "depth"
    default_cfg: str = "depth"
    default_loss: str = "laplace_nll"

    def get_validator(self) -> Any:
        if self.model is None:
            self.build_model()
        return _MetricValidator(self.model, _eval_depth_err)

    def visualize(
        self, frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        from neuro_pilot.utils.visualize import render_depth_frame

        return render_depth_frame(frame_bgr, output, **kwargs)


@Registry.register_task("lane")
class LaneTask(GraphDrivingTask):
    """Row-anchor ego-lane detection (`lane`)."""

    task_name: str = "lane"
    default_cfg: str = "lane"
    default_loss: str = "waypoint"

    def visualize(
        self, frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        from neuro_pilot.utils.visualize import render_lane_frame

        return render_lane_frame(frame_bgr, output, **kwargs)


@Registry.register_task("cipo")
@Registry.register_task("drive")
class CipoTask(GraphDrivingTask):
    """Vehicle/CIPO object detection and lead distance estimation (`cipo`, `drive`)."""

    task_name: str = "cipo"
    default_cfg: str = "cipo"
    default_loss: str = "laplace_nll"

    def visualize(
        self, frame_bgr: np.ndarray, output: Dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        from neuro_pilot.utils.visualize import render_cipo_frame

        return render_cipo_frame(frame_bgr, output, **kwargs)
