"""VisionPilot perception predictors (AutoSpeed2, AutoSteer2, AutoDrive, Scene3D)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from neuro_pilot.engine.predictors.base import PREDICTOR_REGISTRY, BaseGraphPredictor
from neuro_pilot.engine.predictors.utils import (
    colorize_depth,
    load_bgr,
    nms_xywh,
    sigmoid,
    to_imagenet_tensor,
    to_rgb_01,
)
from neuro_pilot.nn.zoo.graph import GraphModel


@PREDICTOR_REGISTRY.register("cipo")
@PREDICTOR_REGISTRY.register("visionpilot_autospeed2")
@PREDICTOR_REGISTRY.register("autospeed")
class CipoPredictor(BaseGraphPredictor):
    """4-class traffic object detector with NMS box decoding."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        bgr = load_bgr(source)
        return {"input": to_rgb_01(bgr, 1024, 512).to(self.device)}

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        raw = raw_outputs["output"].detach().cpu().numpy()
        conf = float(kwargs.get("conf", 0.45))
        iou = float(kwargs.get("iou", 0.45))
        return {"boxes": nms_xywh(raw, conf_thres=conf, iou_thres=iou), "raw": raw}


@PREDICTOR_REGISTRY.register("lane")
@PREDICTOR_REGISTRY.register("visionpilot_autosteer2")
@PREDICTOR_REGISTRY.register("autosteer")
class LanePredictor(BaseGraphPredictor):
    """Row-anchor ego-lane predictor returning valid `(x, y)` points on `1024x512`."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        bgr = load_bgr(source)
        return {"input": to_rgb_01(bgr, 1024, 512).to(self.device)}

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        lane_value = raw_outputs["lane_value"].detach().cpu().numpy().reshape(-1)
        height = raw_outputs["height"].detach().cpu().numpy().reshape(-1)
        thresh = float(kwargs.get("height_thres", 0.5))
        ys = np.linspace(0.0, 511.0, len(lane_value))
        points = [
            (float(xv * 1024.0), float(yv))
            for xv, yv, hv in zip(lane_value, ys, height)
            if hv >= thresh
        ]
        return {"points": points, "lane_value": lane_value, "height": height}


@PREDICTOR_REGISTRY.register("drive")
@PREDICTOR_REGISTRY.register("visionpilot_autodrive")
@PREDICTOR_REGISTRY.register("autodrive")
class DrivePredictor(BaseGraphPredictor):
    """Temporal Siamese CIPO distance, curvature, and lead-flag predictor."""

    def __init__(
        self,
        model: GraphModel,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(model, device=device)
        self._prev_tensor: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self._prev_tensor = None

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        curr_t = to_imagenet_tensor(load_bgr(source), 1024, 512).to(self.device)
        prev_src = kwargs.get("prev_img")
        if prev_src is not None:
            prev_t = to_imagenet_tensor(load_bgr(prev_src), 1024, 512).to(self.device)
        elif self._prev_tensor is not None:
            prev_t = self._prev_tensor
        else:
            prev_t = curr_t
        self._prev_tensor = curr_t.detach()
        return {"image_prev": prev_t, "image_curr": curr_t}

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        dist_norm = float(raw_outputs["distance"].detach().cpu().reshape(-1)[0].item())
        curv_norm = float(raw_outputs["curvature"].detach().cpu().reshape(-1)[0].item())
        flag_logit = float(raw_outputs["flag_logit"].detach().cpu().reshape(-1)[0].item())
        return {
            "distance_m": 150.0 * (1.0 - float(np.clip(dist_norm, 0.0, 1.0))),
            "curvature": curv_norm * 0.21,
            "cipo_prob": float(sigmoid(flag_logit)),
        }


@PREDICTOR_REGISTRY.register("depth")
@PREDICTOR_REGISTRY.register("visionpilot_scene3d_lite")
@PREDICTOR_REGISTRY.register("scene3d")
class DepthPredictor(BaseGraphPredictor):
    """Monocular depth estimator returning raw depth grid and Inferno heatmap."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        bgr = load_bgr(source)
        return {"input": to_imagenet_tensor(bgr, 640, 320).to(self.device)}

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        depth = raw_outputs["output"].detach().cpu().numpy()[0, 0]
        return {"depth": depth, "depth_vis": colorize_depth(depth)}


@PREDICTOR_REGISTRY.register("seg")
@PREDICTOR_REGISTRY.register("seg_l")
@PREDICTOR_REGISTRY.register("ddrnet23_slim")
@PREDICTOR_REGISTRY.register("eomt_large")
class SegPredictor(BaseGraphPredictor):
    """Semantic driving segmentation predictor (`seg` DDRNet23-Slim & `seg_l` EoMT-Large)."""

    def __init__(
        self,
        model: GraphModel,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(model, device=device)
        self._use_fp16 = (
            self.device.type == "cuda"
            and getattr(self.model, "model_key", "") == "seg_l"
            and hasattr(self.model, "half")
        )
        if self._use_fp16:
            self.model.half()

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            img = source.get("image")
            if isinstance(img, torch.Tensor):
                self._last_hw = (int(img.shape[-2]), int(img.shape[-1]))
            out = super().preprocess(source)
        else:
            bgr = load_bgr(source)
            is_large = getattr(self.model, "model_key", "") == "seg_l"
            width = int(kwargs.get("width", 896 if is_large else 1024))
            height = int(kwargs.get("height", 512))
            self._last_hw = (height, width)
            out = {"image": to_rgb_01(bgr, width, height).to(self.device)}
        if self._use_fp16 and "image" in out:
            out["image"] = out["image"].half()
        return out

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        logits_t = raw_outputs["mask"].detach()
        default_h, default_w = getattr(self, "_last_hw", (logits_t.shape[-2], logits_t.shape[-1]))
        out_h = int(kwargs.get("height", default_h))
        out_w = int(kwargs.get("width", default_w))
        if logits_t.shape[-2:] != (out_h, out_w):
            logits_up = torch.nn.functional.interpolate(
                logits_t.float(), size=(out_h, out_w), mode="bilinear", align_corners=False
            )
        else:
            logits_up = logits_t.float()
        class_map = torch.argmax(logits_up[0], dim=0).cpu().numpy().astype(np.uint8)
        return {"class_map": class_map, "logits": logits_t[0].cpu().numpy()}
