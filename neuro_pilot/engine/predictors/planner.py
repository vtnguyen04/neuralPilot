"""Dual-camera E2E trajectory planner and cabin driver monitoring predictors."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch

from neuro_pilot.engine.predictors.base import PREDICTOR_REGISTRY, BaseGraphPredictor
from neuro_pilot.engine.predictors.utils import bgr_to_yuv6, load_bgr, nms_xywh, sigmoid
from neuro_pilot.nn.zoo.graph import GraphModel
from neuro_pilot.utils.camera import (
    get_virtual_warp_matrix,
    resolve_camera_intrinsics,
    warp_to_model_view,
)

OP_SLICES: Dict[str, slice] = {
    "lane_lines": slice(0, 528),
    "lane_lines_prob": slice(528, 536),
    "road_edges": slice(536, 800),
    "meta": slice(800, 855),
    "desire_pred": slice(855, 887),
    "pose": slice(887, 899),
    "wide_from_device_euler": slice(899, 905),
    "road_transform": slice(905, 917),
    "plan": slice(917, 1907),
    "lead": slice(1907, 2051),
    "lead_prob": slice(2051, 2054),
    "desire_state": slice(2054, 2062),
    "action": slice(2062, 2066),
    "vision_feat": slice(2066, 2578),
    "pad": slice(2578, 2580),
}


def parse_openpilot_output(raw_2580: np.ndarray) -> Dict[str, np.ndarray]:
    """Decode openpilot's packed `[1, 2580]` output vector into named predictions."""
    vec = raw_2580[0]
    plan_raw = vec[OP_SLICES["plan"]]
    ll_raw = vec[OP_SLICES["lane_lines"]]
    re_raw = vec[OP_SLICES["road_edges"]]
    lead_raw = vec[OP_SLICES["lead"]]
    return {
        "plan": plan_raw[: 33 * 15].reshape(33, 15),
        "plan_std": np.exp(np.clip(plan_raw[33 * 15 :], -10.0, 10.0)).reshape(33, 15),
        "lane_lines": ll_raw[: 4 * 33 * 2].reshape(4, 33, 2),
        "lane_lines_std": np.exp(np.clip(ll_raw[4 * 33 * 2 :], -10.0, 10.0)).reshape(4, 33, 2),
        "lane_lines_prob": np.asarray(sigmoid(vec[OP_SLICES["lane_lines_prob"]]))[1::2],
        "lane_lines_prob_near": np.asarray(sigmoid(vec[OP_SLICES["lane_lines_prob"]]))[0::2],
        "road_edges": re_raw[: 2 * 33 * 2].reshape(2, 33, 2),
        "road_edges_std": np.exp(np.clip(re_raw[2 * 33 * 2 :], -10.0, 10.0)).reshape(2, 33, 2),
        "lead": lead_raw[: 3 * 6 * 4].reshape(3, 6, 4),
        "lead_prob": np.asarray(sigmoid(vec[OP_SLICES["lead_prob"]])),
        "desire_state": vec[OP_SLICES["desire_state"]],
        "action": vec[OP_SLICES["action"]],
        "pose": vec[OP_SLICES["pose"]],
        "vision_feat": vec[OP_SLICES["vision_feat"]],
    }


@PREDICTOR_REGISTRY.register("pilot_s")
@PREDICTOR_REGISTRY.register("pilot_l")
@PREDICTOR_REGISTRY.register("pilot_nav")
@PREDICTOR_REGISTRY.register("openpilot_small")
@PREDICTOR_REGISTRY.register("openpilot_chestnut_big")
@PREDICTOR_REGISTRY.register("openpilot_big")
@PREDICTOR_REGISTRY.register("openpilot_n16_nav")
@PREDICTOR_REGISTRY.register("openpilot_nav")
class PlannerPredictor(BaseGraphPredictor):
    """Stateful dual-camera predictor managing rolling temporal policy queues."""

    def __init__(
        self,
        model: GraphModel,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(model, device=device)
        self.feat_dim = int(model.input_specs["state_feat_q"]["shape"][-1])
        self.has_nav = "nav_features" in model.input_specs
        self.state_img_q: torch.Tensor
        self.state_desire_q: torch.Tensor
        self.state_feat_q: torch.Tensor
        self.reset_state()

    def reset_state(self) -> None:
        self.state_img_q = torch.zeros((2, 5, 6, 128, 256), dtype=torch.uint8, device=self.device)
        self.state_desire_q = torch.zeros((132, 1, 8), dtype=torch.float32, device=self.device)
        self.state_feat_q = torch.zeros(
            (128, 1, self.feat_dim), dtype=torch.float32, device=self.device
        )

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)

        narrow_bgr = load_bgr(source)
        wide_src = kwargs.get("wide_img")
        wide_bgr = load_bgr(wide_src) if wide_src is not None else narrow_bgr

        calib_rpy = kwargs.get("calib_rpy")
        if calib_rpy is not None:
            nh, nw = narrow_bgr.shape[:2]
            wh, ww = wide_bgr.shape[:2]
            k_nar = resolve_camera_intrinsics(nw, nh, kwargs.get("intrinsics"), wide=False)
            k_wid = resolve_camera_intrinsics(ww, wh, kwargs.get("wide_intrinsics"), wide=True)
            rpy_arr = np.asarray(calib_rpy, dtype=np.float64)
            wfe = np.asarray(
                kwargs.get("wide_from_device_euler", [0.0, 0.0, 0.0]), dtype=np.float64
            )
            narrow_bgr = warp_to_model_view(
                narrow_bgr, get_virtual_warp_matrix(rpy_arr, k_nar, wide_frame=False)
            )
            wide_bgr = warp_to_model_view(
                wide_bgr, get_virtual_warp_matrix(rpy_arr + wfe, k_wid, wide_frame=True)
            )

        new_img = torch.from_numpy(
            np.stack([bgr_to_yuv6(narrow_bgr), bgr_to_yuv6(wide_bgr)], axis=0)
        ).to(self.device)

        desire_val = kwargs.get("desire")
        desire = (
            torch.as_tensor(desire_val, dtype=torch.float32, device=self.device).reshape(8)
            if desire_val is not None
            else torch.zeros((8,), dtype=torch.float32, device=self.device)
        )
        tc_val = kwargs.get("traffic_convention", [1.0, 0.0])
        traffic = torch.as_tensor(tc_val, dtype=torch.float32, device=self.device).reshape(1, 2)
        at_val = kwargs.get("action_t", [0.275, 0.575])
        action_t = torch.as_tensor(at_val, dtype=torch.float32, device=self.device).reshape(1, 2)

        feed: Dict[str, torch.Tensor] = {
            "new_img": new_img,
            "desire": desire,
            "traffic_convention": traffic,
            "action_t": action_t,
            "state_img_q": self.state_img_q,
            "state_desire_q": self.state_desire_q,
            "state_feat_q": self.state_feat_q,
        }
        if self.has_nav:
            nav_feat = kwargs.get("nav_features")
            nav_instr = kwargs.get("nav_instructions")
            feed["nav_features"] = (
                torch.as_tensor(nav_feat, dtype=torch.float32, device=self.device)
                if nav_feat is not None
                else torch.zeros((1, 256), dtype=torch.float32, device=self.device)
            )
            feed["nav_instructions"] = (
                torch.as_tensor(nav_instr, dtype=torch.float32, device=self.device)
                if nav_instr is not None
                else torch.zeros((1, 150), dtype=torch.float32, device=self.device)
            )
        return feed

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        if kwargs.get("update_state", True):
            self.state_img_q = raw_outputs["next_state_img_q"].detach()
            self.state_desire_q = raw_outputs["next_state_desire_q"].detach().float()
            self.state_feat_q = raw_outputs["next_state_feat_q"].detach().float()

        raw_vec = raw_outputs["outputs"].detach().cpu().numpy()
        parsed = parse_openpilot_output(raw_vec)
        parsed["raw"] = raw_vec
        return parsed


@PREDICTOR_REGISTRY.register("driver")
@PREDICTOR_REGISTRY.register("openpilot_dmonitoring")
@PREDICTOR_REGISTRY.register("dmonitoring")
class DriverPredictor(BaseGraphPredictor):
    """In-cabin driver monitoring predictor accepting BGR/gray frames or raw dict."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        img = load_bgr(source)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        resized = cv2.resize(gray, (1440, 960), interpolation=cv2.INTER_LINEAR)
        flat = torch.from_numpy(np.ascontiguousarray(resized.reshape(1, -1))).to(self.device)
        calib_val = kwargs.get("calib", [0.0, 0.0, 0.0])
        calib = torch.as_tensor(calib_val, dtype=torch.float32, device=self.device).reshape(1, 3)
        return {"input_img": flat, "calib": calib}

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        vec = raw_outputs["outputs"].detach().cpu().numpy()
        return {"outputs": vec, "raw": vec}


@PREDICTOR_REGISTRY.register("pilot_det")
@PREDICTOR_REGISTRY.register("pilot_mhp")
@PREDICTOR_REGISTRY.register("bfmc")
@PREDICTOR_REGISTRY.register("bfmc_mhp")
class MultitaskPredictor(BaseGraphPredictor):
    """Single-camera multi-task predictor returning 2D waypoints and NMS detections."""

    def preprocess(self, source: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        if isinstance(source, dict):
            return super().preprocess(source)
        bgr = load_bgr(source)
        img_shape = self.model.input_specs.get("img", {}).get("shape", [1, 6, 256, 320])
        h_in, w_in = int(img_shape[-2]) * 2, int(img_shape[-1]) * 2
        yuv = torch.from_numpy(bgr_to_yuv6(bgr, width=w_in, height=h_in)).unsqueeze(0)
        cmd_val = int(kwargs.get("command", 0))
        cmd = torch.tensor([cmd_val], dtype=torch.int64, device=self.device)
        return {"img": yuv.to(self.device), "command": cmd}

    def postprocess(
        self, raw_outputs: Dict[str, torch.Tensor], **kwargs: Any
    ) -> Dict[str, Any]:
        conf = float(kwargs.get("conf", 0.30))
        iou = float(kwargs.get("iou", 0.50))
        img_shape = self.model.input_specs.get("img", {}).get("shape", [1, 6, 256, 320])
        h_in, w_in = float(int(img_shape[-2]) * 2), float(int(img_shape[-1]) * 2)
        det_np = raw_outputs["det"].detach().float().cpu().numpy()
        wps = raw_outputs["waypoints"].detach().float().cpu().numpy()[0]
        raw_boxes = nms_xywh(det_np, conf_thres=conf, iou_thres=iou)
        boxes = [
            (
                max(0.0, min(1.0, x1 / w_in)),
                max(0.0, min(1.0, y1 / h_in)),
                max(0.0, min(1.0, x2 / w_in)),
                max(0.0, min(1.0, y2 / h_in)),
                score,
                cls_id,
            )
            for x1, y1, x2, y2, score, cls_id in raw_boxes
        ]
        return {
            "waypoints": wps,
            "detections": boxes,
            "raw_det": det_np,
            "img_wh": (int(w_in), int(h_in)),
        }
