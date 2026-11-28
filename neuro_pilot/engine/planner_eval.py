"""Generic multi-task planner validator (trajectory geometry, per-command, and detection mAP)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from neuro_pilot.data.datasets.planner import PlannerDataset, planner_collate_fn
from neuro_pilot.engine.predictors.utils import bgr_to_yuv6
from neuro_pilot.utils.planner_metrics import (
    decode_nms,
    evaluate_detection,
    frenet_lateral_err,
    sym_curve_dist,
    traj_smoothness,
)
from neuro_pilot.utils.visualize import (
    DEFAULT_CLS_NAMES,
    DEFAULT_CMD_NAMES,
    render_image_grid,
    render_planner_frame,
)

__all__ = [
    "PlannerValidator",
    "decode_nms",
    "evaluate_planner",
    "frenet_lateral_err",
    "sym_curve_dist",
    "traj_smoothness",
]


@torch.no_grad()
def evaluate_planner(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    img_w: int = 640,
    img_h: int = 512,
    class_names: Sequence[str] = DEFAULT_CLS_NAMES,
    cmd_names: Sequence[str] = DEFAULT_CMD_NAMES,
) -> Dict[str, Any]:
    """Evaluate joint trajectory (incl. per-command slices) and multi-class detection mAP."""
    was_training = model.training
    model.eval()
    ades, fdes, entries, lats, mses, curves, jerks, kinks, cmds = ([] for _ in range(9))
    min_mses: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_gts: List[Tuple[np.ndarray, np.ndarray]] = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for batch in val_loader:
        yuv6 = batch["yuv6"].to(device, non_blocking=True)
        cmd = batch["command"].to(device, non_blocking=True)
        gt_t = batch["waypoints"]
        gt_wp = gt_t.numpy()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(img=yuv6, command=cmd)
        pred_t = out["waypoints"].float().cpu()
        pred_wp = pred_t.numpy()
        diff = pred_wp - gt_wp
        dist = np.linalg.norm(diff, axis=-1)
        tail = max(2, gt_wp.shape[1] // 4)
        ades.append(dist.mean(axis=-1))
        fdes.append(dist[:, -1])
        entries.append(dist[:, -tail:].mean(axis=-1))
        lats.append(frenet_lateral_err(pred_wp, gt_wp))
        mses.append((diff ** 2).mean(axis=(1, 2)))
        curves.append(sym_curve_dist(pred_t, gt_t.float()).numpy())
        jk, ek = traj_smoothness(pred_wp)
        jerks.append(jk)
        kinks.append(ek)
        if "mu" in out:
            mu_np = out["mu"].float().cpu().numpy()
            min_mses.append(((mu_np - gt_wp[:, None, :, :]) ** 2).mean(axis=(2, 3)).min(axis=1))
        cmds.append(batch["command"].numpy())
        all_preds.extend(decode_nms(out["det"].float(), img_w=img_w, img_h=img_h))
        _collect_gt_boxes(batch, yuv6.shape[0], all_gts)

    if was_training:
        model.train()
    return _aggregate_metrics(
        ades, fdes, entries, lats, mses, curves, jerks, kinks, cmds,
        min_mses, all_preds, all_gts, class_names, cmd_names,
    )


def _collect_gt_boxes(
    batch: Dict[str, torch.Tensor], bs: int, out_gts: List[Tuple[np.ndarray, np.ndarray]]
) -> None:
    b_idx = batch["batch_idx"].numpy()
    b_cls = batch["cls"].numpy()
    b_box = batch["bboxes"].numpy()
    for i in range(bs):
        m = b_idx == i
        if np.any(m):
            bx = b_box[m]
            xyxy = np.stack(
                [
                    bx[:, 0] - bx[:, 2] * 0.5,
                    bx[:, 1] - bx[:, 3] * 0.5,
                    bx[:, 0] + bx[:, 2] * 0.5,
                    bx[:, 1] + bx[:, 3] * 0.5,
                ],
                axis=1,
            )
            out_gts.append((b_cls[m].astype(np.int64), np.clip(xyxy, 0.0, 1.0)))
        else:
            out_gts.append((np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)))


def _aggregate_metrics(
    ades, fdes, entries, lats, mses, curves, jerks, kinks, cmds,
    min_mses, all_preds, all_gts, class_names, cmd_names,
) -> Dict[str, Any]:
    ade_a, fde_a, cmd_a = np.concatenate(ades), np.concatenate(fdes), np.concatenate(cmds)
    lat_a, mse_a, crv_a = np.concatenate(lats), np.concatenate(mses), np.concatenate(curves)
    jk_a, ek_a, ent_a = np.concatenate(jerks), np.concatenate(kinks), np.concatenate(entries)
    min_a = np.concatenate(min_mses) if min_mses else None
    ade, fde = float(ade_a.mean()), float(fde_a.mean())
    im = cmd_a != 0
    det_res = evaluate_detection(all_preds, all_gts, class_names)
    cmd_metrics: Dict[str, Dict[str, Any]] = {}
    for c in range(len(cmd_names)):
        mc = cmd_a == c
        if not np.any(mc):
            continue
        cm: Dict[str, Any] = {
            "n": int(mc.sum()),
            "ade_px": round(float(ade_a[mc].mean()) * 640.0, 2),
            "med_px": round(float(np.median(ade_a[mc])) * 640.0, 2),
            "p90_px": round(float(np.percentile(ade_a[mc], 90)) * 640.0, 2),
            "fde_px": round(float(fde_a[mc].mean()) * 640.0, 2),
            "entry25_px": round(float(ent_a[mc].mean()) * 640.0, 2),
            "lat_px": round(float(lat_a[mc].mean()) * 640.0, 2),
            "mse": round(float(mse_a[mc].mean()), 6),
            "kink_deg": round(float(ek_a[mc].mean()), 2),
        }
        if min_a is not None:
            cm["min_mse5"] = round(float(min_a[mc].mean()), 6)
        cmd_metrics[cmd_names[c]] = cm
    res: Dict[str, Any] = {
        "score": round(50.0 * det_res["map50"] + 50.0 * max(0.0, 1.0 - ade / 0.12), 3),
        "mse": round(float(mse_a.mean()), 6),
        "ade": round(ade, 5),
        "ade_px640": round(ade * 640.0, 2),
        "fde": round(fde, 5),
        "fde_px640": round(fde * 640.0, 2),
        "inter_ade": round(float(ade_a[im].mean()) if np.any(im) else ade, 5),
        "inter_fde": round(float(fde_a[im].mean()) if np.any(im) else fde, 5),
        "lat_err": round(float(lat_a.mean()), 5),
        "lat_px640": round(float(lat_a.mean()) * 640.0, 2),
        "curve_err": round(float(crv_a.mean()), 5),
        "curve_px640": round(float(crv_a.mean()) * 640.0, 2),
        "jerk_l2": round(float(jk_a.mean()), 5),
        "end_kink_deg": round(float(ek_a.mean()), 2),
        "cmd_ade": {
            cmd_names[c]: round(float(ade_a[cmd_a == c].mean()), 5)
            for c in range(len(cmd_names))
            if np.any(cmd_a == c)
        },
        "cmd_metrics": cmd_metrics,
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in det_res.items()},
    }
    if min_a is not None:
        res["min_mse5"] = round(float(min_a.mean()), 6)
    return res


class PlannerValidator:
    """Callable validator wired into `PlannerTask.get_validator()` and `NeuroPilot.val()`."""

    def __init__(self, model: torch.nn.Module, overrides: Optional[Dict[str, Any]] = None) -> None:
        self.model = model
        self.overrides = overrides or {}
        img_shape = getattr(model, "input_specs", {}).get("img", {}).get("shape", [1, 6, 256, 320])
        self.img_w = int(self.overrides.get("img_w", int(img_shape[-1]) * 2))
        self.img_h = int(self.overrides.get("img_h", int(img_shape[-2]) * 2))
        self.class_names = tuple(self.overrides.get("class_names", DEFAULT_CLS_NAMES))
        self.cmd_names = tuple(self.overrides.get("cmd_names", DEFAULT_CMD_NAMES))

    def __call__(
        self,
        dataloader: Optional[DataLoader] = None,
        val_json: Optional[str | Path] = None,
        save_grid: Optional[str | Path] = None,
        batch_size: int = 8,
        conf: float = 0.30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        device = next(self.model.parameters()).device
        val_path = val_json or self.overrides.get("val_json") or kwargs.get("data")
        if dataloader is None:
            if not val_path:
                raise ValueError("Pass `dataloader` or `val_json` to validate PlannerTask.")
            ds = PlannerDataset(val_path, self.img_w, self.img_h, augment=False)
            dataloader = DataLoader(
                ds, batch_size=int(batch_size), shuffle=False, num_workers=2,
                pin_memory=(device.type == "cuda"), collate_fn=planner_collate_fn,
            )
        metrics = evaluate_planner(
            self.model, dataloader, device, self.img_w, self.img_h,
            class_names=self.class_names, cmd_names=self.cmd_names,
        )
        if save_grid and val_path:
            self.render_val_grid(val_path, save_grid, device=device, conf=float(conf))
        return metrics

    @torch.no_grad()
    def render_val_grid(
        self, val_json: str | Path, out_jpg: str | Path, device: torch.device, conf: float = 0.30
    ) -> Path:
        """Render a stratified 4-command validation grid overlay to `out_jpg`."""
        val_dir = Path(val_json).resolve().parent
        recs = json.loads(Path(val_json).read_text())
        picks, seen = [], set()
        for target_cmd in range(len(self.cmd_names)):
            cands = [r for r in recs if r["command"] == target_cmd and r.get("bboxes")] or [
                r for r in recs if r["command"] == target_cmd
            ]
            for idx in np.linspace(0, max(0, len(cands) - 1), min(3, len(cands)), dtype=int):
                r = cands[int(idx)]
                rid = r.get("id", r.get("img_path"))
                if rid not in seen:
                    picks.append(r)
                    seen.add(rid)
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        tiles = []
        for rec in picks[:12]:
            ip = Path(rec["img_path"])
            bgr = cv2.imread(str(ip if ip.is_absolute() else (val_dir / ip)), cv2.IMREAD_COLOR)
            yuv = torch.from_numpy(bgr_to_yuv6(bgr, self.img_w, self.img_h)).unsqueeze(0).to(device)
            cmd = torch.tensor([int(rec["command"])], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = self.model(img=yuv, command=cmd)
            dets = decode_nms(out["det"].float(), self.img_w, self.img_h, conf_thres=conf)[0]
            vis_out = {"waypoints": out["waypoints"][0].float().cpu().numpy(), "detections": dets}
            tiles.append(render_planner_frame(
                cv2.resize(bgr, (512, 320)), vis_out, command=int(rec["command"]),
                gt_waypoints=np.asarray(rec["waypoints"], dtype=np.float32),
                conf_thres=conf, class_names=self.class_names, cmd_names=self.cmd_names,
                tag=str(rec.get("session", ""))[:16],
            ))
        out_path = Path(out_jpg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), render_image_grid(tiles, cols=3), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return out_path
