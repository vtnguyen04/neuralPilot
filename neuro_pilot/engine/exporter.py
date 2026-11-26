import torch
import torch.nn as nn
import onnx
from neuro_pilot.utils.logger import logger


class TRTEfficientNMS(torch.autograd.Function):
    """TensorRT EfficientNMS plugin wrapper for ONNX export."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold=0.25,
        iou_threshold=0.45,
        max_output_boxes=100,
        background_class=-1,
        score_activation=0,
        box_coding=0,
    ):
        B = boxes.shape[0]
        device = boxes.device
        num_dets = torch.zeros((B, 1), dtype=torch.int32, device=device)
        det_boxes = torch.zeros((B, max_output_boxes, 4), dtype=torch.float32, device=device)
        det_scores = torch.zeros((B, max_output_boxes), dtype=torch.float32, device=device)
        det_classes = torch.zeros((B, max_output_boxes), dtype=torch.int32, device=device)
        return num_dets, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        score_threshold=0.25,
        iou_threshold=0.45,
        max_output_boxes=100,
        background_class=-1,
        score_activation=0,
        box_coding=0,
    ):
        return g.op(
            "TRT::EfficientNMS_TRT",
            boxes,
            scores,
            score_threshold_f=float(score_threshold),
            iou_threshold_f=float(iou_threshold),
            max_output_boxes_i=int(max_output_boxes),
            background_class_i=int(background_class),
            score_activation_i=int(score_activation),
            box_coding_i=int(box_coding),
            outputs=4,
        )


class Exporter:
    """
    Unified Exporter for Neuro Pilot.
    Supports ONNX (standard) and TensorRT engine (via trtexec).
    """

    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
        from neuro_pilot.engine.model import NeuroPilot

        if isinstance(model, NeuroPilot) and hasattr(model, "model"):
            self.model = model.model
        self.device = device
        from .callbacks import CallbackList

        self.callbacks = CallbackList()

    def __call__(self, format="onnx", imgsz=None, **kwargs):
        """Main export entry point."""
        self.callbacks.on_export_start(self)
        logger.info(f"Exporting model to {format}...")
        format = format.lower()
        if format == "onnx":
            path = self.export_onnx(imgsz, **kwargs)
        elif format in {"engine", "trt", "tensorrt"}:
            path = self.export_engine(imgsz, **kwargs)
        elif format in {"torchscript", "ts"}:
            path = self.export_torchscript(**kwargs)
        else:
            raise NotImplementedError(f"Format {format} is not supported yet.")
        self.callbacks.on_export_end(self)
        return path

    def export_torchscript(self, **kwargs):
        """Export a GraphModel to a self-describing `.torchscript` file."""
        from neuro_pilot.nn.zoo.graph import GraphModel

        output_path = kwargs.get("file", "neuro_pilot_model.torchscript")
        if isinstance(self.model, GraphModel):
            out = self.model.export_torchscript(
                output_path, sample_inputs=kwargs.get("sample_inputs")
            )
            logger.info(f"Exported GraphModel TorchScript: {out}")
            return str(out)
        raise NotImplementedError("TorchScript export is currently implemented for GraphModel.")

    def export_onnx(self, imgsz=None, simplify=True, opset=17, end2end=False, **kwargs):
        """Export to ONNX with proper multi-head output handling and optional EfficientNMS."""
        from neuro_pilot.nn.zoo.graph import GraphModel

        output_path = kwargs.get("file", "neuro_pilot_model.onnx")
        if isinstance(self.model, GraphModel):
            out = self.model.export_onnx(
                output_path,
                opset=opset,
                sample_inputs=kwargs.get("sample_inputs"),
                dynamic_axes=kwargs.get("dynamic_axes"),
            )
            logger.info(f"Exported GraphModel ONNX: {out}")
            return str(out)

        imgsz = imgsz or getattr(self.cfg.data, "image_size", 320) if hasattr(self.cfg, "data") else 320
        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)

        skip_heatmap = kwargs.get("skip_heatmap", False)
        end2end = kwargs.get("end2end", end2end)
        score_thresh = kwargs.get("score_thresh", 0.25)
        iou_thresh = kwargs.get("iou_thresh", 0.45)
        max_det = kwargs.get("max_det", 100)

        from neuro_pilot.cfg.schema import HeadConfig

        _head = HeadConfig()
        im = torch.zeros(1, 3, *imgsz).to(self.device).float()
        cmd = torch.zeros(1, _head.num_commands).to(self.device).float()

        self.model.eval()

        class ExportAdapter(nn.Module):
            """Flattens NeuroPilot dict outputs into ordered tuple for ONNX export."""

            def __init__(
                self,
                model,
                skip_heatmap=False,
                end2end=False,
                score_thresh=0.25,
                iou_thresh=0.45,
                max_det=100,
            ):
                super().__init__()
                self.model = model
                self.skip_heatmap = skip_heatmap
                self.end2end = end2end
                self.score_thresh = score_thresh
                self.iou_thresh = iou_thresh
                self.max_det = max_det

            def forward(self, x, cmd):
                out = self.model(x, cmd)

                B, _, H_in, W_in = x.shape
                device = x.device

                if isinstance(out, dict):
                    bboxes = out.get("bboxes")
                    if bboxes is None:
                        bboxes = torch.zeros(B, 18, 0, device=device)

                    waypoints = out.get("waypoints")
                    if waypoints is None:
                        waypoints = torch.zeros(B, _head.num_waypoints, 2, device=device)

                    if not self.skip_heatmap:
                        hm = out.get("heatmap")
                        if isinstance(hm, dict):
                            hm = hm.get("heatmap")
                        if hm is None:
                            hm = torch.zeros(B, 1, H_in, W_in, device=device)
                    else:
                        hm = torch.zeros(B, 1, 1, 1, device=device)

                    classes = out.get("classes")
                    if classes is None:
                        classes = torch.zeros(B, _head.num_commands, device=device)

                    if self.end2end:
                        # bboxes shape: (B, 4 + nc, num_anchors) -> transpose to (B, num_anchors, 4 + nc)
                        bboxes_t = bboxes.permute(0, 2, 1)
                        cxcywh = bboxes_t[..., :4]
                        scores = bboxes_t[..., 4:]

                        # Convert cx, cy, w, h -> x1, y1, x2, y2 (BoxCorner, box_coding=0)
                        x1 = cxcywh[..., 0] - cxcywh[..., 2] / 2
                        y1 = cxcywh[..., 1] - cxcywh[..., 3] / 2
                        x2 = cxcywh[..., 0] + cxcywh[..., 2] / 2
                        y2 = cxcywh[..., 1] + cxcywh[..., 3] / 2
                        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                        num_dets, det_boxes, det_scores, det_classes = TRTEfficientNMS.apply(
                            boxes_xyxy,
                            scores,
                            self.score_thresh,
                            self.iou_thresh,
                            self.max_det,
                            -1,  # background_class
                            0,   # score_activation = 0 (already sigmoid)
                            0,   # box_coding = 0 (BoxCorner x1,y1,x2,y2)
                        )
                        return num_dets, det_boxes, det_scores, det_classes, waypoints, hm, classes

                    return bboxes, waypoints, hm, classes

                return out

        model_wrapper = ExportAdapter(
            self.model,
            skip_heatmap=skip_heatmap,
            end2end=end2end,
            score_thresh=score_thresh,
            iou_thresh=iou_thresh,
            max_det=max_det,
        ).to(self.device)
        model_wrapper.eval()

        dynamic_axes = None
        if end2end:
            output_names = [
                "num_dets",
                "det_boxes",
                "det_scores",
                "det_classes",
                "trajectory",
                "heatmap",
                "classes",
            ]
            if kwargs.get("dynamic", False):
                dynamic_axes = {
                    "image": {0: "batch"},
                    "command": {0: "batch"},
                    "num_dets": {0: "batch"},
                    "det_boxes": {0: "batch"},
                    "det_scores": {0: "batch"},
                    "det_classes": {0: "batch"},
                    "trajectory": {0: "batch"},
                    "heatmap": {0: "batch"},
                    "classes": {0: "batch"},
                }
        else:
            output_names = ["bboxes", "trajectory", "heatmap", "classes"]
            if kwargs.get("dynamic", False):
                dynamic_axes = {
                    "image": {0: "batch"},
                    "command": {0: "batch"},
                    "bboxes": {0: "batch"},
                    "trajectory": {0: "batch"},
                    "heatmap": {0: "batch"},
                    "classes": {0: "batch"},
                }

        logger.info(
            f"Exporting ONNX: imgsz={imgsz}, opset={opset}, end2end={end2end}, skip_heatmap={skip_heatmap}"
        )
        torch.onnx.export(
            model_wrapper,
            (im, cmd),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image", "command"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        if simplify and not end2end:
            try:
                from onnxsim import simplify as onnx_simplify

                model_onnx = onnx.load(output_path)
                model_simp, check = onnx_simplify(model_onnx)
                if check:
                    onnx.save(model_simp, output_path, save_as_external_data=False, all_tensors_to_one_file=True)
                    logger.info("ONNX simplification success.")
            except ImportError:
                logger.warning("onnx-simplifier not found. Skipping simplification.")
            except Exception as e:
                logger.warning(f"onnx-simplifier skipped: {e}")

        try:
            model_onnx = onnx.load(output_path)
            for key, value in {
                "names": str(getattr(self.model, "names", {})),
                "stride": str(int(max(getattr(self.model, "stride", [32])))),
                "imgsz": str(imgsz),
                "skip_heatmap": str(skip_heatmap),
            }.items():
                meta = model_onnx.metadata_props.add()
                meta.key = key
                meta.value = value
            onnx.save(model_onnx, output_path, save_as_external_data=False, all_tensors_to_one_file=True)
            logger.info(f"Export complete: {output_path} (metadata embedded)")
        except Exception as e:
            logger.warning(f"Metadata embedding failed: {e}")

        import os

        data_file = f"{output_path}.data"
        if os.path.exists(data_file):
            try:
                os.remove(data_file)
            except OSError:
                pass

        return output_path

    def export_engine(
        self, imgsz=None, half=True, int8=False, dynamic=False, workspace=4, **kwargs
    ):
        """Export to TensorRT `.engine` (ONNX -> TensorRT via `trtexec` or Python TRT API)."""
        from pathlib import Path
        from neuro_pilot.engine.trt_builder import build_tensorrt_engine
        from neuro_pilot.nn.zoo.graph import GraphModel

        simplify = kwargs.pop("simplify", not kwargs.get("end2end", False))
        raw_file = str(kwargs.pop("file", "neuro_pilot_model.engine"))
        engine_path = Path(raw_file if raw_file.endswith((".engine", ".plan")) else f"{raw_file}.engine")
        onnx_path = Path(
            self.export_onnx(
                imgsz=imgsz,
                simplify=simplify,
                dynamic=dynamic,
                file=str(engine_path.with_suffix(".onnx")),
                **kwargs,
            )
        )

        input_specs = None
        metadata = None
        if isinstance(self.model, GraphModel):
            input_specs = self.model.input_specs
            metadata = {
                "model_key": self.model.model_key,
                "family": self.model.family,
                "input_specs": self.model.input_specs,
            }
        else:
            from neuro_pilot.cfg.schema import HeadConfig

            trt_sz = imgsz or 320
            h, w = trt_sz if isinstance(trt_sz, tuple) else (trt_sz, trt_sz)
            input_specs = {
                "image": {"shape": [1, 3, h, w]},
                "command": {"shape": [1, HeadConfig().num_commands]},
            }

        built = build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            half=half,
            int8=int8,
            dynamic=dynamic,
            workspace_gb=workspace,
            input_specs=input_specs,
            metadata=metadata,
            max_batch=int(kwargs.get("max_batch", 4)),
        )
        return str(built)
