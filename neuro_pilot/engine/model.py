from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union

from neuro_pilot.utils.logger import logger, log_system_info
from neuro_pilot.core.registry import Registry
import neuro_pilot.tasks  # registers all native and Zoo driving tasks
from neuro_pilot.engine.backend.factory import AutoBackend
from neuro_pilot.utils.torch_utils import select_device, default_names


class NeuroPilot(nn.Module):
    """
    Unified NeuroPilot Model Interface.
    Delegates architecture, training, and validation logic to a specific `Task` implementation.

    Attributes:
        task_wrapper (BaseTask): The task implementation.
        model (nn.Module): The underlying PyTorch model.
        overrides (dict): Configuration overrides.
    """

    _system_logged = False

    def __init__(
        self,
        model: Union[str, Path, nn.Module] = "neuralPilot.yaml",
        task: str | None = None,
        scale: str = "n",
        **kwargs,
    ):
        super().__init__()
        if not NeuroPilot._system_logged:
            log_system_info()
            NeuroPilot._system_logged = True

        self.overrides = kwargs
        self.overrides["scale"] = scale
        if task:
            self.overrides["task"] = task
        self.task_wrapper = None
        self.model = None
        self.predictor = None

        self.task_name = task or "multitask"
        self.target_device = select_device(kwargs.get("device", ""))

        if isinstance(model, nn.Module):
            self.model = model
            self._init_task(self.task_name, overrides={"model": model})
            return

        model = str(model)
        model_path = Path(model)
        if model_path.exists() and model_path.suffix in [".pt", ".pth"]:
            self._load(model_path, scale=scale)
        elif model_path.exists() and model_path.suffix in [
            ".onnx", ".engine", ".plan", ".torchscript", ".ts", ".tflite", ".pb",
        ]:
            self._load_compiled(model_path, task=task)
        else:
            auto_pt = self._maybe_download_zoo_weights(model, model_path)
            if auto_pt is not None:
                self._load(auto_pt, scale=scale)
            else:
                self._new(model, scale=scale)

    def _maybe_download_zoo_weights(self, model: str, model_path: Path) -> Path | None:
        """Auto-download pretrained Zoo `.pt` weights for `<key>.pt` or `pretrained=True`."""
        from neuro_pilot.nn.zoo.hub import download_weights, resolve_weight_key

        want_pt = model_path.suffix in {".pt", ".pth"} or bool(self.overrides.get("pretrained"))
        if not want_pt:
            return None
        key = resolve_weight_key(model)
        if key is None:
            return None
        return download_weights(key, dest_dir=self.overrides.get("cache_dir"))

    def _init_task(self, task_name, overrides=None):
        """Initialize the Task Wrapper."""
        try:
            TaskClass: type = Registry.get_task(task_name)
        except ValueError:
            logger.warning(f"Task '{task_name}' not found. Defaulting to 'multitask'.")
            TaskClass = Registry.get_task("multitask")

        from neuro_pilot.utils.monitor import SystemLogger

        self.system_logger = SystemLogger()
        logger.info(f"System Monitor Initialized. CPU: {self.system_logger.get_metrics()['cpu']}%")

        final_overrides = {**self.overrides, **(overrides or {})}

        from neuro_pilot.cfg.schema import load_config, deep_update, AppConfig

        self.cfg_obj = load_config()

        cfg_dict = self.cfg_obj.model_dump()
        app_keys = set(cfg_dict.keys())
        config_overrides = {k: v for k, v in final_overrides.items() if k in app_keys}
        if config_overrides:
            cfg_dict = deep_update(cfg_dict, config_overrides)
            self.cfg_obj = AppConfig(**cfg_dict)

        self.task_wrapper = TaskClass(self.cfg_obj, overrides=final_overrides)

        if not self.model and not (overrides and isinstance(overrides.get("model"), nn.Module)):
            if (
                self.task_name == "multitask"
                and "model_cfg" in final_overrides
                and Path(final_overrides["model_cfg"]).suffix in [".yaml", ".yml"]
            ):
                from neuro_pilot.nn.factory import build_model

                scale = final_overrides.get("scale", "l")
                skip_heatmap = final_overrides.get("skip_heatmap_inference", self.cfg_obj.head.skip_heatmap_inference)
                self.model = build_model(
                    cfg_path=final_overrides["model_cfg"],
                    ch=3,
                    nc=self.cfg_obj.head.num_classes,
                    scale=scale,
                    skip_heatmap_inference=skip_heatmap,
                )
            else:
                self.model = self.task_wrapper.build_model()

        is_pytorch = self.model and not isinstance(self.model, str)
        is_compiled = isinstance(self.model, str) and Path(self.model).suffix in [".onnx", ".engine", ".tflite", ".pb"]

        if is_pytorch:
            if not self.task_wrapper.model:
                self.task_wrapper.model = self.model
            self.model.to(self.target_device)

        if is_pytorch or is_compiled:
            self.backend = AutoBackend.create(self.model, device=self.target_device)

    def _load_compiled(self, model_path: Path, task: str | None = None) -> None:
        """Load compiled ONNX, TensorRT `.engine`, or TorchScript runtime artifact."""
        from neuro_pilot.cfg.schema import load_config
        from neuro_pilot.tasks import load_compiled_zoo_backend

        backend, task_name, wrapper = load_compiled_zoo_backend(
            model_path, self.target_device, self.overrides, task=task
        )
        self.backend = backend
        self.task_name = task_name
        if wrapper is not None:
            self.cfg_obj = load_config()
            wrapper.cfg = self.cfg_obj
            self.model, self.task_wrapper, self.predictor = backend, wrapper, None
            return
        self.model = str(model_path)
        self._init_task(self.task_name)

    def _bind_graph_model(self, graph_model) -> None:
        """Bind a Zoo GraphModel to its inferred driving task and device."""
        from neuro_pilot.cfg.schema import load_config
        from neuro_pilot.tasks import infer_driving_task

        self.model = graph_model.to(self.target_device)
        self.cfg_obj = load_config()
        self.predictor = None
        self.task_name = self.overrides.get("task") or infer_driving_task(
            graph_model.model_key, graph_model.family
        )
        self.task_wrapper = Registry.get_task(self.task_name)(
            self.cfg_obj, overrides=dict(self.overrides)
        )
        self.task_wrapper.model = self.model

    def _new(self, cfg_path: Union[str, Path], scale: str = "n"):
        """Initialize new model from config."""
        from neuro_pilot.engine.tasks import maybe_build_zoo_graph
        from neuro_pilot.utils.checks import check_yaml

        zoo_model = maybe_build_zoo_graph(cfg_path, scale=scale)
        if zoo_model is None:
            cfg_path = check_yaml(cfg_path)
            zoo_model = maybe_build_zoo_graph(cfg_path, scale=scale)
        if zoo_model is not None:
            self._bind_graph_model(zoo_model)
            return
        logger.info(f"Initializing NeuroPilot ({self.task_name}) from {cfg_path} (scale={scale})")
        self._init_task(self.task_name, overrides={"model_cfg": cfg_path, "scale": scale})

    def _load(self, weights_path: Union[str, Path], scale: str = "n"):
        """Load from checkpoint."""
        logger.info(f"Loading NeuroPilot ({self.task_name}) from {weights_path}")

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_key" in ckpt:
            from neuro_pilot.nn.zoo import load_zoo_pt

            self.overrides["_warm_started"] = True
            self._bind_graph_model(load_zoo_pt(weights_path, device=str(self.target_device)))
            return
        model_cfg = ckpt.get("model_cfg")
        if not model_cfg and "args" in ckpt:
            model_cfg = ckpt["args"].get("model_cfg")

        ckpt_scale = ckpt.get("scale")
        if not ckpt_scale and "args" in ckpt:
            ckpt_scale = ckpt["args"].get("scale")

        final_scale = scale if scale != "n" else (ckpt_scale or "n")

        if model_cfg:
            basename = Path(model_cfg).name
            local_path = Path(__file__).resolve().parent.parent / "cfg" / "models" / basename
            if local_path.exists():
                model_cfg = str(local_path)

        overrides = {"model_cfg": model_cfg, "scale": final_scale}
        if not model_cfg:
            del overrides["model_cfg"]

        final_overrides = {**overrides, **self.overrides}

        self._init_task(self.task_name, overrides=final_overrides)
        self.task_wrapper.load_weights(ckpt)
        self.model = self.task_wrapper.model

        if self.model:
            self.model.to(self.target_device)
            if "names" in ckpt:
                self.model.names = ckpt["names"]
                if self.task_wrapper:
                    self.task_wrapper.names = ckpt["names"]
            if "cfg" in ckpt:
                self.cfg_obj = ckpt["cfg"]

        self.backend = AutoBackend.create(self.model, device=self.target_device)

    def train(self, mode: bool = True, **kwargs):
        """
        Dual-purpose method:
        1. If kwargs are provided, runs the training loop (Ultralytics style).
        2. If no kwargs, sets the module to training/eval mode (nn.Module style).
        """
        if not kwargs:
            super().train(mode)
            if self.model:
                self.model.train(mode)
            return self

        if not self.task_wrapper:
            raise RuntimeError("Task not initialized.")

        from neuro_pilot.tasks.base import GraphDrivingTask

        if isinstance(self.task_wrapper, GraphDrivingTask):
            self.overrides.update(kwargs)
            self.task_wrapper.overrides.update(kwargs)
            trainer = self.task_wrapper.get_trainer()
            metrics = trainer.train()
            best = getattr(trainer, "best", None)
            if best is not None and Path(best).exists():
                self._load(best)
            return metrics

        from neuro_pilot.cfg.schema import deep_update, AppConfig

        mapped_kwargs = self._map_kwargs_to_config(kwargs)

        self.overrides = deep_update(self.overrides, mapped_kwargs)
        self.task_wrapper.overrides = deep_update(self.task_wrapper.overrides, mapped_kwargs)

        cfg_dict = self.cfg_obj.model_dump()
        cfg_dict = deep_update(cfg_dict, mapped_kwargs)
        self.cfg_obj = AppConfig(**cfg_dict)
        self.task_wrapper.cfg = self.cfg_obj

        self._handle_resume()

        trainer = self.task_wrapper.get_trainer()
        metrics = trainer.train()

        if trainer.best.exists():
            self._load(trainer.best)

        return metrics

    @staticmethod
    def _build_config_map() -> dict[str, str]:
        """Build a flat mapping: {kwarg_name: section_name} from AppConfig schema."""
        from neuro_pilot.cfg.schema import AppConfig

        config_map: dict[str, str] = {}
        for section_name, field_info in AppConfig.model_fields.items():
            if section_name == "model_config_path":
                continue
            section_cls = field_info.annotation
            if hasattr(section_cls, "model_fields"):
                for key in section_cls.model_fields.keys():
                    config_map[key] = section_name
                    if section_name == "data" and key == "augment":
                        augment_cls = section_cls.model_fields["augment"].annotation
                        if hasattr(augment_cls, "model_fields"):
                            for aug_key in augment_cls.model_fields.keys():
                                config_map[aug_key] = "data.augment"
        return config_map

    def _map_kwargs_to_config(self, kwargs: dict) -> dict:
        """Map user-facing kwargs (epochs, batch, data, ...) into nested config dict."""
        # Shorthand aliases
        _ALIASES = {
            "augment": lambda v: (("data", "augment", "enabled"), v) if isinstance(v, bool) else None,
            "data": lambda v: (("data", "dataset_yaml"), v) if isinstance(v, str) else None,
            "patience": lambda v: (("trainer", "early_stop_patience"), v),
            "epochs": lambda v: (("trainer", "max_epochs"), v),
            "batch": lambda v: (("data", "batch_size"), v),
        }

        config_map = self._build_config_map()
        mapped: dict = {}

        for k, v in kwargs.items():
            # 1. Check aliases
            if k in _ALIASES:
                result = _ALIASES[k](v)
                if result:
                    path, val = result
                    target = mapped
                    for part in path[:-1]:
                        target = target.setdefault(part, {})
                    target[path[-1]] = val
                    continue

            # 2. Map via config schema
            if k in config_map:
                section = config_map[k]
                if "." in section:
                    parts = section.split(".")
                    target = mapped
                    for part in parts:
                        target = target.setdefault(part, {})
                    target[k] = v
                else:
                    target_dict = mapped.setdefault(section, {})
                    if isinstance(target_dict, dict):
                        target_dict[k] = v
                    else:
                        logger.warning(f"Conflict mapping '{k}' to section '{section}'.")
            else:
                mapped[k] = v

        return mapped

    def _handle_resume(self):
        """Resolve resume checkpoint path."""
        from neuro_pilot.cfg.schema import deep_update

        if self.cfg_obj.trainer.resume is True:
            experiment_name = self.cfg_obj.trainer.experiment_name
            last_ckpt = Path("experiments") / experiment_name / "weights" / "last.pt"
            if last_ckpt.exists():
                new_resume = str(last_ckpt)
                logger.info(f"Resuming from inferred checkpoint: {new_resume}")
                self.cfg_obj.trainer.resume = new_resume
                self.task_wrapper.overrides = deep_update(
                    self.task_wrapper.overrides, {"trainer": {"resume": new_resume}}
                )
            else:
                logger.warning(f"Resume requested but {last_ckpt} not found. Starting from scratch.")
                self.cfg_obj.trainer.resume = False
                self.task_wrapper.overrides = deep_update(self.task_wrapper.overrides, {"trainer": {"resume": False}})
        elif self.cfg_obj.trainer.resume:
            logger.info(f"Resuming from specified checkpoint: {self.cfg_obj.trainer.resume}")

    def predict(self, source, **kwargs):
        """Run inference; returns Results (multitask) or decoded dict (Zoo GraphModel)."""
        from neuro_pilot.engine.backend.base import BaseBackend
        from neuro_pilot.nn.zoo.graph import GraphModel

        is_zoo_runtime = isinstance(self.model, GraphModel) or (
            isinstance(self.model, BaseBackend) and bool(self.model.model_key)
        )
        if is_zoo_runtime:
            if self.predictor is None:
                from neuro_pilot.engine.predictors import create_zoo_predictor

                self.predictor = create_zoo_predictor(self.model, device=self.target_device)
            return self.predictor(source, **kwargs)

        if self.predictor is None:
            from neuro_pilot.engine.predictor import Predictor

            self.predictor = Predictor(self.cfg_obj, self.backend, self.target_device)

        return self.predictor(source, **kwargs)

    def predict_video(self, source, out_mp4=None, right_source=None, **kwargs):
        """Stream single/dual driving videos and optionally render zero-text 3D trajectory MP4."""
        from neuro_pilot.engine.video import run_predict_video

        return run_predict_video(self, source, out_mp4=out_mp4, right_source=right_source, **kwargs)

    def export(self, **kwargs):
        """Export the model to ONNX, TensorRT, or TorchScript."""
        from neuro_pilot.engine.exporter import Exporter

        exporter = Exporter(getattr(self, "cfg_obj", None), self.model, self.device)
        return exporter(**kwargs)

    def val(self, **kwargs):
        """Validate locally using the task's validator."""
        from neuro_pilot.tasks.base import GraphDrivingTask

        validator = self.task_wrapper.get_validator()
        if isinstance(self.task_wrapper, GraphDrivingTask):
            return validator(**kwargs)
        dataloader = kwargs.get("dataloader")
        if dataloader is None:
            from neuro_pilot.data import prepare_dataloaders

            _, dataloader = prepare_dataloaders(self.cfg_obj)
        return validator(dataloader)

    def visualize(self, source, output=None, **kwargs):
        """Render task-specific visual overlay for an image path or BGR frame."""
        from neuro_pilot.engine.predictors.utils import load_bgr
        from neuro_pilot.tasks.base import GraphDrivingTask

        bgr = load_bgr(source)
        if output is None:
            output = (
                self.predict(bgr, **kwargs)
                if isinstance(self.task_wrapper, GraphDrivingTask)
                else self.predict(bgr)
            )
        return self.task_wrapper.visualize(bgr, output, **kwargs)

    def benchmark(self, imgsz=640, half=True, batch=1, device=None):
        """Benchmark model performance."""
        import time

        device = device or self.device
        model = self.model.to(device)
        if half and device.type != "cpu":
            model.half()
        img = torch.zeros(batch, 3, imgsz, imgsz, device=device)
        img = img.half() if (half and device.type != "cpu") else img
        cmd = torch.zeros(batch, dtype=torch.long, device=device)
        for _ in range(10):
            model(img, cmd=cmd)
        n = 100
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(n):
            model(img, cmd=cmd)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - t1) / n * 1000
        fps = 1000 / dt * batch
        logger.info(f"Benchmark: {imgsz}x{imgsz}, batch={batch} | {dt:.2f} ms | {fps:.2f} FPS")
        return {"latency_ms": dt, "fps": fps}

    def fuse(self):
        """Fuse layers."""
        if hasattr(self.model, "fuse"):
            self.model.fuse()
        return self

    def info(self, verbose=True):
        return self.model.info(verbose=verbose) if hasattr(self.model, "info") else None

    def save(self, filename: Union[str, Path]):
        """Save model to file."""
        from neuro_pilot.engine.tasks import save_graph_checkpoint
        from neuro_pilot.nn.zoo.graph import GraphModel

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, GraphModel):
            save_graph_checkpoint(self.model, filename)
            return
        payload = {
            "state_dict": self.model.state_dict(),
            "task": self.task_name,
            "overrides": self.overrides,
            "scale": self.overrides.get("scale", "n"),
            "cfg": self.cfg_obj,
        }
        torch.save(payload, filename)

    def attach_adapter(
        self,
        feat_dim: int = 256,
        num_commands: int = 4,
        num_waypoints: int = 10,
        freeze_base: bool = True,
    ):
        """Attach a trainable CommandAdapter head directly onto this loaded model."""
        from neuro_pilot.engine.zoo import ModelZoo

        if self.model is None:
            raise RuntimeError("Model not loaded or initialized.")
        return ModelZoo.attach_adapter(
            self.model,
            feat_dim=feat_dim,
            num_commands=num_commands,
            num_waypoints=num_waypoints,
            freeze_base=freeze_base,
        )

    def to(self, *args, **kwargs):
        """Override to move backend as well."""
        device = select_device(args[0]) if args else select_device(kwargs.get("device", ""))
        self.target_device = device
        if hasattr(self, "backend"):
            self.backend.device = device
            if hasattr(self.backend, "model"):
                self.backend.model.to(device)
        return super().to(device)

    def half(self):
        """Override to set backend to FP16."""
        if self.model and hasattr(self.model, "half"):
            self.model.half()
        if hasattr(self, "backend"):
            self.backend.fp16 = True
            if hasattr(self.backend, "model"):
                self.backend.model.half()
        return self

    @property
    def device(self):
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device("cpu")

    @property
    def names(self):
        """Returns correctly prioritized class names."""
        for source in [self.task_wrapper, self.model]:
            if hasattr(source, "names") and source.names:
                n = source.names
                first_val = next(iter(n.values())) if isinstance(n, dict) else n[0]
                if not str(first_val).startswith("class_"):
                    return n

        if hasattr(self.model, "names"):
            return self.model.names

        num_classes = getattr(self.cfg_obj.head, "num_classes", 14)
        return default_names(num_classes)

    def __call__(self, source, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded or initialized. Please load weights or initialize a config.")
        if self.training:
            return self.model(source, **kwargs)
        return self.predict(source, **kwargs)

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            pass

        if attr in {"model", "task_wrapper", "overrides"}:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

        return getattr(self.model, attr)
