"""Modular YAML-graph E2E & World Model Zoo for NeuroPilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from neuro_pilot.nn.zoo.base import (
    BLOCK_REGISTRY,
    CONVERTER_REGISTRY,
    ZOO_REGISTRY,
    BaseOnnxConverter,
    BaseZooModel,
    OutputParityMetric,
    ParityReport,
)
from neuro_pilot.nn.zoo.blocks import register_all_blocks
from neuro_pilot.nn.zoo.converters import register_all_converters
from neuro_pilot.nn.zoo.graph import GraphModel, is_zoo_yaml
from neuro_pilot.nn.zoo.hub import (
    WEIGHT_SPECS,
    WeightSpec,
    download_weights,
    get_weights_dir,
    list_downloadable,
    resolve_weight_key,
)
from neuro_pilot.nn.zoo.losses import (
    FlowMatchingLoss,
    LaplaceNLL,
    MHPLaplaceLoss,
    WaypointLoss,
)
from neuro_pilot.nn.zoo.parity import (
    convert_onnx_to_pt,
    load_onnx_initializers,
    verify_onnx_to_pt_parity,
)

ZOO_YAML_DIR = Path(__file__).resolve().parents[2] / "cfg" / "models" / "zoo"

# Canonical short model key -> layer-by-layer YAML config filename (plus legacy aliases)
MODEL_YAML_MAP: Dict[str, str] = {
    "pilot_s": "pilot_s.yaml",
    "pilot_l": "pilot_l.yaml",
    "pilot_nav": "pilot_nav.yaml",
    "driver": "driver.yaml",
    "cipo": "cipo.yaml",
    "lane": "lane.yaml",
    "drive": "drive.yaml",
    "depth": "depth.yaml",
    "seg": "seg.yaml",
    "seg_l": "seg_l.yaml",
    "jepa": "jepa.yaml",
    "vq_enc": "vq_enc.yaml",
    "vq_dec": "vq_dec.yaml",
    "ae": "ae.yaml",
    "dit": "dit.yaml",
    "gpt": "gpt.yaml",
    # Backward-compatible aliases for existing checkpoints
    "openpilot_small": "pilot_s.yaml",
    "openpilot_chestnut_big": "pilot_l.yaml",
    "openpilot_big": "pilot_l.yaml",
    "openpilot_n16_nav": "pilot_nav.yaml",
    "openpilot_nav": "pilot_nav.yaml",
    "openpilot_dmonitoring": "driver.yaml",
    "dmonitoring": "driver.yaml",
    "visionpilot_autospeed2": "cipo.yaml",
    "autospeed": "cipo.yaml",
    "visionpilot_autosteer2": "lane.yaml",
    "autosteer": "lane.yaml",
    "visionpilot_autodrive": "drive.yaml",
    "autodrive": "drive.yaml",
    "visionpilot_scene3d_lite": "depth.yaml",
    "scene3d": "depth.yaml",
    "eomt_large": "seg_l.yaml",
    "latent_drive_vjepa": "jepa.yaml",
    "latent_drive": "jepa.yaml",
    "comma_vq_encoder": "vq_enc.yaml",
    "commavq_enc": "vq_enc.yaml",
    "comma_vq_decoder": "vq_dec.yaml",
    "commavq_dec": "vq_dec.yaml",
    "comma_vit_ae_f8c32": "ae.yaml",
    "comma_ae": "ae.yaml",
    "comma_worldmodel_dit": "dit.yaml",
    "comma_dit": "dit.yaml",
    "comma_gpt2m_world_model": "gpt.yaml",
    "comma_gpt": "gpt.yaml",
}


def register_all_zoo_models() -> None:
    """Register all atomic blocks, YAML model factories, and ONNX converters."""
    register_all_blocks()
    register_all_converters()
    for key, yaml_name in MODEL_YAML_MAP.items():
        if key not in ZOO_REGISTRY:
            yaml_file = ZOO_YAML_DIR / yaml_name

            def _factory(
                scale: Optional[str] = None,
                _path: Path = yaml_file,
                **kwargs: Any,
            ) -> GraphModel:
                return GraphModel(_path, scale=scale, **kwargs)

            ZOO_REGISTRY.register(key)(_factory)


def list_zoo_yamls() -> List[Path]:
    """Return all layer-by-layer YAML config files in the Zoo directory."""
    return sorted(ZOO_YAML_DIR.glob("*.yaml"))


def load_zoo_pt(
    pt_path: str | Path,
    device: str = "cpu",
    strict: bool = True,
) -> GraphModel:
    """Load a GraphModel directly from a `.pt` checkpoint without touching any `.onnx` file."""
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_key" not in ckpt:
        raise ValueError(
            f"Checkpoint '{pt_path}' lacks 'model_key'; pass the YAML path to build_zoo_model()."
        )
    cfg = ckpt.get("config") or {}
    model = build_zoo_model(str(ckpt["model_key"]), **cfg)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)
    return model.to(device).eval()


def build_zoo_model(
    cfg_or_key: str | Path,
    scale: Optional[str] = None,
    pt_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> GraphModel:
    """Build a GraphModel from a YAML, model key, or `.pt` checkpoint."""
    register_all_zoo_models()
    candidate = Path(str(cfg_or_key))
    if candidate.suffix.lower() == ".pt" and candidate.is_file():
        return load_zoo_pt(candidate)
    if candidate.suffix.lower() in {".yaml", ".yml"} and candidate.is_file():
        model = GraphModel(candidate, scale=scale, **kwargs)
    elif str(cfg_or_key) in MODEL_YAML_MAP:
        model = GraphModel(ZOO_YAML_DIR / MODEL_YAML_MAP[str(cfg_or_key)], scale=scale, **kwargs)
    else:
        raise KeyError(f"Unknown model or YAML path: {cfg_or_key}")
    if pt_path:
        model.load_pt_checkpoint(pt_path, strict=True)
    return model


__all__ = [
    "BLOCK_REGISTRY",
    "CONVERTER_REGISTRY",
    "ZOO_REGISTRY",
    "MODEL_YAML_MAP",
    "WEIGHT_SPECS",
    "WeightSpec",
    "ZOO_YAML_DIR",
    "BaseZooModel",
    "BaseOnnxConverter",
    "GraphModel",
    "OutputParityMetric",
    "ParityReport",
    "LaplaceNLL",
    "MHPLaplaceLoss",
    "FlowMatchingLoss",
    "WaypointLoss",
    "build_zoo_model",
    "convert_onnx_to_pt",
    "download_weights",
    "get_weights_dir",
    "is_zoo_yaml",
    "list_downloadable",
    "list_zoo_yamls",
    "load_onnx_initializers",
    "load_zoo_pt",
    "register_all_blocks",
    "register_all_converters",
    "register_all_zoo_models",
    "resolve_weight_key",
    "verify_onnx_to_pt_parity",
]
