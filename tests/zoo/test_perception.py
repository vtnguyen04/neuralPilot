"""ONNX->PT FP32 parity, YAML module-list build, and backward tests for VisionPilot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch

from neuro_pilot.nn.zoo import (
    CONVERTER_REGISTRY,
    ZOO_REGISTRY,
    build_zoo_model,
    register_all_zoo_models,
    verify_onnx_to_pt_parity,
)

ROOT = Path(__file__).resolve().parents[2]
YAML_DIR = ROOT / "neuro_pilot" / "cfg" / "models" / "zoo"
DEFAULT_VINFAST = ROOT.parent / "Vinfast" / "onnx_reverse_engineering"
VINFAST_DIR = Path(os.environ.get("NEUROPILOT_VINFAST_DIR", str(DEFAULT_VINFAST)))
ONNX_DIR = VINFAST_DIR / "models" / "visionpilot" / "models"
PT_DIR = VINFAST_DIR / "converted_pt" / "perception"

# (model_key, onnx_file, yaml_file, expected_params, max_abs, max_rel_l2)
SPECS: Tuple[Tuple[str, str, str, int, float, float], ...] = (
    ("lane", "autosteer_2_fp32.onnx", "lane.yaml", 8_215_956, 1e-5, 5e-6),
    ("drive", "autodrive_fp32.onnx", "drive.yaml", 10_576_981, 1e-5, 5e-6),
    ("cipo", "autospeed_2_fp32.onnx", "cipo.yaml", 9_300_140, 1e-3, 1e-6),
    ("depth", "scene3d_lite_fp32.onnx", "depth.yaml", 6_971_153, 2e-4, 2e-6),
)


def test_visionpilot_registry() -> None:
    register_all_zoo_models()
    for key, _, yaml_name, *_ in SPECS:
        assert key in ZOO_REGISTRY and key in CONVERTER_REGISTRY
        assert (YAML_DIR / yaml_name).is_file()


@pytest.mark.parametrize("key,onnx_name,yaml_name,expected_params,max_abs,max_rel_l2", SPECS)
def test_visionpilot_parity(
    key: str,
    onnx_name: str,
    yaml_name: str,
    expected_params: int,
    max_abs: float,
    max_rel_l2: float,
) -> None:
    onnx_path = ONNX_DIR / onnx_name
    if not onnx_path.is_file():
        pytest.skip(f"ONNX file not present: {onnx_path}")
    pt_path = PT_DIR / f"{key}.pt"

    report = verify_onnx_to_pt_parity(
        model_key=key,
        onnx_path=onnx_path,
        pt_output_path=pt_path,
        max_abs_tol=max_abs,
        min_cos_sim=0.999999,
        seed=42,
        device="cpu",
    )
    assert report.num_params == expected_params and report.all_passed
    assert report.missing_keys == [] and report.unexpected_keys == []
    for m in report.outputs:
        assert m.max_abs_diff <= max_abs and m.mean_abs_diff < 1e-5
        assert m.rel_l2_err <= max_rel_l2 and m.cosine_sim >= 0.999999

    # Build from pure module-list YAML and reload converted .pt strict=True
    yaml_model = build_zoo_model(YAML_DIR / yaml_name, pt_path=pt_path).eval()
    with torch.no_grad():
        outs = yaml_model.forward_onnx_dict(yaml_model.make_sample_inputs(seed=42))
    assert set(outs.keys()) == {m.name for m in report.outputs}
    for m in report.outputs:
        assert tuple(outs[m.name].shape) == m.shape and torch.isfinite(outs[m.name]).all()


@pytest.mark.parametrize("key,_onnx,yaml_name,_params,_tol,_rel", SPECS)
def test_visionpilot_backward(
    key: str, _onnx: str, yaml_name: str, _params: int, _tol: float, _rel: float
) -> None:
    model = build_zoo_model(YAML_DIR / yaml_name)
    model.train()
    sample = model.make_sample_inputs(seed=7)
    inputs: Dict[str, torch.Tensor] = {
        k: v.clone().detach().requires_grad_(True) for k, v in sample.items()
    }
    loss = sum(out.float().pow(2).mean() for out in model.forward_onnx_dict(inputs).values())
    loss.backward()
    for inp in inputs.values():
        assert inp.grad is not None and torch.isfinite(inp.grad).all()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
