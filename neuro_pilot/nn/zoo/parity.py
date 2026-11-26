"""ONNX-to-PyTorch (.pt) weight conversion and numerical parity verification harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch

from neuro_pilot.nn.zoo.base import (
    CONVERTER_REGISTRY,
    BaseOnnxConverter,
    BaseZooModel,
    OutputParityMetric,
    ParityReport,
)


def load_onnx_initializers(onnx_path: str | Path) -> Dict[str, np.ndarray]:
    """Load all ONNX initializers (and Constant node tensors) as float32/native numpy arrays."""
    model = onnx.load(str(onnx_path))
    tensors: Dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        tensors[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    tensors[node.output[0]] = numpy_helper.to_array(attr.t)
    return tensors


def convert_onnx_to_pt(
    model_key: str,
    onnx_path: str | Path,
    pt_output_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> Tuple[BaseZooModel, List[str], List[str], int]:
    """Convert an ONNX checkpoint into a verified PyTorch `.pt` checkpoint."""
    converter: BaseOnnxConverter = CONVERTER_REGISTRY.build(model_key)
    model = converter.build_model(config=config)
    onnx_tensors = load_onnx_initializers(onnx_path)
    state_dict = converter.extract_state_dict(onnx_tensors, model)

    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    out_path = Path(pt_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_key": model_key,
            "family": model.family,
            "source_onnx": str(Path(onnx_path).resolve()),
            "config": config or {},
            "state_dict": model.state_dict(),
        },
        str(out_path),
    )
    return model, missing, unexpected, len(state_dict)


def _compute_output_metric(
    name: str,
    pt_arr: np.ndarray,
    ort_arr: np.ndarray,
    max_abs_tol: float,
    min_cos_sim: float,
) -> OutputParityMetric:
    a = pt_arr.astype(np.float64).reshape(-1)
    b = ort_arr.astype(np.float64).reshape(-1)
    diff = np.abs(a - b)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    norm_b = float(np.linalg.norm(b))
    rel_l2 = float(np.linalg.norm(a - b) / max(norm_b, 1e-12))
    denom = float(np.linalg.norm(a) * norm_b)
    cos_sim = 1.0 if denom < 1e-12 and max_abs < 1e-6 else float(np.dot(a, b) / max(denom, 1e-12))
    passed = (max_abs <= max_abs_tol) and (cos_sim >= min_cos_sim)
    return OutputParityMetric(
        name=name,
        shape=tuple(int(x) for x in ort_arr.shape),
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        rel_l2_err=rel_l2,
        cosine_sim=cos_sim,
        passed=passed,
    )


def verify_onnx_to_pt_parity(
    model_key: str,
    onnx_path: str | Path,
    pt_output_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    max_abs_tol: float = 1e-2,
    min_cos_sim: float = 0.9999,
    seed: int = 42,
    device: str = "cpu",
    custom_inputs: Optional[Dict[str, torch.Tensor]] = None,
    providers: Tuple[str, ...] = ("CPUExecutionProvider",),
) -> ParityReport:
    """Convert ONNX -> .pt, reload a fresh model from .pt, and compare against ONNX Runtime."""
    _, missing, unexpected, mapped = convert_onnx_to_pt(
        model_key=model_key,
        onnx_path=onnx_path,
        pt_output_path=pt_output_path,
        config=config,
        strict=True,
    )

    # Instantiate a fresh model and reload strictly from disk to prove serialization fidelity
    converter: BaseOnnxConverter = CONVERTER_REGISTRY.build(model_key)
    pt_model = converter.build_model(config=config)
    pt_model.load_pt_checkpoint(pt_output_path, strict=True)
    pt_model.eval().to(device)

    inputs = custom_inputs if custom_inputs is not None else pt_model.make_sample_inputs(seed=seed)
    pt_inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pt_outputs = pt_model.forward_onnx_dict(pt_inputs)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=list(providers))
    ort_feed = {k: v.detach().cpu().numpy() for k, v in inputs.items()}
    ort_names = [o.name for o in sess.get_outputs()]
    ort_vals = sess.run(ort_names, ort_feed)
    ort_outputs = dict(zip(ort_names, ort_vals))

    metrics: List[OutputParityMetric] = []
    for name in ort_names:
        if name not in pt_outputs:
            raise KeyError(f"PyTorch model '{model_key}' missing ONNX output '{name}'")
        pt_arr = pt_outputs[name].detach().cpu().numpy()
        metric = _compute_output_metric(name, pt_arr, ort_outputs[name], max_abs_tol, min_cos_sim)
        metrics.append(metric)

    all_ok = (not missing) and (not unexpected) and all(m.passed for m in metrics)
    return ParityReport(
        model_key=model_key,
        onnx_path=str(onnx_path),
        pt_path=str(pt_output_path),
        num_params=pt_model.num_parameters(),
        mapped_tensors=mapped,
        missing_keys=missing,
        unexpected_keys=unexpected,
        outputs=metrics,
        all_passed=all_ok,
    )
