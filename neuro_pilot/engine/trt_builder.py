"""TensorRT 10 engine builder supporting `trtexec` CLI (FP16/BF16/INT8) and Python API."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neuro_pilot.engine.backend.base import save_sidecar_metadata
from neuro_pilot.utils.logger import logger

_JETSON_TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")


def resolve_trtexec(explicit: Optional[str] = None) -> Optional[str]:
    """Locate `trtexec` via explicit path, system PATH, or standard JetPack install."""
    if explicit:
        found = shutil.which(explicit)
        if found:
            return found
        if Path(explicit).is_file():
            return str(explicit)
    found = shutil.which("trtexec")
    if found:
        return found
    if _JETSON_TRTEXEC.is_file():
        return str(_JETSON_TRTEXEC)
    return None


def _format_shape_spec(input_specs: Dict[str, Any], batch_size: int) -> str:
    """Build TensorRT `--minShapes/--optShapes/--maxShapes` string from input_specs."""
    parts: List[str] = []
    for name, spec in input_specs.items():
        raw = list(spec["shape"] if isinstance(spec, dict) else spec)
        if raw:
            raw[0] = batch_size
        parts.append(f"{name}:{'x'.join(str(int(d)) for d in raw)}")
    return ",".join(parts)


def build_trtexec_cmd(
    onnx_path: Path,
    engine_path: Path,
    half: bool = True,
    bf16: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    workspace_gb: int = 4,
    workspace_mb: Optional[int] = None,
    input_specs: Optional[Dict[str, Any]] = None,
    max_batch: int = 4,
    trtexec_bin: str = "trtexec",
) -> List[str]:
    """Construct a TensorRT 10 `trtexec` command for any single or multi-input ONNX."""
    mem_mb = int(workspace_mb) if workspace_mb is not None else int(workspace_gb) * 1024
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{mem_mb}M",
    ]
    if bf16:
        cmd.append("--bf16")
    elif half:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")
    if dynamic and input_specs:
        cmd.append(f"--minShapes={_format_shape_spec(input_specs, 1)}")
        cmd.append(f"--optShapes={_format_shape_spec(input_specs, 1)}")
        cmd.append(f"--maxShapes={_format_shape_spec(input_specs, max_batch)}")
    return cmd


def _build_with_python_api(
    onnx_path: Path,
    engine_path: Path,
    half: bool,
    bf16: bool,
    int8: bool,
    dynamic: bool,
    workspace_mb: int,
    input_specs: Optional[Dict[str, Any]],
    max_batch: int,
) -> Path:
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * (1 << 20))
    if bf16 and hasattr(trt.BuilderFlag, "BF16"):
        config.set_flag(trt.BuilderFlag.BF16)
    elif half and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    if dynamic and input_specs:
        profile = builder.create_optimization_profile()
        for name, spec in input_specs.items():
            shape = list(spec["shape"] if isinstance(spec, dict) else spec)
            min_s: Tuple[int, ...] = tuple([1] + shape[1:])
            opt_s: Tuple[int, ...] = tuple([1] + shape[1:])
            max_s: Tuple[int, ...] = tuple([max_batch] + shape[1:])
            profile.set_shape(name, min_s, opt_s, max_s)
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT builder failed for {onnx_path}")
    engine_path.write_bytes(bytes(serialized))
    return engine_path


def _precision_label(half: bool, bf16: bool, int8: bool) -> str:
    if int8:
        return "int8"
    if bf16:
        return "bf16"
    return "fp16" if half else "fp32"


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    half: bool = True,
    bf16: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    workspace_gb: int = 4,
    workspace_mb: Optional[int] = None,
    input_specs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_batch: int = 4,
    trtexec_bin: Optional[str] = None,
) -> Path:
    """Compile an ONNX file to a TensorRT `.engine` and write its `.meta.json` sidecar."""
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    mem_mb = int(workspace_mb) if workspace_mb is not None else int(workspace_gb) * 1024
    trtexec_path = resolve_trtexec(trtexec_bin)
    if trtexec_path:
        cmd = build_trtexec_cmd(
            onnx_path=onnx_path,
            engine_path=engine_path,
            half=half,
            bf16=bf16,
            int8=int8,
            dynamic=dynamic,
            workspace_mb=mem_mb,
            input_specs=input_specs,
            max_batch=max_batch,
            trtexec_bin=trtexec_path,
        )
        logger.info(f"Running TensorRT build: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
    else:
        try:
            _build_with_python_api(
                onnx_path=onnx_path,
                engine_path=engine_path,
                half=half,
                bf16=bf16,
                int8=int8,
                dynamic=dynamic,
                workspace_mb=mem_mb,
                input_specs=input_specs,
                max_batch=max_batch,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT export requires either `trtexec` in PATH or the `tensorrt` Python "
                "package installed. ONNX export succeeded and is ready at: "
                f"{onnx_path}"
            ) from exc

    if metadata:
        save_sidecar_metadata(
            engine_path,
            {**metadata, "precision": _precision_label(half, bf16, int8)},
        )
    logger.info(f"TensorRT engine built: {engine_path}")
    return engine_path
