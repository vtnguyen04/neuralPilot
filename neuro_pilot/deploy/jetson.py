"""Jetson edge deployment helper: ONNX export -> SCP upload -> native TensorRT engine build."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from neuro_pilot.engine.trt_builder import build_trtexec_cmd
from neuro_pilot.utils.logger import logger

_VALID_PRECISIONS = {"bf16", "fp16", "fp32", "int8"}


@dataclass(frozen=True)
class JetsonTarget:
    """SSH connection and TensorRT build target on a Jetson device."""

    host: str
    remote_dir: str
    port: int = 22
    trtexec: str = "/usr/src/tensorrt/bin/trtexec"
    precision: str = "bf16"
    workspace_mb: int = 1024
    ssh_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("JetsonTarget.host must not be empty.")
        if not self.remote_dir.strip():
            raise ValueError("JetsonTarget.remote_dir must not be empty.")
        if self.precision.lower() not in _VALID_PRECISIONS:
            raise ValueError(
                f"Invalid precision '{self.precision}'. Choose from {sorted(_VALID_PRECISIONS)}."
            )


def _quote_remote(path_str: str) -> str:
    """Shell-quote a remote path while keeping leading `~/` shell-expandable."""
    if path_str == "~":
        return "~"
    if path_str.startswith("~/"):
        return "~/" + shlex.quote(path_str[2:])
    return shlex.quote(path_str)


def build_ssh_cmd(target: JetsonTarget, remote_cmd: str) -> List[str]:
    """Build an `ssh` command list executing `remote_cmd` on `target`."""
    cmd = ["ssh", "-p", str(int(target.port))]
    if target.ssh_key:
        cmd.extend(["-i", str(target.ssh_key)])
    cmd.extend([target.host, remote_cmd])
    return cmd


def build_scp_cmd(local_files: Sequence[Path], target: JetsonTarget) -> List[str]:
    """Build an `scp` command list uploading `local_files` to `target.remote_dir`."""
    cmd = ["scp", "-P", str(int(target.port))]
    if target.ssh_key:
        cmd.extend(["-i", str(target.ssh_key)])
    cmd.extend(str(p) for p in local_files)
    remote = target.remote_dir.rstrip("/") + "/"
    cmd.append(f"{target.host}:{remote}")
    return cmd


def build_remote_trtexec_cmd(
    remote_onnx: str,
    remote_engine: str,
    target: JetsonTarget,
    input_specs: Optional[Dict[str, Any]] = None,
    dynamic: bool = False,
    max_batch: int = 1,
) -> str:
    """Build the shell command run on Jetson to compile ONNX into a native TensorRT `.engine`."""
    prec = target.precision.lower()
    parts = build_trtexec_cmd(
        onnx_path=Path(remote_onnx),
        engine_path=Path(remote_engine),
        half=(prec == "fp16"),
        bf16=(prec == "bf16"),
        int8=(prec == "int8"),
        dynamic=dynamic,
        workspace_mb=target.workspace_mb,
        input_specs=input_specs,
        max_batch=max_batch,
        trtexec_bin=target.trtexec,
    )
    quoted = [parts[0]]
    for tok in parts[1:]:
        if tok.startswith("--onnx="):
            quoted.append(f"--onnx={_quote_remote(remote_onnx)}")
        elif tok.startswith("--saveEngine="):
            quoted.append(f"--saveEngine={_quote_remote(remote_engine)}")
        else:
            quoted.append(shlex.quote(tok))
    return " ".join(quoted)


def _ensure_local_onnx(
    source: Union[str, Path, Any],
    work_dir: Path,
    opset: int,
    scale: str,
    name: Optional[str] = None,
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """Export ONNX from a NeuroPilot instance, `.pt`, or model key; pass through `.onnx`."""
    if isinstance(source, (str, Path)):
        src_path = Path(source)
        if src_path.suffix.lower() == ".onnx":
            if not src_path.is_file():
                raise FileNotFoundError(f"ONNX file not found: {src_path}")
            return src_path, None
        from neuro_pilot import NeuroPilot

        model = NeuroPilot(str(source), scale=scale, device="cpu")
        stem = name or src_path.stem or "model"
    else:
        model = source
        inner = getattr(model, "model", None)
        key = getattr(inner, "model_key", "") or "model"
        m_scale = getattr(inner, "scale", None)
        default_stem = f"{key}_{m_scale}" if (m_scale and m_scale not in {"n", "v1"}) else key
        stem = name or default_stem

    work_dir.mkdir(parents=True, exist_ok=True)
    onnx_out = Path(model.export(format="onnx", file=str(work_dir / f"{stem}.onnx"), opset=opset))
    specs = getattr(getattr(model, "model", None), "input_specs", None)
    return onnx_out, specs


def deploy_to_jetson(
    source: Union[str, Path, Any],
    target: JetsonTarget,
    opset: int = 17,
    scale: str = "n",
    work_dir: Union[str, Path] = "runs/deploy",
    name: Optional[str] = None,
    dynamic: bool = False,
    max_batch: int = 1,
    skip_build: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Export ONNX (if needed), SCP to Jetson, and compile native TensorRT `.engine` over SSH."""
    onnx_path, input_specs = _ensure_local_onnx(
        source, Path(work_dir), opset=opset, scale=scale, name=name
    )
    upload_files: List[Path] = [onnx_path]
    meta_sidecar = onnx_path.with_suffix(".meta.json")
    if meta_sidecar.is_file():
        upload_files.append(meta_sidecar)

    remote_base = target.remote_dir.rstrip("/")
    remote_onnx = f"{remote_base}/{onnx_path.name}"
    remote_engine = f"{remote_base}/{onnx_path.stem}.engine"

    mkdir_cmd = build_ssh_cmd(target, f"mkdir -p {_quote_remote(remote_base)}")
    scp_cmd = build_scp_cmd(upload_files, target)
    trt_shell = build_remote_trtexec_cmd(
        remote_onnx=remote_onnx,
        remote_engine=remote_engine,
        target=target,
        input_specs=input_specs,
        dynamic=dynamic,
        max_batch=max_batch,
    )
    trt_cmd = build_ssh_cmd(target, trt_shell)
    plan: List[List[str]] = [mkdir_cmd, scp_cmd] + ([] if skip_build else [trt_cmd])

    if dry_run:
        logger.info(f"Dry-run deploy plan ({len(plan)} steps) -> {target.host}:{remote_engine}")
    else:
        for step in plan:
            logger.info(f"Running: {' '.join(step)}")
            subprocess.run(step, check=True)
        logger.info(f"Jetson deploy complete: {target.host}:{remote_engine}")

    return {
        "local_onnx": str(onnx_path),
        "remote_onnx": remote_onnx,
        "remote_engine": "" if skip_build else remote_engine,
        "precision": target.precision.lower(),
        "commands": plan,
        "dry_run": dry_run,
    }
