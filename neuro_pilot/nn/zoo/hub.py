"""Automatic pretrained weight hub with local cache and 2-tier remote resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "ThanhNguyenVo/neuropilot-weights"
_HF_SELF = f"https://huggingface.co/{DEFAULT_HF_REPO}/resolve/main"
_OPENPILOT_LFS_BATCH = "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs/objects/batch"
_USER_AGENT = "NeuroPilot-Hub/1.3"
_HF_OP = "https://huggingface.co/commaai/openpilot_driving_models/resolve/main"
_HF_AW = "https://huggingface.co/AutowareFoundation"
_HF_VQ = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main"
_SEG_ZIP = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/"
    "qai-hub-models/models/ddrnet23_slim/releases/v0.67.0/ddrnet23_slim-onnx-float.zip"
)
_EOMT_ST = (
    "https://huggingface.co/tue-mps/cityscapes_semantic_eomt_large_1024/"
    "resolve/main/model.safetensors"
)


@dataclass(frozen=True)
class WeightSpec:
    """Metadata describing an upstream permissive checkpoint source."""

    key: str
    kind: str  # "pt" | "onnx" | "zip_onnx" | "lfs_onnx" | "gdrive_onnx" | "safetensors"
    source: str
    license: str
    inner_path: str = ""
    lfs_size: int = 0


WEIGHT_SPECS: Dict[str, WeightSpec] = {
    "pilot_s": WeightSpec(
        "pilot_s", "lfs_onnx",
        "962396e7909556af6a9c7bfb7ad6db01a8d2c8da0760379eb71efda004cb3548",
        "MIT", lfs_size=69699272,
    ),
    "pilot_l": WeightSpec(
        "pilot_l", "onnx",
        f"{_HF_OP}/3f1a78e6-9a79-4d85-b796-1cd573bd5a3a/32768/big_driving_supercombo.onnx",
        "MIT",
    ),
    "pilot_nav": WeightSpec(
        "pilot_nav", "onnx",
        f"{_HF_OP}/96244b1f-6c4e-4adc-848d-eccd0b44b182/12800/big_driving_supercombo.onnx",
        "MIT",
    ),
    "driver": WeightSpec(
        "driver", "lfs_onnx",
        "107163fb8f7846310db4e4b1a2354dd9cd19ef707647463225c8049c3b5c2af0",
        "MIT", lfs_size=7844470,
    ),
    "cipo": WeightSpec(
        "cipo", "onnx", f"{_HF_AW}/auto_speed/resolve/main/models/autospeed_2_fp32.onnx",
        "Apache-2.0",
    ),
    "lane": WeightSpec("lane", "gdrive_onnx", "1u89PujOd89M-l6t_Cvub3BaQsDFKofuY", "Apache-2.0"),
    "drive": WeightSpec("drive", "gdrive_onnx", "1GKqhrNP5xtLBRqrcqL9k1IKnwicSL-N8", "Apache-2.0"),
    "depth": WeightSpec(
        "depth", "onnx", f"{_HF_AW}/scene_3d/resolve/main/models/scene3d_lite_fp32.onnx",
        "Apache-2.0",
    ),
    "seg": WeightSpec(
        "seg", "zip_onnx", _SEG_ZIP, "MIT", inner_path="ddrnet23_slim_onnx/ddrnet23_slim.onnx"
    ),
    "seg_l": WeightSpec("seg_l", "safetensors", _EOMT_ST, "MIT"),
    "jepa": WeightSpec(
        "jepa", "onnx", f"{_HF_AW}/latent_drive/resolve/main/camfront-rptw-lane2-e1.onnx",
        "Apache-2.0",
    ),
    "vq_enc": WeightSpec("vq_enc", "onnx", f"{_HF_VQ}/encoder.onnx", "MIT"),
    "vq_dec": WeightSpec("vq_dec", "onnx", f"{_HF_VQ}/decoder.onnx", "MIT"),
    "pilot_mhp": WeightSpec("pilot_mhp", "pt", f"{_HF_SELF}/pilot_mhp.pt", "Apache-2.0"),
    "pilot_mhp_v2": WeightSpec("pilot_mhp_v2", "pt", f"{_HF_SELF}/pilot_mhp_v2.pt", "Apache-2.0"),
    "pilot_det": WeightSpec("pilot_det", "pt", f"{_HF_SELF}/pilot_det.pt", "Apache-2.0"),
}


def get_weights_dir(cache_dir: Optional[str | Path] = None) -> Path:
    """Return the local directory used to store downloaded `.pt` checkpoints."""
    if cache_dir is not None:
        base = Path(cache_dir).expanduser()
    elif os.environ.get("NEUROPILOT_CACHE_DIR"):
        base = Path(os.environ["NEUROPILOT_CACHE_DIR"]).expanduser()
    else:
        base = Path.home() / ".cache" / "neuro_pilot" / "weights"
    base.mkdir(parents=True, exist_ok=True)
    return base


def list_downloadable() -> List[str]:
    """Return sorted canonical model keys supporting automatic weight download."""
    return sorted(WEIGHT_SPECS.keys())


def resolve_weight_key(name_or_path: str | Path) -> Optional[str]:
    """Resolve a bare key or filename like `pilot_l.pt` to a canonical weight key."""
    raw = str(name_or_path).strip()
    if not raw:
        return None
    p = Path(raw)
    if len(p.parts) > 1 and p.suffix in {".pt", ".pth"}:
        return None
    stem = p.stem if p.suffix in {".pt", ".pth", ".yaml", ".yml"} else raw
    if stem in WEIGHT_SPECS:
        return stem
    from neuro_pilot.nn.zoo import MODEL_YAML_MAP

    yaml_name = MODEL_YAML_MAP.get(stem)
    if yaml_name:
        canonical = Path(yaml_name).stem
        if canonical in WEIGHT_SPECS:
            return canonical
    return None


def _stream_download(url: str, dst: Path, timeout: int = 120) -> None:
    """Stream an HTTPS URL atomically into `dst` using a temporary `.part` file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        part.replace(dst)
    except Exception:
        if part.exists():
            part.unlink()
        raise


def _resolve_lfs_href(oid: str, size: int) -> str:
    """Resolve a public Git-LFS object download URL without authentication."""
    payload = json.dumps(
        {"operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        _OPENPILOT_LFS_BATCH,
        data=payload,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": _USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["objects"][0]["actions"]["download"]["href"])


def _mirror_urls(key: str) -> Tuple[str, ...]:
    """Build candidate direct `.pt` mirror URLs from environment or default HF repo."""
    urls: List[str] = []
    custom_base = os.environ.get("NEUROPILOT_WEIGHTS_URL", "").strip().rstrip("/")
    if custom_base:
        urls.append(f"{custom_base}/{key}.pt")
    hf_repo = os.environ.get("NEUROPILOT_HF_REPO", DEFAULT_HF_REPO).strip()
    if hf_repo:
        urls.append(f"https://huggingface.co/{hf_repo}/resolve/main/{key}.pt")
    return tuple(urls)


def _fetch_upstream_onnx(spec: WeightSpec, work_dir: Path) -> Path:
    """Download the official permissive upstream ONNX file into `work_dir`."""
    onnx_dst = work_dir / f"{spec.key}.onnx"
    if spec.kind == "onnx":
        _stream_download(spec.source, onnx_dst)
    elif spec.kind == "lfs_onnx":
        _stream_download(_resolve_lfs_href(spec.source, spec.lfs_size), onnx_dst)
    elif spec.kind == "gdrive_onnx":
        url = (
            f"https://drive.usercontent.google.com/download"
            f"?id={spec.source}&export=download&confirm=t"
        )
        _stream_download(url, onnx_dst)
    elif spec.kind == "zip_onnx":
        zip_dst = work_dir / f"{spec.key}.zip"
        _stream_download(spec.source, zip_dst)
        with zipfile.ZipFile(zip_dst, "r") as zf:
            target = spec.inner_path
            if target not in zf.namelist():
                matches = [n for n in zf.namelist() if n.endswith(".onnx")]
                if not matches:
                    raise FileNotFoundError(f"No ONNX file found inside archive for '{spec.key}'")
                target = matches[0]
            with zf.open(target) as src, onnx_dst.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    else:
        raise ValueError(f"Unsupported upstream source kind '{spec.kind}' for '{spec.key}'")
    return onnx_dst


def download_weights(
    key: str,
    dest_dir: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """Return local `.pt` checkpoint path, auto-downloading and caching if missing."""
    canonical = resolve_weight_key(key)
    if canonical is None:
        supported = ", ".join(list_downloadable())
        raise KeyError(f"No downloadable checkpoint for '{key}'. Supported: {supported}")

    spec = WEIGHT_SPECS[canonical]
    out_dir = get_weights_dir(dest_dir)
    pt_path = out_dir / f"{canonical}.pt"
    if pt_path.is_file() and not force:
        return pt_path

    for url in _mirror_urls(canonical):
        try:
            logger.info("Downloading pretrained checkpoint '%s' from %s", canonical, url)
            _stream_download(url, pt_path)
            return pt_path
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

    if spec.kind == "pt":
        logger.info("Downloading direct checkpoint '%s' from %s", canonical, spec.source)
        _stream_download(spec.source, pt_path, timeout=300)
        return pt_path

    logger.info("Fetching upstream %s weights for '%s' -> %s", spec.license, canonical, pt_path)
    from neuro_pilot.nn.zoo import convert_onnx_to_pt, register_all_zoo_models
    from neuro_pilot.nn.zoo.converters.eomt import convert_safetensors_to_pt

    register_all_zoo_models()
    with tempfile.TemporaryDirectory(prefix=f"neuropilot_{canonical}_") as tmp:
        tmp_dir = Path(tmp)
        tmp_pt = tmp_dir / f"{canonical}.pt"
        if spec.kind == "safetensors":
            st_file = tmp_dir / f"{canonical}.safetensors"
            _stream_download(spec.source, st_file, timeout=300)
            convert_safetensors_to_pt(canonical, st_file, tmp_pt, half=True)
        else:
            onnx_file = _fetch_upstream_onnx(spec, tmp_dir)
            convert_onnx_to_pt(canonical, onnx_file, tmp_pt, strict=True)
        shutil.move(str(tmp_pt), str(pt_path))
    return pt_path
