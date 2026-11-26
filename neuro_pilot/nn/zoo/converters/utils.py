"""Shared tensor and prefix helpers for ONNX-to-GraphModel converters."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch


def to_tensor(
    array: np.ndarray,
    transpose: bool = False,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """Copy a NumPy array into a contiguous PyTorch tensor with optional transpose/cast."""
    arr = array.T if transpose else array
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def rewrite_prefix(name: str, rules: Sequence[Tuple[str, str]]) -> Optional[str]:
    """Rewrite `name` using the first matching `(src_prefix, dst_prefix)` rule."""
    for src_prefix, dst_prefix in rules:
        if name.startswith(src_prefix):
            return dst_prefix + name[len(src_prefix) :]
    return None
