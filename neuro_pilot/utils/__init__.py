"""
NeuroPilot Utility Toolbox.
Standardized operations, NMS logic, and logging facilities.
"""


def __getattr__(name):
    if name == "logger":
        from .logger import logger as _logger

        return _logger
    if name in {"xywh2xyxy", "xyxy2xywh", "scale_boxes", "scale_coords", "clip_boxes", "clip_coords"}:
        import importlib

        ops = importlib.import_module(".ops", __package__)
        return getattr(ops, name)
    if name in {"non_max_suppression", "decode_and_nms"}:
        import importlib

        nms = importlib.import_module(".nms", __package__)
        return getattr(nms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "logger",
    "xywh2xyxy",
    "xyxy2xywh",
    "scale_boxes",
    "scale_coords",
    "clip_boxes",
    "clip_coords",
    "non_max_suppression",
    "decode_and_nms",
)
