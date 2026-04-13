# Copyright (c) 2026 Vo Thanh Nguyen. All rights reserved.

__version__ = "1.1.2"

def __getattr__(name):
    if name == "NeuroPilot":
        from neuro_pilot.engine.model import NeuroPilot
        return NeuroPilot
    elif name == "Results":
        from neuro_pilot.engine.results import Results
        return Results
    elif name == "check_version":
        from neuro_pilot.utils.checks import check_version
        return check_version
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["NeuroPilot", "Results", "check_version"]
