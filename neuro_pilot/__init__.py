# Copyright (c) 2026 Vo Thanh Nguyen. All rights reserved.

__version__ = "1.4.0"


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
    elif name == "ModelZoo":
        from neuro_pilot.engine.zoo import ModelZoo

        return ModelZoo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelZoo", "NeuroPilot", "Results", "check_version"]
