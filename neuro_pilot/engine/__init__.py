def __getattr__(name):
    if name == "Predictor":
        from .predictor import Predictor
        return Predictor
    elif name == "Trainer":
        from .trainer import Trainer
        return Trainer
    elif name == "Validator":
        from .validator import Validator
        return Validator
    elif name == "Results":
        from .results import Results
        return Results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ("Predictor", "Trainer", "Validator", "Results")
