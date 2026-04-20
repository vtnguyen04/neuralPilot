import logging
from typing import Dict, Type

logger = logging.getLogger(__name__)


class Registry:
    """
    Allows registration of new algorithms, models, and module components globally
    using a decorator pattern to avoid hardcoded imports.
    """

    _BACKBONES: Dict[str, Type] = {}
    _HEADS: Dict[str, Type] = {}
    _LOSSES: Dict[str, Type] = {}
    _NECKS: Dict[str, Type] = {}
    _TASKS: Dict[str, Type] = {}
    _AGGREGATORS: Dict[str, Type] = {}

    @classmethod
    def register_backbone(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            if key in cls._BACKBONES:
                logger.warning(f"Backbone {key} already registered. Overwriting.")
            cls._BACKBONES[key] = obj
            return obj

        return decorator

    @classmethod
    def register_head(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            cls._HEADS[key] = obj
            return obj

        return decorator

    @classmethod
    def register_loss(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            cls._LOSSES[key] = obj
            return obj

        return decorator

    @classmethod
    def register_neck(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            cls._NECKS[key] = obj
            return obj

        return decorator

    @classmethod
    def get_backbone(cls, name: str) -> Type:
        return cls._BACKBONES.get(name)

    @classmethod
    def get_head(cls, name: str) -> Type:
        return cls._HEADS.get(name)

    @classmethod
    def get_loss(cls, name: str) -> Type:
        return cls._LOSSES.get(name)

    @classmethod
    def get_neck(cls, name: str) -> Type:
        return cls._NECKS.get(name)

    @classmethod
    def get(cls, name: str) -> Type:
        """General lookup across all categories."""
        if name in cls._BACKBONES:
            return cls._BACKBONES[name]
        if name in cls._HEADS:
            return cls._HEADS[name]
        if name in cls._NECKS:
            return cls._NECKS[name]
        if name in cls._LOSSES:
            return cls._LOSSES[name]
        if name in cls._TASKS:
            return cls._TASKS[name]
        if name in cls._AGGREGATORS:
            return cls._AGGREGATORS[name]
        return None

    @classmethod
    def register_task(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            if key in cls._TASKS:
                logger.warning(f"Task {key} already registered. Overwriting.")
            cls._TASKS[key] = obj
            return obj

        return decorator

    @classmethod
    def get_task(cls, name: str) -> Type:
        if name not in cls._TASKS:
            raise ValueError(f"Task '{name}' not found in registry. Available: {list(cls._TASKS.keys())}")
        return cls._TASKS[name]

    @classmethod
    def list_tasks(cls) -> list[str]:
        return list(cls._TASKS.keys())

    @classmethod
    def register_aggregator(cls, name: str = None):
        def decorator(obj):
            key = name or obj.__name__
            if key in cls._AGGREGATORS:
                logger.warning(f"Aggregator {key} already registered. Overwriting.")
            cls._AGGREGATORS[key] = obj
            return obj

        return decorator

    @classmethod
    def get_aggregator(cls, name: str) -> Type:
        if name not in cls._AGGREGATORS:
            raise ValueError(f"Aggregator '{name}' not found in registry. Available: {list(cls._AGGREGATORS.keys())}")
        return cls._AGGREGATORS[name]

    @classmethod
    def list_aggregators(cls) -> list[str]:
        return list(cls._AGGREGATORS.keys())


register_backbone = Registry.register_backbone
register_head = Registry.register_head
register_loss = Registry.register_loss
register_neck = Registry.register_neck
register_task = Registry.register_task
register_aggregator = Registry.register_aggregator
