from __future__ import annotations

from typing import Dict, Type
from neuro_pilot.utils.logger import logger


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
    def _register(cls, target_dict: Dict[str, Type], category: str, name: str | None = None):
        """Helper decorator factory to handle registrations across all components."""
        def decorator(obj):
            key = name or obj.__name__
            if key in target_dict:
                logger.warning(f"{category} {key} already registered. Overwriting.")
            target_dict[key] = obj
            return obj

        return decorator

    @classmethod
    def register_backbone(cls, name: str | None = None):
        return cls._register(cls._BACKBONES, "Backbone", name)

    @classmethod
    def register_head(cls, name: str | None = None):
        return cls._register(cls._HEADS, "Head", name)

    @classmethod
    def register_loss(cls, name: str | None = None):
        return cls._register(cls._LOSSES, "Loss", name)

    @classmethod
    def register_neck(cls, name: str | None = None):
        return cls._register(cls._NECKS, "Neck", name)

    @classmethod
    def register_task(cls, name: str | None = None):
        return cls._register(cls._TASKS, "Task", name)

    @classmethod
    def register_aggregator(cls, name: str | None = None):
        return cls._register(cls._AGGREGATORS, "Aggregator", name)

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
        return (
            cls._BACKBONES.get(name)
            or cls._HEADS.get(name)
            or cls._NECKS.get(name)
            or cls._LOSSES.get(name)
            or cls._TASKS.get(name)
            or cls._AGGREGATORS.get(name)
        )

    @classmethod
    def _get_required(cls, target_dict: Dict[str, Type], category: str, name: str) -> Type:
        """Helper to retrieve a required registry entry or raise ValueError."""
        if name not in target_dict:
            raise ValueError(f"{category} '{name}' not found in registry. Available: {list(target_dict.keys())}")
        return target_dict[name]

    @classmethod
    def get_task(cls, name: str) -> Type:
        return cls._get_required(cls._TASKS, "Task", name)

    @classmethod
    def list_tasks(cls) -> list[str]:
        return list(cls._TASKS.keys())

    @classmethod
    def get_aggregator(cls, name: str) -> Type:
        return cls._get_required(cls._AGGREGATORS, "Aggregator", name)

    @classmethod
    def list_aggregators(cls) -> list[str]:
        return list(cls._AGGREGATORS.keys())

    @classmethod
    def list_zoo_models(cls) -> list[str]:
        from neuro_pilot.nn.zoo import ZOO_REGISTRY, register_all_zoo_models

        register_all_zoo_models()
        return ZOO_REGISTRY.keys()

    @classmethod
    def get_zoo_model(cls, name: str) -> Type:
        from neuro_pilot.nn.zoo import ZOO_REGISTRY, register_all_zoo_models

        register_all_zoo_models()
        return ZOO_REGISTRY.get(name)


register_backbone = Registry.register_backbone
register_head = Registry.register_head
register_loss = Registry.register_loss
register_neck = Registry.register_neck
register_task = Registry.register_task
register_aggregator = Registry.register_aggregator
