"""Model registry for zkml-bench."""

from __future__ import annotations

import inspect

import torch.nn as nn

from ..exceptions import ModelRegistrationError

_MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ModelRegistrationError(f"Model '{name}' already registered")
        # Validate constructor has (num_classes, input_shape) signature
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        # Must have 'num_classes' and 'input_shape' params (after self)
        required = {"num_classes", "input_shape"}
        if not required.issubset(set(params)):
            raise ModelRegistrationError(
                f"Model class {cls.__name__} must have 'num_classes' and 'input_shape' constructor params. "
                f"Got: {params}"
            )
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, num_classes: int, input_shape: tuple[int, ...]) -> nn.Module:
    """Instantiate and return a registered model by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {list(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name](num_classes=num_classes, input_shape=input_shape)


def list_models() -> list[dict]:
    """Return list of dicts with 'name' and 'description' (from class docstring)."""
    return [
        {
            "name": name,
            "description": (cls.__doc__ or "").strip().split("\n")[0],
        }
        for name, cls in _MODEL_REGISTRY.items()
    ]
