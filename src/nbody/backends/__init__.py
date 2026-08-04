from __future__ import annotations

import torch

from . import python_pure, torch_compiled, torch_eager, triton_backend
from .base import Backend, BackendUnavailable

BACKEND_NAMES = ("python_pure", "torch_eager", "torch_compiled", "triton")

_REGISTRY = {
    "python_pure": python_pure.BACKEND,
    "torch_eager": torch_eager.BACKEND,
    "torch_compiled": torch_compiled.BACKEND,
    "triton": triton_backend.BACKEND,
}


def get_backend(name: str) -> Backend:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; valid names are {BACKEND_NAMES}") from None


def available_backends(device) -> tuple:
    dev = torch.device(device)
    return tuple(name for name in BACKEND_NAMES if _REGISTRY[name].supports(dev))


__all__ = ["Backend", "BackendUnavailable", "BACKEND_NAMES", "get_backend", "available_backends"]
