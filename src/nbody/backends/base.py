from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from ..state import State


class BackendUnavailable(RuntimeError):
    pass


@runtime_checkable
class Backend(Protocol):
    name: str

    def supports(self, device: torch.device) -> bool: ...

    def accelerations(
        self,
        r: torch.Tensor,
        m: torch.Tensor,
        softening: float = ...,
        *,
        tile_size: int | None = None,
    ) -> torch.Tensor: ...

    def fused_symplectic_euler_step(self, state: "State", dt: float, softening: float = ...) -> "State": ...
