from __future__ import annotations

import torch
from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor.exc import InvalidCxxCompiler

from .. import config
from ..state import State
from .base import BackendUnavailable
from .torch_eager import _accelerations_impl

_compiled_accelerations = torch.compile(_accelerations_impl)


def _run_compiled(r: torch.Tensor, m: torch.Tensor, softening: float, tile_size: int | None) -> torch.Tensor:
    try:
        return _compiled_accelerations(r, m, softening, tile_size)
    except InvalidCxxCompiler as exc:
        raise BackendUnavailable(
            f"torch_compiled: no working C++ compiler found for torch.compile on device {r.device} ({exc})"
        ) from exc
    except BackendCompilerFailed as exc:
        inner = exc.inner_exception
        if isinstance(inner, InvalidCxxCompiler):
            raise BackendUnavailable(
                f"torch_compiled: no working C++ compiler found for torch.compile on device {r.device} ({inner})"
            ) from exc
        raise


class TorchCompiledBackend:
    name = "torch_compiled"

    def supports(self, device) -> bool:
        return True

    def accelerations(
        self,
        r: torch.Tensor,
        m: torch.Tensor,
        softening: float = config.SOFTENING,
        *,
        tile_size: int | None = None,
    ) -> torch.Tensor:
        return _run_compiled(r, m, softening, tile_size)

    def fused_symplectic_euler_step(self, state: State, dt: float, softening: float = config.SOFTENING) -> State:
        a = self.accelerations(state.r, state.m, softening)
        v_new = state.v + dt * a
        r_new = state.r + dt * v_new
        return State(r=r_new, v=v_new, m=state.m)


BACKEND = TorchCompiledBackend()
