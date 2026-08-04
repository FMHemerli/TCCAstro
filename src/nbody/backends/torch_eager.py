from __future__ import annotations

import torch

from .. import config
from .._pairwise import default_tile_size, iter_tiles
from ..state import State
from .base import BackendUnavailable


def _accelerations_impl(r: torch.Tensor, m: torch.Tensor, softening: float, tile_size: int | None) -> torch.Tensor:
    n = r.shape[0]
    if tile_size is None:
        tile_size = default_tile_size(n, r.dtype)

    eps2 = softening * softening
    out = torch.empty_like(r)
    idx = torch.arange(n, device=r.device)

    for start, end in iter_tiles(n, tile_size):
        ri = r[start:end]
        diff = r.unsqueeze(0) - ri.unsqueeze(1)
        dsq = (diff * diff).sum(dim=-1) + eps2
        diag = idx[start:end].unsqueeze(1) == idx.unsqueeze(0)
        dsq = dsq.masked_fill(diag, 1.0)
        inv = dsq.pow(-1.5)
        inv = inv.masked_fill(diag, 0.0)
        minv = m.unsqueeze(0) * inv
        weighted = minv.unsqueeze(-1) * diff
        out[start:end] = config.G * weighted.sum(dim=1)

    return out


class TorchEagerBackend:
    name = "torch_eager"

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
        return _accelerations_impl(r, m, softening, tile_size)

    def fused_symplectic_euler_step(self, state: State, dt: float, softening: float = config.SOFTENING) -> State:
        a = self.accelerations(state.r, state.m, softening)
        v_new = state.v + dt * a
        r_new = state.r + dt * v_new
        return State(r=r_new, v=v_new, m=state.m)


BACKEND = TorchEagerBackend()
