from __future__ import annotations

import torch

from .. import config
from ..state import State
from .base import BackendUnavailable


def _accelerations_scalar(r_list, m_list, softening: float):
    n = len(r_list)
    eps2 = softening * softening
    g = config.G
    out = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(n):
        xi, yi, zi = r_list[i]
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            if j == i:
                continue
            xj, yj, zj = r_list[j]
            dx = xj - xi
            dy = yj - yi
            dz = zj - zi
            dsq = dx * dx + dy * dy + dz * dz + eps2
            inv = dsq ** (-1.5)
            m_inv = m_list[j] * inv
            ax += m_inv * dx
            ay += m_inv * dy
            az += m_inv * dz
        out[i][0] = g * ax
        out[i][1] = g * ay
        out[i][2] = g * az
    return out


class PythonPureBackend:
    name = "python_pure"

    def supports(self, device) -> bool:
        return torch.device(device).type == "cpu"

    def accelerations(
        self,
        r: torch.Tensor,
        m: torch.Tensor,
        softening: float = config.SOFTENING,
        *,
        tile_size: int | None = None,
    ) -> torch.Tensor:
        if r.device.type != "cpu":
            raise BackendUnavailable(f"python_pure backend only supports the cpu device, got {r.device}")
        r_list = r.detach().cpu().tolist()
        m_list = m.detach().cpu().tolist()
        out_list = _accelerations_scalar(r_list, m_list, float(softening))
        return torch.tensor(out_list, dtype=r.dtype, device=r.device)

    def fused_symplectic_euler_step(self, state: State, dt: float, softening: float = config.SOFTENING) -> State:
        a = self.accelerations(state.r, state.m, softening)
        v_new = state.v + dt * a
        r_new = state.r + dt * v_new
        return State(r=r_new, v=v_new, m=state.m)


BACKEND = PythonPureBackend()
