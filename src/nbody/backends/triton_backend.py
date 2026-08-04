from __future__ import annotations

import torch

from .. import config
from .._pairwise import default_tile_size
from ..state import State
from .base import BackendUnavailable

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only without triton installed
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc

_BLOCK_J = 64

if triton is not None:

    @triton.jit
    def _accel_kernel(
        r_ptr,
        m_ptr,
        out_ptr,
        g_ptr,
        eps2_ptr,
        n,
        BLOCK_I: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs_i = pid * BLOCK_I + tl.arange(0, BLOCK_I)
        mask_i = offs_i < n

        xi = tl.load(r_ptr + offs_i * 3 + 0, mask=mask_i, other=0.0)
        yi = tl.load(r_ptr + offs_i * 3 + 1, mask=mask_i, other=0.0)
        zi = tl.load(r_ptr + offs_i * 3 + 2, mask=mask_i, other=0.0)

        g = tl.load(g_ptr)
        eps2 = tl.load(eps2_ptr)

        ax = tl.zeros((BLOCK_I,), dtype=xi.dtype)
        ay = tl.zeros((BLOCK_I,), dtype=xi.dtype)
        az = tl.zeros((BLOCK_I,), dtype=xi.dtype)

        for j0 in range(0, n, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            mask_j = offs_j < n

            xj = tl.load(r_ptr + offs_j * 3 + 0, mask=mask_j, other=0.0)
            yj = tl.load(r_ptr + offs_j * 3 + 1, mask=mask_j, other=0.0)
            zj = tl.load(r_ptr + offs_j * 3 + 2, mask=mask_j, other=0.0)
            mj = tl.load(m_ptr + offs_j, mask=mask_j, other=0.0)

            dx = xj[None, :] - xi[:, None]
            dy = yj[None, :] - yi[:, None]
            dz = zj[None, :] - zi[:, None]

            dsq = dx * dx + dy * dy + dz * dz + eps2

            diag = offs_i[:, None] == offs_j[None, :]
            pair_mask = mask_i[:, None] & mask_j[None, :]
            invalid = diag | (~pair_mask)

            dsq = tl.where(invalid, 1.0, dsq)
            inv_sqrt = tl.rsqrt(dsq)
            inv = inv_sqrt * inv_sqrt * inv_sqrt
            inv = tl.where(invalid, 0.0, inv)

            minv = mj[None, :] * inv

            ax += tl.sum(minv * dx, axis=1)
            ay += tl.sum(minv * dy, axis=1)
            az += tl.sum(minv * dz, axis=1)

        ax = ax * g
        ay = ay * g
        az = az * g

        tl.store(out_ptr + offs_i * 3 + 0, ax, mask=mask_i)
        tl.store(out_ptr + offs_i * 3 + 1, ay, mask=mask_i)
        tl.store(out_ptr + offs_i * 3 + 2, az, mask=mask_i)

else:
    _accel_kernel = None


def _next_pow2(x: int) -> int:
    x = max(1, x)
    return 1 << (x - 1).bit_length()


class TritonBackend:
    name = "triton"

    def supports(self, device) -> bool:
        if triton is None:
            return False
        return torch.device(device).type == "cuda"

    def accelerations(
        self,
        r: torch.Tensor,
        m: torch.Tensor,
        softening: float = config.SOFTENING,
        *,
        tile_size: int | None = None,
    ) -> torch.Tensor:
        if triton is None:
            raise BackendUnavailable(f"triton backend requires the triton package, which is not installed ({_TRITON_IMPORT_ERROR})")
        if r.device.type != "cuda":
            raise BackendUnavailable(f"triton backend requires a cuda device, got {r.device}")

        n = r.shape[0]
        if tile_size is None:
            tile_size = default_tile_size(n, r.dtype)
        block_i = _next_pow2(min(max(1, tile_size), n))

        r_c = r.contiguous()
        m_c = m.contiguous()
        out = torch.empty_like(r_c)
        g_t = torch.tensor(config.G, dtype=r.dtype, device=r.device)
        eps2_t = torch.tensor(softening * softening, dtype=r.dtype, device=r.device)

        grid = (triton.cdiv(n, block_i),)
        _accel_kernel[grid](r_c, m_c, out, g_t, eps2_t, n, BLOCK_I=block_i, BLOCK_J=_BLOCK_J)
        return out

    def fused_symplectic_euler_step(self, state: State, dt: float, softening: float = config.SOFTENING) -> State:
        a = self.accelerations(state.r, state.m, softening)
        v_new = state.v + dt * a
        r_new = state.r + dt * v_new
        return State(r=r_new, v=v_new, m=state.m)


BACKEND = TritonBackend()
