from __future__ import annotations

import torch

from . import config
from ._pairwise import default_tile_size, iter_tiles


def kinetic_energy(state) -> float:
    term = 0.5 * state.m * (state.v * state.v).sum(dim=-1)
    return term.to(torch.float64).sum().item()


def potential_energy(state, softening: float = config.SOFTENING) -> float:
    r = state.r
    m = state.m
    n = r.shape[0]
    eps2 = softening * softening
    tile_size = default_tile_size(n, r.dtype)
    idx = torch.arange(n, device=r.device)

    total = torch.zeros((), dtype=torch.float64, device=r.device)
    for start, end in iter_tiles(n, tile_size):
        ri = r[start:end]
        diff = r.unsqueeze(0) - ri.unsqueeze(1)
        dsq = (diff * diff).sum(dim=-1) + eps2
        diag = idx[start:end].unsqueeze(1) == idx.unsqueeze(0)
        dsq = dsq.masked_fill(diag, 1.0)
        inv_sqrt = dsq.pow(-0.5)
        inv_sqrt = inv_sqrt.masked_fill(diag, 0.0)

        mi = m[start:end]
        term = mi.unsqueeze(1) * m.unsqueeze(0) * inv_sqrt
        total = total + term.to(torch.float64).sum()

    u = -0.5 * config.G * total
    return u.item()


def total_energy(state, softening: float = config.SOFTENING) -> float:
    return kinetic_energy(state) + potential_energy(state, softening)


def linear_momentum(state) -> torch.Tensor:
    term = state.m.unsqueeze(-1) * state.v
    return term.to(torch.float64).sum(dim=0)


def angular_momentum(state) -> torch.Tensor:
    r = state.r
    v = state.v
    cross_x = r[:, 1] * v[:, 2] - r[:, 2] * v[:, 1]
    cross_y = r[:, 2] * v[:, 0] - r[:, 0] * v[:, 2]
    cross_z = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]
    cross = torch.stack((cross_x, cross_y, cross_z), dim=-1)
    term = state.m.unsqueeze(-1) * cross
    return term.to(torch.float64).sum(dim=0)


def center_of_mass(state) -> torch.Tensor:
    r64 = state.r.to(torch.float64)
    m64 = state.m.to(torch.float64)
    weighted = m64.unsqueeze(-1) * r64
    total_mass = m64.sum()
    return weighted.sum(dim=0) / total_mass


def half_mass_radius(state) -> float:
    c = center_of_mass(state)
    r64 = state.r.to(torch.float64)
    d = torch.linalg.vector_norm(r64 - c.unsqueeze(0), dim=-1)
    n = state.n
    sorted_d, _ = torch.sort(d)
    return sorted_d[n // 2 - 1].item()
