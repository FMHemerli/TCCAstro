from __future__ import annotations

import math

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
    m64 = state.m.to(torch.float64)
    d = torch.linalg.vector_norm(r64 - c.unsqueeze(0), dim=-1)

    order = torch.argsort(d)
    m_sorted = m64[order]
    d_sorted = d[order]
    cumulative_mass = torch.cumsum(m_sorted, dim=0)

    half_mass = 0.5 * m64.sum()
    k_star = torch.searchsorted(cumulative_mass, half_mass)
    k_star = torch.clamp(k_star, max=state.n - 1)

    return d_sorted[k_star].item()


def n_live(state) -> int:
    return int((state.m > 0).sum().item())


def mass_spectrum_summary(state) -> dict:
    m64 = state.m.to(torch.float64)
    live = m64[m64 > 0]
    if live.numel() == 0:
        raise ValueError("mass_spectrum_summary: no live particles (all masses are zero)")
    return {
        "n_live": int(live.numel()),
        "total_mass": live.sum().item(),
        "mean_mass": live.mean().item(),
        "min_mass": live.min().item(),
        "max_mass": live.max().item(),
        "std_mass": live.std(unbiased=False).item(),
    }


def scales_from_state(
    state,
    *,
    softening: float = config.SOFTENING,
    radius: float = config.SPHERE_RADIUS,
) -> dict:
    m64 = state.m.to(torch.float64)
    total_mass = m64.sum().item()
    if total_mass <= 0.0:
        raise ValueError("scales_from_state: realized total mass is not positive")

    density = total_mass / ((4.0 / 3.0) * math.pi * radius**3)
    t_ff = math.sqrt(3.0 * math.pi / (32.0 * config.G * density))
    t_conv = 0.5 * t_ff
    v_char = math.sqrt(config.G * total_mass / radius)
    l_scale = total_mass * radius * v_char
    sum_m_sq = (m64 * m64).sum().item()
    u_min_bound = -config.G * (total_mass * total_mass - sum_m_sq) / (2.0 * softening)

    return {
        "total_mass": total_mass,
        "density": density,
        "t_ff": t_ff,
        "t_conv": t_conv,
        "v_char": v_char,
        "l_scale": l_scale,
        "u_min_bound": u_min_bound,
    }
