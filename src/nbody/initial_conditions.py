from __future__ import annotations

import math

import numpy as np
import torch

from . import config, observables, populations
from .state import State


def cold_sphere(
    n: int = config.N_PARTICLES,
    radius: float = config.SPHERE_RADIUS,
    particle_mass: float = config.PARTICLE_MASS,
    seed: int = config.SEED,
    dtype: torch.dtype = torch.float64,
    device="cpu",
) -> State:
    rng = np.random.default_rng(seed)

    u = rng.random(n)
    sample_radius = radius * np.cbrt(u)

    cos_th = 2.0 * rng.random(n) - 1.0
    phi = 2.0 * np.pi * rng.random(n)
    sin_th = np.sqrt(1.0 - cos_th**2)

    positions = np.empty((n, 3), dtype=np.float64)
    positions[:, 0] = sample_radius * sin_th * np.cos(phi)
    positions[:, 1] = sample_radius * sin_th * np.sin(phi)
    positions[:, 2] = sample_radius * cos_th

    positions -= positions.mean(axis=0)
    velocities = np.zeros((n, 3), dtype=np.float64)
    masses = np.full(n, particle_mass, dtype=np.float64)

    r = torch.from_numpy(positions).to(dtype=dtype)
    v = torch.from_numpy(velocities).to(dtype=dtype)
    m = torch.from_numpy(masses).to(dtype=dtype)

    device_t = torch.device(device)
    return State(r=r.to(device=device_t), v=v.to(device=device_t), m=m.to(device=device_t))


def random_sphere(
    n: int = config.N_PARTICLES,
    radius: float = config.SPHERE_RADIUS,
    seed: int = config.SEED,
    mass_spectrum: dict | None = None,
    virial_ratio: float = 0.0,
    f_cut: float = config.F_CUT_DEFAULT,
    mass_seed: int = config.MASS_SEED,
    vel_seed: int = config.VEL_SEED,
    dtype: torch.dtype = torch.float64,
    device="cpu",
) -> State:
    rng = np.random.default_rng(seed)

    u = rng.random(n)
    sample_radius = radius * np.cbrt(u)

    cos_th = 2.0 * rng.random(n) - 1.0
    phi = 2.0 * np.pi * rng.random(n)
    sin_th = np.sqrt(1.0 - cos_th**2)

    positions = np.empty((n, 3), dtype=np.float64)
    positions[:, 0] = sample_radius * sin_th * np.cos(phi)
    positions[:, 1] = sample_radius * sin_th * np.sin(phi)
    positions[:, 2] = sample_radius * cos_th

    positions -= positions.mean(axis=0)

    if mass_spectrum is None:
        masses = np.full(n, config.PARTICLE_MASS, dtype=np.float64)
    else:
        masses = populations.sample_masses(
            n=n,
            alpha=mass_spectrum.get("alpha", config.MASS_ALPHA),
            ratio=mass_spectrum.get("ratio", config.MASS_RATIO),
            particle_mass=mass_spectrum.get("particle_mass", config.PARTICLE_MASS),
            seed=mass_seed,
        )

    if virial_ratio == 0.0:
        velocities = np.zeros((n, 3), dtype=np.float64)
    else:
        # sample_velocities takes the State and evaluates U itself. Passing a precomputed
        # potential would let the caller and the sampler disagree about the softening that
        # normalizes Q, and INV-16(c) needs the lambda that only the sampler can report.
        r_fp64 = torch.from_numpy(positions)
        m_fp64 = torch.from_numpy(masses)
        v_fp64_zero = torch.zeros((n, 3), dtype=torch.float64)
        potential_state = State(r=r_fp64, v=v_fp64_zero, m=m_fp64)
        velocities, _lambda = populations.sample_velocities(
            potential_state,
            radius=radius,
            q=virial_ratio,
            f_cut=f_cut,
            softening=config.SOFTENING,
            g=config.G,
            seed=vel_seed,
        )

    r = torch.from_numpy(positions).to(dtype=dtype)
    v = torch.from_numpy(velocities).to(dtype=dtype)
    m = torch.from_numpy(masses).to(dtype=dtype)

    device_t = torch.device(device)
    return State(r=r.to(device=device_t), v=v.to(device=device_t), m=m.to(device=device_t))


def two_body_kepler(
    a: float = config.KEPLER_A,
    e: float = config.KEPLER_E,
    particle_mass: float = config.PARTICLE_MASS,
    dtype: torch.dtype = torch.float64,
    device="cpu",
) -> State:
    mu = config.G * (2.0 * particle_mass)
    r_apo = a * (1.0 + e)
    v_apo = math.sqrt(mu / a * (1.0 - e) / (1.0 + e))

    positions = np.array(
        [[+r_apo / 2.0, 0.0, 0.0], [-r_apo / 2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    velocities = np.array(
        [[0.0, +v_apo / 2.0, 0.0], [0.0, -v_apo / 2.0, 0.0]],
        dtype=np.float64,
    )
    masses = np.full(2, particle_mass, dtype=np.float64)

    r = torch.from_numpy(positions).to(dtype=dtype)
    v = torch.from_numpy(velocities).to(dtype=dtype)
    m = torch.from_numpy(masses).to(dtype=dtype)

    device_t = torch.device(device)
    return State(r=r.to(device=device_t), v=v.to(device=device_t), m=m.to(device=device_t))


def two_body_circular(
    separation: float = config.CIRC_SEPARATION,
    softening: float = config.SOFTENING,
    particle_mass: float = config.PARTICLE_MASS,
    dtype: torch.dtype = torch.float64,
    device="cpu",
) -> State:
    mu = config.G * (2.0 * particle_mass)
    omega = math.sqrt(mu / (separation**2 + softening**2) ** 1.5)
    v_each = omega * separation / 2.0

    positions = np.array(
        [[+separation / 2.0, 0.0, 0.0], [-separation / 2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    velocities = np.array(
        [[0.0, +v_each, 0.0], [0.0, -v_each, 0.0]],
        dtype=np.float64,
    )
    masses = np.full(2, particle_mass, dtype=np.float64)

    r = torch.from_numpy(positions).to(dtype=dtype)
    v = torch.from_numpy(velocities).to(dtype=dtype)
    m = torch.from_numpy(masses).to(dtype=dtype)

    device_t = torch.device(device)
    return State(r=r.to(device=device_t), v=v.to(device=device_t), m=m.to(device=device_t))
