from __future__ import annotations

import math

import numpy as np
import torch

from . import config, observables

_ALPHA_ONE_TOL = 1e-12
_ALPHA_TWO_TOL = 1e-12


def g_factor(alpha: float, ratio: float) -> float:
    if abs(alpha - 1.0) < _ALPHA_ONE_TOL:
        return (ratio - 1.0) / math.log(ratio)
    if abs(alpha - 2.0) < _ALPHA_TWO_TOL:
        return math.log(ratio) / (1.0 - 1.0 / ratio)
    exp1 = 1.0 - alpha
    exp2 = 2.0 - alpha
    return ((1.0 - alpha) / (2.0 - alpha)) * (ratio**exp2 - 1.0) / (ratio**exp1 - 1.0)


def mass_min_from_mean(
    alpha: float = config.MASS_ALPHA,
    ratio: float = config.MASS_RATIO,
    particle_mass: float = config.PARTICLE_MASS,
) -> float:
    return particle_mass / g_factor(alpha, ratio)


def power_law_cdf(m, m_min: float, m_max: float, alpha: float):
    m = np.asarray(m, dtype=np.float64)
    if abs(alpha - 1.0) < _ALPHA_ONE_TOL:
        return np.log(m / m_min) / np.log(m_max / m_min)
    exp = 1.0 - alpha
    return (np.power(m, exp) - m_min**exp) / (m_max**exp - m_min**exp)


def power_law_inv_cdf(u, m_min: float, m_max: float, alpha: float):
    u = np.asarray(u, dtype=np.float64)
    if abs(alpha - 1.0) < _ALPHA_ONE_TOL:
        return m_min * np.power(m_max / m_min, u)
    exp = 1.0 - alpha
    base = m_min**exp + u * (m_max**exp - m_min**exp)
    return np.power(base, 1.0 / exp)


def _inv_cdf_on_subinterval(u, a: float, b: float, m_min: float, m_max: float, alpha: float):
    f_a = power_law_cdf(a, m_min, m_max, alpha)
    f_b = power_law_cdf(b, m_min, m_max, alpha)
    return power_law_inv_cdf(f_a + u * (f_b - f_a), m_min, m_max, alpha)


def mass_big_threshold(n: int, m_min: float, m_max: float, alpha: float) -> float:
    tail_prob = 2.0 / n
    return float(power_law_inv_cdf(1.0 - tail_prob, m_min, m_max, alpha))


def _log_binomial_pmf(n: int, k: int, p: float) -> float:
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * math.log(p) + (n - k) * math.log(1.0 - p)
    )


def mass_k_weights(n: int, tail_prob: float) -> tuple[float, float, float]:
    log_weights = [_log_binomial_pmf(n, k, tail_prob) for k in (1, 2, 3)]
    peak = max(log_weights)
    unnormalized = [math.exp(log_w - peak) for log_w in log_weights]
    total = sum(unnormalized)
    return tuple(w / total for w in unnormalized)


def sample_masses(
    n: int = config.N_PARTICLES,
    alpha: float = config.MASS_ALPHA,
    ratio: float = config.MASS_RATIO,
    particle_mass: float = config.PARTICLE_MASS,
    seed: int = config.MASS_SEED,
) -> np.ndarray:
    if n < 3:
        raise ValueError(f"sample_masses requires n >= 3 (k can be as large as 3), got {n}")

    m_min = mass_min_from_mean(alpha=alpha, ratio=ratio, particle_mass=particle_mass)
    m_max = ratio * m_min
    tail_prob = 2.0 / n
    m_big = mass_big_threshold(n, m_min, m_max, alpha)

    rng_m = np.random.default_rng(seed)

    weights = mass_k_weights(n, tail_prob)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0

    u_k = rng_m.random()
    k = 1 + int(np.searchsorted(cumulative, u_k))

    u_tail = rng_m.random(k)
    m_tail = _inv_cdf_on_subinterval(u_tail, m_big, m_max, m_min, m_max, alpha)

    u_body = rng_m.random(n - k)
    m_body = _inv_cdf_on_subinterval(u_body, m_min, m_big, m_min, m_max, alpha)

    masses = np.concatenate([m_tail, m_body])
    perm = rng_m.permutation(n)
    masses = masses[perm]

    return masses


def truncated_second_moment_factor(x: float) -> float:
    if x <= 0.0:
        return 0.0
    y = 0.5 * x * x
    sqrt_y = math.sqrt(y)
    gamma_three_half = 0.5 * math.sqrt(math.pi) * math.erf(sqrt_y) - sqrt_y * math.exp(-y)
    return 3.0 - 2.0 * (y**1.5) * math.exp(-y) / gamma_three_half


def solve_sigma(
    v_cut: float,
    target_second_moment: float,
    *,
    x_c_min: float = config.X_C_MIN,
    tol: float = 1e-14,
    max_iter: int = 200,
) -> float:
    if v_cut <= 0.0:
        raise ValueError(f"v_cut must be positive, got {v_cut}")
    if target_second_moment <= 0.0:
        raise ValueError(f"target_second_moment must be positive, got {target_second_moment}")

    def moment(sigma: float) -> float:
        return sigma * sigma * truncated_second_moment_factor(v_cut / sigma)

    hi = v_cut / x_c_min
    lo = hi * 1.0e-12
    moment_hi = moment(hi)
    if moment_hi < target_second_moment:
        raise ValueError(
            f"target second moment {target_second_moment} is not reachable within the "
            f"non-degenerate regime x_c = v_cut/sigma >= {x_c_min} "
            f"(moment at x_c_min is {moment_hi}); Q exceeds the Q_USABLE_COEFF * f_cut**2 bound "
            "and should have been rejected before calling solve_sigma"
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if moment(mid) > target_second_moment:
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= tol * hi:
            break

    return 0.5 * (lo + hi)


def sample_velocities(
    state,
    *,
    q: float = config.Q_DEFAULT,
    f_cut: float = config.F_CUT_DEFAULT,
    radius: float = config.SPHERE_RADIUS,
    softening: float = config.SOFTENING,
    g: float = config.G,
    seed: int = config.VEL_SEED,
) -> tuple[np.ndarray, float]:
    m = state.m.detach().to("cpu", dtype=torch.float64).numpy()
    n = m.shape[0]

    if q == 0.0:
        return np.zeros((n, 3), dtype=np.float64), 1.0

    q_max = config.Q_USABLE_COEFF * f_cut**2
    if q > q_max:
        raise ValueError(
            f"virial ratio Q={q} exceeds the non-degeneracy bound Q_max={q_max} for f_cut={f_cut} "
            f"(Q_USABLE_COEFF * f_cut**2, requiring x_c = v_cut/sigma >= {config.X_C_MIN}); "
            "the truncated Maxwellian degenerates into the uniform velocity ball at this point"
        )

    u_potential = observables.potential_energy(state, softening=softening)
    if u_potential >= 0.0:
        raise ValueError(
            f"sample_velocities expects a bound configuration (negative potential energy), "
            f"got U={u_potential}"
        )
    u_abs = -u_potential

    m_total = m.sum()
    v_esc = math.sqrt(2.0 * g * m_total / radius)
    v_cut = f_cut * v_esc
    target_v2 = q * u_abs / m_total

    sigma = solve_sigma(v_cut, target_v2)

    rng_v = np.random.default_rng(seed)

    v = np.empty((n, 3), dtype=np.float64)
    pending = np.arange(n)
    while pending.size > 0:
        draw = rng_v.normal(loc=0.0, scale=sigma, size=(pending.size, 3))
        speed = np.sqrt((draw * draw).sum(axis=1))
        accepted = speed <= v_cut
        v[pending[accepted]] = draw[accepted]
        pending = pending[~accepted]

    v_mean = (m[:, None] * v).sum(axis=0) / m_total
    v = v - v_mean

    speed_sq = (v * v).sum(axis=1)
    k_prime = 0.5 * float((m * speed_sq).sum())
    k_target = 0.5 * q * u_abs
    lam = math.sqrt(k_target / k_prime)
    v = v * lam

    return v, lam
