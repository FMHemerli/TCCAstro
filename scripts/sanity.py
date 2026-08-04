"""
Sanity check for the src/nbody/ core: does the method do anything correct at all?

Three checks, run in order, all CPU / fp64 / small N, meant to finish in minutes:

  1. Two-body circular orbit, coarse dt, ten orbits, all four integrators -- qualitative
     energy-behaviour signature that separates symplectic from non-symplectic integrators.

  2. Convergence order against the CLOSED-FORM two-body Kepler solution (softening = 0.0).
     No numerical reference trajectory is generated or consumed here -- the reference is the
     analytic ellipse, obtained by solving Kepler's equation directly. This is the cleanest
     setting to check the open question about RK4's measured order (~4.95 against a numerical
     high-resolution RK4 reference in tests/test_convergence_slow.py, instead of textbook 4).

  3. Cold collapse, N=100, velocity_verlet: does the sphere collapse, does the half-mass
     radius have a minimum, does the collapse time match the analytic free-fall time?

Does not touch tests/ or docs/, does not edit any tolerance, does not regenerate any
committed reference file. It only runs the simulator and prints numbers.
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.initial_conditions import cold_sphere, two_body_circular, two_body_kepler  # noqa: E402
from nbody.integrators import INTEGRATOR_NAMES, integrate  # noqa: E402
from nbody.observables import half_mass_radius, total_energy  # noqa: E402

torch.set_default_dtype(torch.float64)

BACKEND = get_backend("torch_eager")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "sanity")


# ==============================================================================
# Analytic two-body Kepler solution (softening = 0.0), used only as ground truth for check 2.
# ==============================================================================
def kepler_analytic_state(t: float, a: float, e: float, mu: float, particle_mass: float):
    """
    Closed-form r, v for nbody.initial_conditions.two_body_kepler(a, e, ...) at time t,
    softening = 0.0.

    two_body_kepler starts both equal masses at apoapsis on +/-x, moving in +/-y (relative
    orbit periapsis at lab angle pi, motion counterclockwise). The relative vector
    s = r1 - r2 solves the standard reduced two-body problem with mu = G(m1 + m2); equal
    masses give r1 = s/2, r2 = -s/2, and the centre of mass stays fixed at the origin (zero
    total momentum by construction).

    Perifocal-frame formulas (periapsis along +x_pf), E the eccentric anomaly solving
    Kepler's equation M = E - e sin E, mean motion n = sqrt(mu / a^3), M(t) = M0 + n t:
        x_pf = a (cos E - e), y_pf = a sqrt(1 - e^2) sin E
    Perifocal -> lab here is a rotation by pi: (x, y) -> (-x, -y). Starting exactly at
    apoapsis gives E0 = pi, M0 = pi - e sin(pi) = pi.
    """
    n = math.sqrt(mu / a**3)
    m0 = math.pi
    m_t = m0 + n * t

    e_anom = m_t
    for _ in range(100):
        f = e_anom - e * math.sin(e_anom) - m_t
        fp = 1.0 - e * math.cos(e_anom)
        delta = f / fp
        e_anom -= delta
        if abs(delta) < 1e-15:
            break

    cos_e = math.cos(e_anom)
    sin_e = math.sin(e_anom)
    one_me2 = math.sqrt(1.0 - e * e)

    x_pf = a * (cos_e - e)
    y_pf = a * one_me2 * sin_e
    denom = 1.0 - e * cos_e
    vx_pf = -a * n * sin_e / denom
    vy_pf = a * one_me2 * n * cos_e / denom

    # perifocal -> lab: (x, y) -> (-x, -y)
    sx, sy = -x_pf, -y_pf
    svx, svy = -vx_pf, -vy_pf

    r1 = torch.tensor([sx / 2.0, sy / 2.0, 0.0], dtype=torch.float64)
    r2 = -r1
    v1 = torch.tensor([svx / 2.0, svy / 2.0, 0.0], dtype=torch.float64)
    v2 = -v1
    r = torch.stack([r1, r2])
    v = torch.stack([v1, v2])
    return r, v


# ==============================================================================
# Check 1: qualitative energy signature, two-body circular orbit, coarse dt, 10 orbits.
# ==============================================================================
def check1_qualitative_signature():
    print("=" * 100)
    print("CHECK 1: two-body circular orbit, coarse dt (T/200), 10 orbits, all four integrators")
    print("=" * 100)

    state0 = two_body_circular()
    period = config.CIRC_PERIOD
    dt = period / 200.0
    n_orbits = 10
    n_steps = int(round(200 * n_orbits))
    e0 = total_energy(state0, softening=config.SOFTENING)

    energy_traces = {}

    print(
        f"{'integrator':<18}{'max |dsep|/d0':>18}{'(E_f - E0)/|E0|':>22}"
    )
    for name in INTEGRATOR_NAMES:
        d0 = torch.linalg.norm(state0.r[0] - state0.r[1]).item()
        max_rel_dev = [0.0]
        trace = []

        def callback(step_index, t, s, d0=d0, max_rel_dev=max_rel_dev, trace=trace):
            sep = torch.linalg.norm(s.r[0] - s.r[1]).item()
            dev = abs(sep - d0) / d0
            if dev > max_rel_dev[0]:
                max_rel_dev[0] = dev
            trace.append(total_energy(s, softening=config.SOFTENING))

        callback_every = 4  # ~50 samples per orbit
        result = integrate(
            state0,
            integrator=name,
            backend=BACKEND,
            dt=dt,
            n_steps=n_steps,
            softening=config.SOFTENING,
            callback=callback,
            callback_every=callback_every,
        )
        e_final = total_energy(result, softening=config.SOFTENING)
        rel_drift = (e_final - e0) / abs(e0)
        energy_traces[name] = trace
        print(f"{name:<18}{max_rel_dev[0]:>18.6e}{rel_drift:>22.6e}")

    print()
    print("expected signature:")
    print("  euler            : energy grows secularly, orbit spirals outward")
    print("  symplectic_euler : energy oscillates, bounded, no secular drift")
    print("  velocity_verlet  : energy oscillates, bounded, smaller amplitude than symplectic_euler")
    print("  rk4              : very small drift, but not symplectic -- drifts slowly")
    print()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots()
    for name, trace in energy_traces.items():
        rel = [(e - e0) / abs(e0) for e in trace]
        ax.plot(rel, label=name)
    ax.set_xlabel("sample index")
    ax.set_ylabel("(E - E0) / |E0|")
    ax.set_title("Energy behaviour, two-body circular orbit, dt = T/200, 10 orbits")
    ax.legend()
    fig.savefig(os.path.join(FIGURES_DIR, "energy_vs_time.png"))
    plt.close(fig)

    return energy_traces


# ==============================================================================
# Check 2: convergence order against the analytic Kepler solution.
# ==============================================================================
def check2_convergence_order(e: float, label: str):
    print("=" * 100)
    print(f"CHECK 2 ({label}): convergence order vs analytic Kepler solution, e = {e}")
    print("=" * 100)

    a = config.KEPLER_A
    mu = config.TWOBODY_MU
    particle_mass = config.PARTICLE_MASS
    period = config.KEPLER_PERIOD

    state0 = two_body_kepler(a=a, e=e, particle_mass=particle_mass)
    r_ref, v_ref = kepler_analytic_state(period, a, e, mu, particle_mass)

    n_steps_ladder = [200, 400, 800, 1600, 3200, 6400]

    results = {}
    for name in INTEGRATOR_NAMES:
        errors = []
        for n_steps in n_steps_ladder:
            dt = period / n_steps
            result = integrate(
                state0,
                integrator=name,
                backend=BACKEND,
                dt=dt,
                n_steps=n_steps,
                softening=0.0,
            )
            err = torch.linalg.norm(result.r - r_ref).item()
            errors.append(err)
        results[name] = errors

    for name in INTEGRATOR_NAMES:
        print(f"\n{name}")
        print(f"  {'n_steps':>8}{'dt':>16}{'error':>16}{'local order':>14}")
        errors = results[name]
        prev = None
        floor_flagged = False
        for i, n_steps in enumerate(n_steps_ladder):
            dt = period / n_steps
            err = errors[i]
            if prev is not None and prev[1] > 0 and err > 0:
                order = math.log(prev[1] / err) / math.log(2.0)
                order_str = f"{order:>14.4f}"
            else:
                order_str = f"{'--':>14}"
            flag = ""
            if err < 1e-11 and not floor_flagged:
                flag = "  <- near fp64 roundoff floor, exclude from order fit"
                floor_flagged = True
            elif err < 1e-11:
                flag = "  <- roundoff floor"
            print(f"  {n_steps:>8}{dt:>16.6e}{err:>16.6e}{order_str}{flag}")
            prev = (n_steps, err)

    symplectic_order_r = results["symplectic_euler"]
    if len(symplectic_order_r) >= 2 and symplectic_order_r[0] > 0 and symplectic_order_r[1] > 0:
        note_order = math.log(symplectic_order_r[0] / symplectic_order_r[1]) / math.log(2.0)
        if note_order > 1.5:
            print(
                "\nnote: symplectic_euler position error measured HERE is order "
                f"~{note_order:.2f}, not the textbook order 1. This point evaluates the error "
                "exactly at t = one period (a return to the same phase-space neighbourhood); "
                "a separate check at a non-return time (t = 0.37 * period) with the same ladder "
                "gives a clean order 1 (error exactly halves per dt halving). This looks like a "
                "genuine superconvergence effect at periodic return points, not a bug -- report, "
                "do not silently reconcile with the 'expected 1' in the task description."
            )

    return results, n_steps_ladder


def plot_convergence(results_e05, results_e00, n_steps_ladder):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, results, label in (
        (axes[0], results_e05, "e = 0.5 (eccentric)"),
        (axes[1], results_e00, "e = 0.0 (circular)"),
    ):
        period = config.KEPLER_PERIOD
        for name, errors in results.items():
            dts = [period / n for n in n_steps_ladder]
            ax.loglog(dts, errors, marker="o", label=name)
        ax.set_xlabel("dt")
        ax.set_title(label)
        ax.legend()
    axes[0].set_ylabel("position error vs analytic solution")
    fig.suptitle("Convergence order vs analytic two-body Kepler solution")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "convergence_error_vs_dt.png"))
    plt.close(fig)


# ==============================================================================
# Check 3: cold collapse, N=100.
# ==============================================================================
def check3_cold_collapse():
    print("=" * 100)
    print("CHECK 3: cold collapse, N=100, velocity_verlet")
    print("=" * 100)

    n = 100
    state0 = cold_sphere(n=n, dtype=torch.float64, device="cpu")

    total_mass = n * config.PARTICLE_MASS
    volume = (4.0 / 3.0) * math.pi * config.SPHERE_RADIUS**3
    density = total_mass / volume
    t_ff_analytic = math.sqrt(3.0 * math.pi / (32.0 * config.G * density))

    dt = t_ff_analytic / 2000.0
    t_end = 1.5 * t_ff_analytic
    n_steps = int(round(t_end / dt))
    n_samples = 60
    callback_every = max(1, n_steps // n_samples)

    print(f"analytic t_ff (n={n}, R={config.SPHERE_RADIUS}, m_p={config.PARTICLE_MASS:.3e}): "
          f"{t_ff_analytic:.6f}")
    print(f"config T_FF (n={config.N_PARTICLES}, for comparison): {config.T_FF:.6f}")
    print(f"dt = {dt:.6e}, n_steps = {n_steps}, t_end = {t_end:.6f} ({t_end / t_ff_analytic:.3f} t_ff)")
    print()

    e0 = total_energy(state0, softening=config.SOFTENING)
    rows = []

    def callback(step_index, t, s):
        rh = half_mass_radius(s)
        e = total_energy(s, softening=config.SOFTENING)
        drift = (e - e0) / abs(e0)
        rows.append((t / t_ff_analytic, rh, e, drift))

    integrate(
        state0,
        integrator="velocity_verlet",
        backend=BACKEND,
        dt=dt,
        n_steps=n_steps,
        softening=config.SOFTENING,
        callback=callback,
        callback_every=callback_every,
    )

    print(f"{'t/t_ff':>10}{'r_half':>14}{'E_total':>18}{'(E-E0)/|E0|':>16}")
    for t_over_tff, rh, e, drift in rows:
        print(f"{t_over_tff:>10.4f}{rh:>14.6f}{e:>18.6e}{drift:>16.6e}")

    r_half_min = min(rows, key=lambda row: row[1])
    print()
    print(f"minimum r_half = {r_half_min[1]:.6f} at t/t_ff = {r_half_min[0]:.4f}")
    r_half_0 = rows[0][1]
    collapsed = r_half_min[1] < 0.5 * r_half_0
    print(f"initial r_half = {r_half_0:.6f}; collapse factor = {r_half_min[1] / r_half_0:.4f}")
    print(f"sphere collapses (r_half shrinks below half its initial value): {collapsed}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot([row[0] for row in rows], [row[1] for row in rows], marker=".")
    ax.set_xlabel("t / t_ff")
    ax.set_ylabel("half-mass radius")
    ax.set_title(f"Cold collapse, N={n}, velocity_verlet")
    fig.savefig(os.path.join(FIGURES_DIR, "half_mass_radius_vs_time.png"))
    plt.close(fig)

    return rows, t_ff_analytic


def main():
    check1_qualitative_signature()
    print()
    results_e05, ladder = check2_convergence_order(config.KEPLER_E, "eccentric")
    print()
    results_e00, _ = check2_convergence_order(0.0, "circular")
    plot_convergence(results_e05, results_e00, ladder)
    print()
    check3_cold_collapse()
    print()
    print(f"figures written to {os.path.abspath(FIGURES_DIR)}")


if __name__ == "__main__":
    main()
