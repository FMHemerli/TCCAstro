"""
RUN_COLLAPSE-scale invariants: docs/integradores.md INV-4 (energy behaviour), INV-9 (t_collapse
vs t_ff), the RUN_COLLAPSE variant of INV-3 (angular momentum, normalized by L_SCALE), and the
operational form of INV-2(c) (momentum drift bound over a full run).

Uses N=1000, dt=DT_COLLAPSE=5e-4 s, n_steps=N_STEPS_COLLAPSE=12600 (T_END = 3 t_ff), exactly
Section 4.5's RUN_COLLAPSE. Each integrator is run exactly once (fp64); the run is shared across
INV-4, INV-2(c), and the collapse variant of INV-3 via a single callback that samples energy,
momentum, angular momentum and half-mass radius together at the OUT_DT cadence (every 20 steps),
so this file does four full collapses in fp64, plus one extra fp32 collapse with
velocity_verlet for INV-9's fp32 cross-check (mirroring the document's own choice to validate
fp32 against velocity_verlet specifically, Section 7 INV-9).

Marked slow: N=1000 is O(N^2)=10^6 pairwise terms per force evaluation; 12600 steps per
integrator (50400 for rk4's 4 evals/step) means on the order of 10^10-10^11 pairwise-interaction
operations total across this file. Expect several minutes on CPU. Deselect with
`-m "not slow"` for per-commit CI.
"""
import math

import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.integrators import integrate  # noqa: E402
from nbody.observables import (  # noqa: E402
    angular_momentum,
    half_mass_radius,
    kinetic_energy,
    linear_momentum,
    total_energy,
)

from tolerances import (  # noqa: E402
    INV3_COLLAPSE_L_EULER_MIN,
    INV3_COLLAPSE_L_RK4_MAX,
    INV3_COLLAPSE_L_RK4_MIN,
    INV3_COLLAPSE_L_SYMPLECTIC_MAX,
    INV3_COLLAPSE_L_VERLET_MAX,
    INV4_EULER_FINAL_MIN,
    INV4_OSCILLATORY_FINAL_OVER_PEAK_MAX,
    INV4_RK4_FINAL_OVER_PEAK_MIN,
    INV4_RK4_PEAK_MAX,
    INV4_SYMPLECTIC_PEAK_MAX,
    INV4_VERLET_PEAK_MAX,
    INV9_FP32_RHALF_MIN_REL_TOL,
    INV9_FP32_TCOLLAPSE_ABS_TOL_TFF_FRACTION,
    INV9_RHALF_MIN_HIGH,
    INV9_RHALF_MIN_LOW,
    INV9_TCOLLAPSE_OVER_TFF_TOL,
    TOL_MOM_DRIFT_FP32,
    TOL_MOM_DRIFT_FP64,
)

pytestmark = pytest.mark.slow

BACKEND = get_backend("torch_eager")


def run_collapse(integrator, dtype):
    state = cold_sphere(n=config.N_PARTICLES, seed=config.SEED, dtype=dtype, device="cpu")
    e0 = total_energy(state, softening=config.SOFTENING)
    l0 = angular_momentum(state)
    m_tot = state.m.sum().item()

    times, energies, p_over_bound, l_norms, r_halfs = [], [], [], [], []

    def callback(step_index, time, s):
        times.append(time)
        energies.append(total_energy(s, softening=config.SOFTENING))
        p_norm = torch.linalg.norm(linear_momentum(s)).item()
        k = kinetic_energy(s)
        v_rms = math.sqrt(max(2.0 * k / m_tot, 0.0))
        denom = m_tot * v_rms if v_rms > 0 else float("inf")
        p_over_bound.append(p_norm / denom if denom > 0 else 0.0)
        l_norms.append(torch.linalg.norm(angular_momentum(s)).item())
        r_halfs.append(half_mass_radius(s))

    callback_every = int(round(config.OUT_DT / config.DT_COLLAPSE))
    final = integrate(
        state, integrator=integrator, backend=BACKEND, dt=config.DT_COLLAPSE,
        n_steps=config.N_STEPS_COLLAPSE, softening=config.SOFTENING, callback=callback,
        callback_every=callback_every,
    )
    return {
        "e0": e0,
        "l0_norm": torch.linalg.norm(l0).item(),
        "times": times,
        "rel_energy": [(e - e0) / abs(e0) for e in energies],
        "p_over_bound": p_over_bound,
        "l_norms": l_norms,
        "r_halfs": r_halfs,
        "final": final,
    }


def parabolic_t_collapse(times, r_halfs):
    k_star = min(range(len(r_halfs)), key=lambda i: r_halfs[i])
    assert 0 < k_star < len(r_halfs) - 1, (
        "t_collapse minimum found at the sampling grid boundary; cannot parabolically refine "
        "(RUN_COLLAPSE should show the minimum well inside [0, 3 t_ff])"
    )
    y_minus, y_0, y_plus = r_halfs[k_star - 1], r_halfs[k_star], r_halfs[k_star + 1]
    out_dt = times[k_star] - times[k_star - 1]
    denom = y_minus - 2 * y_0 + y_plus
    t_grid = times[k_star]
    if denom == 0:
        return t_grid, y_0
    t_refined = t_grid + (out_dt / 2.0) * (y_minus - y_plus) / denom
    return t_refined, y_0


@pytest.fixture(scope="module")
def euler_run():
    return run_collapse("euler", torch.float64)


@pytest.fixture(scope="module")
def symplectic_run():
    return run_collapse("symplectic_euler", torch.float64)


@pytest.fixture(scope="module")
def verlet_run():
    return run_collapse("velocity_verlet", torch.float64)


@pytest.fixture(scope="module")
def rk4_run():
    return run_collapse("rk4", torch.float64)


class TestINV4EnergyBehavior:
    def test_euler_grows_monotonically_to_at_least_0_3(self, euler_run):
        trace = euler_run["rel_energy"]
        assert trace[-1] >= INV4_EULER_FINAL_MIN, f"final ΔE/E0={trace[-1]:.3e}"
        violations = sum(1 for a, b in zip(trace, trace[1:]) if b < a)
        # "monotono" per Section 7; allow a small number of non-monotonic samples out of ~630
        # to absorb any single-ULP-scale noise in the observable itself, without weakening the
        # secular-growth claim.
        assert violations <= 2, (
            f"euler's energy trace should be monotonically increasing (Section 7, INV-4); "
            f"found {violations} decreasing steps"
        )

    def test_symplectic_euler_peak_and_final_over_peak(self, symplectic_run):
        trace = symplectic_run["rel_energy"]
        peak = max(abs(x) for x in trace)
        assert peak <= INV4_SYMPLECTIC_PEAK_MAX, f"peak={peak:.3e}"
        assert abs(trace[-1]) <= peak * INV4_OSCILLATORY_FINAL_OVER_PEAK_MAX, (
            f"final={trace[-1]:.3e}, peak={peak:.3e}"
        )

    def test_velocity_verlet_peak_and_final_over_peak(self, verlet_run):
        trace = verlet_run["rel_energy"]
        peak = max(abs(x) for x in trace)
        assert peak <= INV4_VERLET_PEAK_MAX, f"peak={peak:.3e}"
        assert abs(trace[-1]) <= peak * INV4_OSCILLATORY_FINAL_OVER_PEAK_MAX, (
            f"final={trace[-1]:.3e}, peak={peak:.3e}"
        )

    def test_rk4_peak_and_final_negative_and_comparable_to_peak(self, rk4_run):
        trace = rk4_run["rel_energy"]
        peak = max(abs(x) for x in trace)
        assert peak <= INV4_RK4_PEAK_MAX, f"peak={peak:.3e}"
        assert trace[-1] <= 0, f"final={trace[-1]:.3e} should be negative"
        assert abs(trace[-1]) >= peak * INV4_RK4_FINAL_OVER_PEAK_MIN, (
            f"final={trace[-1]:.3e}, peak={peak:.3e}"
        )


class TestINV2cMomentumDriftBound:
    @pytest.mark.parametrize("run_fixture", ["euler_run", "symplectic_run", "verlet_run", "rk4_run"])
    def test_momentum_stays_within_operational_bound(self, run_fixture, request):
        run = request.getfixturevalue(run_fixture)
        max_ratio = max(run["p_over_bound"])
        assert max_ratio <= TOL_MOM_DRIFT_FP64, (
            f"{run_fixture}: max ||P(t)||/(M_tot v_rms(t)) = {max_ratio:.3e} exceeds "
            f"{TOL_MOM_DRIFT_FP64:.1e}"
        )


class TestINV3CollapseAngularMomentum:
    def test_velocity_verlet(self, verlet_run):
        drift = verlet_run["l_norms"][-1] / config.L_SCALE
        assert drift <= INV3_COLLAPSE_L_VERLET_MAX, f"drift={drift:.3e}"

    def test_symplectic_euler(self, symplectic_run):
        drift = symplectic_run["l_norms"][-1] / config.L_SCALE
        assert drift <= INV3_COLLAPSE_L_SYMPLECTIC_MAX, f"drift={drift:.3e}"

    def test_rk4(self, rk4_run):
        drift = rk4_run["l_norms"][-1] / config.L_SCALE
        assert INV3_COLLAPSE_L_RK4_MIN <= drift <= INV3_COLLAPSE_L_RK4_MAX, f"drift={drift:.3e}"

    def test_euler(self, euler_run):
        drift = euler_run["l_norms"][-1] / config.L_SCALE
        assert drift >= INV3_COLLAPSE_L_EULER_MIN, f"drift={drift:.3e}"


class TestINV9CollapseTime:
    def test_t_collapse_over_t_ff_fp64(self, verlet_run):
        t_collapse, r_half_min = parabolic_t_collapse(verlet_run["times"], verlet_run["r_halfs"])
        ratio = t_collapse / config.T_FF
        assert abs(ratio - 1.0) <= INV9_TCOLLAPSE_OVER_TFF_TOL, (
            f"t_collapse/t_ff={ratio:.4f}, expected within {INV9_TCOLLAPSE_OVER_TFF_TOL} of 1.0"
        )
        assert INV9_RHALF_MIN_LOW <= r_half_min <= INV9_RHALF_MIN_HIGH, (
            f"r_half_min={r_half_min:.4f} m outside [{INV9_RHALF_MIN_LOW}, {INV9_RHALF_MIN_HIGH}]"
        )

    def test_fp32_cross_check_against_fp64(self, verlet_run):
        fp32_run = run_collapse("velocity_verlet", torch.float32)
        t64, rhalf64 = parabolic_t_collapse(verlet_run["times"], verlet_run["r_halfs"])
        t32, rhalf32 = parabolic_t_collapse(fp32_run["times"], fp32_run["r_halfs"])

        abs_tol = INV9_FP32_TCOLLAPSE_ABS_TOL_TFF_FRACTION * config.T_FF
        assert abs(t32 - t64) <= abs_tol, (
            f"t_collapse fp32={t32:.4f}s vs fp64={t64:.4f}s, diff exceeds {abs_tol:.4f}s"
        )
        rel_diff = abs(rhalf32 / rhalf64 - 1.0)
        assert rel_diff <= INV9_FP32_RHALF_MIN_REL_TOL, (
            f"r_half_min fp32={rhalf32:.4f} vs fp64={rhalf64:.4f}, relative diff "
            f"{rel_diff:.4f} exceeds {INV9_FP32_RHALF_MIN_REL_TOL}"
        )

        # Section 7, INV-4: the *peak* of |ΔE/E0| is reproducible in fp32 to ~0.1%, even though
        # trajectories decorrelate; a loose cross-check here, since this file already spends the
        # cost of a full fp32 collapse for the t_collapse check above.
        peak64 = max(abs(x) for x in verlet_run["rel_energy"])
        peak32 = max(abs(x) for x in fp32_run["rel_energy"])
        assert peak32 == pytest.approx(peak64, rel=0.05)

        # INV-2(c) operational bound, fp32 variant, reusing this same fp32 run (Section 7).
        max_ratio_fp32 = max(fp32_run["p_over_bound"])
        assert max_ratio_fp32 <= TOL_MOM_DRIFT_FP32, (
            f"fp32 velocity_verlet: max ||P(t)||/(M_tot v_rms(t)) = {max_ratio_fp32:.3e} "
            f"exceeds {TOL_MOM_DRIFT_FP32:.1e}"
        )
