"""
Two-body invariants: docs/integradores.md INV-3, INV-5, INV-6, and a fast qualitative subset of
INV-7 (the full multi-level amplitude-ratio table for INV-7 is expensive -- 20 periods at four
step counts per integrator -- and lives in test_energy_amplitude_slow.py, marked slow).

INV-3 here uses the exact configuration the document's numeric thresholds were measured at
(TWOBODY_ECC, 2000 steps/period, 20 periods, fp64): shortening it would invalidate the
thresholds, since the RK4 window [1e-13, 1e-9] and the symplectic bound are tied to that specific
duration. N=2 keeps each force evaluation trivial, so despite the 40000-step count this stays a
per-commit-affordable test on CPU.
"""
import math

import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.initial_conditions import two_body_circular, two_body_kepler  # noqa: E402
from nbody.integrators import integrate  # noqa: E402
from nbody.observables import angular_momentum, total_energy  # noqa: E402

from tolerances import (  # noqa: E402
    INV3_L_EULER_MIN,
    INV3_L_RK4_MAX,
    INV3_L_RK4_MIN,
    INV3_L_SYMPLECTIC_EULER_MAX,
    INV3_L_VELOCITY_VERLET_MAX,
    INV5_RETURN_ERROR_MAX_AT_1000_STEPS,
    INV5_RETURN_ERROR_RATIO_MAX,
    INV5_RETURN_ERROR_RATIO_MIN,
    INV6_EPI_MAX_OMEGA_DT,
    INV6_EPI_RATIO_MAX,
    INV6_EPI_RATIO_MIN,
    INV6_EPS,
    INV6_GAMMA,
    TOL_ENERGY_FP64,
)

# Not a table value: the document gives no explicit tolerance for ||L|| vs. the analytic value
# at t=0 in INV-5. This is a single fp64 evaluation of a cross product and a two-term sum, so
# it is held to the same order of precision as TOL-ACCEL/TOL-GRAD single-evaluation checks
# elsewhere in this suite (EXTRAPOLATED, not transcribed from the spec).
INV5_L_AT_IC_REL_TOL = 1e-9

BACKEND = get_backend("torch_eager")


# =============================================================================================
# INV-5: two-body Kepler, eps = 0.0 exactly.
# =============================================================================================
class TestINV5Kepler:
    E_EXACT = -config.G * config.PARTICLE_MASS**2 / (2 * config.KEPLER_A)
    L_EXACT = (config.PARTICLE_MASS / 2) * (config.KEPLER_A * (1 + config.KEPLER_E)) * 0.2109356932

    def _return_error(self, steps_per_period):
        state = two_body_kepler()
        dt = config.KEPLER_PERIOD / steps_per_period
        result = integrate(state, integrator="velocity_verlet", backend=BACKEND, dt=dt,
                            n_steps=steps_per_period, softening=0.0)
        max_dev = torch.linalg.norm(result.r - state.r, dim=1).max().item()
        r_apo = config.KEPLER_A * (1 + config.KEPLER_E)
        return max_dev / r_apo

    def test_return_error_at_1000_steps_per_period(self):
        rel_err = self._return_error(1000)
        assert rel_err <= INV5_RETURN_ERROR_MAX_AT_1000_STEPS, (
            f"INV-5: return error {rel_err:.3e} exceeds "
            f"{INV5_RETURN_ERROR_MAX_AT_1000_STEPS:.1e}"
        )

    def test_return_error_scales_as_h_squared(self):
        err_1000 = self._return_error(1000)
        err_5000 = self._return_error(5000)
        ratio = err_1000 / err_5000
        assert INV5_RETURN_ERROR_RATIO_MIN <= ratio <= INV5_RETURN_ERROR_RATIO_MAX, (
            f"INV-5: h^2 scaling ratio {ratio:.2f} outside "
            f"[{INV5_RETURN_ERROR_RATIO_MIN}, {INV5_RETURN_ERROR_RATIO_MAX}]"
        )

    def test_energy_at_ic_matches_analytic_value(self):
        state = two_body_kepler()
        e = total_energy(state, softening=0.0)
        assert e == pytest.approx(self.E_EXACT, rel=TOL_ENERGY_FP64 * 100), (
            f"E(0)={e:.6e} vs E_exact={self.E_EXACT:.6e}"
        )

    def test_angular_momentum_at_ic_matches_analytic_value(self):
        state = two_body_kepler()
        l_vec = angular_momentum(state)
        l_mag = torch.linalg.norm(l_vec).item()
        assert l_mag == pytest.approx(self.L_EXACT, rel=INV5_L_AT_IC_REL_TOL), (
            f"||L(0)||={l_mag:.6e} vs analytic {self.L_EXACT:.6e}"
        )


# =============================================================================================
# INV-6: two-body circular, softened (eps = 0.05).
# =============================================================================================
class TestINV6CircularSoftened:
    def test_epicyclic_amplitude_matches_the_predicted_value(self):
        """
        The softened circular orbit is NOT integrated as a circle: velocity_verlet places the pair
        on an epicycle of radial amplitude (omega*dt)^2 / (4*gamma). docs/integradores.md, TOL-EPI.

        Asserting a fixed bound on that amplitude, as the retracted INV6_SEPARATION_REL_TOL = 1e-6
        did, is an assertion about dt rather than about the code: the same correct implementation
        fails it at spp <= 4426 and passes at spp >= 4427. What is a statement about the code is
        the ratio of measured amplitude to predicted amplitude, which is dimensionless and
        dt-independent inside the domain of validity.

        Two ways to get this wrong against correct code, both called out in the document:
        the amplitude must be sampled EVERY step (at 40 samples per epicycle the extremum is
        missed by ~0.3%), and it is the deviation from d, not from the discrete orbit's own radius
        R_h > d -- t = 0 sits at the minimum of the oscillation, so the maximum deviation is twice
        the radial eccentricity, and measuring against R_h would halve it and give ratio 0.5.
        """
        spp = 2000
        n_periods = 10
        d0 = config.CIRC_SEPARATION
        dt = config.CIRC_PERIOD / spp
        n_steps = spp * n_periods

        omega = 2.0 * math.pi / config.CIRC_PERIOD
        assert omega * dt <= INV6_EPI_MAX_OMEGA_DT, (
            f"INV-6: omega*dt = {omega * dt:.4f} is outside TOL-EPI's domain of validity "
            f"({INV6_EPI_MAX_OMEGA_DT}); the second-order prediction does not apply"
        )

        max_relative_dev = [0.0]

        def callback(step_index, time, s):
            sep = torch.linalg.norm(s.r[0] - s.r[1]).item()
            dev = abs(sep - d0) / d0
            if dev > max_relative_dev[0]:
                max_relative_dev[0] = dev

        integrate(state := two_body_circular(), integrator="velocity_verlet", backend=BACKEND,
                  dt=dt, n_steps=n_steps, softening=INV6_EPS,
                  callback=callback, callback_every=1)

        a_meas = max_relative_dev[0]
        a_epi = (omega * dt) ** 2 / (4.0 * INV6_GAMMA)
        f_prec = 4.0 * math.sqrt(n_steps) * 2.220446049250313e-16
        ratio = (a_meas - f_prec) / a_epi

        assert INV6_EPI_RATIO_MIN <= ratio <= INV6_EPI_RATIO_MAX, (
            f"INV-6 / TOL-EPI: measured epicyclic amplitude {a_meas:.6e} against predicted "
            f"{a_epi:.6e} gives ratio {ratio:.4f}, outside "
            f"[{INV6_EPI_RATIO_MIN}, {INV6_EPI_RATIO_MAX}]"
        )

    def test_softened_period_differs_from_kepler_period_by_expected_amount(self):
        # Discriminating power of INV-6 (Section 7): with eps=0 the period would be
        # KEPLER_PERIOD, a 0.19% difference from CIRC_PERIOD. This is a sanity check on the
        # fixture values themselves, not on integrate().
        rel_diff = abs(config.CIRC_PERIOD - config.KEPLER_PERIOD) / config.KEPLER_PERIOD
        assert rel_diff == pytest.approx(0.0019, abs=2e-4)


# =============================================================================================
# INV-3: angular momentum conservation, TWOBODY_ECC, eps = 0.05.
# =============================================================================================
class TestINV3AngularMomentum:
    STEPS_PER_PERIOD = 2000
    N_PERIODS = 20

    def _run(self, integrator):
        # TWOBODY_ECC (Section 7, INV-7): identical r, v to TWOBODY_KEPLER (Section 4.8: "as
        # condicoes iniciais permanecem as de INV-5"); only eps changes, from 0.0 to 0.05.
        state = two_body_kepler()
        dt = config.KEPLER_PERIOD / self.STEPS_PER_PERIOD
        n_steps = self.STEPS_PER_PERIOD * self.N_PERIODS
        l0 = angular_momentum(state)
        l0_norm = torch.linalg.norm(l0).item()
        result = integrate(state, integrator=integrator, backend=BACKEND, dt=dt,
                            n_steps=n_steps, softening=0.05)
        l_final = angular_momentum(result)
        drift = torch.linalg.norm(l_final - l0).item() / l0_norm
        return drift

    def test_symplectic_euler_conserves_l(self):
        drift = self._run("symplectic_euler")
        assert drift <= INV3_L_SYMPLECTIC_EULER_MAX, f"drift={drift:.3e}"

    def test_velocity_verlet_conserves_l(self):
        drift = self._run("velocity_verlet")
        assert drift <= INV3_L_VELOCITY_VERLET_MAX, f"drift={drift:.3e}"

    def test_rk4_l_drift_within_window(self):
        drift = self._run("rk4")
        assert INV3_L_RK4_MIN <= drift <= INV3_L_RK4_MAX, (
            f"drift={drift:.3e} outside [{INV3_L_RK4_MIN:.1e}, {INV3_L_RK4_MAX:.1e}]"
        )

    def test_euler_l_drift_is_large(self):
        drift = self._run("euler")
        assert drift >= INV3_L_EULER_MIN, f"drift={drift:.3e}"

    def test_symplectics_conserve_l_orders_of_magnitude_better_than_euler(self):
        symp_drift = self._run("symplectic_euler")
        euler_drift = self._run("euler")
        assert euler_drift / symp_drift >= 1e6, (
            "the ~12-order-of-magnitude separation between symplectic and explicit Euler "
            "(Section 7, INV-3) should be clearly visible even with a loose margin"
        )


# =============================================================================================
# INV-7 (fast qualitative subset only -- see test_energy_amplitude_slow.py for the full table).
# =============================================================================================
class TestINV7QualitativeEnergyBehavior:
    """
    Cheap, single-step-count version of INV-7's qualitative claims (not the precise amplitude
    ratios, which require the full multi-level sweep at 1000/2000/4000/8000 steps per period --
    see test_energy_amplitude_slow.py). Deliberately avoids comparing the single last sampled
    point to the peak (that comparison is phase-dependent at only a handful of periods, and can
    look arbitrarily good or bad depending on exactly where in the oscillation cycle the run
    happens to end): instead it compares the oscillation envelope (max |ΔE/E0| observed) in the
    first period against the envelope in the last period. A method with no secular drift keeps
    the envelope roughly constant across periods; a method with secular drift proportional to
    elapsed time (Section 3.4, RK4) grows its envelope from the first period to the last.
    """
    STEPS_PER_PERIOD = 1000
    N_PERIODS = 5
    SAMPLES_PER_PERIOD = 20

    def _run_energy_trace(self, integrator):
        state = two_body_kepler()
        e0 = total_energy(state, softening=0.05)
        dt = config.KEPLER_PERIOD / self.STEPS_PER_PERIOD
        n_steps = self.STEPS_PER_PERIOD * self.N_PERIODS
        callback_every = self.STEPS_PER_PERIOD // self.SAMPLES_PER_PERIOD
        trace = []

        def callback(step_index, time, s):
            trace.append(total_energy(s, softening=0.05))

        integrate(state, integrator=integrator, backend=BACKEND, dt=dt, n_steps=n_steps,
                  softening=0.05, callback=callback, callback_every=callback_every)
        rel_trace = [(e - e0) / abs(e0) for e in trace]
        return rel_trace

    def test_euler_grows_monotonically_and_ends_positive(self):
        trace = self._run_energy_trace("euler")
        assert trace[-1] > 0
        assert all(b >= a for a, b in zip(trace, trace[1:])), (
            "euler's energy error should grow monotonically (Section 3.1, Section 7 INV-4)"
        )

    def test_symplectic_euler_envelope_does_not_grow_across_periods(self):
        trace = self._run_energy_trace("symplectic_euler")
        first_period_peak = max(abs(x) for x in trace[: self.SAMPLES_PER_PERIOD])
        last_period_peak = max(abs(x) for x in trace[-self.SAMPLES_PER_PERIOD:])
        assert first_period_peak > 0
        ratio = last_period_peak / first_period_peak
        assert ratio <= 3.0, (
            f"symplectic_euler's oscillation envelope grew {ratio:.2f}x from the first to the "
            f"last of {self.N_PERIODS} periods -- expected bounded oscillation, no secular drift"
        )

    def test_velocity_verlet_envelope_does_not_grow_across_periods(self):
        trace = self._run_energy_trace("velocity_verlet")
        first_period_peak = max(abs(x) for x in trace[: self.SAMPLES_PER_PERIOD])
        last_period_peak = max(abs(x) for x in trace[-self.SAMPLES_PER_PERIOD:])
        assert first_period_peak > 0
        ratio = last_period_peak / first_period_peak
        assert ratio <= 3.0, (
            f"velocity_verlet's oscillation envelope grew {ratio:.2f}x from the first to the "
            f"last of {self.N_PERIODS} periods -- expected bounded oscillation, no secular drift"
        )

    def test_rk4_final_negative(self):
        trace = self._run_energy_trace("rk4")
        assert trace[-1] < 0, "RK4 secular drift is negative in this configuration (Section 3.4)"
