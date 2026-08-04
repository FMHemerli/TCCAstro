"""
INV-7 full table: docs/integradores.md Section 7, TWOBODY_ECC (eps = 0.05), fp64, 20 nominal
periods, at the exact step-count ladders the document measured (1000/2000/4000/8000 for the
oscillatory methods, 1000/2000/4000 for rk4's ratio -- 8000 is excluded from RK4's ratio check
per the document's own note that its amplitude has hit the fp64 round-off floor there).

This reproduces the amplitude-scaling exponent (order 1 for symplectic_euler, order 2 for
velocity_verlet, >= order ~3.6 for rk4) via the amplitude of the ΔE/E0 oscillation, and the
final/amplitude ratios that distinguish bounded oscillation (symplectic methods) from secular
drift (rk4, euler).

Marked slow: 20 periods at up to 8000 steps/period, times four step-count levels, times four
integrators (rk4 at 4 force evals/step) is on the order of two million force evaluations. Even
at N=2 (trivial per-evaluation cost), Python/torch per-call dispatch overhead makes this a
multi-minute test. Deselect with `-m "not slow"` for per-commit CI.
"""
import pytest

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.initial_conditions import two_body_kepler  # noqa: E402
from nbody.integrators import integrate  # noqa: E402
from nbody.observables import total_energy  # noqa: E402

from tolerances import (  # noqa: E402
    INV7_EULER_FINAL_MIN,
    INV7_RK4_FINAL_OVER_AMPLITUDE_MIN,
    INV7_RK4_RATIO_MIN,
    INV7_SYMPLECTIC_FINAL_OVER_AMPLITUDE_MAX,
    INV7_SYMPLECTIC_RATIO_CENTER,
    INV7_SYMPLECTIC_RATIO_TOL,
    INV7_VERLET_FINAL_OVER_AMPLITUDE_MAX,
    INV7_VERLET_RATIO_CENTER,
    INV7_VERLET_RATIO_TOL,
)

pytestmark = pytest.mark.slow

BACKEND = get_backend("torch_eager")
N_PERIODS = 20
SAMPLES_PER_PERIOD = 20


def energy_amplitude_and_final(integrator, steps_per_period):
    state = two_body_kepler()
    e0 = total_energy(state, softening=0.05)
    dt = config.KEPLER_PERIOD / steps_per_period
    n_steps = steps_per_period * N_PERIODS
    callback_every = max(1, steps_per_period // SAMPLES_PER_PERIOD)
    trace = []

    def callback(step_index, time, s):
        trace.append(total_energy(s, softening=0.05))

    integrate(state, integrator=integrator, backend=BACKEND, dt=dt, n_steps=n_steps,
              softening=0.05, callback=callback, callback_every=callback_every)
    rel_trace = [(e - e0) / abs(e0) for e in trace]
    amplitude = max(rel_trace) - min(rel_trace)
    final = rel_trace[-1]
    return amplitude, final


class TestSymplecticEulerAmplitudeTable:
    STEPS = (1000, 2000, 4000, 8000)

    @pytest.fixture(scope="class")
    def results(self):
        return {n: energy_amplitude_and_final("symplectic_euler", n) for n in self.STEPS}

    def test_final_bounded_by_amplitude_at_each_level(self, results):
        for n, (amplitude, final) in results.items():
            assert abs(final) <= amplitude * INV7_SYMPLECTIC_FINAL_OVER_AMPLITUDE_MAX, (
                f"n={n}: |final|={abs(final):.3e} exceeds amplitude/50={amplitude / 50:.3e}"
            )

    def test_amplitude_scales_as_h_to_the_first_power(self, results):
        for n in self.STEPS[:-1]:
            ratio = results[n][0] / results[2 * n][0]
            assert (
                INV7_SYMPLECTIC_RATIO_CENTER - INV7_SYMPLECTIC_RATIO_TOL
                <= ratio
                <= INV7_SYMPLECTIC_RATIO_CENTER + INV7_SYMPLECTIC_RATIO_TOL
            ), f"amplitude ratio n={n}->{2*n}: {ratio:.3f}, expected 2.0 +/- 0.15"


class TestVelocityVerletAmplitudeTable:
    STEPS = (1000, 2000, 4000, 8000)

    @pytest.fixture(scope="class")
    def results(self):
        return {n: energy_amplitude_and_final("velocity_verlet", n) for n in self.STEPS}

    def test_final_bounded_by_amplitude_at_each_level(self, results):
        for n, (amplitude, final) in results.items():
            assert abs(final) <= amplitude * INV7_VERLET_FINAL_OVER_AMPLITUDE_MAX, (
                f"n={n}: |final|={abs(final):.3e} exceeds amplitude/1000={amplitude / 1000:.3e}"
            )

    def test_amplitude_scales_as_h_squared(self, results):
        for n in self.STEPS[:-1]:
            ratio = results[n][0] / results[2 * n][0]
            assert (
                INV7_VERLET_RATIO_CENTER - INV7_VERLET_RATIO_TOL
                <= ratio
                <= INV7_VERLET_RATIO_CENTER + INV7_VERLET_RATIO_TOL
            ), f"amplitude ratio n={n}->{2*n}: {ratio:.3f}, expected 4.0 +/- 0.3"


class TestRK4AmplitudeTable:
    RATIO_STEPS = (1000, 2000, 4000)  # 8000 excluded: amplitude hits fp64 round-off floor there.

    @pytest.fixture(scope="class")
    def results(self):
        return {n: energy_amplitude_and_final("rk4", n) for n in self.RATIO_STEPS}

    def test_final_comparable_to_amplitude_and_negative(self, results):
        for n, (amplitude, final) in results.items():
            assert final < 0, f"n={n}: rk4 final ΔE/E0 should be negative, got {final:.3e}"
            assert abs(final) >= amplitude * INV7_RK4_FINAL_OVER_AMPLITUDE_MIN, (
                f"n={n}: |final|={abs(final):.3e} should be >= amplitude/3={amplitude / 3:.3e} "
                f"(deriva secular, nao oscilacao limitada)"
            )

    def test_amplitude_scaling_ratio_at_least_twelve(self, results):
        for n in self.RATIO_STEPS[:-1]:
            ratio = results[n][0] / results[2 * n][0]
            assert ratio >= INV7_RK4_RATIO_MIN, (
                f"amplitude ratio n={n}->{2*n}: {ratio:.2f}, expected >= "
                f"{INV7_RK4_RATIO_MIN} (order >= 3.6, per document's own relaxed criterion)"
            )


class TestEulerFinalGrowth:
    STEPS = (1000, 2000, 4000, 8000)

    @pytest.mark.parametrize("steps_per_period", STEPS)
    def test_final_positive_and_large(self, steps_per_period):
        _, final = energy_amplitude_and_final("euler", steps_per_period)
        assert final >= INV7_EULER_FINAL_MIN, (
            f"n={steps_per_period}: euler final ΔE/E0={final:.3e} below {INV7_EULER_FINAL_MIN}"
        )
