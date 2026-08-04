"""
nbody.integrators against docs/integradores.md Section 3 (the four update recurrences, exactly
as written, including the two "armadilha de implementacao" traps called out there) and
docs/api-contract.md's integrate() signature and error contracts.

Each recurrence is re-derived independently in this file directly from Section 3's pseudocode
and checked against integrate()'s output after a small, fixed number of steps on a random
(nonzero velocity) system. Cold_sphere is deliberately NOT used here: its v=0 at t=0 would make
euler and symplectic_euler produce an identical first position update, which is exactly the
kind of degenerate case that would hide the "write v before r without a temporary" trap Section
3.1 warns about.
"""
import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.integrators import (  # noqa: E402
    FORCE_EVALS_PER_STEP,
    FORCE_EVALS_STARTUP,
    INTEGRATOR_NAMES,
    integrate,
)
from nbody.state import State  # noqa: E402

from tolerances import TOL_IMPL_SHORT_FP64  # noqa: E402

REFERENCE_BACKEND = get_backend("torch_eager")


def random_state(n=6, seed=42, dtype=torch.float64):
    rng = np.random.default_rng(seed)
    r = torch.tensor(rng.normal(scale=2.0, size=(n, 3)), dtype=dtype)
    v = torch.tensor(rng.normal(scale=0.5, size=(n, 3)), dtype=dtype)
    m = torch.full((n,), config.PARTICLE_MASS, dtype=dtype)
    return State(r=r, v=v, m=m)


def accel(r, m, softening):
    return REFERENCE_BACKEND.accelerations(r, m, softening=softening)


def relative_error(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


class TestEulerRecurrence:
    """Section 3.1: a^n = A(r^n); r^(n+1) = r^n + h v^n; v^(n+1) = v^n + h a^n.
    v is NOT used after being updated to build r -- must use v^n, not v^(n+1)."""

    def test_matches_reference_recurrence_over_three_steps(self):
        h = 1e-3
        eps = 0.05
        state = random_state()
        r, v, m = state.r.clone(), state.v.clone(), state.m
        for _ in range(3):
            a = accel(r, m, eps)
            r_next = r + h * v  # explicitly uses v^n, the trap Section 3.1 warns about
            v_next = v + h * a
            r, v = r_next, v_next

        result = integrate(state, integrator="euler", backend=REFERENCE_BACKEND, dt=h,
                            n_steps=3, softening=eps)
        assert relative_error(result.r, r) <= TOL_IMPL_SHORT_FP64
        assert relative_error(result.v, v) <= TOL_IMPL_SHORT_FP64

    def test_differs_from_symplectic_euler_with_nonzero_velocity(self):
        # If euler silently becomes semi-implicit (the trap), it would coincide with
        # symplectic_euler's r update on step 1. With v0 != 0 they must differ.
        h = 1e-3
        eps = 0.05
        state = random_state()
        r_euler = integrate(state, integrator="euler", backend=REFERENCE_BACKEND, dt=h,
                             n_steps=1, softening=eps).r
        r_symp = integrate(state, integrator="symplectic_euler", backend=REFERENCE_BACKEND,
                            dt=h, n_steps=1, softening=eps).r
        assert not torch.allclose(r_euler, r_symp, rtol=1e-8, atol=1e-12)


class TestSymplecticEulerRecurrence:
    """Section 3.2 ("velocity first"): a^n = A(r^n); v^(n+1) = v^n + h a^n;
    r^(n+1) = r^n + h v^(n+1). Uses the JUST-UPDATED velocity to advance position."""

    def test_matches_reference_recurrence_over_three_steps(self):
        h = 1e-3
        eps = 0.05
        state = random_state()
        r, v, m = state.r.clone(), state.v.clone(), state.m
        for _ in range(3):
            a = accel(r, m, eps)
            v_next = v + h * a
            r_next = r + h * v_next  # uses v^(n+1), NOT v^n -- this is what makes it symplectic
            r, v = r_next, v_next

        result = integrate(state, integrator="symplectic_euler", backend=REFERENCE_BACKEND,
                            dt=h, n_steps=3, softening=eps)
        assert relative_error(result.r, r) <= TOL_IMPL_SHORT_FP64
        assert relative_error(result.v, v) <= TOL_IMPL_SHORT_FP64


class TestVelocityVerletRecurrence:
    """Section 3.3, KDK form: v^(n+1/2) = v^n + (h/2) a^n; r^(n+1) = r^n + h v^(n+1/2);
    a^(n+1) = A(r^(n+1)); v^(n+1) = v^(n+1/2) + (h/2) a^(n+1)."""

    def test_matches_reference_recurrence_over_three_steps(self):
        h = 1e-3
        eps = 0.05
        state = random_state()
        r, v, m = state.r.clone(), state.v.clone(), state.m
        a = accel(r, m, eps)
        for _ in range(3):
            v_half = v + (h / 2) * a
            r_next = r + h * v_half
            a_next = accel(r_next, m, eps)
            v_next = v_half + (h / 2) * a_next
            r, v, a = r_next, v_next, a_next

        result = integrate(state, integrator="velocity_verlet", backend=REFERENCE_BACKEND,
                            dt=h, n_steps=3, softening=eps)
        assert relative_error(result.r, r) <= TOL_IMPL_SHORT_FP64
        assert relative_error(result.v, v) <= TOL_IMPL_SHORT_FP64


class TestRK4Recurrence:
    """Section 3.4: cross-coupling trap -- the argument of A() at stage k(n+1)_v is built from
    k(n)_r (the position increment), not k(n)_v."""

    def test_matches_reference_recurrence_over_two_steps(self):
        h = 1e-3
        eps = 0.05
        state = random_state()
        r, v, m = state.r.clone(), state.v.clone(), state.m
        for _ in range(2):
            k1_r = v
            k1_v = accel(r, m, eps)
            k2_r = v + (h / 2) * k1_v
            k2_v = accel(r + (h / 2) * k1_r, m, eps)
            k3_r = v + (h / 2) * k2_v
            k3_v = accel(r + (h / 2) * k2_r, m, eps)
            k4_r = v + h * k3_v
            k4_v = accel(r + h * k3_r, m, eps)
            r_next = r + (h / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
            v_next = v + (h / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            r, v = r_next, v_next

        result = integrate(state, integrator="rk4", backend=REFERENCE_BACKEND, dt=h,
                            n_steps=2, softening=eps)
        assert relative_error(result.r, r) <= TOL_IMPL_SHORT_FP64
        assert relative_error(result.v, v) <= TOL_IMPL_SHORT_FP64

    def test_cross_coupling_trap_is_detectable(self):
        # A buggy RK4 that uses k(n)_v instead of k(n)_r as the position argument to A() at each
        # stage is a *different*, order-2-consistent method (Section 3.4). Sanity-check that our
        # own reference recurrence (used above) actually differs from that buggy variant on this
        # system, so the comparison above has discriminating power.
        h = 1e-3
        eps = 0.05
        state = random_state()
        r, v, m = state.r.clone(), state.v.clone(), state.m

        # Correct.
        k1_r, k1_v = v, accel(r, m, eps)
        r_correct_stage2_arg = r + (h / 2) * k1_r

        # Buggy: uses k1_v instead of k1_r as the position argument.
        r_buggy_stage2_arg = r + (h / 2) * k1_v

        assert not torch.allclose(r_correct_stage2_arg, r_buggy_stage2_arg, rtol=1e-6)


class TestForceEvaluationCount:
    """Section 3.5's n_force/step table is a mandatory cost property, not just documentation:
    velocity_verlet's acceleration reuse in particular (Section 3.3, 'obrigatorio') is only
    observable by counting calls, since mathematically a fresh evaluation and a reused one are
    bit-identical (accelerations() is a deterministic pure function of r, m)."""

    class CountingBackend:
        def __init__(self, inner):
            self._inner = inner
            self.name = inner.name
            self.calls = 0

        def supports(self, device):
            return self._inner.supports(device)

        def accelerations(self, r, m, softening=config.SOFTENING, *, tile_size=None):
            self.calls += 1
            return self._inner.accelerations(r, m, softening=softening, tile_size=tile_size)

        def fused_symplectic_euler_step(self, state, dt, softening=config.SOFTENING):
            return self._inner.fused_symplectic_euler_step(state, dt, softening=softening)

    @pytest.mark.parametrize("name", ["euler", "symplectic_euler", "velocity_verlet", "rk4"])
    @pytest.mark.parametrize("n_steps", [0, 5])
    def test_call_count_matches_force_evals_per_step(self, name, n_steps):
        counting = self.CountingBackend(REFERENCE_BACKEND)
        state = random_state(n=8)
        integrate(state, integrator=name, backend=counting, dt=1e-3, n_steps=n_steps,
                  softening=0.05)
        # Section 3.5 (docs/integradores.md) is explicit that the border cost is normative,
        # not inferred: FORCE_EVALS_PER_STEP (k, per-step steady-state cost) and
        # FORCE_EVALS_STARTUP (s, one-time pre-condition cost, Section 3.3) are BOTH exposed by
        # the API contract as separate tables, and the cost formula is
        #     n_force(method, M) = M * FORCE_EVALS_PER_STEP[name] + FORCE_EVALS_STARTUP[name]
        # stated to hold "inclusive para M = 0" (Section 3.5), where velocity_verlet's
        # precondition a^0 = A(r^0) alone costs exactly 1 and the other three integrators, which
        # have no precondition, cost exactly 0. Read s from FORCE_EVALS_STARTUP rather than
        # hardcoding which method has a nonzero startup cost.
        expected = n_steps * FORCE_EVALS_PER_STEP[name] + FORCE_EVALS_STARTUP[name]
        assert counting.calls == expected, (
            f"{name}, n_steps={n_steps}: expected {expected} force evaluations "
            f"(FORCE_EVALS_PER_STEP[{name}]={FORCE_EVALS_PER_STEP[name]}, "
            f"FORCE_EVALS_STARTUP[{name}]={FORCE_EVALS_STARTUP[name]}), got {counting.calls}"
        )


class TestErrorContracts:
    def test_unknown_integrator_name_raises_value_error(self):
        state = random_state(n=3)
        with pytest.raises(ValueError) as excinfo:
            integrate(state, integrator="not_a_real_integrator", backend=REFERENCE_BACKEND,
                      dt=1e-3, n_steps=1)
        message = str(excinfo.value)
        for name in INTEGRATOR_NAMES:
            assert name in message, f"error message should list valid integrator names, got: {message!r}"

    def test_unknown_backend_name_raises_value_error(self):
        state = random_state(n=3)
        with pytest.raises(ValueError):
            integrate(state, integrator="euler", backend="not_a_real_backend", dt=1e-3, n_steps=1)

    def test_backend_accepted_as_name_string(self):
        state = random_state(n=3)
        result_by_name = integrate(state, integrator="euler", backend="torch_eager", dt=1e-3,
                                    n_steps=1, softening=0.05)
        result_by_object = integrate(state, integrator="euler", backend=REFERENCE_BACKEND,
                                      dt=1e-3, n_steps=1, softening=0.05)
        assert torch.equal(result_by_name.r, result_by_object.r)
        assert torch.equal(result_by_name.v, result_by_object.v)


class TestCallback:
    def test_callback_invoked_every_callback_every_steps(self):
        state = random_state(n=4)
        calls = []

        def callback(step_index, time, s):
            calls.append((step_index, time))

        dt = 1e-3
        n_steps = 12
        callback_every = 3
        integrate(state, integrator="euler", backend=REFERENCE_BACKEND, dt=dt,
                  n_steps=n_steps, softening=0.05, callback=callback,
                  callback_every=callback_every)

        assert len(calls) >= 1, "callback should be invoked at least once"
        steps = [c[0] for c in calls]
        assert steps == sorted(steps), "callback step indices must be non-decreasing"
        for i in range(1, len(steps)):
            assert steps[i] - steps[i - 1] == callback_every, (
                f"consecutive callback step indices should differ by callback_every="
                f"{callback_every}, got {steps[i-1]} -> {steps[i]}"
            )
        for step_index, time in calls:
            assert time == pytest.approx(step_index * dt, rel=1e-9, abs=1e-15), (
                f"callback time {time} should equal step_index*dt = {step_index * dt}"
            )

    def test_no_callback_is_a_no_op(self):
        state = random_state(n=4)
        result = integrate(state, integrator="euler", backend=REFERENCE_BACKEND, dt=1e-3,
                            n_steps=5, softening=0.05, callback=None)
        assert result.n == state.n
