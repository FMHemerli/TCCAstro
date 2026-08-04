from __future__ import annotations

from . import config
from .backends import get_backend
from .state import State

INTEGRATOR_NAMES = ("euler", "symplectic_euler", "velocity_verlet", "rk4")

FORCE_EVALS_PER_STEP = {
    "euler": 1,
    "symplectic_euler": 1,
    "velocity_verlet": 1,
    "rk4": 4,
}

FORCE_EVALS_STARTUP = {
    "euler": 0,
    "symplectic_euler": 0,
    "velocity_verlet": 1,
    "rk4": 0,
}


def _resolve_backend(backend):
    if isinstance(backend, str):
        return get_backend(backend)
    return backend


def _step_euler(state: State, backend, dt: float, softening: float) -> State:
    a = backend.accelerations(state.r, state.m, softening)
    r_new = state.r + dt * state.v
    v_new = state.v + dt * a
    return State(r=r_new, v=v_new, m=state.m)


def _step_symplectic_euler(state: State, backend, dt: float, softening: float) -> State:
    a = backend.accelerations(state.r, state.m, softening)
    v_new = state.v + dt * a
    r_new = state.r + dt * v_new
    return State(r=r_new, v=v_new, m=state.m)


def _step_velocity_verlet(state: State, backend, dt: float, softening: float, a_current):
    v_half = state.v + (dt / 2.0) * a_current
    r_new = state.r + dt * v_half
    a_new = backend.accelerations(r_new, state.m, softening)
    v_new = v_half + (dt / 2.0) * a_new
    return State(r=r_new, v=v_new, m=state.m), a_new


def _step_rk4(state: State, backend, dt: float, softening: float) -> State:
    r0 = state.r
    v0 = state.v
    m = state.m

    k1_r = v0
    k1_v = backend.accelerations(r0, m, softening)

    k2_r = v0 + (dt / 2.0) * k1_v
    k2_v = backend.accelerations(r0 + (dt / 2.0) * k1_r, m, softening)

    k3_r = v0 + (dt / 2.0) * k2_v
    k3_v = backend.accelerations(r0 + (dt / 2.0) * k2_r, m, softening)

    k4_r = v0 + dt * k3_v
    k4_v = backend.accelerations(r0 + dt * k3_r, m, softening)

    r_new = r0 + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_new = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return State(r=r_new, v=v_new, m=m)


def integrate(
    state: State,
    *,
    integrator: str,
    backend,
    dt: float,
    n_steps: int,
    softening: float = config.SOFTENING,
    callback=None,
    callback_every: int | None = None,
) -> State:
    if integrator not in INTEGRATOR_NAMES:
        raise ValueError(f"unknown integrator {integrator!r}; valid names are {INTEGRATOR_NAMES}")

    backend_obj = _resolve_backend(backend)

    every = None
    if callback is not None:
        every = callback_every if callback_every is not None else 1

    current = state
    a_current = None
    if integrator == "velocity_verlet":
        a_current = backend_obj.accelerations(current.r, current.m, softening)

    if every is not None:
        callback(0, 0.0, current)

    for step in range(n_steps):
        if integrator == "euler":
            current = _step_euler(current, backend_obj, dt, softening)
        elif integrator == "symplectic_euler":
            current = _step_symplectic_euler(current, backend_obj, dt, softening)
        elif integrator == "velocity_verlet":
            current, a_current = _step_velocity_verlet(current, backend_obj, dt, softening, a_current)
        elif integrator == "rk4":
            current = _step_rk4(current, backend_obj, dt, softening)

        if every is not None and (step + 1) % every == 0:
            callback(step + 1, (step + 1) * dt, current)

    return current
