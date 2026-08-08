from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import collisions, config
from .backends import get_backend
from .state import State


@dataclass
class CollisionRunStats:
    """Long-run collision accumulators, owned by the integration loop (Sec. 9.1.1, point 1:
    resolve() is pure and returns only this pass's deltas; the loop -- not collisions.py --
    is where they accumulate). n_elastic/n_merge/n_fragment are exact event counts.
    delta_e_int and delta_l_spin accumulate in fp64 regardless of the state's own dtype (Sec.
    7: "o diagnostico de energia colisional ... e sempre acumulado em fp64"). c_coll_max and
    f_reject_max are running maxima over the passes seen so far (Sec. 4.4.5, 4.5: both are
    reported quantities, never bloqueantes)."""

    n_elastic: int = 0
    n_merge: int = 0
    n_fragment: int = 0
    delta_e_int: float = 0.0
    delta_l_spin: torch.Tensor | None = None
    c_coll_max: float = 0.0
    f_reject_max: float = 0.0

    def update(self, outcome: "collisions.CollisionOutcome") -> None:
        self.n_elastic += outcome.n_elastic
        self.n_merge += outcome.n_merge
        self.n_fragment += outcome.n_fragment
        self.delta_e_int += outcome.delta_e_int
        self.delta_l_spin = (
            outcome.delta_l_spin.clone()
            if self.delta_l_spin is None
            else self.delta_l_spin + outcome.delta_l_spin
        )
        self.c_coll_max = max(self.c_coll_max, outcome.c_coll_max)
        self.f_reject_max = max(self.f_reject_max, outcome.f_reject)

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


def _step_velocity_verlet_collision(
    state: State,
    backend,
    dt: float,
    softening: float,
    a_current,
    model: "collisions.CollisionModel",
    generator,
):
    # Sec. 4.5: the collision pass C_h sits inside the drift, between the two half-kicks.
    # v_half is v^(n+1/2); collision_pass takes it over (detect + pair_disjoint + resolve)
    # and returns the fully-drifted (r^(n+1), v^(n+1/2)_updated) pair. The second half-kick
    # below is unconditional and identical to the collisionless path -- exactly one force
    # evaluation per step, whether or not any event fired this step.
    v_half = state.v + (dt / 2.0) * a_current
    detect_state = State(r=state.r, v=v_half, m=state.m)
    outcome = collisions.collision_pass(detect_state, dt, model, generator, softening)
    post = outcome.state
    a_new = backend.accelerations(post.r, post.m, softening)
    v_new = post.v + (dt / 2.0) * a_new
    return State(r=post.r, v=v_new, m=post.m), a_new, outcome


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
    collision: "collisions.CollisionModel | None" = None,
    collision_rng: np.random.Generator | None = None,
) -> "State | tuple[State, CollisionRunStats]":
    if integrator not in INTEGRATOR_NAMES:
        raise ValueError(f"unknown integrator {integrator!r}; valid names are {INTEGRATOR_NAMES}")

    if collision is not None:
        if softening == 0.0:
            raise ValueError(
                "integrate(): collision resolution requires softening > 0.0 (Sec. 5.2, "
                "INV-28). With softening == 0.0, a dead slot (m = 0) coincident with a live "
                "particle produces 0 * inf = NaN in the acceleration field."
            )
        if integrator != "velocity_verlet":
            raise ValueError(
                "integrate(): collision resolution is only defined inside the "
                f"velocity_verlet drift (Sec. 4.5, Sec. 9.1.1), got integrator={integrator!r}"
            )
    elif collision_rng is not None:
        raise ValueError("integrate(): collision_rng was given but collision is None")

    backend_obj = _resolve_backend(backend)

    every = None
    if callback is not None:
        every = callback_every if callback_every is not None else 1

    current = state
    a_current = None
    if integrator == "velocity_verlet":
        a_current = backend_obj.accelerations(current.r, current.m, softening)

    # The collision stream must survive across separate integrate() calls (Sec. 4.7.1,
    # INV-32: exactly 2 draws per accepted event, continuous over the run). A caller that
    # advances a simulation in chunks (e.g. the real-time viewer, one call per frame) has to
    # own the generator itself and pass the same object back in on every call; a fresh
    # np.random.default_rng(collision.seed) built here on every call would restart the stream
    # at every chunk boundary instead of continuing it. collision_rng is that caller-owned
    # object. When omitted, a generator seeded from collision.seed is created here, which
    # reproduces the previous single-call behaviour bit for bit for any caller that still
    # makes one integrate() call per run.
    if collision is not None:
        rng_c = collision_rng if collision_rng is not None else np.random.default_rng(collision.seed)
    else:
        rng_c = None
    stats = CollisionRunStats() if collision is not None else None

    if every is not None:
        callback(0, 0.0, current)

    for step in range(n_steps):
        if integrator == "euler":
            current = _step_euler(current, backend_obj, dt, softening)
        elif integrator == "symplectic_euler":
            current = _step_symplectic_euler(current, backend_obj, dt, softening)
        elif integrator == "velocity_verlet":
            if collision is None:
                current, a_current = _step_velocity_verlet(current, backend_obj, dt, softening, a_current)
            else:
                current, a_current, outcome = _step_velocity_verlet_collision(
                    current, backend_obj, dt, softening, a_current, collision, rng_c
                )
                stats.update(outcome)
        elif integrator == "rk4":
            current = _step_rk4(current, backend_obj, dt, softening)

        if every is not None and (step + 1) % every == 0:
            callback(step + 1, (step + 1) * dt, current)

    if collision is None:
        return current
    return current, stats
