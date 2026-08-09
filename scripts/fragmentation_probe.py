"""
Diagnostic instrumentation for the fragmentation channel.

Written to answer a specific observation: watching scripts/realtime.py with --collisions, a
body appears to split off beside the dominant body, at a distance that looks wrong, and
vanishes again a few steps later. This script establishes whether that is a rendering
artifact or the collision model, and measures four things the specification never bounds.

It changes no physics. It replicates integrators._step_velocity_verlet_collision step by
step -- half-kick, detect, pair_disjoint, resolve, half-kick -- so that every accepted event
can be inspected immediately before and after resolve(). The trajectory it produces is the
same one integrate() produces; the loop is unrolled only so the events are observable.

The four measurements:

  1. PLACEMENT GEOMETRY. resolve() places the two fragments exactly in contact using their
     NEW contact radii, R = R_ref (m/m_bar)^(1/3). Because that is concave in m, an even
     split of an UNEQUAL pair gives R_a + R_b larger than the colliding pair's own contact
     separation -- the fragments are pushed further apart than the bodies ever were. Closed
     form, in the incoming mass ratio q = m_big/m_small and the split f:

         (R_a + R_b) / (R_i + R_j) = [f^(1/3) + (1-f)^(1/3)]
                                     / [(q/(1+q))^(1/3) + (1/(1+q))^(1/3)]

     with a ceiling of 2^(2/3) = 1.5874 as q -> inf, f -> 1/2. The specification notes only
     that neither R_a+R_b >= R_i+R_j nor its converse holds universally (Sec. 4.9, as an
     aside about the sign of the E_int increment); it never draws the geometric conclusion.

  2. SEPARATION SPEED. u' = |u| sqrt(mu/mu') conserves the CM-frame kinetic energy exactly.
     For an unequal pair split evenly that makes the fragments separate far SLOWER than the
     impact:

         |u'| / |u| = sqrt( [q/(1+q)^2] / [f(1-f)] )

     The specification's claim that the placement "impede recolisão imediata" rests on the
     SIGN of u' only. Nothing anywhere checks that its magnitude clears the pair.

  3. RE-EVENTS ON THE SAME PAIR. Whether the two slots meet again, how many steps later,
     and through which channel. The approach guard (dr.dv < 0) blocks a pair only while it
     is still separating; once mutual gravity turns it around the pair is a candidate again,
     and it was left exactly touching.

  4. ENERGY GATE. Whether the event had the energy to do what the model says it did, as
     T_cm / E_bind with E_bind = (3/5) G m^2 / R for the larger body. Read the caveat in
     the report: the bodies are point masses with no internal structure, so E_bind is a
     yardstick imported from outside the model, not a quantity the simulation contains.

Usage:

    python scripts/fragmentation_probe.py                      # N=1000, 3 t_ff, what the viewer shows
    python scripts/fragmentation_probe.py --cold               # the Sec. 4.13.6 campaign protocol
    python scripts/fragmentation_probe.py --n 400 --steps 3000 # quick look

The default initial condition deliberately matches scripts/realtime.py without --cold -- the
Salpeter spectrum at Q = Q_DEFAULT -- because the observation this script was written for was
made watching that. --cold switches to the campaign's equal-mass, Q = 0 protocol.

Output: a report on stdout and results/2026/fragmentation_probe_{spectrum,cold}.csv with one
row per fragmentation event, tagged by which initial condition was run.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody import collisions, config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.initial_conditions import cold_sphere, random_sphere  # noqa: E402
from nbody.state import State  # noqa: E402

torch.set_default_dtype(torch.float64)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "2026")


def results_path(cold: bool) -> str:
    """Tagged by initial condition. The two protocols are different measurements of different
    populations, so a fixed filename would let one silently replace the other."""
    return os.path.join(
        RESULTS_DIR, f"fragmentation_probe_{'cold' if cold else 'spectrum'}.csv"
    )


RE_EVENT_WINDOW = 200  # steps; how far back to look for the same pair meeting again


# ==============================================================================
# Closed forms, so the report can state what the measurement should be
# ==============================================================================
def placement_ratio(q: float, f: float) -> float:
    """(R_a + R_b) / (R_i + R_j) for incoming mass ratio q >= 1 and split f."""
    return (f ** (1 / 3) + (1 - f) ** (1 / 3)) / (
        (q / (1 + q)) ** (1 / 3) + (1 / (1 + q)) ** (1 / 3)
    )


def speed_ratio(q: float, f: float) -> float:
    """|u'| / |u| = sqrt(mu/mu') for incoming mass ratio q >= 1 and split f."""
    return math.sqrt((q / (1 + q) ** 2) / (f * (1 - f)))


def contact_radius(m: float, r_ref: float) -> float:
    return r_ref * (m / config.PARTICLE_MASS) ** (1 / 3)


def binding_energy(m: float, r_ref: float) -> float:
    """(3/5) G m^2 / R, uniform sphere. An imported scale -- see the module docstring."""
    return 0.6 * config.G * m * m / contact_radius(m, r_ref)


# ==============================================================================
# The instrumented run
# ==============================================================================
def build_initial_state(n: int, seed: int, cold: bool, virial_ratio: float):
    """Cold equal-mass sphere, or the Salpeter/virial realization -- the same two initial
    conditions scripts/realtime.py offers, so a measurement here can be tied to either what
    the viewer shows by default or what the Sec. 4.13.7 campaign ran.

    They are NOT interchangeable and the difference matters for reading the numbers below:
    the viewer defaults to random_sphere with the mass spectrum and Q = Q_DEFAULT, while the
    campaign protocol is "massas iguais, Q = 0" (Sec. 4.13.6). Mass ratios of several hundred
    to one are reached in both, but by different routes -- from the spectrum at t = 0 in one,
    only through collisional growth in the other.
    """
    if cold:
        return cold_sphere(n=n, seed=seed)
    return random_sphere(n=n, mass_spectrum={}, virial_ratio=virial_ratio, seed=seed)


def run(n, n_steps, dt, softening, seed, collision_seed, cold, virial_ratio):
    backend = get_backend("torch_eager")
    state = build_initial_state(n, seed, cold, virial_ratio)
    model = collisions.CollisionModel(
        v_coh=collisions.v_coh_from_state(state, config.SPHERE_RADIUS), seed=collision_seed
    )
    rng = np.random.default_rng(model.seed)
    a = backend.accelerations(state.r, state.m, softening)

    events: list[dict] = []
    re_events: list[tuple[str, str, int]] = []
    last_event: dict[int, tuple[int, str, int]] = {}  # slot -> (step, channel, partner)
    candidate_steps = 0
    reject_max = 0.0
    max_candidates = 0
    rejecting_steps = 0
    total_candidates = 0
    total_accepted = 0

    t0 = time.perf_counter()
    for step in range(n_steps):
        v_half = state.v + (dt / 2.0) * a
        pre = State(r=state.r, v=v_half, m=state.m)

        candidates = collisions.detect(pre, dt, model)
        accepted = collisions.pair_disjoint(candidates)
        if candidates.n:
            candidate_steps += 1
            max_candidates = max(max_candidates, candidates.n)
            reject_max = max(reject_max, accepted.f_reject)
            total_candidates += candidates.n
            total_accepted += accepted.n
            if accepted.f_reject > 0.0:
                rejecting_steps += 1

        m_before = pre.m.clone()
        v_before = v_half.clone()
        outcome = collisions.resolve(pre, dt, model, accepted, rng, softening)
        post = outcome.state

        for k in range(accepted.n):
            i, j = int(accepted.i[k]), int(accepted.j[k])
            m_i, m_j = m_before[i].item(), m_before[j].item()
            m_a, m_b = post.m[i].item(), post.m[j].item()

            # Channel is inferred from the mass bookkeeping, not from the counts resolve()
            # returns: those are aggregates and cannot be attributed to a pair.
            if m_b == 0.0:
                channel = "merge"
            elif abs(m_a - m_i) <= 1e-12 * m_i:
                channel = "elastic"
            else:
                channel = "frag"

            for slot in (i, j):
                previous = last_event.get(slot)
                if previous is not None:
                    prev_step, prev_channel, partner = previous
                    if partner in (i, j) and step - prev_step <= RE_EVENT_WINDOW:
                        re_events.append((prev_channel, channel, step - prev_step))
            last_event[i] = (step, channel, j)
            last_event[j] = (step, channel, i)

            if channel != "frag":
                continue

            big_m = m_i + m_j
            m_big, m_small = max(m_i, m_j), min(m_i, m_j)
            q = m_big / m_small
            f = min(m_a, m_b) / big_m
            mu = m_i * m_j / big_m
            u = torch.linalg.norm(v_before[j] - v_before[i]).item()
            t_cm = 0.5 * mu * u * u

            old_contact = contact_radius(m_i, model.r_ref) + contact_radius(m_j, model.r_ref)
            new_contact = contact_radius(m_a, model.r_ref) + contact_radius(m_b, model.r_ref)

            events.append(
                {
                    "step": step,
                    "t_over_tff": step * dt / config.T_FF,
                    "m_big_over_mbar": m_big / config.PARTICLE_MASS,
                    "mass_ratio_q": q,
                    "split_f": f,
                    "old_contact_sep": old_contact,
                    "new_contact_sep": new_contact,
                    "placement_ratio": new_contact / old_contact,
                    "placement_ratio_closed_form": placement_ratio(q, f),
                    "u": u,
                    "u_prime": torch.linalg.norm(post.v[j] - post.v[i]).item(),
                    "speed_ratio": torch.linalg.norm(post.v[j] - post.v[i]).item() / u,
                    "speed_ratio_closed_form": speed_ratio(q, f),
                    "t_cm_over_ebind_big": t_cm / binding_energy(m_big, model.r_ref),
                    "t_cm_over_ebind_pair": t_cm / binding_energy(big_m, model.r_ref),
                }
            )

        a = backend.accelerations(post.r, post.m, softening)
        state = State(r=post.r, v=post.v + (dt / 2.0) * a, m=post.m)

    return {
        "events": events,
        "re_events": re_events,
        "candidate_steps": candidate_steps,
        "max_candidates": max_candidates,
        "reject_max": reject_max,
        "rejecting_steps": rejecting_steps,
        "total_candidates": total_candidates,
        "total_accepted": total_accepted,
        "wall": time.perf_counter() - t0,
        "final_state": state,
        "ic_label": None,  # filled in by main()
    }


# ==============================================================================
# Report
# ==============================================================================
def describe(values):
    v = np.asarray(values, dtype=float)
    parts = (np.min(v), np.median(v), np.percentile(v, 90), np.max(v))
    return "".join(f"{p:>12.4g}" for p in parts)


def report(result, n, n_steps, dt):
    events = result["events"]
    print("=" * 96)
    print(f"fragmentation probe -- N={n}, {n_steps} steps, dt={dt:g} s, "
          f"t_end={n_steps * dt / config.T_FF:.2f} t_ff, wall={result['wall']:.0f}s")
    print(f"initial condition: {result['ic_label']}")
    print("=" * 96)

    print()
    print("PAIRING (the 'several bodies colliding at once, resolved pair by pair' hypothesis)")
    print(f"  steps with at least one candidate : {result['candidate_steps']} / {n_steps}")
    print(f"  most candidates in any one step   : {result['max_candidates']}")
    print(f"  steps where a candidate was denied: {result['rejecting_steps']}")
    print(f"  max f_reject in any one step      : {result['reject_max']:.4f}")
    # f_reject_total over the whole run, not the per-step maximum: this is the aggregate the
    # specification asks for (Sec. 4.5, an open [A] pending a stage-3 measurement with
    # resolve() enabled). Note the run below is NOT the stage-3 protocol unless --cold is
    # given, so it does not by itself close that item.
    if result["total_candidates"]:
        denied = result["total_candidates"] - result["total_accepted"]
        print(f"  f_reject_total over the run       : {denied / result['total_candidates']:.4f}"
              f"  ({denied} of {result['total_candidates']} candidates deferred a step)")
    if result["reject_max"] == 0.0:
        print("  -> no candidate was ever denied in this run. Beware of concluding too much "
              "from that:")
        print("     rejection needs three bodies in contact at once, which only happens in the "
              "dense")
        print("     core, so a run that stops before the runaway will report zero whatever the "
              "model does.")

    if not events:
        print("\nno fragmentation events in this run; nothing further to report")
        return

    print()
    print(f"FRAGMENTATION EVENTS: {len(events)}")
    print(f"  {'':34}{'min':>12}{'median':>12}{'p90':>12}{'max':>12}")
    for key, label in (
        ("mass_ratio_q", "incoming mass ratio q"),
        ("split_f", "split f (lighter fragment)"),
        ("placement_ratio", "placement / old contact sep"),
        ("speed_ratio", "|u'| / |u|"),
        ("t_cm_over_ebind_big", "T_cm / E_bind(larger body)"),
    ):
        print(f"  {label:34}{describe([e[key] for e in events])}")

    farther = sum(1 for e in events if e["placement_ratio"] > 1.0)
    slower = sum(1 for e in events if e["speed_ratio"] < 1.0)
    starved = sum(1 for e in events if e["t_cm_over_ebind_big"] < 1.0)
    print()
    print(f"  placed FARTHER apart than the colliding pair was : {farther}/{len(events)} "
          f"({100 * farther / len(events):.0f}%)   ceiling of the closed form: {2 ** (2/3):.4f}")
    print(f"  separating SLOWER than the impact speed          : {slower}/{len(events)} "
          f"({100 * slower / len(events):.0f}%)")
    print(f"  T_cm below E_bind of the body being split        : {starved}/{len(events)} "
          f"({100 * starved / len(events):.0f}%)")

    # The closed forms are printed against the measurement rather than asserted, so a future
    # change to resolve() shows up here as a disagreement instead of passing silently.
    worst_placement = max(
        abs(e["placement_ratio"] - e["placement_ratio_closed_form"]) for e in events
    )
    worst_speed = max(abs(e["speed_ratio"] - e["speed_ratio_closed_form"]) for e in events)
    print()
    print("  closed form vs measurement, worst absolute disagreement over all events:")
    print(f"    placement ratio {worst_placement:.2e}      |u'|/|u| {worst_speed:.2e}")

    print()
    print("TEN EVENTS ON THE HEAVIEST BODIES")
    print(f"  {'step':>7}{'m_big/mbar':>12}{'q':>9}{'f':>7}{'place':>8}"
          f"{'|u|':>8}{'u_p/u':>8}{'T_cm/E_bind':>13}")
    for e in sorted(events, key=lambda e: -e["m_big_over_mbar"])[:10]:
        print(f"  {e['step']:>7}{e['m_big_over_mbar']:>12.1f}{e['mass_ratio_q']:>9.1f}"
              f"{e['split_f']:>7.3f}{e['placement_ratio']:>8.2f}{e['u']:>8.1f}"
              f"{e['speed_ratio']:>8.3f}{e['t_cm_over_ebind_big']:>13.2e}")

    re_events = result["re_events"]
    print()
    print(f"RE-EVENTS ON THE SAME PAIR (within {RE_EVENT_WINDOW} steps): {len(re_events)}")
    if re_events:
        gaps = np.array([g for _, _, g in re_events])
        print(f"  gap in steps: min={gaps.min()} median={np.median(gaps):.0f} max={gaps.max()}")
        print("  channel transitions: " + ", ".join(
            f"{a}->{b}: {c}" for (a, b), c in Counter((a, b) for a, b, _ in re_events).most_common(6)
        ))

    print()
    print("CAVEAT ON THE ENERGY GATE. The bodies are point masses; the model contains no")
    print("internal binding energy. E_bind = (3/5)Gm^2/R uses the model's own contact radius")
    chi = config.R_REF_DEFAULT / config.SOFTENING
    print("but is a yardstick imported from outside it, and every contact happens inside the")
    print(f"Plummer softening (chi = r_ref/eps = {chi:.2g}), where the force is not Newtonian.")
    print("Read the column as 'how far the event is from being energetically possible', not")
    print("as a conservation violation.")


def write_csv(events, path):
    if not events:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(events[0]))
        writer.writeheader()
        for row in events:
            writer.writerow(row)
    print(f"\nwrote {len(events)} rows to {os.path.normpath(path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n", type=int, default=config.N_PARTICLES)
    parser.add_argument("--steps", type=int, default=config.N_STEPS_COLLAPSE)
    parser.add_argument("--dt", type=float, default=config.DT_COLLAPSE)
    parser.add_argument("--softening", type=float, default=config.SOFTENING)
    parser.add_argument("--seed", type=int, default=config.SEED, help="positions realization")
    parser.add_argument("--collision-seed", type=int, default=config.COLLISION_SEED)
    parser.add_argument(
        "--cold",
        action="store_true",
        help="equal masses at rest (cold_sphere), the Sec. 4.13.6 campaign protocol; the "
        "default instead matches what scripts/realtime.py shows without --cold",
    )
    parser.add_argument("--virial-q", type=float, default=config.Q_DEFAULT)
    args = parser.parse_args()

    result = run(args.n, args.steps, args.dt, args.softening, args.seed, args.collision_seed,
                 args.cold, args.virial_q)
    result["ic_label"] = (
        "cold_sphere -- equal masses, Q = 0 (Sec. 4.13.6 campaign protocol)"
        if args.cold
        else f"random_sphere -- Salpeter spectrum, Q = {args.virial_q:g} "
        f"(scripts/realtime.py default)"
    )
    report(result, args.n, args.steps, args.dt)
    write_csv(result["events"], results_path(args.cold))


if __name__ == "__main__":
    main()
