"""
Stage-3 collision campaign: the K_SEEDS ensemble of docs/simulacao-estocastica.md Sec. 4.13.7.

This script did not exist. The ensemble was run, its results are in Sec. 4.13.7 as prose and
tables, and config.py:100-105 says plainly that the stage-3 wiring "does not exist yet". So
the campaign's own numbers could not be regenerated or checked by anyone reading the
repository, and INV-31(C6) -- min_i m_i / m_bar >= 1e-3 in at least 3 of 4 seeds -- is
recorded there as NAO REPORTADO, "cannot be given as passed". Both of those are what this
script is for.

Protocol, fixed by Sec. 4.13.6 and not a free choice here: cold sphere (equal masses, Q = 0),
chi = 0.1, dt = DT_COLLAPSE, 3 t_ff, the POSITIONS realization held fixed at config.SEED
while only the COLLISION seed varies. That scope is normative when citing INV-31: the
ensemble measures robustness to the collision model's stochastic choices, NOT robustness
across realizations of the initial condition.

Reference values to reproduce (Sec. 4.13.7, run of 2026-08-08):

    seed       N_final   max m / M_real   t_runaway/t_ff   |E_int|/|E_0|
    20190225   774       0.3213           1.9517           10.833
    20190226   783       0.3032           1.9874            9.050
    20190227   915       0.0161           never             0.026   <- |p| = nan, NOT physical
    20190228   798       0.2926           1.9636            7.834

    channels, aggregated over the ensemble: 27.1% elastic, 6.4% fusion, 66.5% fragmentation,
    11331 events

REPRODUCTION VERDICT (2026-08-09, recorded in Sec. 4.13.7.1). INV-31(C6) -- the open clause
-- PASSES 4/4 in both configurations run, by two orders of margin. The per-seed table does
NOT reproduce, on cpu/torch_eager or on cuda/torch_compiled, and the two disagree with each
other as well. Determinism, the OUT_DT chunking, the detect() guard removal and the
"v_coh = V_CHAR" ambiguity were each tested and each eliminated as the cause; what is left is
reduction order in the force kernel's row sums, ~1e-16 at the first step, amplified over
~25 Lyapunov times. The published per-seed values are specific to an environment that was
never recorded. The ensemble-level claims all reproduce.

The 20190227 row is excluded from the physics in Sec. 4.13.7: it ended with |p_final| = nan
and never reproduced in three reruns (which gave N_final = 785, max m = 307.6 m_bar). This
script does not special-case it. integrate() now raises FloatingPointError with the index of
the first non-finite step (integrators.py:241-248), so if it recurs the step index is
reported instead of a silently poisoned trajectory; the run is recorded as failed and the
others continue.

Chunking: the run is advanced in OUT_DT slices rather than one integrate() call, so the time
series needed for t_runaway and max|E_int| can be sampled. This is bit-for-bit equivalent to
a single call ONLY because integrate() accepts a caller-owned collision_rng, advanced in
place, keeping the draw stream continuous across chunk boundaries (integrators.py:176-188).
A fresh generator per chunk would restart the stream every slice and change the trajectory.

Usage:

    python scripts/collision_campaign.py                 # the full 4-seed ensemble, ~35 min
    python scripts/collision_campaign.py --seeds 20190225 --n 200 --n-steps 2000   # smoke

Output: results/2026/collision_campaign_<device>_<backend>_<vcoh>.csv (one row per seed) and
a matching _series.csv (the sampled time series). The filename carries the configuration
because the per-seed numbers differ between configurations by construction.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody import collisions, config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.integrators import integrate  # noqa: E402
from nbody.observables import n_live, total_energy  # noqa: E402

torch.set_default_dtype(torch.float64)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "2026")


def results_paths(device: str, v_coh_mode: str, backend: str) -> tuple[str, str]:
    """Output paths tagged by the configuration that produced them.

    A fixed filename would be wrong here rather than merely untidy: this campaign's whole
    point is comparing configurations, and the numbers differ BETWEEN them by construction
    (see run_one_seed on device and reduction order). Letting a cuda run silently overwrite a
    cpu one would destroy exactly the comparison the script exists to make.
    """
    tag = f"{device}_{backend}_{v_coh_mode}".replace("-", "")
    return (
        os.path.join(RESULTS_DIR, f"collision_campaign_{tag}.csv"),
        os.path.join(RESULTS_DIR, f"collision_campaign_{tag}_series.csv"),
    )

# The four collision seeds of the Sec. 4.13.7 ensemble. COLLISION_SEED is the first.
K_SEEDS = (20190225, 20190226, 20190227, 20190228)

# INV-31(C7): t_runaway is the first t at which the heaviest body reaches this fraction of
# the realized total mass. Sec. 4.13.9 fixed the transition duration as t_end - t_runaway.
RUNAWAY_FRACTION = 0.10
# INV-31(C6): floor on the lightest surviving body, in units of m_bar.
C6_MIN_MASS_OVER_MBAR = 1e-3


class EnsembleTotals:
    """Accumulator across the OUT_DT chunks of one trajectory.

    Not nbody.integrators.CollisionRunStats: that class's update() consumes a
    CollisionOutcome (one resolve() pass), so two CollisionRunStats cannot be composed. Each
    integrate() call hands back its own CollisionRunStats for the chunk it advanced, and
    those are what this adds up. Counts and delta_e_int are additive; c_coll_max and
    f_reject_max are running maxima, so they compose the same way whatever the chunk size.
    """

    def __init__(self):
        self.n_elastic = 0
        self.n_merge = 0
        self.n_fragment = 0
        self.delta_e_int = 0.0
        self.c_coll_max = 0.0
        self.f_reject_max = 0.0

    def add_chunk(self, stats) -> None:
        self.n_elastic += stats.n_elastic
        self.n_merge += stats.n_merge
        self.n_fragment += stats.n_fragment
        self.delta_e_int += stats.delta_e_int
        self.c_coll_max = max(self.c_coll_max, stats.c_coll_max)
        self.f_reject_max = max(self.f_reject_max, stats.f_reject_max)


def resolve_v_coh(state, mode: str) -> float:
    """The regime map's velocity scale, by either of the two readings of "v_coh = V_CHAR".

    They are not the same number. `from-state` recomputes sqrt(G M_real / R_0) from the
    realized masses, which is what nbody.collisions.v_coh_from_state does and what the API
    contract prescribes; `v-char` uses the rounded literal config.V_CHAR = 3.2799885. For the
    equal-mass cold sphere they differ by 1.1e-5 relative -- ten orders of magnitude above
    round-off.

    MEASURED 2026-08-09, seed 20190225, full 3 t_ff protocol: the two give IDENTICAL results
    (N_final = 781, max m/M = 0.3154, 3138 events, |E_int|/|E0| = 10.077). The option was
    added on the hypothesis that this difference explained why the published Sec. 4.13.7
    numbers do not reproduce, and that hypothesis is refuted. The reason is that v_coh does
    not perturb the continuous trajectory at all: it only shifts p_frag, by 1.6e-6, so a draw
    flips only when u1 lands within 1.6e-6 of a channel boundary -- about a 0.5% chance over
    3138 events. None flipped.

    The option is kept because it makes the ambiguity in "v_coh = V_CHAR" explicit and lets
    the null result be re-derived rather than taken on trust.
    """
    if mode == "v-char":
        return config.V_CHAR
    return collisions.v_coh_from_state(state, config.SPHERE_RADIUS)


def run_one_seed(collision_seed, n, n_steps, dt, out_dt, softening, positions_seed, backend,
                 v_coh_mode="from-state", device="cpu"):
    """One trajectory. Returns a summary dict and the sampled series.

    device matters for reproducing published numbers, not just for speed: the force kernel
    sums per row, so a different device or backend reduces in a different order, and the
    collapse decorrelates on a Lyapunov time of ~0.12 t_ff. Over a 3 t_ff run that is ~25
    e-foldings, so two devices agree on the distribution and not on any individual seed.
    """
    state = cold_sphere(n=n, radius=config.SPHERE_RADIUS,
                        particle_mass=config.PARTICLE_MASS, seed=positions_seed,
                        device=device)
    model = collisions.CollisionModel(
        r_ref=config.R_REF_DEFAULT,
        m_bar=config.PARTICLE_MASS,
        v_coh=resolve_v_coh(state, v_coh_mode),
        seed=collision_seed,
    )
    # Owned here, passed to every integrate() call, never rebuilt -- see the module docstring.
    rng = np.random.default_rng(model.seed)

    e0 = total_energy(state, softening)
    steps_per_chunk = max(1, int(round(out_dt / dt)))

    totals = EnsembleTotals()
    series = []
    t_runaway = None
    max_e_int_ratio = 0.0
    failure = ""

    t0 = time.perf_counter()
    step = 0
    while step < n_steps:
        chunk = min(steps_per_chunk, n_steps - step)
        try:
            state, chunk_stats = integrate(
                state, integrator="velocity_verlet", backend=backend, dt=dt, n_steps=chunk,
                softening=softening, collision=model, collision_rng=rng,
            )
        except FloatingPointError as exc:
            # Sec. 4.13.7's 20190227 mode. Located, not silently absorbed: the seed is
            # recorded as failed with the step index and the ensemble continues without it.
            failure = f"FloatingPointError at global step ~{step}: {exc}"
            break

        totals.add_chunk(chunk_stats)
        step += chunk
        t = step * dt

        m64 = state.m.to(torch.float64)
        max_m_fraction = (m64.max() / m64.sum()).item()
        e_int_ratio = abs(totals.delta_e_int) / abs(e0)
        max_e_int_ratio = max(max_e_int_ratio, e_int_ratio)
        if t_runaway is None and max_m_fraction >= RUNAWAY_FRACTION:
            t_runaway = t

        series.append({
            "collision_seed": collision_seed,
            "t_over_tff": t / config.T_FF,
            "n_live": n_live(state),
            "max_m_over_mtotal": max_m_fraction,
            "e_int_over_e0": e_int_ratio,
        })

    wall = time.perf_counter() - t0
    m64 = state.m.to(torch.float64)
    live = m64[m64 > 0]

    return {
        "collision_seed": collision_seed,
        "positions_seed": positions_seed,
        "v_coh_mode": v_coh_mode,
        "device": device,
        "n": n,
        "n_steps": n_steps,
        "dt": dt,
        "n_final": int(live.numel()),
        "max_m_over_mtotal": (live.max() / m64.sum()).item() if live.numel() else float("nan"),
        # INV-31(C6), the clause Sec. 4.13.7 records as NAO REPORTADO.
        "min_m_over_mbar": (live.min() / config.PARTICLE_MASS).item() if live.numel() else float("nan"),
        "n_below_c6_floor": int((live / config.PARTICLE_MASS < C6_MIN_MASS_OVER_MBAR).sum().item()),
        "t_runaway_over_tff": t_runaway / config.T_FF if t_runaway is not None else float("nan"),
        "transition_over_tff": (n_steps * dt - t_runaway) / config.T_FF if t_runaway is not None else float("nan"),
        "max_e_int_over_e0": max_e_int_ratio,
        "n_merge": totals.n_merge,
        "n_elastic": totals.n_elastic,
        "n_fragment": totals.n_fragment,
        "n_events": totals.n_merge + totals.n_elastic + totals.n_fragment,
        "mass_rel_error": abs(m64.sum().item() - config.PARTICLE_MASS * n) / (config.PARTICLE_MASS * n),
        "f_reject_max": totals.f_reject_max,
        "c_coll_max": totals.c_coll_max,
        "wall_seconds": wall,
        "failure": failure,
    }, series


def report(rows):
    print()
    print("=" * 108)
    print("ENSEMBLE SUMMARY")
    print("=" * 108)
    print(f"{'seed':>10}{'N_final':>9}{'max m/M':>10}{'min m/mbar':>12}{'t_run/tff':>11}"
          f"{'trans/tff':>11}{'|E_int|/|E0|':>14}{'events':>8}{'wall':>8}")
    for r in rows:
        note = "  FAILED" if r["failure"] else ""
        print(f"{r['collision_seed']:>10}{r['n_final']:>9}{r['max_m_over_mtotal']:>10.4f}"
              f"{r['min_m_over_mbar']:>12.4g}{r['t_runaway_over_tff']:>11.4f}"
              f"{r['transition_over_tff']:>11.4f}{r['max_e_int_over_e0']:>14.3f}"
              f"{r['n_events']:>8}{r['wall_seconds']:>8.0f}{note}")

    clean = [r for r in rows if not r["failure"]]
    if not clean:
        print("\nevery seed failed; no ensemble statistics")
        return

    merges = sum(r["n_merge"] for r in clean)
    elastics = sum(r["n_elastic"] for r in clean)
    frags = sum(r["n_fragment"] for r in clean)
    total = merges + elastics + frags
    print()
    if total == 0:
        # Reachable from any run too short to reach contact -- the smoke invocations in the
        # docstring are exactly that. Reporting it plainly beats dividing by zero, and beats
        # printing three zero percentages that look like a measured channel distribution.
        print(f"no collision events over {len(clean)} clean seed(s): the run is too short to "
              f"reach contact, and no channel statistic is defined")
        return
    print(f"channels aggregated over {len(clean)} clean seed(s), {total} events: "
          f"{100 * elastics / total:.1f}% elastic, {100 * merges / total:.1f}% fusion, "
          f"{100 * frags / total:.1f}% fragmentation")

    print()
    print("INV-31 clauses this run can speak to:")
    print(f"  (C1) no seed ends with N_final == 1        : "
          f"{'PASS' if all(r['n_final'] > 1 for r in clean) else 'FAIL'}")
    median_events = float(np.median([r["n_events"] for r in clean]))
    print(f"  (C2) median events >= 50                   : "
          f"{'PASS' if median_events >= 50 else 'FAIL'}  (median {median_events:.0f})")
    ok_c5 = total and min(elastics, merges, frags) / total >= 0.05
    print(f"  (C5) every channel >= 5% aggregated        : {'PASS' if ok_c5 else 'FAIL'}")
    n_c6 = sum(1 for r in clean if r["min_m_over_mbar"] >= C6_MIN_MASS_OVER_MBAR)
    print(f"  (C6) min m/m_bar >= 1e-3 in >= 3 of 4      : "
          f"{'PASS' if n_c6 >= 3 else 'FAIL'}  ({n_c6} of {len(clean)} clean seeds)")
    print("       ^ Sec. 4.13.7 records this clause as NAO REPORTADO; this is the measurement")
    print("         it was missing. It is only decisive with the full 4-seed protocol.")
    print(f"  (C7) runaway detected                      : "
          f"{sum(1 for r in clean if not math.isnan(r['t_runaway_over_tff']))} of {len(clean)} seeds")
    worst_mass = max(r["mass_rel_error"] for r in clean)
    print(f"  mass conserved, worst relative error       : {worst_mass:.3e}")


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {os.path.normpath(path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(K_SEEDS))
    parser.add_argument("--n", type=int, default=config.N_PARTICLES)
    parser.add_argument("--n-steps", type=int, default=config.N_STEPS_COLLAPSE)
    parser.add_argument("--dt", type=float, default=config.DT_COLLAPSE)
    parser.add_argument("--out-dt", type=float, default=config.OUT_DT)
    parser.add_argument("--softening", type=float, default=config.SOFTENING)
    parser.add_argument("--positions-seed", type=int, default=config.SEED,
                        help="held FIXED across the ensemble by the protocol; varying it "
                             "measures something else (Sec. 4.13.7)")
    parser.add_argument("--backend", default="torch_eager")
    parser.add_argument("--device", default="cpu",
                        help="cpu or cuda; see run_one_seed on why this changes the numbers")
    parser.add_argument(
        "--v-coh", choices=("from-state", "v-char"), default="from-state",
        help="which reading of 'v_coh = V_CHAR' to use; see resolve_v_coh",
    )
    args = parser.parse_args()

    backend = get_backend(args.backend)
    print(f"stage-3 collision campaign: {len(args.seeds)} seeds, N={args.n}, "
          f"{args.n_steps} steps, dt={args.dt:g} s "
          f"({args.n_steps * args.dt / config.T_FF:.3f} t_ff), backend={args.backend}")
    print(f"cold sphere (equal masses, Q=0), positions seed {args.positions_seed} held fixed, "
          f"chi = {config.R_REF_DEFAULT / config.SOFTENING:g}, v_coh = {args.v_coh}, "
          f"device = {args.device}")

    rows, all_series = [], []
    for seed in args.seeds:
        print(f"\n  seed {seed} ...", flush=True)
        row, series = run_one_seed(seed, args.n, args.n_steps, args.dt, args.out_dt,
                                   args.softening, args.positions_seed, backend, args.v_coh,
                                   args.device)
        if row["failure"]:
            print(f"    {row['failure']}")
        else:
            print(f"    N_final={row['n_final']}  max m/M={row['max_m_over_mtotal']:.4f}  "
                  f"events={row['n_events']}  {row['wall_seconds']:.0f}s")
        rows.append(row)
        all_series.extend(series)

    report(rows)
    print()
    summary_path, series_path = results_paths(args.device, args.v_coh, args.backend)
    write_csv(summary_path, rows)
    write_csv(series_path, all_series)


if __name__ == "__main__":
    main()
