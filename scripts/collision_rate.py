"""
Collision-rate campaign: measures the quantity that fixes the contact-radius parameter chi,
with detection turned on and NO resolution channel implemented -- candidate pairs are
counted and discarded every step, the state evolves exactly as the collisionless collapse
would. This is what makes the measurement a clean baseline for choosing chi
(docs/simulacao-estocastica.md, Sec. 4.1: "Este documento nao fixa chi; fixa o que medi-lo
significa").

The step loop below is velocity-Verlet KDK, reproduced here by hand rather than through
nbody.integrators.integrate(): detection needs the mid-step velocity v^(n+1/2) (Sec. 4.3 --
"a hipotese de movimento retilineo e exata para o drift do Verlet ... se dv for tomado como
a diferenca das velocidades de meio passo"), which integrate() does not expose. With zero
resolution applied, this loop is bit-for-bit the same KDK step integrate() performs for
velocity_verlet; only an extra, side-effect-free measurement (nbody.collisions.detect +
pair_disjoint) is inserted at the drift.

Sweeps chi = R_ref / SOFTENING over the grid in CHI_GRID, for two mass populations (equal
mass and the Sec. 2 mass spectrum), always at Q = 0 (cold collapse) so the only axis varied
here is chi x mass spectrum -- virial ratio is Sec. 3's independent axis and is held at its
INV-17 base case. dt = DT_COLLISION is fixed across the whole sweep (Sec. 8), not rescaled
per chi, since the campaign's second purpose is to check whether C_coll measured over this
sweep bears out the dt = DT_COLLAPSE/4 decision of Sec. 4.4.

DETECTIONS VS. ENCOUNTERS, AND WHY THE DISTINCTION IS LOAD-BEARING HERE. Sec. 4.5 resolves a
collision in the same step it is detected, at t*, inside the drift -- the pair is moved apart
by whatever the outcome map does, and Sec. 4.5 explicitly VETOES leaving a residual overlap
in place ("separar ate contato exato" is forbidden precisely because the adopted scheme never
lets an overlap persist). This script implements no outcome map: detection runs, candidates
are counted, and the state is left untouched. Consequently a pair that comes into contact
does NOT separate afterwards -- it can stay swept-detected, unresolved, for as long as its
orbit keeps it inside R_i+R_j, which for a slow or bound pair can be hundreds of consecutive
steps. Counting every step's detection as an "event" therefore counts one physical contact
once per step it persists, inflating totals by orders of magnitude and, worse, producing a
detection rate that looks flat from t=0 instead of bursting at core crossing.

The fix applied throughout this script: maintain the set of pairs currently overlapping
(`in_contact`, keyed on the constant per-particle contact radii, since mass is never touched
in this no-resolution baseline) and count a new ENCOUNTER only on the "not overlapping" ->
"overlapping" transition. A pair leaves `in_contact` once its post-step separation clears
R_i+R_j. DETECTIONS (the raw, step-repeated swept-candidate count) are still recorded --
n_detections_total -- because the ratio n_detections_total / n_encounters_total is itself
informative: it is the mean number of steps a contact persists unresolved, i.e. a proxy for
the contact duration that the real resolution step would eliminate by construction.
max_simultaneous_accepted_pairs and sweep_only_fraction have the same persistence bias
(a long-lived stuck pair inflates "simultaneous accepted" counts across many unrelated later
steps, and dilutes sweep_only_fraction because a persistently overlapping pair reports
"still overlapping at end of step" on every repeat, not just its first). Both are recomputed
on the encounter transition instead of the raw per-step stream; the raw, biased quantity is
also kept (suffixed _raw) for transparency. f_reject is Sec. 4.5's own per-candidate,
whole-execution aggregate (INV-19(d)) and is left as literally defined, since that IS the
quantity INV-19 constrains -- but the same persistence effect dilutes it here (a stuck,
uncontested pair contributes many zero-conflict candidate instances to both its numerator and
denominator), so the measured value should be read as a probably-optimistic bound on the
resolved model's true conflict rate, not an exact one.

LIMITATION THIS BASELINE CANNOT REMOVE (declare, do not discover later). Encounter counting
recovers the RATE at which contacts begin, which is the quantity Sec. 4.1 needs to fix chi.
It does not, and cannot, recover the SEQUENCE of events a resolved run would show beyond the
very first encounter of any given particle: resolving that encounter changes its velocity
(and, for merge/fragment, its mass and slot), so every trajectory downstream of a resolved
event would diverge from what this unresolved baseline computes. The event count and its
timing up to and including a particle's first encounter are meaningful; anything conditioned
on "what happens to this particle after it has already collided once" is not, in this script.

Output: results/2026/collision_rate.csv (row_type in {"event_bin", "summary"}: one
event_bin row per (chi, population, time bin) with the ENCOUNTER count in that bin -- the
main deliverable is this temporal distribution, Sec. 4.1 predicts a burst at core crossing --
and one summary row per (chi, population) with the scalar diagnostics: totals (both
detections and encounters), encounters per particle per bounce (Sec. 4.1), max simultaneous
new encounters in a single step, f_reject (Sec. 4.5 / INV-19), C_coll measured (Sec. 4.4 /
INV-18), the fraction of newly-accepted encounters that a static end-of-step overlap test
would have missed, and the core number density measured at peak compression (the [A] mark in
Sec. 4.1 this campaign is asked to resolve: 1.4e3 m^-3 per docs/integradores.md Sec. 4.3 vs.
2852 m^-3 from the direct calculation in docs/simulacao-estocastica.md Sec. 4.1).
figures/collision_rate.png (encounters per t_ff vs t/t_ff, all chi values, one panel per
population), built only from the written CSV.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import platform
import re
import socket
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.collisions import CollisionModel, contact_radii, detect, pair_disjoint  # noqa: E402
from nbody.initial_conditions import random_sphere  # noqa: E402
from nbody.observables import center_of_mass, half_mass_radius, n_live, scales_from_state  # noqa: E402
from nbody.state import State  # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "2026", "collision_rate.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

BACKEND_NAME = "torch_eager"
DEVICE = "cuda"
DTYPE = torch.float64

CHI_GRID = (0.05, 0.1, 0.2, 0.5, 1.0)
POPULATIONS = ("equal_mass", "mass_spectrum")

DT = config.DT_COLLISION
N_STEPS = config.N_STEPS_COLLISION
BOUNCE_HALF_WINDOW_TFF = 0.05  # Sec. 4.1: [t_bounce - 0.05 t_ff, t_bounce + 0.05 t_ff]
N_TIME_BINS = 150  # ~150 bins over the ~3 t_ff run -> bin width ~0.02 t_ff

CSV_FIELDNAMES = [
    "row_type",
    "chi",
    "r_ref",
    "population",
    "n_particles",
    "n_live",
    "softening",
    "seed",
    "mass_seed",
    "backend",
    "device",
    "dtype",
    "dt",
    "n_steps",
    "t_end",
    "t_ff_realized",
    "total_mass_realized",
    "bin_index",
    "t_over_tff_bin_center",
    "encounters_in_bin",
    "n_detections_total",
    "n_encounters_total",
    "detections_per_encounter",
    "n_coll_per_particle",
    "n_live_at_bounce",
    "t_bounce_over_tff",
    "max_simultaneous_new_encounters",
    "max_simultaneous_accepted_pairs_raw",
    "f_reject_total",
    "n_candidates_total",
    "n_accepted_total",
    "n_rejected_total",
    "c_coll_max",
    "c_coll_max_t_over_tff",
    "sweep_only_fraction",
    "n_new_accepted_encounters",
    "r_half_min",
    "r_half_min_t_over_tff",
    "core_n_in_half_mass_radius",
    "core_number_density",
    "wall_time_s",
    "hostname",
    "os_platform",
    "cpu_model",
    "gpu_name",
    "gpu_runtime_version",
    "torch_version",
    "python_version",
    "timestamp_utc",
]


def env_metadata() -> dict:
    gpu_name, gpu_runtime = "", ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if torch.version.hip is not None:
            gpu_runtime = f"hip {torch.version.hip}"
        elif torch.version.cuda is not None:
            gpu_runtime = f"cuda {torch.version.cuda}"
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            m = re.search(r"^model name\s*:\s*(.+)$", fh.read(), re.MULTILINE)
        if m:
            cpu_model = m.group(1).strip()
    except OSError as exc:
        cpu_model = f"unavailable ({exc})"
    return {
        "hostname": socket.gethostname(),
        "os_platform": platform.platform(),
        "cpu_model": cpu_model,
        "gpu_name": gpu_name,
        "gpu_runtime_version": gpu_runtime,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def build_initial_state(population: str, device: str) -> State:
    if population == "equal_mass":
        return random_sphere(
            n=config.N_PARTICLES,
            radius=config.SPHERE_RADIUS,
            seed=config.SEED,
            mass_spectrum=None,
            virial_ratio=0.0,
            dtype=DTYPE,
            device=device,
        )
    if population == "mass_spectrum":
        return random_sphere(
            n=config.N_PARTICLES,
            radius=config.SPHERE_RADIUS,
            seed=config.SEED,
            mass_spectrum={
                "alpha": config.MASS_ALPHA,
                "ratio": config.MASS_RATIO,
                "particle_mass": config.PARTICLE_MASS,
            },
            virial_ratio=0.0,
            mass_seed=config.MASS_SEED,
            dtype=DTYPE,
            device=device,
        )
    raise ValueError(f"unknown population {population!r}")


def run_one(chi: float, population: str, backend, env: dict) -> list[dict]:
    r_ref = chi * config.SOFTENING
    model = CollisionModel(r_ref=r_ref, m_bar=config.PARTICLE_MASS)

    state = build_initial_state(population, DEVICE)
    n_live0 = n_live(state)
    scales = scales_from_state(state, softening=config.SOFTENING, radius=config.SPHERE_RADIUS)
    t_ff = scales["t_ff"]

    # Mass is never touched in this no-resolution baseline, so per-particle contact radii are
    # constant for the whole run; this lets the "has this tracked pair left contact yet" check
    # below index directly instead of recomputing model radii every step.
    radii_all = contact_radii(state.m, model)

    bin_width_s = (N_STEPS * DT) / N_TIME_BINS
    bin_encounters = [0] * N_TIME_BINS

    n_candidates_total = 0
    n_accepted_total = 0
    n_rejected_total = 0
    n_encounters_total = 0
    n_sweep_only = 0
    n_new_accepted_encounters = 0
    max_simultaneous_accepted_raw = 0
    max_simultaneous_new_encounters = 0
    c_coll_max = 0.0
    c_coll_max_t = float("nan")

    r_half_min = float("inf")
    r_half_min_t = float("nan")
    r_half_min_step = -1
    r_half_min_state = None

    # Pairs currently overlapping (|r_i - r_j| < R_i + R_j), tracked across steps so a
    # persistent overlap is counted as ONE encounter, not once per step it survives
    # unresolved. See module docstring, "DETECTIONS VS. ENCOUNTERS".
    in_contact: set[tuple[int, int]] = set()

    a_current = backend.accelerations(state.r, state.m, config.SOFTENING)

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        v_half = state.v + 0.5 * DT * a_current
        detect_state = State(r=state.r, v=v_half, m=state.m)

        candidates = detect(detect_state, DT, model)
        accepted = pair_disjoint(candidates)

        n_cand = candidates.n
        n_acc = accepted.n
        n_candidates_total += n_cand
        n_accepted_total += n_acc
        n_rejected_total += n_cand - n_acc
        if n_acc > max_simultaneous_accepted_raw:
            max_simultaneous_accepted_raw = n_acc

        if n_cand > 0:
            c_coll = (candidates.rel_speed * DT / candidates.contact_radius_sum).max().item()
            if c_coll > c_coll_max:
                c_coll_max = c_coll
                c_coll_max_t = (step * DT) / t_ff

        # New encounters: candidates (pre-pairing -- disjoint-pairing conflicts are a
        # resolution-order artifact, not a statement about whether a pair is physically
        # touching) not already tracked as overlapping. A pair enters `in_contact`
        # regardless of whether pair_disjoint accepts or defers it this step.
        cand_i = candidates.i.tolist()
        cand_j = candidates.j.tolist()
        cand_t = candidates.t_star.tolist()

        new_pairs = []
        for k in range(n_cand):
            pair = (cand_i[k], cand_j[k])
            if pair not in in_contact:
                new_pairs.append((pair, cand_t[k]))
                in_contact.add(pair)

        if len(new_pairs) > max_simultaneous_new_encounters:
            max_simultaneous_new_encounters = len(new_pairs)

        if new_pairs:
            n_encounters_total += len(new_pairs)
            for pair, t_star in new_pairs:
                t_event = step * DT + t_star
                b = min(int(t_event / bin_width_s), N_TIME_BINS - 1)
                bin_encounters[b] += 1

        r_new = state.r + DT * v_half
        a_new = backend.accelerations(r_new, state.m, config.SOFTENING)
        v_new = v_half + 0.5 * DT * a_new
        new_state = State(r=r_new, v=v_new, m=state.m)

        # sweep-only fraction, restricted to encounters that are both new and accepted this
        # step (i.e. the ones a resolution map would act on at this t*, per Sec. 4.5): would a
        # static end-of-step overlap test on r_new have caught this same pair?
        if new_pairs and n_acc > 0:
            accepted_set = set(zip(accepted.i.tolist(), accepted.j.tolist()))
            new_accepted = [pair for pair, _ in new_pairs if pair in accepted_set]
            if new_accepted:
                ai = torch.tensor([p[0] for p in new_accepted], device=DEVICE, dtype=torch.int64)
                aj = torch.tensor([p[1] for p in new_accepted], device=DEVICE, dtype=torch.int64)
                final_sep = torch.linalg.vector_norm(r_new[ai] - r_new[aj], dim=-1)
                r_sum = radii_all[ai] + radii_all[aj]
                n_sweep_only += int((final_sep >= r_sum).sum().item())
                n_new_accepted_encounters += len(new_accepted)

        # A tracked pair leaves `in_contact` once its post-step separation clears R_i+R_j --
        # the transition that ends this encounter's persistence window and lets the same pair
        # count as a fresh encounter if it later comes back into contact.
        if in_contact:
            tracked = list(in_contact)
            ti = torch.tensor([p[0] for p in tracked], device=DEVICE, dtype=torch.int64)
            tj = torch.tensor([p[1] for p in tracked], device=DEVICE, dtype=torch.int64)
            sep_now = torch.linalg.vector_norm(r_new[ti] - r_new[tj], dim=-1)
            r_sum_now = radii_all[ti] + radii_all[tj]
            keep_mask = (sep_now < r_sum_now).tolist()
            in_contact = {pair for pair, keep in zip(tracked, keep_mask) if keep}

        state = new_state
        a_current = a_new

        r_half = half_mass_radius(state)
        if r_half < r_half_min:
            r_half_min = r_half
            r_half_min_t = ((step + 1) * DT) / t_ff
            r_half_min_step = step + 1
            r_half_min_state = state

    torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - t0

    t_bounce = r_half_min_t * t_ff
    window_lo = t_bounce - BOUNCE_HALF_WINDOW_TFF * t_ff
    window_hi = t_bounce + BOUNCE_HALF_WINDOW_TFF * t_ff
    bin_centers_s = [(b + 0.5) * bin_width_s for b in range(N_TIME_BINS)]
    n_encounters_in_bounce_window = sum(
        count for center, count in zip(bin_centers_s, bin_encounters) if window_lo <= center <= window_hi
    )
    n_coll_per_particle = 2.0 * n_encounters_in_bounce_window / n_live0

    core_center = center_of_mass(r_half_min_state)
    r64 = r_half_min_state.r.to(torch.float64)
    m64 = r_half_min_state.m.to(torch.float64)
    live_mask = m64 > 0
    d = torch.linalg.vector_norm(r64 - core_center.unsqueeze(0), dim=-1)
    core_n = int(((d <= r_half_min) & live_mask).sum().item())
    core_volume = (4.0 / 3.0) * math.pi * r_half_min**3
    core_density = core_n / core_volume

    f_reject_total = n_rejected_total / n_candidates_total if n_candidates_total > 0 else 0.0
    sweep_only_fraction = n_sweep_only / n_new_accepted_encounters if n_new_accepted_encounters > 0 else 0.0
    detections_per_encounter = n_accepted_total / n_encounters_total if n_encounters_total > 0 else float("nan")

    common = {
        "chi": chi,
        "r_ref": r_ref,
        "population": population,
        "n_particles": config.N_PARTICLES,
        "n_live": n_live0,
        "softening": config.SOFTENING,
        "seed": config.SEED,
        "mass_seed": config.MASS_SEED if population == "mass_spectrum" else "",
        "backend": BACKEND_NAME,
        "device": DEVICE,
        "dtype": "float64",
        "dt": DT,
        "n_steps": N_STEPS,
        "t_end": N_STEPS * DT,
        "t_ff_realized": t_ff,
        "total_mass_realized": scales["total_mass"],
    }
    common.update(env)

    rows = []
    for b in range(N_TIME_BINS):
        row = dict.fromkeys(CSV_FIELDNAMES, "")
        row.update(common)
        row["row_type"] = "event_bin"
        row["bin_index"] = b
        row["t_over_tff_bin_center"] = bin_centers_s[b] / t_ff
        row["encounters_in_bin"] = bin_encounters[b]
        rows.append(row)

    summary = dict.fromkeys(CSV_FIELDNAMES, "")
    summary.update(common)
    summary["row_type"] = "summary"
    summary["n_detections_total"] = n_accepted_total
    summary["n_encounters_total"] = n_encounters_total
    summary["detections_per_encounter"] = detections_per_encounter
    summary["n_coll_per_particle"] = n_coll_per_particle
    summary["n_live_at_bounce"] = n_live0
    summary["t_bounce_over_tff"] = r_half_min_t
    summary["max_simultaneous_new_encounters"] = max_simultaneous_new_encounters
    summary["max_simultaneous_accepted_pairs_raw"] = max_simultaneous_accepted_raw
    summary["f_reject_total"] = f_reject_total
    summary["n_candidates_total"] = n_candidates_total
    summary["n_accepted_total"] = n_accepted_total
    summary["n_rejected_total"] = n_rejected_total
    summary["c_coll_max"] = c_coll_max
    summary["c_coll_max_t_over_tff"] = c_coll_max_t
    summary["sweep_only_fraction"] = sweep_only_fraction
    summary["n_new_accepted_encounters"] = n_new_accepted_encounters
    summary["r_half_min"] = r_half_min
    summary["r_half_min_t_over_tff"] = r_half_min_t
    summary["core_n_in_half_mass_radius"] = core_n
    summary["core_number_density"] = core_density
    summary["wall_time_s"] = wall_time_s
    rows.append(summary)

    print(
        f"  chi={chi:<5} pop={population:<14} t_ff={t_ff:.6f}s t_bounce={r_half_min_t:.4f}t_ff "
        f"n_encounters={n_encounters_total:>6} n_detections={n_accepted_total:>7} "
        f"(x{detections_per_encounter:.1f}) N_coll/particle={n_coll_per_particle:.4f} "
        f"max_simult_new={max_simultaneous_new_encounters:>4} f_reject={f_reject_total:.4f} "
        f"C_coll_max={c_coll_max:.4f} sweep_only={sweep_only_fraction:.4f} "
        f"core_n_v={core_density:.1f} wall={wall_time_s:.1f}s",
        flush=True,
    )

    return rows


def main() -> None:
    global N_STEPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=N_STEPS, help=f"override step count (default {N_STEPS})")
    parser.add_argument(
        "--chi-list", type=str, default=",".join(str(c) for c in CHI_GRID), help="comma-separated chi values"
    )
    parser.add_argument(
        "--populations", type=str, default=",".join(POPULATIONS), help="comma-separated population labels"
    )
    args = parser.parse_args()

    N_STEPS = args.n_steps
    chi_list = [float(c) for c in args.chi_list.split(",")]
    populations = [p.strip() for p in args.populations.split(",")]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/ROCm device not available on this host; refusing to silently fall back "
            "to a different device for a campaign whose backend/device choice is normative."
        )

    torch.set_default_dtype(DTYPE)
    backend = get_backend(BACKEND_NAME)
    env = env_metadata()

    print("=" * 100)
    print(
        f"collision-rate campaign: N={config.N_PARTICLES}, dtype=fp64, eps={config.SOFTENING}, "
        f"dt={DT}, n_steps={N_STEPS}, backend={BACKEND_NAME}/{DEVICE}"
    )
    print(f"chi grid: {chi_list}, populations: {populations}")
    for k, v in env.items():
        print(f"  {k:<24}{v}")
    print("=" * 100)

    all_rows = []
    for population in populations:
        for chi in chi_list:
            all_rows.extend(run_one(chi, population, backend, env))

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\nwrote {len(all_rows)} rows to {os.path.abspath(RESULTS_PATH)}")

    make_figure(all_rows, chi_list, populations)


def make_figure(all_rows, chi_list, populations) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(populations), figsize=(7 * len(populations), 5), squeeze=False)
    axes = axes[0]

    for ax, population in zip(axes, populations):
        for chi in chi_list:
            bins = [
                row
                for row in all_rows
                if row["row_type"] == "event_bin" and row["population"] == population and row["chi"] == chi
            ]
            bins.sort(key=lambda row: row["bin_index"])
            xs = [row["t_over_tff_bin_center"] for row in bins]
            ys = [row["encounters_in_bin"] for row in bins]
            if xs:
                ax.plot(xs, ys, label=f"chi={chi}", linewidth=1)
        ax.set_xlabel("t / t_ff")
        ax.set_ylabel("new encounters per bin")
        ax.set_title(f"population={population}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Collision encounter rate vs time, N={config.N_PARTICLES}, eps={config.SOFTENING} m, "
        f"dt={DT} s, {BACKEND_NAME}/{DEVICE}, no resolution "
        f"(encounter = first step a pair overlaps, not every step it stays unresolved)"
    )
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "collision_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
