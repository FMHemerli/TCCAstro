"""Scaling law for the velocity_verlet / rk4 energy-error crossover.

Velocity Verlet holds |dE/E0| in a bounded band B ~ dt^2; RK4 drifts secularly at a rate
R ~ dt^4. The crossover time t_c = B / R is therefore expected to scale as dt^-2.

t_c is computed, not observed: both B and R are measurable over a short horizon, so
integrating for hundreds of t_ff to watch the two curves meet is unnecessary.

Both scalings are asymptotic. A level whose RK4 drift is not linear (R^2 below the acceptance
threshold) lies outside the regime where the argument applies; it is reported and excluded,
and the first level to fail marks where the asymptotic regime ends.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.integrators import FORCE_EVALS_PER_STEP, integrate  # noqa: E402
from nbody.observables import total_energy  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(REPO, "results", "2026", "crossover_scaling_law.csv")
FIG_PATH = os.path.join(REPO, "figures", "crossover_scaling_law.png")

BUDGETS = (10500, 7425, 5250, 3712, 2625)   # force evaluations per t_ff
MEASURE_TFF = 10.0
SAMPLES_PER_TFF = 200
TRANSIENT_CUT_TFF = 1.0      # the first centre crossing is a one-off; the regime starts after it
BAND_PERCENTILE = 95.0
R2_ACCEPT = 0.99
FLOOR = 1e-13                # N * eps_mach, relative; nothing below this is interpretable
STRUCTURAL_HORIZON_TFF = 100.0

HEADER_NOTE = (
    "t_c = B/R computed from a 10 t_ff measurement, not observed. Interpretability floor "
    f"~{FLOOR:g} relative (fp64). Only |dE/E0| is tracked; no trajectory error (Lyapunov time "
    "~1 crossing time, so pointwise error saturates and would be noise). Beyond "
    f"~{STRUCTURAL_HORIZON_TFF:g} t_ff the system has passed several relaxation times: "
    "integrator-vs-integrator comparison remains valid, structural claims about the cluster do not."
)


def run_trace(integrator, backend, budget, n_tff):
    """Integrate n_tff free-fall times, returning (t/t_ff, |dE/E0|) samples."""
    state = cold_sphere(n=config.N_PARTICLES, seed=config.SEED,
                        dtype=torch.float64, device="cuda")
    e0 = total_energy(state, config.SOFTENING)
    steps_per_tff = max(1, round(budget / FORCE_EVALS_PER_STEP[integrator]))
    dt = config.T_FF / steps_per_tff
    n_steps = int(round(steps_per_tff * n_tff))
    every = max(1, round(steps_per_tff / SAMPLES_PER_TFF))

    trace = []

    def callback(step_index, t, s):
        trace.append((t / config.T_FF, abs((total_energy(s, config.SOFTENING) - e0) / abs(e0))))

    t_wall = time.perf_counter()
    integrate(state, integrator=integrator, backend=backend, dt=dt, n_steps=n_steps,
              softening=config.SOFTENING, callback=callback, callback_every=every)
    torch.cuda.synchronize()
    return trace, dt, time.perf_counter() - t_wall


def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (q / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def linear_fit(xs, ys):
    """Least squares y = a*x + b, returning (a, b, r_squared)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def main():
    backend = get_backend("torch_compiled")
    print(f"N={config.N_PARTICLES} fp64 torch_compiled/cuda  eps={config.SOFTENING} m  "
          f"janela de medicao {MEASURE_TFF} t_ff\n")
    print(f"{'budget':>8}{'dt_verlet':>13}{'B (p95)':>13}{'R /t_ff':>13}{'R2':>9}"
          f"{'t_c':>11}  status")

    levels = []
    for budget in BUDGETS:
        v_trace, dt_v, wv = run_trace("velocity_verlet", backend, budget, MEASURE_TFF)
        r_trace, dt_r, wr = run_trace("rk4", backend, budget, MEASURE_TFF)

        band = percentile([d for t, d in v_trace if t > TRANSIENT_CUT_TFF], BAND_PERCENTILE)
        fit_pts = [(t, d) for t, d in r_trace if t > TRANSIENT_CUT_TFF]
        rate, intercept, r2 = linear_fit([t for t, _ in fit_pts], [d for _, d in fit_pts])

        accepted = r2 >= R2_ACCEPT and rate > 0
        t_c = band / rate if rate > 0 else float("nan")
        print(f"{budget:>8}{dt_v:>13.4e}{band:>13.4e}{rate:>13.4e}{r2:>9.4f}{t_c:>11.1f}  "
              f"{'aceito' if accepted else 'EXCLUIDO'}")

        levels.append({
            "row_type": "level", "force_per_tff": budget, "dt_verlet": dt_v, "dt_rk4": dt_r,
            "measure_tff": MEASURE_TFF,
            "band_estimator": f"p{BAND_PERCENTILE:g}_after_{TRANSIENT_CUT_TFF:g}tff_cut",
            "band_B": band, "rk4_rate_R_per_tff": rate, "rk4_fit_intercept": intercept,
            "rk4_fit_r_squared": r2, "rk4_fit_n_points": len(fit_pts),
            "t_c_computed": t_c, "accepted": accepted, "r2_accept_threshold": R2_ACCEPT,
            "wall_time_s": wv + wr, "n_particles": config.N_PARTICLES,
            "softening": config.SOFTENING, "seed": config.SEED, "dtype": "float64",
            "backend": "torch_compiled", "device": "cuda",
        })

    accepted = [lv for lv in levels if lv["accepted"]]
    print()
    if len(accepted) >= 2:
        exponent, log_c, r2_fit = linear_fit(
            [math.log(lv["dt_verlet"]) for lv in accepted],
            [math.log(lv["t_c_computed"]) for lv in accepted],
        )
        print(f"expoente medido: {exponent:.4f}   previsto: -2   R2={r2_fit:.5f}   "
              f"({len(accepted)} niveis aceitos)")
    else:
        exponent = log_c = r2_fit = float("nan")
        print(f"niveis aceitos insuficientes ({len(accepted)}) para ajustar a lei")

    first_rejected = next((lv["force_per_tff"] for lv in levels if not lv["accepted"]), None)
    print("regime assintotico falha a partir de: "
          f"{first_rejected if first_rejected else 'nenhum nivel reprovado'}")

    rows = [{"row_type": "header_note", "notes": HEADER_NOTE}]
    rows += levels
    rows.append({
        "row_type": "fit", "fit_exponent": exponent, "fit_predicted_exponent": -2.0,
        "fit_r_squared": r2_fit, "fit_n_points": len(accepted),
        "first_rejected_budget": first_rejected if first_rejected else "",
        "notes": "least squares of log(t_c) vs log(dt_verlet), accepted levels only",
    })

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {os.path.abspath(CSV_PATH)}")

    if accepted:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog([lv["dt_verlet"] for lv in accepted],
                  [lv["t_c_computed"] for lv in accepted], "o", label="niveis aceitos")
        rejected = [lv for lv in levels if not lv["accepted"]]
        if rejected:
            ax.loglog([lv["dt_verlet"] for lv in rejected],
                      [lv["t_c_computed"] for lv in rejected], "x", color="red",
                      label="excluidos (fora do regime assintotico)")
        if len(accepted) >= 2:
            xs = [lv["dt_verlet"] for lv in accepted]
            fit_x = [min(xs), max(xs)]
            ax.loglog(fit_x, [math.exp(log_c) * x ** exponent for x in fit_x], "-",
                      label=f"ajuste, expoente {exponent:.3f}")
            ax.loglog(fit_x, [accepted[0]["t_c_computed"]
                              * (x / accepted[0]["dt_verlet"]) ** -2.0 for x in fit_x], "--",
                      color="gray", label="previsto, expoente -2")
        ax.set_xlabel("dt do velocity_verlet [s]")
        ax.set_ylabel("t_c / t_ff   (banda do verlet / taxa do rk4)")
        ax.set_title(f"Lei de escala do cruzamento, N={config.N_PARTICLES}, fp64, "
                     f"eps={config.SOFTENING} m")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
        fig.savefig(FIG_PATH, dpi=150)
        print(f"wrote {os.path.abspath(FIG_PATH)}")


if __name__ == "__main__":
    main()
