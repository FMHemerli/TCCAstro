"""
Long-run energy-drift study: does |dE/E0| grow linearly in time for rk4 and euler over
many free-fall times, while velocity_verlet and symplectic_euler stay in a bounded band
that does not grow from crossing to crossing?

Cold spherical collapse, N=1000, fp64, eps=0.05 (config.SOFTENING, "o da especificacao"),
torch_compiled/cuda. Four integrators, single metric: |dE/E0(t)|.

This is not the integrator_study.py experiment and does not reuse its reference or its
err_r metric. Here the initial state is cold (v=0 for every particle), so the total
energy of the continuous problem is E(t) = U_0 exactly -- an identity, not an
approximation (docs/tcc-2019-extrato.md, "Forca e potencial", confirms the 2019 force and
potential are already the Plummer pair and that the identity held exactly in 2019 too).
No reference trajectory is built or needed: any nonzero |dE/E0| is integrator error, full
stop.

This experiment answers a specific claim from the 2019 monograph
(docs/tcc-2019-extrato.md, "Criterio de erro" and "Justificativa documentada para Verlet e
Runge-Kutta"): the monograph fixed a 5% energy-drift tolerance (Sec. 3.5, declared
arbitrary) and named Verlet/RK4 as methods that could reduce error at extra cost
(Sec. 3.4). The main deliverable is, per integrator, the first t/t_ff at which |dE/E0|
crosses that 5% tolerance -- the column that compares the two eras directly.

Equal cost per unit physical time: dt is chosen per integrator so that all four spend the
same number of force evaluations per t_ff (FORCE_EVALS_PER_STEP), i.e. rk4 (k=4) takes
4x larger steps than the three k=1 integrators. FORCE_EVALS_STARTUP (only
velocity_verlet, s=1, once per run) is a one-time cost folded into n_force_total but not
into the per-t_ff budget -- negligible over a multi-t_ff run, reported for transparency.

FORCE_PER_TFF = 10500 is not a free choice: it reproduces the smallest budget row in
results/2026/integrator_study.csv (budget_nominal=12600 force evals over
T_END=1.2*t_ff => 12600/1.2 = 10500 force evals per t_ff exactly, with no remainder for
either k=1 or k=4). That is the budget the task's own prediction (rk4 ~2.1e-7/t_ff,
velocity_verlet band ~3.1e-5) was read off, so it is the budget this script must use to
test that prediction, not an independently chosen density.

Two stages, controlled by --n-tff (default 10 = Stage 1). Stage 1 samples densely and
fits: (a) a line to |dE/E0|(t/t_ff) after the first t_ff for rk4 and euler -- reporting
slope and fit quality, testing whether the drift is linear at all; (b) a line to the
per-t_ff windowed maximum of |dE/E0| for velocity_verlet and symplectic_euler -- testing
whether that band grows from crossing to crossing or stays flat. Stage 1 must be reported
and approved before Stage 2 (--n-tff up to ~200, or to the predicted rk4/velocity_verlet
crossing at ~150 t_ff) is run; this script does not gate that itself, the operator does.

Caveats that apply to every row beyond a few t_ff (see "Ressalvas do agente fisico" in the
task): point-wise trajectory error is not tracked here at all (no reference, no err_r);
if it had been, it would have saturated well inside this horizon since the Lyapunov time
of a self-gravitating N-body system is of order one crossing time. Beyond ~100 t_ff the
cluster has been through several relaxation times -- comparison BETWEEN integrators (same
initial state and same physical time for all four) remains valid, but no STRUCTURAL claim
about the cluster (density profile, virial state, etc.) is supported by this run; none is
made. The diagnostic floor is ~N*eps_mach ~ 1e-13 relative in fp64; nothing below that is
interpreted as signal.

Output: results/2026/longrun_energy.csv (row_type in {"sample", "summary"}: one sample
row per recorded callback point, one summary row per integrator with the fits and the
5%-crossing result) and figures/longrun_energy_drift.png (|dE/E0| vs t/t_ff, log y, all
four integrators, 5% line marked).
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
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.backends.base import BackendUnavailable  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.integrators import FORCE_EVALS_PER_STEP, FORCE_EVALS_STARTUP, INTEGRATOR_NAMES, integrate  # noqa: E402
from nbody.observables import linear_momentum, total_energy  # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "2026", "longrun_energy.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

BACKEND_NAME = "torch_compiled"
DEVICE = "cuda"
DTYPE = torch.float64

N_PARTICLES = config.N_PARTICLES
SOFTENING = config.SOFTENING
SEED = config.SEED

# Force evaluations spent per t_ff, identical for all four integrators (see module
# docstring for the derivation: reproduces integrator_study.csv budget=12600 at
# T_END=1.2 t_ff).
FORCE_PER_TFF_DEFAULT = 10500
SAMPLES_PER_TFF_DEFAULT = 300
DRIFT_TOLERANCE = 0.05  # 2019 monograph, Sec. 3.5, declared arbitrary

# Which analysis applies to which integrator (see module docstring).
LINEAR_DRIFT_INTEGRATORS = ("euler", "rk4")
BAND_TREND_INTEGRATORS = ("symplectic_euler", "velocity_verlet")

CSV_FIELDNAMES = [
    "row_type",
    "integrator",
    "stage",
    "n_particles",
    "softening",
    "seed",
    "backend",
    "device",
    "dtype",
    "t_ff",
    "force_per_tff_nominal",
    "force_evals_per_step",
    "force_evals_startup",
    "dt",
    "steps_per_tff",
    "n_tff_simulated",
    "n_steps_total",
    "n_force_total",
    "t_end",
    "sample_index",
    "t",
    "t_over_tff",
    "dE_E0",
    "abs_dE_E0",
    "n_samples",
    "callback_every",
    "diagnostic_overhead_fraction",
    "wall_time_s",
    "finite",
    "max_abs_dE_E0",
    "final_dE_E0",
    "final_t_over_tff",
    "drift_tolerance",
    "crossed_tolerance",
    "t_over_tff_cross_tolerance",
    "fit_type",
    "fit_domain_start_t_over_tff",
    "fit_n_points",
    "fit_slope_per_tff",
    "fit_intercept",
    "fit_r_squared",
    "characteristic_scale",
    "trend_over_horizon_fraction",
    "momentum_final_norm_scaled",
    "notes",
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


def run_cost(integrator: str, force_per_tff: int, n_tff: int) -> dict:
    k = FORCE_EVALS_PER_STEP[integrator]
    s = FORCE_EVALS_STARTUP[integrator]
    steps_per_tff = force_per_tff // k
    if steps_per_tff * k != force_per_tff:
        raise ValueError(
            f"force_per_tff={force_per_tff} is not evenly divisible by k={k} for {integrator!r}; "
            "equal-cost-per-t_ff requires an exact ratio, choose a force_per_tff divisible by 4"
        )
    dt = config.T_FF / steps_per_tff
    n_steps_total = steps_per_tff * n_tff
    n_force_total = n_steps_total * k + s
    return {
        "steps_per_tff": steps_per_tff,
        "dt": dt,
        "n_steps_total": n_steps_total,
        "n_force_total": n_force_total,
    }


def linear_fit(xs, ys) -> dict:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    n = x.size
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": float("nan"), "n": n}
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": r_squared, "n": n}


def crossing_time(trace, tolerance: float):
    """First t/t_ff at which |dE/E0| crosses tolerance, linearly interpolated between the
    two straddling samples. Returns None if the tolerance is never crossed in this trace."""
    prev_t_tff, prev_abs = None, None
    for t_tff, abs_drift in trace:
        if prev_abs is not None and prev_abs < tolerance <= abs_drift:
            if abs_drift == prev_abs:
                return t_tff
            frac = (tolerance - prev_abs) / (abs_drift - prev_abs)
            return prev_t_tff + frac * (t_tff - prev_t_tff)
        prev_t_tff, prev_abs = t_tff, abs_drift
    return None


def analyze_band_trend(trace, n_tff: int) -> dict:
    """Windowed max of |dE/E0| per t_ff bin (crossing-to-crossing), linear fit of that
    windowed max against bin center in t_ff units."""
    bins_max = [None] * n_tff
    for t_tff, abs_drift in trace:
        idx = int(math.floor(t_tff))
        if idx == n_tff:
            idx = n_tff - 1  # last sample lands exactly on the upper edge
        if 0 <= idx < n_tff:
            if bins_max[idx] is None or abs_drift > bins_max[idx]:
                bins_max[idx] = abs_drift
    xs = [i + 0.5 for i, v in enumerate(bins_max) if v is not None]
    ys = [v for v in bins_max if v is not None]
    fit = linear_fit(xs, ys)
    scale = float(np.mean(ys)) if ys else float("nan")
    horizon = (max(xs) - min(xs)) if len(xs) >= 2 else float("nan")
    trend_fraction = (
        abs(fit["slope"] * horizon) / scale if (scale and scale > 0.0 and not math.isnan(fit["slope"])) else float("nan")
    )
    return {
        "fit_type": "windowed_max_per_tff_trend",
        "fit_domain_start_t_over_tff": xs[0] if xs else float("nan"),
        "fit": fit,
        "characteristic_scale": scale,
        "trend_over_horizon_fraction": trend_fraction,
    }


def analyze_linear_drift(trace) -> dict:
    """Linear fit of |dE/E0|(t/t_ff) restricted to t/t_ff >= 1.0, i.e. after the first
    t_ff, so the initial transient is excluded from the slope estimate."""
    domain = [(t_tff, abs_drift) for t_tff, abs_drift in trace if t_tff >= 1.0]
    xs = [t for t, _ in domain]
    ys = [d for _, d in domain]
    fit = linear_fit(xs, ys)
    return {
        "fit_type": "linear_drift_after_1tff",
        "fit_domain_start_t_over_tff": xs[0] if xs else float("nan"),
        "fit": fit,
        "characteristic_scale": float("nan"),
        "trend_over_horizon_fraction": float("nan"),
    }


def run_one(integrator: str, force_per_tff: int, n_tff: int, samples_per_tff: int, state0, backend, e0: float) -> dict:
    cost = run_cost(integrator, force_per_tff, n_tff)
    steps_per_tff = cost["steps_per_tff"]
    dt = cost["dt"]
    n_steps_total = cost["n_steps_total"]
    callback_every = max(1, steps_per_tff // samples_per_tff)

    trace_full = []  # (t, dE_E0) signed
    trace_abs_tff = []  # (t/t_ff, |dE_E0|)

    def callback(step_index, t, s):
        e = total_energy(s, softening=SOFTENING)
        signed = (e - e0) / abs(e0)
        trace_full.append((t, signed))
        trace_abs_tff.append((t / config.T_FF, abs(signed)))

    t0 = time.perf_counter()
    final_state = integrate(
        state0,
        integrator=integrator,
        backend=backend,
        dt=dt,
        n_steps=n_steps_total,
        softening=SOFTENING,
        callback=callback,
        callback_every=callback_every,
    )
    torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - t0

    finite = bool(torch.isfinite(final_state.r).all() and torch.isfinite(final_state.v).all())
    notes = []
    if not finite:
        notes.append("diverged: non-finite r or v at end of run")

    if trace_full:
        max_abs_drift = max(abs(d) for _, d in trace_full)
        final_t, final_drift = trace_full[-1]
        final_t_tff = final_t / config.T_FF
    else:
        max_abs_drift = float("nan")
        final_drift = float("nan")
        final_t_tff = float("nan")

    if finite:
        p_final = linear_momentum(final_state)
        p_norm_scaled = torch.linalg.norm(p_final).item() / (config.TOTAL_MASS * config.V_CHAR)
    else:
        p_norm_scaled = float("nan")

    t_cross = crossing_time(trace_abs_tff, DRIFT_TOLERANCE)
    crossed = t_cross is not None
    if not crossed:
        notes.append(f"did not cross {DRIFT_TOLERANCE:.0%} by t_max={n_tff} t_ff")

    if integrator in LINEAR_DRIFT_INTEGRATORS:
        analysis = analyze_linear_drift(trace_abs_tff)
    else:
        analysis = analyze_band_trend(trace_abs_tff, n_tff)

    return {
        "final_state": final_state,
        "dt": dt,
        "steps_per_tff": steps_per_tff,
        "n_steps_total": n_steps_total,
        "n_force_total": cost["n_force_total"],
        "callback_every": callback_every,
        "trace_full": trace_full,
        "trace_abs_tff": trace_abs_tff,
        "n_samples": len(trace_full),
        "max_abs_dE_E0": max_abs_drift,
        "final_dE_E0": final_drift,
        "final_t_over_tff": final_t_tff,
        "wall_time_s": wall_time_s,
        "finite": finite,
        "crossed_tolerance": crossed,
        "t_over_tff_cross_tolerance": t_cross,
        "analysis": analysis,
        "momentum_final_norm_scaled": p_norm_scaled,
        "notes": "; ".join(notes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tff", type=int, default=10, help="number of t_ff to simulate (default: 10, Stage 1)")
    parser.add_argument("--stage", type=str, default="stage1", help="label recorded in the CSV and figure title")
    parser.add_argument("--force-per-tff", type=int, default=FORCE_PER_TFF_DEFAULT)
    parser.add_argument("--samples-per-tff", type=int, default=SAMPLES_PER_TFF_DEFAULT)
    parser.add_argument(
        "--stage2-horizons",
        type=str,
        default="150,200",
        help="comma-separated t_ff horizons to project Stage-2 wall time for (reporting only, not run)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/ROCm device not available on this host; refusing to silently fall back "
            "to a different device for a study whose backend choice is normative."
        )

    torch.set_default_dtype(DTYPE)
    backend = get_backend(BACKEND_NAME)
    env = env_metadata()

    print("=" * 100)
    print(f"longrun energy-drift study: N={N_PARTICLES}, dtype=fp64, eps={SOFTENING}, backend={BACKEND_NAME}/{DEVICE}")
    print(f"stage={args.stage}, n_tff={args.n_tff}, force_per_tff={args.force_per_tff}, "
          f"samples_per_tff={args.samples_per_tff}, drift_tolerance={DRIFT_TOLERANCE:.0%}")
    print(f"t_ff = {config.T_FF:.6f} s; t_end simulated = {args.n_tff * config.T_FF:.6f} s")
    for k, v in env.items():
        print(f"  {k:<24}{v}")
    print("=" * 100)

    state0 = cold_sphere(n=N_PARTICLES, seed=SEED, dtype=DTYPE, device=DEVICE)
    e0 = total_energy(state0, softening=SOFTENING)
    print(f"E0 = U0 (cold start, v=0 for all particles) = {e0:.10e} J")

    # Warmup: one call, timed and reported separately, never mixed into measurements below.
    # Single call is sufficient for torch.compile -- same (N, dtype, device) shape reused
    # for every subsequent call in this script, so no further compilation occurs.
    warmup_cost = run_cost("rk4", args.force_per_tff, 1)
    t0 = time.perf_counter()
    _ = integrate(state0, integrator="rk4", backend=backend, dt=warmup_cost["dt"], n_steps=2, softening=SOFTENING)
    torch.cuda.synchronize()
    print(f"warmup (torch.compile trace + 2 rk4 steps): {time.perf_counter() - t0:.2f} s")
    print()

    rows = []
    traces_for_plot = {}
    per_integrator_wall = {}
    for integrator in INTEGRATOR_NAMES:
        print(f"running {integrator:<18} n_tff={args.n_tff} ...", flush=True)
        try:
            result = run_one(integrator, args.force_per_tff, args.n_tff, args.samples_per_tff, state0, backend, e0)
        except BackendUnavailable as exc:
            raise RuntimeError(f"backend {BACKEND_NAME!r} on {DEVICE!r} unavailable mid-run: {exc}") from exc

        common = {
            "integrator": integrator,
            "stage": args.stage,
            "n_particles": N_PARTICLES,
            "softening": SOFTENING,
            "seed": SEED,
            "backend": BACKEND_NAME,
            "device": DEVICE,
            "dtype": "float64",
            "t_ff": config.T_FF,
            "force_per_tff_nominal": args.force_per_tff,
            "force_evals_per_step": FORCE_EVALS_PER_STEP[integrator],
            "force_evals_startup": FORCE_EVALS_STARTUP[integrator],
            "dt": result["dt"],
            "steps_per_tff": result["steps_per_tff"],
            "n_tff_simulated": args.n_tff,
            "n_steps_total": result["n_steps_total"],
            "n_force_total": result["n_force_total"],
            "t_end": args.n_tff * config.T_FF,
        }
        common.update(env)

        for i, (t, signed) in enumerate(result["trace_full"]):
            row = dict.fromkeys(CSV_FIELDNAMES, "")
            row.update(common)
            row["row_type"] = "sample"
            row["sample_index"] = i
            row["t"] = t
            row["t_over_tff"] = t / config.T_FF
            row["dE_E0"] = signed
            row["abs_dE_E0"] = abs(signed)
            rows.append(row)

        analysis = result["analysis"]
        fit = analysis["fit"]
        summary = dict.fromkeys(CSV_FIELDNAMES, "")
        summary.update(common)
        summary["row_type"] = "summary"
        summary["n_samples"] = result["n_samples"]
        summary["callback_every"] = result["callback_every"]
        summary["diagnostic_overhead_fraction"] = result["n_samples"] / result["n_force_total"]
        summary["wall_time_s"] = result["wall_time_s"]
        summary["finite"] = result["finite"]
        summary["max_abs_dE_E0"] = result["max_abs_dE_E0"]
        summary["final_dE_E0"] = result["final_dE_E0"]
        summary["final_t_over_tff"] = result["final_t_over_tff"]
        summary["drift_tolerance"] = DRIFT_TOLERANCE
        summary["crossed_tolerance"] = result["crossed_tolerance"]
        summary["t_over_tff_cross_tolerance"] = (
            result["t_over_tff_cross_tolerance"] if result["crossed_tolerance"] else ""
        )
        summary["fit_type"] = analysis["fit_type"]
        summary["fit_domain_start_t_over_tff"] = analysis["fit_domain_start_t_over_tff"]
        summary["fit_n_points"] = fit["n"]
        summary["fit_slope_per_tff"] = fit["slope"]
        summary["fit_intercept"] = fit["intercept"]
        summary["fit_r_squared"] = fit["r_squared"]
        summary["characteristic_scale"] = analysis["characteristic_scale"]
        summary["trend_over_horizon_fraction"] = analysis["trend_over_horizon_fraction"]
        summary["momentum_final_norm_scaled"] = result["momentum_final_norm_scaled"]
        summary["notes"] = result["notes"]
        rows.append(summary)

        traces_for_plot[integrator] = result["trace_abs_tff"]
        per_integrator_wall[integrator] = result["wall_time_s"]

        cross_str = (
            f"{result['t_over_tff_cross_tolerance']:.3f} t_ff"
            if result["crossed_tolerance"]
            else f"not crossed by {args.n_tff} t_ff"
        )
        print(
            f"  n_steps={result['n_steps_total']:>8} n_force={result['n_force_total']:>8} "
            f"wall={result['wall_time_s']:>8.3f}s max|dE/E0|={result['max_abs_dE_E0']:.3e} "
            f"fit_type={analysis['fit_type']} slope/tff={fit['slope']:.3e} R2={fit['r_squared']:.4f} "
            f"crosses {DRIFT_TOLERANCE:.0%} at {cross_str}"
        )

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nwrote {len(rows)} rows ({len(rows) - len(INTEGRATOR_NAMES)} sample + {len(INTEGRATOR_NAMES)} summary) "
          f"to {os.path.abspath(RESULTS_PATH)}")

    make_figure(traces_for_plot, args)

    print()
    print("Stage-2 wall-time projection (linear extrapolation of this stage's per-t_ff cost; "
          "sampling density held fixed, so diagnostic overhead fraction is unchanged):")
    horizons = [float(h) for h in args.stage2_horizons.split(",")]
    for horizon in horizons:
        total = 0.0
        parts = []
        for integrator in INTEGRATOR_NAMES:
            per_tff = per_integrator_wall[integrator] / args.n_tff
            est = per_tff * horizon
            total += est
            parts.append(f"{integrator}={est:.1f}s")
        print(f"  {horizon:.0f} t_ff: total ~= {total:.1f} s ({total / 60.0:.1f} min)  [" + ", ".join(parts) + "]")


def make_figure(traces_for_plot, args) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for integrator in INTEGRATOR_NAMES:
        trace = traces_for_plot.get(integrator, [])
        if not trace:
            continue
        ts = [t for t, _ in trace]
        drifts = [d if d != 0.0 else 1e-20 for _, d in trace]
        ax.semilogy(ts, drifts, label=integrator, linewidth=1, marker=".", markersize=2)
    ax.axhline(DRIFT_TOLERANCE, color="black", linestyle="--", linewidth=1, label=f"{DRIFT_TOLERANCE:.0%} (TCC 2019, Sec. 3.5)")
    ax.set_xlabel("t / t_ff")
    ax.set_ylabel("|dE/E0|")
    ax.set_title(
        f"Energy drift vs time, N={N_PARTICLES}, eps={SOFTENING} m, fp64, {BACKEND_NAME}/{DEVICE}\n"
        f"equal cost: {args.force_per_tff} force evals / t_ff, stage={args.stage}, n_tff={args.n_tff}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "longrun_energy_drift.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
