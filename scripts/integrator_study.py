"""
Equal-cost comparison of the four integrators on the cold spherical collapse, N = 1000,
fp64, T_END = 1.2 t_ff. Closed scope: which integrator gives the best trajectory/energy
behaviour per force evaluation. No tiling, no GPU-vs-CPU, no backend re-benchmarking.

Backend: torch_compiled/cuda, chosen from the existing results/2026/bench_n1000.csv
(median_time_s at N=1000 fp64: torch_compiled/cuda 8.368e-4 s, torch_eager/cuda
1.557e-3 s, torch_compiled/cpu 1.390e-3 s) -- not re-measured here.

Cost axis (docs/integradores.md Sec. 8.1-8.2): n_force = n_steps * FORCE_EVALS_PER_STEP
+ FORCE_EVALS_STARTUP (rk4 k=4; others k=1; velocity_verlet alone has startup s=1, the
KDK seed). BUDGETS are nominal; the real n_force per run is what gets reported/plotted.

Reference: RK4, fp64, dt_ref far finer than any study dt, scoring only final-position
error err_r = ||r - r_ref||_F / (sqrt(N) * R0) at T_END (Sec. 6.1 metric). Self-convergence
at 2x dt_ref checked once, not campaigned. RK4's own err_r is self-referential (reference
is also rk4) -- flagged in notes, same effect refuted as a bogus "order 4.95" in
scripts/sanity.py; report, do not use for convergence order.

Timing: one integrate() call per (integrator, budget), time.perf_counter() around it,
torch.cuda.synchronize() before the clock stops. The call also carries a ~150-sample
energy-drift callback; its cost is ~constant in absolute terms across integrators at a
fixed budget, so it does not bias the cross-integrator comparison, and its size relative
to n_force is reported per row (diagnostic_overhead_fraction, <=1.2%).

Output: results/2026/integrator_study.csv, figures/energy_vs_time_intermediate_budget.png,
figures/error_vs_nforce.png.
"""
from __future__ import annotations

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
from nbody.backends.base import BackendUnavailable  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.integrators import FORCE_EVALS_PER_STEP, FORCE_EVALS_STARTUP, INTEGRATOR_NAMES, integrate  # noqa: E402
from nbody.observables import linear_momentum, total_energy  # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "2026", "integrator_study.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

BACKEND_NAME = "torch_compiled"
DEVICE = "cuda"
DTYPE = torch.float64

N_PARTICLES = config.N_PARTICLES
SOFTENING = config.SOFTENING
SEED = config.SEED
SPHERE_RADIUS = config.SPHERE_RADIUS

T_END = 1.2 * config.T_FF
BUDGETS = [12600, 25200, 50400, 100800]
INTERMEDIATE_BUDGET = 50400  # upper of the two log-midpoint budgets; see module docstring

N_SAMPLES_TARGET = 150

# Reference: RK4, dt far finer than any per-step dt used in the study (finest study dt is
# T_END / 100800 for the 1-eval methods). Self-convergence checked at 2x this dt.
N_STEPS_REF = 300000
N_STEPS_REF_CHECK = 150000


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


def run_cost(integrator: str, budget: int) -> tuple[int, float, int]:
    k = FORCE_EVALS_PER_STEP[integrator]
    s = FORCE_EVALS_STARTUP[integrator]
    n_steps = budget // k
    dt = T_END / n_steps
    n_force = n_steps * k + s
    return n_steps, dt, n_force


def build_reference(state0, backend) -> dict:
    """RK4, fp64, dt_ref << any study dt. Self-convergence checked at 2x dt_ref."""
    dt_ref = T_END / N_STEPS_REF
    t0 = time.perf_counter()
    ref_state = integrate(
        state0, integrator="rk4", backend=backend, dt=dt_ref, n_steps=N_STEPS_REF, softening=SOFTENING
    )
    torch.cuda.synchronize()
    ref_wall = time.perf_counter() - t0

    dt_check = T_END / N_STEPS_REF_CHECK
    check_state = integrate(
        state0, integrator="rk4", backend=backend, dt=dt_check, n_steps=N_STEPS_REF_CHECK, softening=SOFTENING
    )
    torch.cuda.synchronize()

    norm = math.sqrt(N_PARTICLES) * SPHERE_RADIUS
    self_conv_err = torch.linalg.norm(ref_state.r - check_state.r).item() / norm

    print(f"reference built: rk4, dt_ref={dt_ref:.6e}, n_steps={N_STEPS_REF}, wall={ref_wall:.2f}s")
    print(f"self-convergence (vs dt_ref x2, n_steps={N_STEPS_REF_CHECK}): err = {self_conv_err:.6e}")

    return {
        "state": ref_state,
        "dt_ref": dt_ref,
        "n_steps_ref": N_STEPS_REF,
        "self_conv_err": self_conv_err,
    }


def err_r(state, ref_state) -> float:
    norm = math.sqrt(N_PARTICLES) * SPHERE_RADIUS
    return torch.linalg.norm(state.r - ref_state.r).item() / norm


def run_one(integrator: str, budget: int, state0, backend, e0: float) -> dict:
    n_steps, dt, n_force = run_cost(integrator, budget)
    callback_every = max(1, n_steps // N_SAMPLES_TARGET)

    trace = []

    def callback(step_index, t, s):
        e = total_energy(s, softening=SOFTENING)
        trace.append((t, (e - e0) / abs(e0)))

    t0 = time.perf_counter()
    final_state = integrate(
        state0,
        integrator=integrator,
        backend=backend,
        dt=dt,
        n_steps=n_steps,
        softening=SOFTENING,
        callback=callback,
        callback_every=callback_every,
    )
    torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - t0

    finite = bool(torch.isfinite(final_state.r).all() and torch.isfinite(final_state.v).all())
    notes = []
    if not finite:
        notes.append("diverged: non-finite r or v at T_END")

    if finite:
        drifts = [d for _, d in trace]
        max_abs_drift = max(abs(d) for d in drifts) if drifts else float("nan")
        final_drift = drifts[-1] if drifts else float("nan")
        p_final = linear_momentum(final_state)
        p_norm = torch.linalg.norm(p_final).item()
        p_norm_scaled = p_norm / (config.TOTAL_MASS * config.V_CHAR)
    else:
        max_abs_drift = float("nan")
        final_drift = float("nan")
        p_norm = float("nan")
        p_norm_scaled = float("nan")

    if integrator == "rk4":
        notes.append(
            "err_r vs reference is self-referential (reference is also rk4); biased, "
            "do not use for convergence-order claims (see scripts/sanity.py)"
        )

    return {
        "final_state": final_state,
        "n_steps": n_steps,
        "dt": dt,
        "n_force": n_force,
        "callback_every": callback_every,
        "n_samples": len(trace),
        "max_abs_dE_E0": max_abs_drift,
        "final_dE_E0": final_drift,
        "momentum_final_norm": p_norm,
        "momentum_final_norm_scaled": p_norm_scaled,
        "wall_time_s": wall_time_s,
        "trace": trace,
        "notes": "; ".join(notes),
        "finite": finite,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/ROCm device not available on this host; refusing to silently fall back "
            "to a different device for a study whose backend choice is normative."
        )

    torch.set_default_dtype(DTYPE)
    backend = get_backend(BACKEND_NAME)
    env = env_metadata()

    print("=" * 100)
    print(f"integrator study: N={N_PARTICLES}, dtype=fp64, backend={BACKEND_NAME}, device={DEVICE}")
    print(f"T_END = {T_END:.6f} s = 1.2 t_ff (t_ff = {config.T_FF:.6f} s)")
    print(f"budgets (n_force nominal): {BUDGETS}")
    for k, v in env.items():
        print(f"  {k:<24}{v}")
    print("=" * 100)

    state0 = cold_sphere(n=N_PARTICLES, seed=SEED, dtype=DTYPE, device=DEVICE)
    e0 = total_energy(state0, softening=SOFTENING)

    # Warmup: one call, timed and reported separately, never mixed into measurements below.
    # A single call is sufficient for torch.compile here -- same (N, dtype, device) shape
    # is reused for every subsequent call in this script, so no further compilation occurs.
    t0 = time.perf_counter()
    _ = integrate(state0, integrator="rk4", backend=backend, dt=T_END / N_STEPS_REF, n_steps=2, softening=SOFTENING)
    torch.cuda.synchronize()
    print(f"warmup (torch.compile trace + 2 rk4 steps): {time.perf_counter() - t0:.2f} s")
    print()

    try:
        ref = build_reference(state0, backend)
    except BackendUnavailable as exc:
        raise RuntimeError(f"backend {BACKEND_NAME!r} on {DEVICE!r} unavailable: {exc}") from exc
    print()

    rows = []
    traces_intermediate = {}
    for integrator in INTEGRATOR_NAMES:
        for budget in BUDGETS:
            print(f"running {integrator:<18} budget={budget:>7} ...", flush=True)
            try:
                result = run_one(integrator, budget, state0, backend, e0)
            except BackendUnavailable as exc:
                raise RuntimeError(f"backend {BACKEND_NAME!r} on {DEVICE!r} unavailable mid-run: {exc}") from exc

            e_r = err_r(result["final_state"], ref["state"]) if result["finite"] else float("nan")

            row = {
                "integrator": integrator,
                "budget_nominal": budget,
                "n_steps": result["n_steps"],
                "dt": result["dt"],
                "n_force": result["n_force"],
                "t_end": T_END,
                "t_end_over_tff": T_END / config.T_FF,
                "softening": SOFTENING,
                "n_particles": N_PARTICLES,
                "seed": SEED,
                "backend": BACKEND_NAME,
                "device": DEVICE,
                "dtype": "float64",
                "callback_every": result["callback_every"],
                "n_samples": result["n_samples"],
                "diagnostic_overhead_fraction": result["n_samples"] / result["n_force"],
                "max_abs_dE_E0": result["max_abs_dE_E0"],
                "final_dE_E0": result["final_dE_E0"],
                "err_r_final": e_r,
                "ref_integrator": "rk4",
                "ref_dt": ref["dt_ref"],
                "ref_n_steps": ref["n_steps_ref"],
                "ref_self_convergence_err_r": ref["self_conv_err"],
                "momentum_final_norm": result["momentum_final_norm"],
                "momentum_final_norm_scaled": result["momentum_final_norm_scaled"],
                "wall_time_s": result["wall_time_s"],
                "notes": result["notes"],
            }
            row.update(env)
            rows.append(row)

            if budget == INTERMEDIATE_BUDGET:
                traces_intermediate[integrator] = result["trace"]

            print(
                f"  n_steps={result['n_steps']:>7} n_force={result['n_force']:>7} "
                f"wall={result['wall_time_s']:>8.3f}s max|dE/E0|={result['max_abs_dE_E0']:.3e} "
                f"final dE/E0={result['final_dE_E0']:+.3e} err_r={e_r:.3e}"
            )

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nwrote {len(rows)} rows to {os.path.abspath(RESULTS_PATH)}")

    make_figures(rows, traces_intermediate)


def make_figures(rows, traces_intermediate) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for integrator in INTEGRATOR_NAMES:
        trace = traces_intermediate.get(integrator, [])
        if not trace:
            continue
        ts = [t / config.T_FF for t, _ in trace]
        drifts = [abs(d) if d != 0.0 else 1e-20 for _, d in trace]
        ax.semilogy(ts, drifts, label=integrator, marker=".", markersize=3, linewidth=1)
    ax.set_xlabel("t / t_ff")
    ax.set_ylabel("|dE/E0|")
    ax.set_title(
        f"Energy drift vs time, N={N_PARTICLES}, budget={INTERMEDIATE_BUDGET} force evals\n"
        f"eps={SOFTENING} m, T_END=1.2 t_ff, fp64, {BACKEND_NAME}/{DEVICE}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path_a = os.path.join(FIGURES_DIR, "energy_vs_time_intermediate_budget.png")
    fig.savefig(path_a, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for integrator in INTEGRATOR_NAMES:
        pts = sorted(
            (r["n_force"], r["err_r_final"])
            for r in rows
            if r["integrator"] == integrator and not math.isnan(r["err_r_final"])
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, label=integrator, marker="o")
    ax.set_xlabel("n_force (force evaluations, real cost)")
    ax.set_ylabel("err_r(T_END) vs high-resolution rk4 reference")
    ax.set_title(
        f"Final-position error vs cost, N={N_PARTICLES}, eps={SOFTENING} m\n"
        f"T_END=1.2 t_ff, fp64, {BACKEND_NAME}/{DEVICE} (rk4 curve is self-referential, see notes)"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path_b = os.path.join(FIGURES_DIR, "error_vs_nforce.png")
    fig.savefig(path_b, dpi=150)
    plt.close(fig)

    print(f"wrote {os.path.abspath(path_a)}")
    print(f"wrote {os.path.abspath(path_b)}")


if __name__ == "__main__":
    main()
