"""
N sweep of one force evaluation, ``backend.accelerations(r, m, softening)``, across the
six-rung ladder documented in tests/test_backends.py (python_pure/cpu, torch_eager/cpu,
torch_compiled/cpu, torch_eager/cuda, torch_compiled/cuda, triton/cuda), in fp32 and fp64,
for N in powers of two starting at 512. This extends scripts/bench.py (a single N=1000
snapshot) into a scaling campaign; every methodological convention below is inherited from
that script and from the coordinating brief for this task.

Per-backend N ceiling (the sweep grid is not the same set for every rung -- this is by
design, not an omission):

  - python_pure/cpu:            512, 1024, 2048, 4096 (mandatory), 8192 (optional, included
                                 here with fewer repeats -- see N_REPEATS below). N=16384+
                                 is deliberately excluded: python_pure is a scalar O(N^2)
                                 Python loop, and the 2019 legacy data (results/2019/
                                 benchmarks.csv) shows a single N=32768 call costing
                                 648-1767 s on comparable hardware; a full repeated-median
                                 campaign at that N is out of budget for this pure-Python
                                 rung.
  - torch_eager/cpu, torch_compiled/cpu:    512 .. 32768.
  - torch_eager/cuda, torch_compiled/cuda, triton/cuda:    512 .. 65536.

Methodology (unchanged from scripts/bench.py, restated here because this is a standalone
script):

  - Warmup is exactly one call, timed and reported in its own column (warmup_time_s), never
    mixed into the repeated measurements, and never reused across N or across dtype: every
    (backend, device, dtype, N) quadruple gets its own warmup call. For torch_compiled this
    call includes tracing + Inductor compilation; for triton it includes autotune + kernel
    compilation. Because the tiled pairwise kernel's tiling loop (nbody._pairwise.iter_tiles)
    has a different iteration count at every distinct N, torch_compiled genuinely retraces at
    every N regardless of PyTorch's "automatic dynamic shapes" heuristic -- this is expected
    and is recorded, not suppressed.
  - torch._dynamo.reset() is called immediately before every torch_compiled warmup call (for
    both cpu and cuda rows -- the compiled callable is a single module-level object shared
    across devices). This is a deliberate harness intervention, not a change to src/nbody:
    without it, torch.compile's default recompile_limit (8 distinct
    guard-specializations per compiled function; confirmed via
    torch._dynamo.config.cache_size_limit == 8 on this host) would be exceeded partway
    through this sweep (up to 16 distinct (N, dtype) shape/dtype combinations are attempted
    against the same compiled object for the cuda rung), at which point Dynamo silently falls
    back to eager execution for further calls without raising. That silent fallback would
    mislabel eager timings as "torch_compiled" -- exactly the kind of substitution this
    project's measurement discipline forbids. Resetting before every point guarantees each
    warmup call is a genuine, isolated one-shot compile.
  - After warmup, N_REPEATS(backend, N) repeated calls are timed (see the repeats policy
    below). Median and IQR (numpy.percentile, linear interpolation) are reported, plus the
    full raw sample list -- never a mean/stdev summary.
  - CUDA calls are followed by torch.cuda.synchronize() before the clock stops, both for the
    warmup call and every repeat.
  - Any backend.accelerations() call may raise BackendUnavailable; this is caught by name
    (never a bare except) and turned into an unavailable row with the exception message
    recorded verbatim. Out-of-memory conditions (torch.OutOfMemoryError, a CPU-side
    MemoryError, or a RuntimeError whose message names "out of memory") are caught
    specifically and also turned into unavailable rows with the reason recorded -- the sweep
    continues past them rather than crashing. No other exception is caught.
  - A correctness sanity column, max_abs_diff_vs_torch_eager_fp64, compares each row's output
    against a torch_eager/cpu/fp64 reference computed once per N (same seed, same N) --
    recomputed at every N since the reference particle configuration itself depends on N.

Repeats policy (N_REPEATS_TENSOR/N_REPEATS_PYTHON_PURE from bench.py are a starting point;
they are scaled down as N grows so that a single row at large N does not cost minutes, while
never dropping below enough repeats to report a meaningful median+IQR):

  tensor backends (torch_eager, torch_compiled -- cpu and cuda; triton -- cuda):
      N <= 4096:  50 repeats
      N == 8192:  30 repeats
      N == 16384: 20 repeats
      N == 32768: 12 repeats
      N == 65536: 10 repeats
  python_pure/cpu:
      N <= 4096:  15 repeats
      N == 8192:  8 repeats (optional point; O(N^2) scalar-Python cost budget)

Measurement order: outer loop over N ascending, inner loop over the six rungs in their fixed
table order, innermost loop dtype fp32 then fp64 -- so every backend sees the same relative
position in the overall time-ordered run at every N (no backend monopolizes the GPU for a
long contiguous block while another is measured only after long idle/thermal drift). A
(backend, device) row is skipped entirely (no row emitted) for N values outside that rung's
declared ceiling above -- this is the sweep grid definition, not a dropped measurement.

Disk cache handling: ~/.triton/cache and /tmp/torchinductor_* are cleared completely before
this script starts (done by the operator running this script, not by the script itself, so
that clearing never races with an in-flight compile). The disk_cache_state column records
this as a constant string on every row.

Output: results/2026/sweep_n.csv (tidy/flat, one row per (backend, device, dtype, N), full
raw samples kept, environment columns repeated on every row) and a log-log figure,
figures/sweep_n_loglog.png, generated by re-reading the just-written CSV from disk (never
from the in-memory row list), so the figure is always reproducible from the committed data
alone.
"""
from __future__ import annotations

import csv
import datetime
import os
import platform
import re
import socket
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.backends.base import BackendUnavailable  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "2026", "sweep_n.csv")
FIGURE_PATH = os.path.join(os.path.dirname(__file__), "..", "figures", "sweep_n_loglog.png")

SOFTENING = config.SOFTENING
SEED = config.SEED

DISK_CACHE_STATE = "cleared_before_run"

# The six-rung ladder from tests/test_backends.py's own docstring table, fixed order.
RUNGS = [
    ("python_pure", "cpu"),
    ("torch_eager", "cpu"),
    ("torch_compiled", "cpu"),
    ("torch_eager", "cuda"),
    ("torch_compiled", "cuda"),
    ("triton", "cuda"),
]
DTYPES = [torch.float32, torch.float64]

# Per-rung N ceiling. Not the same grid for every rung -- see module docstring.
RUNG_N_VALUES = {
    ("python_pure", "cpu"): [512, 1024, 2048, 4096, 8192],
    ("torch_eager", "cpu"): [512, 1024, 2048, 4096, 8192, 16384, 32768],
    ("torch_compiled", "cpu"): [512, 1024, 2048, 4096, 8192, 16384, 32768],
    ("torch_eager", "cuda"): [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    ("torch_compiled", "cuda"): [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    ("triton", "cuda"): [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
}
ALL_N_VALUES = sorted({n for values in RUNG_N_VALUES.values() for n in values})

CUDA_AVAILABLE = torch.cuda.is_available()


def n_repeats_for(backend_name: str, n: int) -> int:
    """Repeats policy: scaled down as N grows, see module docstring for the table."""
    if backend_name == "python_pure":
        return 15 if n <= 4096 else 8
    if n <= 4096:
        return 50
    if n == 8192:
        return 30
    if n == 16384:
        return 20
    if n == 32768:
        return 12
    return 10  # n == 65536


# ==============================================================================
# Environment metadata -- captured once, attached to every row. Duplicated from
# scripts/bench.py deliberately (this script is standalone, same convention as bench.py).
# ==============================================================================
def cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return f"unavailable ({exc})"
    m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else platform.processor() or "unknown"


def collect_environment() -> dict:
    triton_version = ""
    try:
        import triton

        triton_version = triton.__version__
    except ImportError as exc:
        triton_version = f"not installed ({exc})"

    gpu_name = ""
    torch_gpu_runtime_version = ""
    if CUDA_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        if torch.version.hip is not None:
            torch_gpu_runtime_version = f"hip {torch.version.hip}"
        elif torch.version.cuda is not None:
            torch_gpu_runtime_version = f"cuda {torch.version.cuda}"

    return {
        "hostname": socket.gethostname(),
        "os_platform": platform.platform(),
        "cpu_model": cpu_model_name(),
        "cpu_logical_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": CUDA_AVAILABLE,
        "gpu_name": gpu_name,
        "torch_gpu_runtime_version": torch_gpu_runtime_version,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "triton_version": triton_version,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


BASE_ROW_KEYS = [
    "backend",
    "device",
    "dtype",
    "n_particles",
    "softening",
    "seed",
    "n_repeats",
    "available",
    "unavailable_reason",
    "warmup_time_s",
    "median_time_s",
    "iqr_time_s",
    "q25_time_s",
    "q75_time_s",
    "min_time_s",
    "max_time_s",
    "samples_time_s",
    "max_abs_diff_vs_torch_eager_fp64",
    "disk_cache_state",
]


# ==============================================================================
# Timing of a single (backend, device, dtype) call.
# ==============================================================================
def time_one_call(backend, r, m, softening, device: torch.device) -> tuple[float, torch.Tensor]:
    t0 = time.perf_counter()
    out = backend.accelerations(r, m, softening)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, out


def _is_oom_message(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def bench_row(
    backend_name: str,
    device_str: str,
    dtype: torch.dtype,
    n: int,
    reference_fp64_cpu: torch.Tensor,
) -> dict:
    device = torch.device(device_str)
    backend = get_backend(backend_name)
    n_repeats = n_repeats_for(backend_name, n)

    row = {key: "" for key in BASE_ROW_KEYS}
    row.update(
        {
            "backend": backend_name,
            "device": device_str,
            "dtype": str(dtype).replace("torch.", ""),
            "n_particles": n,
            "softening": SOFTENING,
            "seed": SEED,
            "n_repeats": n_repeats,
            "available": False,
            "disk_cache_state": DISK_CACHE_STATE,
        }
    )

    if device_str == "cuda" and not CUDA_AVAILABLE:
        row["unavailable_reason"] = "no CUDA device detected on this host (torch.cuda.is_available() is False)"
        return row

    if not backend.supports(device):
        row["unavailable_reason"] = f"{backend_name}.supports({device}) is False"
        return row

    # Data setup happens outside the timed region: this is input generation, not the thing
    # being measured.
    state = cold_sphere(n=n, seed=SEED, dtype=dtype, device=device)

    # Deliberate harness intervention for torch_compiled only -- see module docstring.
    # Outside the timed region: this is cache management, not part of the measured call.
    if backend_name == "torch_compiled":
        torch._dynamo.reset()

    try:
        warmup_time, warmup_out = time_one_call(backend, state.r, state.m, SOFTENING, device)
        samples = []
        last_out = warmup_out
        for _ in range(n_repeats):
            dt, out = time_one_call(backend, state.r, state.m, SOFTENING, device)
            samples.append(dt)
            last_out = out
    except BackendUnavailable as exc:
        row["unavailable_reason"] = str(exc)
        return row
    except torch.OutOfMemoryError as exc:
        row["unavailable_reason"] = f"OOM (torch.OutOfMemoryError): {exc}"
        return row
    except MemoryError as exc:
        row["unavailable_reason"] = f"OOM (CPU MemoryError): {exc}"
        return row
    except RuntimeError as exc:
        if _is_oom_message(exc):
            row["unavailable_reason"] = f"OOM (RuntimeError): {exc}"
            return row
        raise

    diff = last_out.detach().to("cpu", dtype=torch.float64) - reference_fp64_cpu
    max_abs_diff = diff.abs().max().item()

    q25, q75 = np.percentile(samples, [25, 75])
    row.update(
        {
            "available": True,
            "warmup_time_s": warmup_time,
            "median_time_s": float(np.median(samples)),
            "iqr_time_s": float(q75 - q25),
            "q25_time_s": float(q25),
            "q75_time_s": float(q75),
            "min_time_s": min(samples),
            "max_time_s": max(samples),
            "samples_time_s": ";".join(repr(s) for s in samples),
            "max_abs_diff_vs_torch_eager_fp64": max_abs_diff,
        }
    )
    return row


def make_figure(csv_path: str, out_path: str) -> None:
    """Regenerate the log-log figure purely from the CSV on disk (never from in-memory rows)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Fixed categorical order + colorblind-safe palette (Okabe-Ito), one color per
    # (backend, device) rung, assigned in the same fixed order as RUNGS above.
    rung_order = [
        ("python_pure", "cpu"),
        ("torch_eager", "cpu"),
        ("torch_compiled", "cpu"),
        ("torch_eager", "cuda"),
        ("torch_compiled", "cuda"),
        ("triton", "cuda"),
    ]
    rung_colors = {
        ("python_pure", "cpu"): "#000000",
        ("torch_eager", "cpu"): "#0072B2",
        ("torch_compiled", "cpu"): "#56B4E9",
        ("torch_eager", "cuda"): "#D55E00",
        ("torch_compiled", "cuda"): "#E69F00",
        ("triton", "cuda"): "#009E73",
    }
    rung_labels = {
        ("python_pure", "cpu"): "python_pure / cpu",
        ("torch_eager", "cpu"): "torch_eager / cpu",
        ("torch_compiled", "cpu"): "torch_compiled / cpu",
        ("torch_eager", "cuda"): "torch_eager / cuda",
        ("torch_compiled", "cuda"): "torch_compiled / cuda",
        ("triton", "cuda"): "triton / cuda",
    }

    dtypes = ["float32", "float64"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, dtype in zip(axes, dtypes):
        for backend_name, device_str in rung_order:
            sub = df[
                (df["backend"] == backend_name)
                & (df["device"] == device_str)
                & (df["dtype"] == dtype)
            ].sort_values("n_particles")
            if sub.empty:
                continue
            color = rung_colors[(backend_name, device_str)]
            label = rung_labels[(backend_name, device_str)]

            avail = sub[sub["available"] == True]  # noqa: E712
            unavail = sub[sub["available"] == False]  # noqa: E712

            if not avail.empty:
                n = avail["n_particles"].to_numpy()
                med = avail["median_time_s"].to_numpy()
                q25 = avail["q25_time_s"].to_numpy()
                q75 = avail["q75_time_s"].to_numpy()
                ax.plot(n, med, marker="o", markersize=4, linewidth=1.5, color=color, label=label)
                ax.fill_between(n, q25, q75, color=color, alpha=0.18, linewidth=0)

            if not unavail.empty:
                # Mark unavailable/OOM points distinctly at the top of the visible range
                # rather than dropping them silently from the plot's implied domain.
                n = unavail["n_particles"].to_numpy()
                ylim_top = ax.get_ylim()[1] if ax.lines else 1.0
                ax.scatter(
                    n,
                    [ylim_top] * len(n),
                    marker="x",
                    color=color,
                    s=50,
                    zorder=5,
                )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N (particles)")
        ax.set_title(f"dtype = {dtype}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("median wall-clock time per force evaluation (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Force evaluation time vs N, log-log, shaded band = IQR (q25-q75).\n"
        "'x' markers = unavailable/OOM at that N (see disk_cache_state / unavailable_reason "
        "columns in results/2026/sweep_n.csv)."
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    env = collect_environment()
    fieldnames = BASE_ROW_KEYS + list(env.keys())

    print("=" * 100)
    print("environment")
    print("=" * 100)
    for key, value in env.items():
        print(f"  {key:<28}{value}")
    print(f"  disk_cache_state           {DISK_CACHE_STATE}")
    print()

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    fh = open(RESULTS_PATH, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    fh.flush()

    campaign_t0 = time.perf_counter()
    n_rows = 0

    for n in ALL_N_VALUES:
        # Reference output for the correctness sanity column: torch_eager, cpu, fp64, same
        # seed, recomputed at every N since the reference configuration depends on N.
        reference_state = cold_sphere(n=n, seed=SEED, dtype=torch.float64, device="cpu")
        reference_backend = get_backend("torch_eager")
        reference_fp64_cpu = reference_backend.accelerations(reference_state.r, reference_state.m, SOFTENING)

        for backend_name, device_str in RUNGS:
            if n not in RUNG_N_VALUES[(backend_name, device_str)]:
                continue
            for dtype in DTYPES:
                dtype_str = str(dtype).replace("torch.", "")
                t_row0 = time.perf_counter()
                print(f"N={n:<7} {backend_name:<16}{device_str:<6}{dtype_str:<10} ...", end=" ", flush=True)
                row = bench_row(backend_name, device_str, dtype, n, reference_fp64_cpu)
                row.update(env)
                writer.writerow(row)
                fh.flush()
                n_rows += 1
                t_row1 = time.perf_counter()
                if row["available"]:
                    print(
                        f"warmup={row['warmup_time_s']:.4e}s median={row['median_time_s']:.4e}s "
                        f"iqr={row['iqr_time_s']:.4e}s reps={row['n_repeats']} "
                        f"row_wall={t_row1 - t_row0:.2f}s",
                        flush=True,
                    )
                else:
                    print(f"UNAVAILABLE: {row['unavailable_reason']} row_wall={t_row1 - t_row0:.2f}s", flush=True)

    fh.close()
    campaign_t1 = time.perf_counter()
    print()
    print(f"wrote {n_rows} rows to {os.path.abspath(RESULTS_PATH)}")
    print(f"total campaign wall time: {campaign_t1 - campaign_t0:.1f} s")

    make_figure(os.path.abspath(RESULTS_PATH), os.path.abspath(FIGURE_PATH))
    print(f"wrote figure to {os.path.abspath(FIGURE_PATH)}")


if __name__ == "__main__":
    main()
