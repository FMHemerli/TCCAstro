"""
First trustworthy timing number for this project: one force evaluation,
``backend.accelerations(r, m, softening)``, at N = 1000, for every (backend, device)
rung on the ladder documented in tests/test_backends.py, in fp32 and fp64.

This is deliberately NOT a campaign. No N sweep, no config layer, no abstraction beyond
what fits directly in this file. The physical correctness of src/nbody/ was already
established by scripts/sanity.py; the question here is purely wall-clock cost of one
implementation choice versus another -- which is what the original 2019 TCC was actually
about, and which the 2019 notebooks measured without warmup separation, without
repetition, and without recording the environment. See results/2019/benchmarks.csv
(scripts/extract_legacy_results.py) for that dataset's own admission of this gap.

Methodology:

  - The six (backend, device) rungs are fixed by tests/test_backends.py's own docstring
    table. python_pure/cpu, torch_eager/cpu, torch_compiled/cpu, torch_eager/cuda,
    torch_compiled/cuda, triton/cuda. backend.supports(device) is still checked before
    calling (it is a structural contract check, not a hardware probe -- see
    tests/test_backends.py::TestSupportsIsHonest); CUDA physical presence is checked
    separately via torch.cuda.is_available().
  - Warmup is exactly one call, timed and reported in its own column
    (warmup_time_s), never mixed into the repeated measurements. For torch_compiled
    this call includes tracing + Inductor compilation; for triton it includes autotune
    + kernel compilation (tens of seconds, observed directly on this host before writing
    this script). A follow-up timing sweep, run by hand, showed the call immediately
    after warmup is already at steady state for both compiled backends, so a single
    warmup call is enough here -- this is an empirical check on this host, not a
    general guarantee.
  - After warmup, N_REPEATS_TENSOR (tensor backends) or N_REPEATS_PYTHON_PURE
    (python_pure, which is a scalar O(N^2) Python loop and was measured to cost
    ~0.13 s/call at N=1000 on this host, so fewer repeats are used) repeated calls are
    timed. Median and IQR (numpy.percentile, linear interpolation) are reported, plus
    the full raw sample list -- never a mean/stdev summary that discards the
    right-skew of scheduling noise.
  - CUDA calls are followed by torch.cuda.synchronize() before the clock stops, both
    for the warmup call and every repeat; without it a "CUDA time" is just a queueing
    time.
  - Any backend.accelerations() call may raise BackendUnavailable; this is caught by
    name (never a bare except) and turned into an unavailable row with the exception
    message recorded verbatim. No other exception is caught -- an unnamed failure means
    the script stops, loudly.
  - A correctness sanity column, max_abs_diff_vs_torch_eager_fp64, compares each row's
    output against a torch_eager/cpu/fp64 reference computed once on the same initial
    condition. This is not a pass/fail gate; a large number here would be reported, not
    hidden, and does not stop the benchmark.

Output: printed table (condensed, environment metadata shown once) plus
results/2026/bench_n1000.csv (full columns, one row per (backend, device, dtype), plus
the environment columns repeated on every row since results are tidy/flat by convention).
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

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "2026", "bench_n1000.csv"
)

N_PARTICLES = 1000
SOFTENING = config.SOFTENING
SEED = config.SEED

# The six-rung ladder from tests/test_backends.py's own docstring table.
RUNGS = [
    ("python_pure", "cpu"),
    ("torch_eager", "cpu"),
    ("torch_compiled", "cpu"),
    ("torch_eager", "cuda"),
    ("torch_compiled", "cuda"),
    ("triton", "cuda"),
]
DTYPES = [torch.float32, torch.float64]

N_REPEATS_TENSOR = 50
N_REPEATS_PYTHON_PURE = 15  # O(N^2) pure-Python loop; measured ~0.13 s/call at N=1000

CUDA_AVAILABLE = torch.cuda.is_available()


# ==============================================================================
# Environment metadata -- captured once, attached to every row.
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
        # torch.version.hip is set on ROCm builds (this host); torch.version.cuda on
        # official CUDA builds. Record whichever is populated, and say which one it is.
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


# ==============================================================================
# Timing of a single (backend, device, dtype) row.
# ==============================================================================
def time_one_call(backend, r, m, softening, device: torch.device) -> float:
    t0 = time.perf_counter()
    out = backend.accelerations(r, m, softening)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, out


def bench_row(backend_name: str, device_str: str, dtype: torch.dtype, reference_fp64_cpu: torch.Tensor) -> dict:
    device = torch.device(device_str)
    backend = get_backend(backend_name)
    n_repeats = N_REPEATS_PYTHON_PURE if backend_name == "python_pure" else N_REPEATS_TENSOR

    row = {
        "backend": backend_name,
        "device": device_str,
        "dtype": str(dtype).replace("torch.", ""),
        "n_particles": N_PARTICLES,
        "softening": SOFTENING,
        "seed": SEED,
        "n_repeats": n_repeats,
        "available": False,
        "unavailable_reason": "",
        "warmup_time_s": "",
        "median_time_s": "",
        "iqr_time_s": "",
        "q25_time_s": "",
        "q75_time_s": "",
        "min_time_s": "",
        "max_time_s": "",
        "samples_time_s": "",
        "max_abs_diff_vs_torch_eager_fp64": "",
    }

    if device_str == "cuda" and not CUDA_AVAILABLE:
        row["unavailable_reason"] = "no CUDA device detected on this host (torch.cuda.is_available() is False)"
        return row

    if not backend.supports(device):
        row["unavailable_reason"] = f"{backend_name}.supports({device}) is False"
        return row

    # Data setup happens outside the timed region: this is input generation, not the
    # thing being measured. cold_sphere draws from numpy in float64 internally and
    # casts down afterwards, so the same seed gives the same physical particle
    # positions (to within cast precision) across every dtype/device combination.
    state = cold_sphere(n=N_PARTICLES, seed=SEED, dtype=dtype, device=device)

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


def main() -> None:
    env = collect_environment()

    print("=" * 100)
    print("environment")
    print("=" * 100)
    for key, value in env.items():
        print(f"  {key:<28}{value}")
    print()

    # Reference output for the correctness sanity column: torch_eager, cpu, fp64, same IC.
    reference_state = cold_sphere(n=N_PARTICLES, seed=SEED, dtype=torch.float64, device="cpu")
    reference_backend = get_backend("torch_eager")
    reference_fp64_cpu = reference_backend.accelerations(reference_state.r, reference_state.m, SOFTENING)

    rows = []
    for backend_name, device_str in RUNGS:
        for dtype in DTYPES:
            print(f"running {backend_name}/{device_str}/{str(dtype).replace('torch.', '')} ...", flush=True)
            row = bench_row(backend_name, device_str, dtype, reference_fp64_cpu)
            row.update(env)
            rows.append(row)

    print()
    print("=" * 100)
    print(f"N = {N_PARTICLES}, softening = {SOFTENING}, seed = {SEED}")
    print("=" * 100)
    header = (
        f"{'backend':<16}{'device':<8}{'dtype':<10}{'reps':>6}{'warmup_s':>14}"
        f"{'median_s':>14}{'iqr_s':>14}{'max|diff|':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["available"]:
            print(
                f"{row['backend']:<16}{row['device']:<8}{row['dtype']:<10}{row['n_repeats']:>6}"
                f"{row['warmup_time_s']:>14.6e}{row['median_time_s']:>14.6e}{row['iqr_time_s']:>14.6e}"
                f"{row['max_abs_diff_vs_torch_eager_fp64']:>14.3e}"
            )
        else:
            print(
                f"{row['backend']:<16}{row['device']:<8}{row['dtype']:<10}"
                f"{'-- unavailable: ' + row['unavailable_reason']}"
            )

    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print()
    print(f"wrote {len(rows)} rows to {os.path.abspath(RESULTS_PATH)}")


if __name__ == "__main__":
    main()
