"""Extract structured benchmark data from the 2019 TCC legacy notebooks.

The four notebooks in ``legacy/notebooks-2019/`` are primary evidence: they are
byte-identical to tag ``v2019-tcc`` and MUST NOT be modified. This script only
reads them (as JSON, per the nbformat spec) and parses the execution outputs
that are already recorded in the files -- it does not execute any notebook
code and does not invent any value that is not literally present in a cell
source or a cell output.

Three CSV files are produced under ``results/2019/``:

``benchmarks.csv``
    One row per (machine, implementation, run_phase) timing measurement,
    parsed from the "Codigo executado em X s" lines that ``run_nbody``
    printed. For Numba-jitted implementations, ``run_nbody`` was called with
    the default ``flag_compile=True`` and prints two timings per cell: a
    first call that includes JIT compilation (``run_phase=compile_warmup``)
    and a second, already-compiled call (``run_phase=measured``). For pure
    Python and plain NumPy, ``flag_compile=False`` was passed explicitly and
    only one timing is printed (``run_phase=measured``). Only rows with
    ``run_phase=measured`` are used as inputs to ``speedup_vs_python``.

``speedups.csv``
    One row per "Speed up of ..." cell in the "Resultados e comparacoes"
    section of each notebook: the ratio expression (e.g. ``t_py / t_nb``),
    the two implementations it resolves to, and the value the notebook
    itself computed (or, if the cell was commented out / raised an error,
    a note explaining why no value exists).

``environments.csv``
    One row per notebook with the environment metadata each notebook printed
    about itself (uname, lsb_release, lscpu, GCC/Python/NumPy/Numba
    versions, GPU detection via ``cuda.detect()``), plus the three separate
    version strings found in the artifact (markdown title, filename,
    ``nbody_version`` literal in the ``run_nbody`` source and as printed).

Every field that is not literally present in the notebook is left empty and
explained in a ``notes`` column -- never filled with a plausible guess.
Re-running this script against the same notebooks always regenerates
byte-for-byte identical CSVs (rows are sorted deterministically before
writing).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "legacy" / "notebooks-2019"
RESULTS_DIR = REPO_ROOT / "results" / "2019"

# Every executed benchmark in these notebooks uses M = 1 step and a single
# run per configuration: no repetition, no median, no dispersion measure.
# This is a structural property of the whole 2019 dataset, verified by
# reading every "Teste de nbody em ..." cell: none of them loops or repeats
# the call to run_nbody. Recorded verbatim on every measured row so it
# travels with the number instead of living only in this docstring.
SINGLE_RUN_NOTE = (
    "execucao unica (M=1 passo), sem repeticao, sem mediana, sem dispersao; "
    "nao comparavel diretamente a medidas de 2026 que reportam mediana e IQR "
    "sobre repeticoes"
)

FILENAME_MACHINE_RE = re.compile(
    r"_(Idapad320|DellG3|GoogleColab|Neuromancer)_v(?P<fnver>[\d.]+)\.ipynb$"
)

# Maps the function object passed as the first positional argument of
# run_nbody(...) to a stable implementation identifier. ".py_func" means the
# uncompiled Python function underlying a @njit-decorated definition was
# used (Numba bypassed), matching the notebooks' own "Python puro" /
# "Python NumPy" (no Numba) test cells.
FUNC_TO_IMPL = {
    "gravitationalNbody_NumbaCPU.py_func": "python_pure",
    "gravitationalNbody_NumPy_NumbaCPU.py_func": "numpy_pure",
    "gravitationalNbody_NumbaCPU": "numba_cpu_serial",
    "gravitationalNbody_NumPy_NumbaCPU": "numpy_numba_cpu_serial",
    "gravitationalNbody_NumbaCPUparallel": "numba_cpu_parallel",
    "gravitationalNbody_NumPy_NumbaCPUparallel": "numpy_numba_cpu_parallel",
    "gravitationalNbody_NumbaGPU64": "numba_cuda_float64",
}
ALL_IMPLEMENTATIONS = sorted(set(FUNC_TO_IMPL.values()))

ASSIGN_RE = re.compile(
    r"(?P<var_all>t_all_\w+)\s*,\s*(?P<var_t>t_\w+)\s*,\s*(?P<var_err>erro_\w+)"
    r"\s*=\s*run_nbody\(\s*(?P<func>gravitationalNbody_\w+(?:\.py_func)?)",
    re.MULTILINE,
)
TIME_RE = re.compile(r"C[oó]digo executado em ([\d.eE+-]+) s")
N_RE = re.compile(r"N = (\d+) corpos, dt = ([\d.eE+-]+)s, M = (\d+) interaç")
HTOD_RE = re.compile(r"Copied data from CPU to GPU after ([\d.eE+-]+) s")
KERNEL_RE = re.compile(r"GPU calculations done after ([\d.eE+-]+) s")
DTOH_RE = re.compile(r"Copied data from GPU to CPU after ([\d.eE+-]+) s")
SPEEDUP_LINE_RE = re.compile(r"^\s*(?P<num>t_\w+)\s*/\s*(?P<den>t_\w+)\s*$")
UNAME_HOST_RE = re.compile(r"^Linux\s+(\S+)\s")
LSCPU_MODEL_RE = re.compile(r"(?:Nome do modelo|Model name):\s*(.+)")
LSCPU_CPUS_RE = re.compile(r"^CPU\(s\):\s*(\d+)", re.MULTILINE)
CUDA_DEVICE_RE = re.compile(r"id\s+0\s+b'([^']+)'")
CUDA_CC_RE = re.compile(r"compute capability:\s*([\d.]+)")
MARKDOWN_TITLE_VERSION_RE = re.compile(r"v(?P<ver>\d+\.\d+\.\d+)")
NBODY_VERSION_SOURCE_RE = re.compile(r'nbody_version\s*=\s*"([\d.]+)"')
NBODY_VERSION_PRINTED_RE = re.compile(r"N-Body \(C\) Flavio Manoel, v([\d.]+)")
FLAG_COMPILE_FALSE_RE = re.compile(r"flag_compile\s*=\s*False")
FLAG_ARRAY_FALSE_RE = re.compile(r"flag_array\s*=\s*False")
FLAG_ENERGY_TRUE_RE = re.compile(r"flag_Energy\s*=\s*True")

MACHINE_MARKDOWN_KEYWORDS = {
    "Idapad320": "Ideapad 320",
    "DellG3": "Dell G3",
    "GoogleColab": "Google Colab",
    "Neuromancer": "Neuromancer",
}


class ExtractionError(Exception):
    """Raised when a notebook does not match an expected structural pattern.

    Always carries the notebook filename and the cell index so a parse
    failure can be traced back to the exact place it happened; never
    swallowed, never substituted with a guessed value.
    """


def notebook_files() -> list[Path]:
    files = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not files:
        raise ExtractionError(f"no .ipynb files found under {NOTEBOOKS_DIR}")
    return files


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def active_source(cell: dict) -> str:
    """Source with fully commented-out lines removed.

    Used to decide whether a code cell actually executed the statement it
    contains, as opposed to merely containing it inside a "#..." line kept
    for reference (every notebook here keeps several disabled variants next
    to the one that was run).
    """
    lines = cell_source(cell).splitlines()
    return "\n".join(l for l in lines if not l.strip().startswith("#"))


def stream_text(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append("".join(out.get("text", [])))
    return "".join(parts)


def execute_result_text(cell: dict) -> Optional[str]:
    for out in cell.get("outputs", []):
        if out.get("output_type") == "execute_result":
            plain = out.get("data", {}).get("text/plain")
            if plain:
                text = "".join(plain) if isinstance(plain, list) else plain
                return text.strip().strip("'\"")
    return None


def error_info(cell: dict) -> Optional[tuple[str, str]]:
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            return out.get("ename", ""), out.get("evalue", "")
    return None


@dataclass
class NotebookContext:
    machine: str
    filename: str
    filename_version: str
    nb: dict
    var_to_impl: dict = field(default_factory=dict)  # t_py -> "python_pure"
    var_to_cell: dict = field(default_factory=dict)  # t_py -> cell index


def build_context(path: Path) -> NotebookContext:
    m = FILENAME_MACHINE_RE.search(path.name)
    if not m:
        raise ExtractionError(f"filename does not match expected pattern: {path.name}")
    machine = m.group(1)
    filename_version = m.group("fnver")
    nb = load_notebook(path)
    return NotebookContext(
        machine=machine, filename=path.name, filename_version=filename_version, nb=nb
    )


def index_assignments(ctx: NotebookContext) -> None:
    """Populate ctx.var_to_impl / ctx.var_to_cell from every ACTIVE
    "t_all_x, t_x, erro_x = run_nbody(func, ...)" assignment in the notebook.
    Commented-out assignments (never executed) are deliberately excluded.
    """
    for i, cell in enumerate(ctx.nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src_active = active_source(cell)
        for am in ASSIGN_RE.finditer(src_active):
            func = am.group("func")
            impl = FUNC_TO_IMPL.get(func)
            if impl is None:
                raise ExtractionError(
                    f"{ctx.filename} cell {i}: unrecognised run_nbody function '{func}'"
                )
            if impl == "python_pure" and not FLAG_ARRAY_FALSE_RE.search(src_active):
                raise ExtractionError(
                    f"{ctx.filename} cell {i}: expected flag_array=False for "
                    f"the pure-Python (.py_func, no NumPy arrays) variant"
                )
            var_t = am.group("var_t")
            ctx.var_to_impl[var_t] = impl
            ctx.var_to_cell[var_t] = i


def extract_benchmark_rows(ctx: NotebookContext) -> list[dict]:
    rows = []
    for i, cell in enumerate(ctx.nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src_active = active_source(cell)
        am = ASSIGN_RE.search(src_active)
        if am is None:
            continue
        func = am.group("func")
        impl = FUNC_TO_IMPL[func]
        text = stream_text(cell)
        time_matches = TIME_RE.findall(text)
        n_matches = N_RE.findall(text)
        if not time_matches:
            raise ExtractionError(
                f"{ctx.filename} cell {i}: active run_nbody call for '{func}' "
                f"produced no 'Codigo executado em' output to parse"
            )
        if not n_matches:
            raise ExtractionError(
                f"{ctx.filename} cell {i}: could not parse N/dt/M header line "
                f"from cell output"
            )
        expects_single = bool(FLAG_COMPILE_FALSE_RE.search(src_active))
        expected_count = 1 if expects_single else 2
        if len(time_matches) != expected_count:
            raise ExtractionError(
                f"{ctx.filename} cell {i}: expected {expected_count} timing "
                f"line(s) (flag_compile={'False' if expects_single else 'True (default)'}) "
                f"but found {len(time_matches)}"
            )
        n_particles_set = {int(nm[0]) for nm in n_matches}
        dt_set = {nm[1] for nm in n_matches}
        n_steps_set = {int(nm[2]) for nm in n_matches}
        if len(n_particles_set) != 1 or len(dt_set) != 1 or len(n_steps_set) != 1:
            raise ExtractionError(
                f"{ctx.filename} cell {i}: inconsistent N/dt/M across the "
                f"printed header lines of a single cell"
            )
        n_particles = n_particles_set.pop()
        dt_value = dt_set.pop()
        n_steps = n_steps_set.pop()

        htod = HTOD_RE.findall(text)
        kernel = KERNEL_RE.findall(text)
        dtoh = DTOH_RE.findall(text)
        object_mode_fallback = "NumbaWarning" in text and "object mode" in text

        flag_energy_true = bool(FLAG_ENERGY_TRUE_RE.search(src_active))

        phases = ["compile_warmup", "measured"] if expected_count == 2 else ["measured"]
        for idx, phase in enumerate(phases):
            notes = [SINGLE_RUN_NOTE, "flag_Energy=False (validacao de energia desativada nesta execucao)"]
            if flag_energy_true:
                # Should never trigger given the notebooks as written, but
                # if it ever does, that is exactly the kind of thing that
                # must be surfaced, not silently dropped.
                notes = [n for n in notes if not n.startswith("flag_Energy=False")]
                notes.append("flag_Energy=True nesta celula (excecao ao padrao do dataset)")
            if impl == "numba_cuda_float64" and object_mode_fallback:
                notes.append(
                    "saida contem NumbaWarning de fallback para object mode na "
                    "funcao wrapper gravitationalNbody_NumbaGPU64 (timer() nao "
                    "tipavel em nopython); o tempo medido inclui overhead de "
                    "interpretador Python ao redor dos kernels @cuda.jit"
                )
            rows.append(
                {
                    "machine": ctx.machine,
                    "implementation": impl,
                    "n_particles": n_particles,
                    "n_steps": n_steps,
                    "dt": dt_value,
                    "dtype": "float64",
                    "run_phase": phase,
                    "time_s": time_matches[idx],
                    "gpu_htod_s": htod[idx] if htod else "",
                    "gpu_kernel_s": kernel[idx] if kernel else "",
                    "gpu_dtoh_s": dtoh[idx] if dtoh else "",
                    "speedup_vs_python": "",  # filled in a second pass
                    "source_notebook": ctx.filename,
                    "source_cell": i,
                    "notes": "; ".join(notes),
                }
            )
    return rows


def extract_speedup_rows(ctx: NotebookContext) -> list[dict]:
    rows = []
    cells = ctx.nb["cells"]
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        raw = cell_source(cell).strip()
        m = SPEEDUP_LINE_RE.match(raw.lstrip("#").strip()) if raw else None
        if m is None:
            continue
        num_var, den_var = m.group("num"), m.group("den")
        is_commented = raw.startswith("#")

        label = ""
        if i > 0 and cells[i - 1].get("cell_type") == "markdown":
            label = cell_source(cells[i - 1]).strip()

        num_impl = ctx.var_to_impl.get(num_var)
        den_impl = ctx.var_to_impl.get(den_var)

        value = ""
        notes = []
        if is_commented:
            notes.append("celula comentada no notebook, nunca executada")
        else:
            err = error_info(cell)
            result = execute_result_text(cell)
            if err is not None:
                ename, evalue = err
                notes.append(f"erro na execucao: {ename}: {evalue}")
            elif result is not None:
                value = result
            else:
                raise ExtractionError(
                    f"{ctx.filename} cell {i}: active speedup expression "
                    f"'{num_var} / {den_var}' has neither an execute_result "
                    f"nor an error output"
                )
        if num_impl is None:
            notes.append(f"variavel '{num_var}' nunca atribuida por um run_nbody ativo neste notebook")
        if den_impl is None:
            notes.append(f"variavel '{den_var}' nunca atribuida por um run_nbody ativo neste notebook")

        rows.append(
            {
                "machine": ctx.machine,
                "numerator_var": num_var,
                "numerator_implementation": num_impl or "",
                "denominator_var": den_var,
                "denominator_implementation": den_impl or "",
                "speedup": value,
                "label": label,
                "executed": "false" if is_commented else "true",
                "source_notebook": ctx.filename,
                "source_cell": i,
                "notes": "; ".join(notes),
            }
        )
    return rows


def fill_speedup_vs_python(
    benchmark_rows: list[dict], speedup_rows: list[dict], ctx: NotebookContext
) -> None:
    measured = {
        r["implementation"]: r
        for r in benchmark_rows
        if r["machine"] == ctx.machine and r["run_phase"] == "measured"
    }
    reported = {}
    for sr in speedup_rows:
        if sr["machine"] != ctx.machine or sr["executed"] != "true" or not sr["speedup"]:
            continue
        if sr["numerator_implementation"] == "python_pure" and sr["denominator_implementation"]:
            reported[sr["denominator_implementation"]] = (sr["speedup"], sr["source_cell"])

    for row in benchmark_rows:
        if row["machine"] != ctx.machine or row["run_phase"] != "measured":
            continue
        impl = row["implementation"]
        if impl == "python_pure":
            row["speedup_vs_python"] = "1.0"
            row["notes"] += "; speedup_vs_python = 1.0 por definicao (linha de base)"
            continue
        if impl in reported:
            value, src_cell = reported[impl]
            row["speedup_vs_python"] = value
            row["notes"] += f"; speedup_vs_python lido diretamente da celula {src_cell} do notebook"
        elif "python_pure" in measured:
            t_py = float(measured["python_pure"]["time_s"])
            t_x = float(row["time_s"])
            row["speedup_vs_python"] = repr(t_py / t_x)
            row["notes"] += (
                "; speedup_vs_python derivado pelo extrator (t_py medido / t_x medido); "
                "nao ha celula no notebook reportando esta razao diretamente"
            )


def missing_implementation_rows(ctx: NotebookContext, benchmark_rows: list[dict]) -> list[dict]:
    present = {r["implementation"] for r in benchmark_rows if r["machine"] == ctx.machine}
    missing = sorted(set(ALL_IMPLEMENTATIONS) - present)
    rows = []
    for impl in missing:
        func_name = next(f for f, v in FUNC_TO_IMPL.items() if v == impl)
        base_func = func_name.split(".")[0]
        def_present = any(
            cell.get("cell_type") == "code" and f"def {base_func}" in cell_source(cell)
            for cell in ctx.nb["cells"]
        )
        error_notes = []
        for cell in ctx.nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            err = error_info(cell)
            if err and ("gpu" in cell_source(cell).lower() or "gpu" in "".join(err).lower()):
                error_notes.append(f"{err[0]}: {err[1]}")
        notes = [
            f"nao medido: nenhuma celula ativa chama run_nbody com {base_func} neste notebook",
        ]
        notes.append(
            f"definicao de {base_func} {'presente' if def_present else 'AUSENTE'} no notebook"
        )
        if error_notes:
            notes.append("celula(s) subsequente(s) relacionada(s) levantam: " + "; ".join(sorted(set(error_notes))))
        rows.append(
            {
                "machine": ctx.machine,
                "implementation": impl,
                "n_particles": "",
                "n_steps": "",
                "dt": "",
                "dtype": "float64",
                "run_phase": "not_measured",
                "time_s": "",
                "gpu_htod_s": "",
                "gpu_kernel_s": "",
                "gpu_dtoh_s": "",
                "speedup_vs_python": "",
                "source_notebook": ctx.filename,
                "source_cell": "",
                "notes": "; ".join(notes),
            }
        )
    return rows


def extract_environment_row(ctx: NotebookContext) -> dict:
    nb = ctx.nb
    cells = nb["cells"]
    title_cell = cell_source(cells[0]) if cells else ""
    title_version_matches = MARKDOWN_TITLE_VERSION_RE.findall(title_cell)
    title_version = title_version_matches[0] if title_version_matches else ""

    def find_stream(prefix_src: str) -> str:
        for cell in cells:
            if cell.get("cell_type") == "code" and cell_source(cell).strip() == prefix_src:
                return stream_text(cell)
        return ""

    def find_execute_result(src_line: str) -> str:
        for cell in cells:
            if cell.get("cell_type") == "code" and cell_source(cell).strip() == src_line:
                r = execute_result_text(cell)
                if r is not None:
                    return r
        return ""

    uname_full = find_stream("!uname -a").strip()
    uname_host_m = UNAME_HOST_RE.match(uname_full)
    uname_host = uname_host_m.group(1) if uname_host_m else ""

    lsb_release_full = find_stream("!lsb_release -a").strip()
    lscpu_full = find_stream("!lscpu")
    cpu_model_m = LSCPU_MODEL_RE.search(lscpu_full)
    cpu_model = cpu_model_m.group(1).strip() if cpu_model_m else ""
    cpu_count_m = LSCPU_CPUS_RE.search(lscpu_full)
    cpu_count = cpu_count_m.group(1) if cpu_count_m else ""

    gcc_version = find_stream("!gcc --version").splitlines()[0].strip() if find_stream("!gcc --version") else ""
    python_version = find_stream("!python --version").strip()
    numpy_version = find_execute_result("np.__version__")
    numba_version = find_execute_result("nb.__version__")
    numba_threads = find_execute_result("nb.config.NUMBA_DEFAULT_NUM_THREADS")

    cuda_detect_text = find_stream("cuda.detect()")
    gpu_name_m = CUDA_DEVICE_RE.search(cuda_detect_text)
    gpu_name_detected = gpu_name_m.group(1) if gpu_name_m else ""
    gpu_cc_m = CUDA_CC_RE.search(cuda_detect_text)
    gpu_cc_detected = gpu_cc_m.group(1) if gpu_cc_m else ""
    gpu_memory = find_execute_result("cuda.current_context(1).get_memory_info()")

    nbody_version_source = ""
    for cell in cells:
        if cell.get("cell_type") == "code":
            m = NBODY_VERSION_SOURCE_RE.search(cell_source(cell))
            if m:
                nbody_version_source = m.group(1)
                break
    nbody_version_printed_set = set()
    for cell in cells:
        if cell.get("cell_type") == "code":
            for m in NBODY_VERSION_PRINTED_RE.finditer(stream_text(cell)):
                nbody_version_printed_set.add(m.group(1))
    nbody_version_printed = ",".join(sorted(nbody_version_printed_set))

    keyword = MACHINE_MARKDOWN_KEYWORDS[ctx.machine]
    declared_text = ""
    for cell in cells[:6]:
        if cell.get("cell_type") == "markdown" and keyword in cell_source(cell):
            declared_text = cell_source(cell).strip()
            break

    notes = []
    if uname_host and "neuromancer" in uname_host.lower() and ctx.machine != "Neuromancer":
        notes.append(
            f"uname reporta hostname '{uname_host}', que colide (case-insensitive) com o "
            f"nome do notebook dedicado 'Neuromancer' (outra maquina, AMD FX-6100); "
            f"a proveniencia deste notebook '{ctx.machine}' nao e auditavel apenas pelo arquivo"
        )
    if not declared_text:
        notes.append(
            f"nenhuma celula markdown deste notebook descreve o hardware efetivamente "
            f"detectado (uname host='{uname_host}', GPU='{gpu_name_detected}'); as celulas "
            f"markdown de ambiente descrevem apenas Dell XPS15/Dell G3/Ideapad 320/Google Colab"
        )
    if declared_text and gpu_name_detected:
        declared_lower = declared_text.lower()
        name_lower = gpu_name_detected.lower()
        cc_lower = f"cc{gpu_cc_detected}".lower() if gpu_cc_detected else ""
        name_mismatch = name_lower not in declared_lower
        cc_mismatch = bool(cc_lower) and cc_lower not in declared_lower
        if name_mismatch or cc_mismatch:
            declared_flat = declared_text.replace("\n", " | ")
            notes.append(
                f"cuda.detect() mostra GPU '{gpu_name_detected}' CC{gpu_cc_detected}, que "
                f"diverge do texto markdown declarado para esta maquina "
                f"('{declared_flat[:200]}...')"
            )
    versions = {title_version, ctx.filename_version, nbody_version_source}
    versions.discard("")
    if len(versions) > 1:
        notes.append(
            f"versoes divergentes no artefato: titulo markdown='{title_version}', "
            f"nome de arquivo='{ctx.filename_version}', nbody_version no codigo-fonte="
            f"'{nbody_version_source}', nbody_version impresso na saida='{nbody_version_printed}'"
        )

    return {
        "machine": ctx.machine,
        "source_notebook": ctx.filename,
        "uname_full": uname_full,
        "uname_hostname": uname_host,
        "lsb_release_full": lsb_release_full.replace("\n", " | "),
        "cpu_model_lscpu": cpu_model,
        "cpu_count_lscpu": cpu_count,
        "gcc_version": gcc_version,
        "python_version": python_version,
        "numpy_version": numpy_version,
        "numba_version": numba_version,
        "numba_default_num_threads": numba_threads,
        "gpu_name_detected": gpu_name_detected,
        "gpu_compute_capability_detected": gpu_cc_detected,
        "gpu_memory_info_detected": gpu_memory,
        "markdown_title_full": title_cell.replace("\n", " | "),
        "markdown_title_version": title_version,
        "filename_version": ctx.filename_version,
        "nbody_version_source_literal": nbody_version_source,
        "nbody_version_printed": nbody_version_printed,
        "declared_machine_markdown_text": declared_text.replace("\n", " | "),
        "notes": "; ".join(notes),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    files = notebook_files()

    all_benchmark_rows: list[dict] = []
    all_speedup_rows: list[dict] = []
    all_environment_rows: list[dict] = []

    for path in files:
        ctx = build_context(path)
        index_assignments(ctx)

        bench_rows = extract_benchmark_rows(ctx)
        speedup_rows = extract_speedup_rows(ctx)
        fill_speedup_vs_python(bench_rows, speedup_rows, ctx)
        bench_rows += missing_implementation_rows(ctx, bench_rows)

        env_row = extract_environment_row(ctx)

        all_benchmark_rows += bench_rows
        all_speedup_rows += speedup_rows
        all_environment_rows.append(env_row)

    impl_order = {impl: idx for idx, impl in enumerate(ALL_IMPLEMENTATIONS)}
    phase_order = {"measured": 0, "compile_warmup": 1, "not_measured": 2}
    all_benchmark_rows.sort(
        key=lambda r: (
            r["machine"],
            impl_order.get(r["implementation"], 99),
            phase_order.get(r["run_phase"], 99),
        )
    )
    all_speedup_rows.sort(key=lambda r: (r["machine"], r["source_cell"]))
    all_environment_rows.sort(key=lambda r: r["machine"])

    benchmark_fields = [
        "machine",
        "implementation",
        "n_particles",
        "n_steps",
        "dt",
        "dtype",
        "run_phase",
        "time_s",
        "gpu_htod_s",
        "gpu_kernel_s",
        "gpu_dtoh_s",
        "speedup_vs_python",
        "source_notebook",
        "source_cell",
        "notes",
    ]
    speedup_fields = [
        "machine",
        "numerator_var",
        "numerator_implementation",
        "denominator_var",
        "denominator_implementation",
        "speedup",
        "label",
        "executed",
        "source_notebook",
        "source_cell",
        "notes",
    ]
    environment_fields = [
        "machine",
        "source_notebook",
        "uname_full",
        "uname_hostname",
        "lsb_release_full",
        "cpu_model_lscpu",
        "cpu_count_lscpu",
        "gcc_version",
        "python_version",
        "numpy_version",
        "numba_version",
        "numba_default_num_threads",
        "gpu_name_detected",
        "gpu_compute_capability_detected",
        "gpu_memory_info_detected",
        "markdown_title_full",
        "markdown_title_version",
        "filename_version",
        "nbody_version_source_literal",
        "nbody_version_printed",
        "declared_machine_markdown_text",
        "notes",
    ]

    write_csv(RESULTS_DIR / "benchmarks.csv", all_benchmark_rows, benchmark_fields)
    write_csv(RESULTS_DIR / "speedups.csv", all_speedup_rows, speedup_fields)
    write_csv(RESULTS_DIR / "environments.csv", all_environment_rows, environment_fields)

    print(f"wrote {len(all_benchmark_rows)} rows to {RESULTS_DIR / 'benchmarks.csv'}")
    print(f"wrote {len(all_speedup_rows)} rows to {RESULTS_DIR / 'speedups.csv'}")
    print(f"wrote {len(all_environment_rows)} rows to {RESULTS_DIR / 'environments.csv'}")


if __name__ == "__main__":
    main()
