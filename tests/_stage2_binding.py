"""
Shared infrastructure for tests/test_mass_spectrum.py, tests/test_scales_from_state.py,
tests/test_random_sphere.py, tests/test_half_mass_radius_median.py,
tests/test_collision_detection.py and tests/test_collision_pairing.py.

Not collected by pytest (no test_ prefix); imported like tests/conftest.py's helpers are.

Why this file exists
---------------------
docs/simulacao-estocastica.md Section 9.1 gives the FULL signature of exactly one new callable,
`random_sphere`. For everything else it adds -- `nbody.populations.sample_masses`,
`mass_min_from_mean`, `sample_velocities`, `solve_sigma`, `nbody.observables.scales_from_state`,
and the whole `nbody.collisions` module (`CollisionModel`, `detect`, `pair_disjoint`, `resolve`)
-- it gives only the name (and, for `CollisionModel`, a prose list of the fields it "contains":
`r_ref, v_coh, b, w, frag_f_min, eta, seed") but not a fixed parameter list or return type for the
callables. This is a real gap in the specification (flagged in the final report), not something
this suite is permitted to close by reading src/nbody/populations.py or src/nbody/collisions.py:
the whole point of this agent is that the tests must be able to disagree with the implementation,
which requires deriving them from the document alone.

The compromise adopted here: call these functions by matching their PUBLIC, DECLARED parameter
names (via `inspect.signature`) against a small set of semantic roles built from the mathematical
notation the specification itself uses ("mass_seed", "alpha", "R"/"mass_ratio", "PARTICLE_MASS",
"Q"/"virial_ratio", "r_ref", "dt", ...). Reading a function's declared parameter names is reading
its signature -- exactly the "forma" docs/api-contract.md says is the object of this contract --
not its algorithm. If a required (no-default) parameter cannot be matched to any offered role,
`ContractGap` is raised with the exact signature and the roles that were tried, and call sites
turn that into a loud, clearly-labelled pytest.skip rather than a silent pass or a guessed
positional call.

For `detect` and `pair_disjoint`, the specification additionally does not fix the TYPE of a
"candidate"/"event" record (dict? namedtuple? dataclass? a tensor row?), so `try_calls` and
`extract_pair_records` below extend the same philosophy to calling conventions and return-value
shapes: try the most literal reading of the document's own notation first, and raise `ContractGap`
-- never guess silently -- when nothing matches.
"""
from __future__ import annotations

import inspect
import math
import re

import numpy as np
import torch


class ContractGap(Exception):
    """
    Raised when a Section 9.1 function's declared parameter list cannot be bound to the
    semantic roles this suite can derive from the specification's own notation, without
    reading the implementation. Test call sites must catch this and pytest.skip with the
    message attached -- never swallow it into a pass.
    """


def _tokens(name: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _match_score(ptoks: set, rtoks: set) -> int:
    if ptoks == rtoks:
        return 3
    if ptoks <= rtoks or rtoks <= ptoks:
        return 2
    if ptoks & rtoks:
        return 1
    return 0


def bind_by_role(func, roles: dict) -> dict:
    """
    Match func's declared parameters to `roles` (role-name -> value) by word-token overlap,
    preferring the MOST SPECIFIC match (exact token-set equality, then containment, then any
    overlap) rather than the first role that shares any token -- otherwise "mass_ratio" and
    "particle_mass" would both spuriously match a "mass_seed" role on the shared token "mass".
    Several role names may point at the same value (synonyms). Parameters with a default that
    match no role are simply left to their default. Parameters WITHOUT a default that match no
    role raise ContractGap (the call would otherwise be a guess, not a derivation).
    """
    sig = inspect.signature(func)
    kwargs = {}
    unmatched = []
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ptoks = _tokens(name)
        best_score, best_value = 0, None
        for role, value in roles.items():
            score = _match_score(ptoks, _tokens(role))
            if score > best_score:
                best_score, best_value = score, value
        if best_score > 0:
            kwargs[name] = best_value
        elif param.default is inspect.Parameter.empty:
            unmatched.append(name)
    if unmatched:
        raise ContractGap(
            f"{getattr(func, '__name__', func)}{sig}: required parameter(s) {unmatched} could "
            f"not be matched to any role in {sorted(roles)}. docs/simulacao-estocastica.md "
            f"Section 9.1 names this function but does not fix its parameter list, and this "
            f"suite must not read src/nbody/populations.py to discover it."
        )
    return kwargs


def call_by_role(func, roles: dict):
    return func(**bind_by_role(func, roles))


def get_field(obj, *names):
    """
    Best-effort extraction of a named quantity from a Section 9.1 function's return value, whose
    exact type (dict? dataclass? namedtuple?) is not fixed by the specification. Tries each
    candidate name as a dict key and then as an attribute.
    """
    candidates = []
    for n in names:
        candidates.extend([n, n.lower(), n.upper()])
    if isinstance(obj, dict):
        for c in candidates:
            if c in obj:
                return obj[c]
    for c in candidates:
        if hasattr(obj, c):
            return getattr(obj, c)
    raise ContractGap(
        f"could not find any of {names} on the return value of a Section 9.1 function "
        f"(type {type(obj)!r}, public attributes={[a for a in dir(obj) if not a.startswith('_')]})"
    )


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", dtype=torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)


def call_sample_masses(sample_masses_fn, n, seed, alpha=None, mass_ratio=None, particle_mass=None):
    roles = {
        "n": n, "n_particles": n, "num_particles": n, "size": n, "count": n,
        "mass_seed": seed, "seed": seed, "rng_seed": seed,
    }
    if alpha is not None:
        roles.update({"alpha": alpha, "exponent": alpha, "power": alpha})
    if mass_ratio is not None:
        roles.update({"mass_ratio": mass_ratio, "ratio": mass_ratio, "truncation_ratio": mass_ratio})
    if particle_mass is not None:
        roles.update({
            "particle_mass": particle_mass, "mean_mass": particle_mass,
            "target_mass": particle_mass, "mbar": particle_mass,
        })
    return to_numpy(call_by_role(sample_masses_fn, roles))


def call_mass_min_from_mean(fn, alpha, mass_ratio, particle_mass):
    roles = {
        "alpha": alpha, "exponent": alpha, "power": alpha,
        "mass_ratio": mass_ratio, "ratio": mass_ratio, "truncation_ratio": mass_ratio,
        "particle_mass": particle_mass, "mean_mass": particle_mass,
        "target_mass": particle_mass, "mbar": particle_mass,
    }
    result = call_by_role(fn, roles)
    return float(result)


# -------------------------------------------------------------------------------------------
# Independent statistics: Kolmogorov-Smirnov p-value and chi-square p-value, implemented from
# their standard textbook definitions (no scipy in this environment). These are generic
# numerical-analysis utilities, not derived from nbody in any way.
# -------------------------------------------------------------------------------------------
def ks_statistic(sorted_samples: np.ndarray, cdf_fn) -> float:
    """One-sample Kolmogorov-Smirnov D statistic against the CDF `cdf_fn`."""
    n = len(sorted_samples)
    i = np.arange(1, n + 1)
    f = cdf_fn(sorted_samples)
    d_plus = np.max(i / n - f)
    d_minus = np.max(f - (i - 1) / n)
    return float(max(d_plus, d_minus))


def ks_pvalue(d: float, n: int) -> float:
    """
    Asymptotic two-sided Kolmogorov distribution p-value (Marsaglia-Kolmogorov series), with
    the standard finite-sample correction factor (Stephens 1970). Matches textbook critical
    values (e.g. scipy.stats.kstwobign.sf) to several significant digits for the n and D ranges
    used in this suite.
    """
    if d <= 0.0:
        return 1.0
    t = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    if t < 1e-6:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        total += (-1) ** (k - 1) * math.exp(-2.0 * k * k * t * t)
    p = 2.0 * total
    return min(max(p, 0.0), 1.0)


def _gammainc_p_series(a: float, x: float) -> float:
    """Regularized LOWER incomplete gamma P(a, x), series form (valid for x < a + 1)."""
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(2000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammainc_q_cf(a: float, x: float) -> float:
    """Regularized UPPER incomplete gamma Q(a, x), continued-fraction form (x >= a + 1)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(stat: float, dof: float) -> float:
    """
    Chi-square survival function (upper tail p-value), via the regularized incomplete gamma
    function (standard series/continued-fraction algorithm; e.g. Press et al., Numerical
    Recipes, "gser"/"gcf"). Validated against textbook critical values: chi2_sf(16.92, 9) ~=
    0.05, chi2_sf(21.67, 9) ~= 0.01.
    """
    a, x = dof / 2.0, stat / 2.0
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gammainc_p_series(a, x)
    return _gammainc_q_cf(a, x)


def chi_square_gof(observed_counts: np.ndarray, expected_probs, n: int):
    """Pearson chi-square goodness-of-fit statistic and p-value against fixed category probs."""
    observed_counts = np.asarray(observed_counts, dtype=np.float64)
    expected_probs = np.asarray(expected_probs, dtype=np.float64)
    expected_counts = expected_probs * n
    stat = float(np.sum((observed_counts - expected_counts) ** 2 / expected_counts))
    dof = len(expected_probs) - 1
    return stat, chi2_sf(stat, dof)


def try_calls(func, attempts):
    """
    Try calling `func` with each of `attempts` (a list of (args, kwargs) tuples) in order,
    returning (result, index) for the first attempt that does not raise TypeError/AttributeError.
    Raises ContractGap listing every failed attempt if none succeed.

    Used for `nbody.collisions.CollisionModel`/`detect`/`pair_disjoint`, whose parameter lists
    Section 9.1 does not fix (unlike `random_sphere`): the calling convention itself has to be
    discovered by trying the readings the document's own notation supports, not read from
    src/nbody/collisions.py.
    """
    errors = []
    for idx, (args, kwargs) in enumerate(attempts):
        try:
            return func(*args, **kwargs), idx
        except (TypeError, AttributeError) as exc:
            errors.append(f"  attempt {idx}: args={args!r} kwargs={sorted(kwargs)} -> "
                           f"{type(exc).__name__}: {exc}")
    joined = "\n".join(errors)
    raise ContractGap(
        f"{getattr(func, '__name__', func)}: none of {len(attempts)} candidate calling "
        f"conventions succeeded:\n{joined}"
    )


def _get_any(obj, names):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def extract_pair_records(candidates):
    """
    Best-effort extraction of (i, j, t_star) triples from an iterable of candidate/event records
    whose element type Section 9.1 does not fix (dict, namedtuple, dataclass, a raw (i, j, t)
    tuple in either field order, a tensor row, ...).

    Returns a list of (i, j, t_star) with i, j as int and t_star as float or None when no
    time-like field could be identified. Raises ContractGap when i/j cannot be identified for
    some element -- this is a genuine gap to report, not something to paper over with a guess.
    """
    records = []
    for c in candidates:
        i = _get_any(c, ("i", "idx_i", "a", "first", "p", "p1"))
        j = _get_any(c, ("j", "idx_j", "b", "second", "q", "p2"))
        t_star = _get_any(c, ("t_star", "t", "tstar", "time", "t_min"))
        if i is None or j is None:
            try:
                seq = [float(x) for x in c]
            except TypeError:
                seq = None
            if seq and len(seq) >= 2:
                int_like = [k for k, x in enumerate(seq) if x == int(x)]
                if len(int_like) >= 2:
                    i, j = int(seq[int_like[0]]), int(seq[int_like[1]])
                    if t_star is None:
                        remaining = [x for k, x in enumerate(seq) if k not in int_like[:2]]
                        if remaining:
                            t_star = remaining[0]
        if i is None or j is None:
            raise ContractGap(
                f"cannot extract (i, j) from candidate element {c!r} of type {type(c)!r}; "
                f"nbody.collisions does not fix a candidate/event record type (Section 9.1)"
            )
        records.append((int(i), int(j), None if t_star is None else float(t_star)))
    return records


def extract_numbers(text: str):
    """All floats/ints appearing in a string, for loosely checking exception messages name the
    quantities the spec requires (Q, f_cut, Q_max) without pinning exact wording/formatting."""
    return [float(m) for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def any_close(numbers, target, rel_tol=1e-3):
    return any(math.isclose(v, target, rel_tol=rel_tol) for v in numbers)
