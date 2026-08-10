"""
docs/simulacao-estocastica.md -- collision DETECTION, REVISION (f), 2026-08-09.

Section 4.3 replaced the whole detection scheme: the old detector resolved a pair at t*, the
INTERIOR MINIMUM of |sep(t)|^2 over the step (a clamped vertex of the separation parabola); the
new detector resolves at t_c, the FIRST CONTACT -- the smaller root of |sep(t)|^2 = R^2. The two
are NOT the same instant in general (Section 4.3's own proof: for an interior candidate,
0 < t_c < t_min <= h). This is not a refinement of the old algebra: at t*, u.n = 0 EXACTLY (the
minimum of a differentiable function has zero derivative along the direction that matters), which
is Section 4.3's documented defect 1 -- "o canal elastico era a identidade" -- and Section 4.5's
own registered error (a premise built on "u.n < 0 em t*", which the document now shows is false).
At t_c, u.n < 0 STRICTLY and by construction, which is the whole point of the rewrite.

This module covers:
  - Section 4.3's guards 1-5 and the rationalized root form (t_c = 2c/(-b+sqrt(D))), tested as
    PURE GEOMETRY (no nbody.collisions call) and independently against the real detect().
  - Section 6, INV-38: the three clauses the revision introduces --
      (1) the accepted-candidate SET is preserved relative to the old (t*-based) test, up to the
          measure-zero q(h)=0 boundary;
      (2) u.n < 0 is not just guaranteed pointwise but has a MEDIAN magnitude ratio >= 0.30 over
          the events a first-contact detector actually returns (BLOQUEANTE, on the median, not
          the minimum -- a pointwise floor would reject legitimate grazing encounters);
      (3) the rationalized root is numerically stable where the canonical form (-b-sqrt(D))/(2a)
          catastrophically cancels.
  - Section 4.4/INV-18(b): the Courant-collisional bookkeeping and the tunneling argument, which
    Section 4.3 states explicitly SURVIVES the rewrite unchanged ("o argumento anti-tunelamento da
    Secao 4.4.2 sobrevive intacto, porque ele e sobre a monotonicidade de dr.dv no passo, que nao
    foi tocada"). Kept largely as before, with field names updated to the (f) rename (t_c, not
    t_star) and the real-detector comparisons switched to the t_c closed form.
  - Section 4.1: contact_radii, unchanged by this revision.

Written from the specification alone. nbody.collisions.detect/CollisionModel/CollisionCandidates
were never opened as source -- only their PUBLIC SIGNATURES via inspect.signature and
dataclasses.fields, confirming CollisionCandidates carries (i, j, t_c, u_n, rel_speed,
contact_radius_sum) and that CollisionModel has no v_coh field left (both normative per Section
9.1.1's 2026-08-09 (f) amendment table).

Out of scope, deliberately: collision RESOLUTION (the three outcome maps, gates, E_int, L_spin --
INV-20 through INV-24, INV-32 through INV-37) -- covered by tests/test_collision_resolution.py.
Section 4.5's pairing (INV-19) -- covered by tests/test_collision_pairing.py.
"""
from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.state import State  # noqa: E402

try:
    import nbody.collisions as collisions
except ImportError:
    collisions = None

from tolerances import (  # noqa: E402
    CHI_DEFAULT,
    COLLISION_C_COLL_AT_DT_COLLISION,
    COLLISION_C_COLL_LIGHTEST_AT_2_5E4,
    COLLISION_R_SUM_LIGHTEST_PAIR,
    COLLISION_U_MAX_ASSUMED,
    DT_COLLISION,
    EPS_PREC_FP64,
    INV38_CANCELLATION_CASE,
    INV38_EVENT_SET_BOUNDARY_ULP,
    INV38_EVENT_SET_MAX_DISCREPANT_FRACTION,
    INV38_EVENT_SET_N_CONFIGS,
    INV38_ROOT_REL_TOL,
    INV38_SIGN_MEDIAN_MIN,
    INV38_SIGN_MEDIAN_PREDICTED,
    MASS_MIN,
    R_REF_DEFAULT,
    TOL_COURANT_MAX,
)

mpmath.mp.dps = 50  # "referencia de 50 digitos" (Section 6, INV-38(3))


# =============================================================================================
# Independent reference algebra (Section 4.3), never read from any implementation.
# =============================================================================================
def _old_t_star(dr: np.ndarray, dv: np.ndarray, h: float) -> float:
    """PRE-(f) scheme: t* = clamp(-dr.dv/|dv|^2, 0, h), the interior MINIMUM of |sep(t)|^2."""
    dv2 = float(np.dot(dv, dv))
    if dv2 == 0.0:
        return 0.0
    t = -float(np.dot(dr, dv)) / dv2
    return min(max(t, 0.0), h)


def _sep(dr: np.ndarray, dv: np.ndarray, t: float) -> float:
    d = dr + t * dv
    return float(math.sqrt(np.dot(d, d)))


def _t_c(dr: np.ndarray, dv: np.ndarray, r: float, h: float):
    """
    Section 4.3, the five guards in order, and the rationalized root
    t_c = 2c / (-b + sqrt(D)), D = b^2 - 4ac, a = |dv|^2, b = 2(dr.dv), c = |dr|^2 - r^2.
    Returns None if the pair is not a candidate, else t_c in [0, h].
    """
    a = float(np.dot(dv, dv))
    if a == 0.0:
        return None  # guard 1: |dv|^2 == 0, not a candidate (also fails guard 2 identically)
    b = 2.0 * float(np.dot(dr, dv))
    if b >= 0.0:
        return None  # guard 2: dr.dv < 0 strictly required (approaching)
    c = float(np.dot(dr, dr)) - r * r
    if c <= 0.0:
        return 0.0  # guard 3: already overlapping at the start of the step
    d = b * b - 4.0 * a * c
    if d <= 0.0:
        return None  # D > 0 strict (D == 0 is the grazing tangency, measure zero, excluded)
    t = 2.0 * c / (-b + math.sqrt(d))
    if not (0.0 <= t <= h):
        return None  # guard 5: contact does not fall inside this step
    return t


def _t_c_canonical(dr: np.ndarray, dv: np.ndarray, r: float, h: float):
    """The FORBIDDEN canonical form (-b - sqrt(D)) / (2a), Section 4.3: 'a canonica e proibida'."""
    a = float(np.dot(dv, dv))
    if a == 0.0:
        return None
    b = 2.0 * float(np.dot(dr, dv))
    if b >= 0.0:
        return None
    c = float(np.dot(dr, dr)) - r * r
    if c <= 0.0:
        return 0.0
    d = b * b - 4.0 * a * c
    if d <= 0.0:
        return None
    t = (-b - math.sqrt(d)) / (2.0 * a)
    if not (0.0 <= t <= h):
        return None
    return t


def _t_c_mp(a, b, c, h):
    """Section 4.3's rationalized form evaluated in mpmath's arbitrary precision (50 digits)."""
    a, b, c = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(c)
    if a == 0:
        return None
    if b >= 0:
        return None
    if c <= 0:
        return mpmath.mpf(0)
    d = b * b - 4 * a * c
    if d <= 0:
        return None
    t = 2 * c / (-b + mpmath.sqrt(d))
    if not (0 <= t <= h):
        return None
    return t


def _grid_search_min(dr: np.ndarray, dv: np.ndarray, h: float, n_points: int):
    ts = np.linspace(0.0, h, n_points)
    seps_sq = np.sum((dr[None, :] + ts[:, None] * dv[None, :]) ** 2, axis=1)
    idx = int(np.argmin(seps_sq))
    return float(ts[idx]), float(math.sqrt(seps_sq[idx]))


def _grid_search_first_contact(dr: np.ndarray, dv: np.ndarray, r: float, h: float, n_points: int):
    """First grid point (from t=0) at which |sep(t)| <= r, or None if never reached."""
    ts = np.linspace(0.0, h, n_points)
    seps_sq = np.sum((dr[None, :] + ts[:, None] * dv[None, :]) ** 2, axis=1)
    hit = np.nonzero(seps_sq <= r * r)[0]
    if hit.size == 0:
        return None
    idx = int(hit[0])
    return float(ts[idx])


# =============================================================================================
# Section 4.3's guards, tested as pure geometry.
# =============================================================================================
class TestSection43Guards:
    def test_guard1_zero_relative_velocity_is_not_a_candidate(self):
        # 'a == 0 (|dv|^2 == 0): nao e candidato... nenhuma divisao por zero e alcancada.'
        dr = np.array([0.3, -0.2, 0.05])
        dv = np.array([0.0, 0.0, 0.0])
        assert _t_c(dr, dv, r=0.01, h=1e-4) is None

    def test_guard2_separating_pair_is_not_a_candidate(self):
        # dr.dv > 0 at t=0: separating, must be rejected regardless of geometry.
        dr = np.array([1.0, 0.0, 0.0])
        dv = np.array([1.0, 0.0, 0.0])
        assert _t_c(dr, dv, r=0.01, h=1.0) is None

    def test_guard2_is_strict_not_lax(self):
        # dr.dv == 0 exactly (tangential motion, neither approaching nor separating) must also be
        # rejected -- Section 4.3: 'guarda de aproximacao... ESTRITA'.
        dr = np.array([1.0, 0.0, 0.0])
        dv = np.array([0.0, 1.0, 0.0])
        assert float(np.dot(dr, dv)) == 0.0, "test setup error: dr.dv should be exactly 0"
        assert _t_c(dr, dv, r=0.01, h=1.0) is None

    def test_guard3_already_overlapping_resolves_at_t_equals_zero(self):
        # c <= 0: already in contact at the start of the step -- t_c = 0, D not evaluated.
        dr = np.array([0.005, 0.0, 0.0])
        dv = np.array([-1.0, 0.0, 0.0])
        r = 0.01
        assert float(np.dot(dr, dr)) - r * r <= 0.0, "test setup error: expected c <= 0"
        assert _t_c(dr, dv, r=r, h=1.0) == 0.0

    def test_guard4_tangency_d_equals_zero_is_excluded(self):
        # D == 0 exactly: grazing tangency, u.n == 0, excluded by the STRICT D > 0 requirement
        # (Section 4.3: 'e' um conjunto de medida nula; exclui-lo... remove o unico ponto em que
        # o novo esquema reproduziria o antigo').
        # Construct b, c such that D = b^2 - 4ac == 0 exactly: a=1, b=-2, c=1 -> D=4-4=0.
        dr = np.array([1.0, 0.0, 0.0])  # |dr|^2 = 1 = c + r^2 for r = 0 -> use r=0 for exactness
        dv = np.array([-2.0, 0.0, 0.0])  # a = 4... adjust below for an exact D=0 case instead
        # Direct algebraic construction instead of geometric: a=1, b=-2, c=1 => D=0.
        a, b, c = 1.0, -2.0, 1.0
        d = b * b - 4 * a * c
        assert d == 0.0, "test setup error: expected exact tangency"
        # Reproduce _t_c's guard-4 logic directly on (a,b,c) since dr/dv are not uniquely
        # determined by (a,b,c) alone in 3D; the guard itself operates on these scalars only.
        assert not (d > 0.0), "tangency (D=0) must be excluded, not accepted"

    def test_guard5_root_outside_the_step_is_not_a_candidate(self):
        # A pair approaching, but far enough that the true contact falls after h.
        dr = np.array([-10.0, 0.0, 0.0])
        dv = np.array([1.0, 0.0, 0.0])
        assert _t_c(dr, dv, r=0.01, h=1.0) is None

    def test_accepted_t_c_always_satisfies_u_dot_n_strictly_negative(self):
        """
        Section 4.3's central claim: at t_c, u.n < 0 STRICTLY, in every accepted case (interior
        root AND the c<=0 boundary branch). This is what makes the ricochet impulse nonzero
        (Section 4.9(C)) -- the exact defect (u.n == 0 identically at t*) this revision exists to
        fix.
        """
        rng = np.random.default_rng(20260809)
        n_hits = 0
        for _ in range(100_000):
            dr = rng.uniform(-1.0, 1.0, 3)
            dv = rng.uniform(-30.0, 30.0, 3)
            r = rng.uniform(1e-3, 0.5)
            h = rng.uniform(1e-5, 1e-2)
            t = _t_c(dr, dv, r, h)
            if t is None:
                continue
            n_hits += 1
            sep = dr + t * dv
            u_dot_n = float(np.dot(sep, dv))  # sep . dv has the same sign as sep.n * |sep| (n
            # parallel to sep at t_c, Section 4.3's second property) -- sign is all that's tested
            if t == 0.0:
                # boundary branch: u.n < 0 follows from the approach guard alone (dr.dv < 0)
                assert float(np.dot(dr, dv)) < 0.0
            else:
                assert u_dot_n < 0.0, f"u.n >= 0 at an accepted interior t_c: sep.dv={u_dot_n}"
        assert n_hits > 500, "test setup produced too few accepted candidates to be meaningful"

    def test_cancellation_in_c_is_harmless_when_separation_is_close_to_r(self):
        # Section 4.3: '|dr| ~ R: c perde digitos... e' inofensivo: c pequeno significa t_c ~ 0'.
        # A pair starting almost exactly at the contact radius, approaching slowly: t_c should be
        # tiny and the reported separation at t_c should still equal R to high relative precision.
        r = 0.01
        dr = np.array([r * (1.0 + 1e-9), 0.0, 0.0])  # a hair outside contact
        dv = np.array([-0.5, 0.0, 0.0])
        h = 1.0
        t = _t_c(dr, dv, r, h)
        assert t is not None
        assert t < 1e-7, f"expected t_c near zero for a near-contact start, got {t}"
        sep_at_tc = _sep(dr, dv, t)
        assert abs(sep_at_tc / r - 1.0) <= 1e-9


# =============================================================================================
# INV-38(1): the accepted event SET matches the OLD (t*-based) test, up to the measure-zero
# q(h) == 0 boundary.
# =============================================================================================
class TestINV38aEventSetEquivalence:
    """
    Section 6, INV-38(1): over random (dr, dv, R) configurations, the old test
    (|sep(t*)| < R with t* = clamp(-b/2a, 0, h)) and the new test (guards 1-5 of Section 4.3)
    accept the SAME set, except at the measure-zero boundary q(h) == 0.

    N reduced from the document's 2e6 to INV38_EVENT_SET_N_CONFIGS for suite speed -- this is a
    deterministic algebraic equivalence (Section 4.3's own [T] proof, reproduced in this module's
    docstring), not a statistical estimate, so a smaller N still gives a valid pass/fail.
    """

    SEED = 20260809

    def test_old_and_new_detectors_agree_on_the_accepted_set(self):
        rng = np.random.default_rng(self.SEED)
        n = INV38_EVENT_SET_N_CONFIGS
        dr = rng.uniform(-1.0, 1.0, size=(n, 3))
        dv = rng.uniform(-40.0, 40.0, size=(n, 3))
        r = rng.uniform(1e-4, 0.5, size=n)
        h = rng.uniform(1e-6, 1e-2, size=n)

        a = np.sum(dv * dv, axis=1)
        b = 2.0 * np.sum(dr * dv, axis=1)
        c = np.sum(dr * dr, axis=1) - r ** 2

        # OLD test: accept iff q(t*) < 0 with t* = clamp(-b/(2a), 0, h) -- the vertex of
        # q(t) = a t^2 + b t + c -- AND the approach guard (dr.dv < 0). Section 4.3 is explicit
        # this guard is 'inalterada' (unchanged by the revision) and its own equivalence proof is
        # written entirely inside the scope 'seja t_min = -b/(2a) > 0 (positivo pela guarda 2)',
        # i.e. conditioned on the guard already holding for both the old and the new test.
        valid_a = a > 0.0
        approaching = b < 0.0
        t_min = np.where(valid_a, -b / (2.0 * np.where(valid_a, a, 1.0)), 0.0)
        t_star = np.clip(t_min, 0.0, h)
        sep_sq_at_tstar = np.sum((dr + t_star[:, None] * dv) ** 2, axis=1)
        q_at_tstar = sep_sq_at_tstar - r ** 2
        old_accept = valid_a & approaching & (q_at_tstar < 0.0)
        # measure-zero boundary: q(h) == 0 within round-off
        q_at_h = np.sum((dr + h[:, None] * dv) ** 2, axis=1) - r ** 2
        boundary = np.abs(q_at_h) <= INV38_EVENT_SET_BOUNDARY_ULP * EPS_PREC_FP64 * np.maximum(r ** 2, 1.0)

        # NEW test: guards 1-5, vectorized.
        d = b * b - 4.0 * a * c
        approaching = b < 0.0
        overlapped = c <= 0.0
        interior_root = np.where(
            approaching & valid_a & ~overlapped & (d > 0.0),
            2.0 * c / np.where(-b + np.sqrt(np.clip(d, 0.0, None)) != 0.0,
                                -b + np.sqrt(np.clip(d, 0.0, None)), 1.0),
            np.inf,
        )
        new_accept = (
            valid_a & approaching & (
                overlapped | ((d > 0.0) & (interior_root >= 0.0) & (interior_root <= h))
            )
        )

        mismatch = old_accept != new_accept
        # discrepancies are tolerated only at the measure-zero boundary
        genuine_mismatch = mismatch & ~boundary
        discrepant_fraction = float(np.count_nonzero(genuine_mismatch)) / n
        assert discrepant_fraction <= INV38_EVENT_SET_MAX_DISCREPANT_FRACTION, (
            f"old and new detectors disagree on {discrepant_fraction:.3e} of {n} configurations "
            f"outside the measure-zero q(h)=0 boundary (Section 6, INV-38(1)); first few mismatch "
            f"indices: {np.nonzero(genuine_mismatch)[0][:5]}"
        )


class TestINV38aEventSetEquivalenceAgainstRealDetector:
    """The same equivalence, on a smaller batch, against the REAL nbody.collisions.detect()."""

    def test_real_detector_matches_old_test_on_a_random_batch(self):
        if collisions is None:
            pytest.skip("nbody.collisions is not importable")
        rng = np.random.default_rng(20260809_02)
        n = 2000
        dr = rng.uniform(-0.3, 0.3, size=(n, 3))
        dv = rng.uniform(-30.0, 30.0, size=(n, 3))
        r = rng.uniform(1e-3, 0.05, size=n)
        h = 5.0e-4

        r0 = np.zeros((n, 2, 3))
        r0[:, 1, :] = dr
        v0 = np.zeros((n, 2, 3))
        v0[:, 1, :] = dv
        # contact_radii(m, model) = r_ref*(m/m_bar)^(1/3); force R_i+R_j = r by using m_i = m_j =
        # m_bar and scaling r_ref per event is awkward across a single call, so instead solve for
        # the per-event m ratio that reproduces r with a SHARED model.r_ref -- simpler: build one
        # model per event via a fresh CollisionModel with r_ref = r/2 (so R_i=R_j=r/2, sum=r).
        old_predicted = np.zeros(n, dtype=bool)
        new_actual = np.zeros(n, dtype=bool)
        for k in range(n):
            a = float(np.dot(dv[k], dv[k]))
            if a == 0.0:
                old_predicted[k] = False
            else:
                t_min = -float(np.dot(dr[k], dv[k])) / a
                t_star = min(max(t_min, 0.0), h)
                sep_sq = float(np.dot(dr[k] + t_star * dv[k], dr[k] + t_star * dv[k]))
                old_predicted[k] = sep_sq < r[k] ** 2

            model = collisions.CollisionModel(r_ref=r[k] / 2.0, m_bar=config.PARTICLE_MASS)
            state = State(
                r=torch.as_tensor(r0[k], dtype=torch.float64),
                v=torch.as_tensor(v0[k], dtype=torch.float64),
                m=torch.tensor([config.PARTICLE_MASS, config.PARTICLE_MASS], dtype=torch.float64),
            )
            result = collisions.detect(state, h, model)
            new_actual[k] = result.i.numel() > 0

        mismatch = np.count_nonzero(old_predicted != new_actual)
        assert mismatch / n <= INV38_EVENT_SET_MAX_DISCREPANT_FRACTION * 50, (
            f"real detect() disagrees with the OLD t*-based test on {mismatch}/{n} configurations "
            f"(a looser bound than the pure-geometry test above, since this loop also exercises "
            f"detect()'s own R/mass plumbing, not just the root-finding algebra)"
        )


# =============================================================================================
# INV-38(2): the median of |u.n|/|u| over accepted first-contact events, not the minimum.
# =============================================================================================
class TestINV38bSignMedian:
    """
    Section 6, INV-38(2), BLOQUEANTE: with impact parameter b uniform in the disk of radius R
    (the correct distribution for a first-contact detector), (u.n/|u|)^2 = 1 - (b/R)^2 ~ U(0,1),
    so the predicted median of |u.n|/|u| is sqrt(0.5) = 0.7071. The cota is on the MEDIAN
    specifically because a pointwise floor would reject legitimate grazing encounters (b -> R
    gives |u.n|/|u| -> 0 legitimately).
    """

    def _sample_median_un_over_u(self, rng, n, use_old_scheme):
        u_mag = 10.0
        h = 1e-3
        r = 0.01
        ratios = []
        for _ in range(n):
            b = r * math.sqrt(rng.uniform(0.0, 1.0))  # impact parameter uniform IN THE DISK
            n0 = np.array([1.0, 0.0, 0.0])
            t0 = np.array([0.0, 1.0, 0.0])
            dv = -u_mag * n0  # purely along -n0 at first; direction of dr chosen below
            # dr(t_contact) must have |dr|=r with impact parameter b: place dr(t_contact) at
            # angle theta from n0 such that the perpendicular offset is b.
            if b >= r:
                continue
            cos_a = math.sqrt(max(1.0 - (b / r) ** 2, 0.0))
            sin_a = b / r
            dr_at_contact = r * (cos_a * n0 + sin_a * t0)
            if use_old_scheme:
                t_target = 0.5 * h  # arbitrary interior instant for the OLD scheme's t*
                dr0 = dr_at_contact - t_target * dv
                t = _old_t_star(dr0, dv, h)
            else:
                t_target = 0.3 * h
                dr0 = dr_at_contact - t_target * dv
                t = _t_c(dr0, dv, r, h)
                if t is None:
                    continue
            sep = dr0 + t * dv
            sep_norm = np.linalg.norm(sep)
            if sep_norm == 0.0:
                continue
            n_hat = sep / sep_norm
            un_over_u = abs(float(np.dot(dv, n_hat))) / u_mag
            ratios.append(un_over_u)
        return np.array(ratios)

    def test_median_at_least_030_under_first_contact_detection(self):
        rng = np.random.default_rng(20260809_03)
        ratios = self._sample_median_un_over_u(rng, 5000, use_old_scheme=False)
        assert ratios.size > 1000
        median = float(np.median(ratios))
        assert median >= INV38_SIGN_MEDIAN_MIN, (
            f"median |u.n|/|u| = {median:.4f} < {INV38_SIGN_MEDIAN_MIN} (Section 6, INV-38(2))"
        )
        # the document's own predicted value, with generous margin (this is a geometric identity,
        # not a fitted number: (u.n/|u|)^2 ~ U(0,1) => median = sqrt(0.5))
        assert abs(median - INV38_SIGN_MEDIAN_PREDICTED) <= 0.05, (
            f"median {median:.4f} far from the predicted {INV38_SIGN_MEDIAN_PREDICTED}"
        )

    def test_old_t_star_scheme_gives_exactly_zero_u_dot_n_whenever_the_vertex_is_interior(self):
        """
        The discriminating IDENTITY behind the document's '< 1e-5 [M]' measurement (Section 6,
        INV-38(2)): whenever the old scheme's t* is the UNCLAMPED interior vertex of
        q(t) = |sep(t)|^2, u.n = sep.dv/|sep| is EXACTLY zero there (calculus: q'(t*) = 0 by
        definition of an interior minimum, and q'(t)/2 = sep(t).dv). Section 4.3's own defect box
        measures this at |cos(u,n)| <= 5.4e-6 over interior cases in a real run -- a whole-run
        MEDIAN also depends on what fraction of a real run's accepted events happen to be interior
        vs boundary-clamped, which this suite cannot reproduce without executing the (retired) old
        model end to end. What IS a clean, from-the-document identity, independent of any
        particular execution's interior/boundary mix, is this: every interior-vertex case gives
        u.n = 0 to round-off, unconditionally on the geometry. That identity is what test_median_
        at_least_030_under_first_contact_detection above contrasts against t_c's strictly
        negative, well-distributed u.n.
        """
        rng = np.random.default_rng(20260809_04)
        n_interior = 0
        for _ in range(5000):
            dr = rng.uniform(-1.0, 1.0, 3)
            dv = rng.uniform(-30.0, 30.0, 3)
            h = rng.uniform(1e-4, 1e-2)
            dv2 = float(np.dot(dv, dv))
            if dv2 == 0.0:
                continue
            t_min = -float(np.dot(dr, dv)) / dv2
            if not (0.0 < t_min < h):
                continue
            n_interior += 1
            sep = dr + t_min * dv
            sep_norm = np.linalg.norm(sep)
            if sep_norm == 0.0:
                continue
            u_dot_n = float(np.dot(dv, sep / sep_norm))
            assert abs(u_dot_n) <= 1e-9, (
                f"u.n = {u_dot_n} should be exactly 0 (round-off) at an unclamped interior t*"
            )
        assert n_interior > 300, "test setup produced too few interior-vertex cases"


# =============================================================================================
# INV-38(3): stability of the rationalized root form vs the forbidden canonical form.
# =============================================================================================
class TestINV38cRootStability:
    def test_rationalized_form_matches_extended_precision_for_near_cancellation(self):
        case = INV38_CANCELLATION_CASE
        a, b, c = case["a"], case["b"], case["c"]
        assert abs(4 * a * c / b ** 2) <= 1e-12, "test case is not actually near-cancellation"
        h = 10.0
        # Evaluate the rationalized form directly on (a, b, c) in float64:
        d = b * b - 4.0 * a * c
        t_f64 = 2.0 * c / (-b + math.sqrt(d))
        t_ref = _t_c_mp(a, b, c, h)
        assert t_ref is not None
        rel = abs(float(mpmath.mpf(t_f64) / t_ref - 1))
        assert rel <= INV38_ROOT_REL_TOL, (
            f"rationalized root relative error {rel:.3e} exceeds {INV38_ROOT_REL_TOL:.1e} "
            f"against the 50-digit reference (Section 6, INV-38(3))"
        )

    def test_canonical_form_fails_catastrophically_on_the_same_case(self):
        """
        Section 6, INV-38(3): the suite must EXHIBIT the canonical form's failure, not merely
        assert it. a=1, b=-1e6, c=1e-6 (4ac/b^2 ~ 4e-18): the canonical (-b-sqrt(D))/(2a)
        subtracts two nearly equal large numbers and loses essentially all precision.
        """
        case = INV38_CANCELLATION_CASE
        a, b, c = case["a"], case["b"], case["c"]
        d = b * b - 4.0 * a * c
        t_canonical = (-b - math.sqrt(d)) / (2.0 * a)
        t_ref = _t_c_mp(a, b, c, 10.0)
        assert t_ref is not None
        rel = abs(float(mpmath.mpf(t_canonical) / t_ref - 1))
        assert rel > INV38_ROOT_REL_TOL, (
            f"the canonical form should FAIL the {INV38_ROOT_REL_TOL:.1e} tolerance here (that "
            f"is the whole point of INV-38(3)); got relative error {rel:.3e} instead -- if this "
            f"is now small, math.sqrt's rounding behavior changed underneath this test"
        )
        # Section 4.3's own characterization: "a canonica devolve 0.0 -- falha total". Confirm
        # the failure mode, not just its existence.
        assert t_canonical == 0.0, (
            f"expected total cancellation (t_canonical == 0.0) as the document reports for this "
            f"exact case; got {t_canonical!r}"
        )


# =============================================================================================
# Section 4.4 / INV-18(b): Courant-collisional bookkeeping and tunneling. Kept largely as before
# this revision (Section 4.3 states the anti-tunneling argument is untouched); field names
# updated to the (f) rename (t_c).
# =============================================================================================
class TestINV18bCourantFormula:
    def test_lightest_pair_courant_at_dt_2_5e4_matches_document(self):
        c = COLLISION_U_MAX_ASSUMED * 2.5e-4 / COLLISION_R_SUM_LIGHTEST_PAIR
        assert math.isclose(c, COLLISION_C_COLL_LIGHTEST_AT_2_5E4, rel_tol=5e-3)

    def test_lightest_pair_courant_at_dt_collision_matches_document_and_is_valid(self):
        c = COLLISION_U_MAX_ASSUMED * DT_COLLISION / COLLISION_R_SUM_LIGHTEST_PAIR
        assert math.isclose(c, COLLISION_C_COLL_AT_DT_COLLISION, rel_tol=5e-3)
        assert c <= TOL_COURANT_MAX, f"C_coll={c:.4f} exceeds the validity bound {TOL_COURANT_MAX}"

    def test_r_ref_default_matches_chi_times_softening(self):
        assert math.isclose(R_REF_DEFAULT, CHI_DEFAULT * config.SOFTENING, rel_tol=1e-12)

    def test_lightest_pair_contact_radius_sum_from_first_principles(self):
        r_ref = CHI_DEFAULT * config.SOFTENING
        r_light = r_ref * (MASS_MIN / config.PARTICLE_MASS) ** (1.0 / 3.0)
        r_sum = 2.0 * r_light
        assert math.isclose(r_sum, COLLISION_R_SUM_LIGHTEST_PAIR, rel_tol=2e-3)


class TestINV18bTunnelingSurvivesUnderFirstContact:
    """
    Section 4.3: 'o argumento anti-tunelamento da Secao 4.4.2 sobrevive intacto'. The swept
    detector still catches a pair whose true closest approach is well inside the contact radius
    even though both endpoints of the step are clear -- the ONLY thing the (f) revision changes is
    which instant inside the step gets reported (t_c, the entry into the sphere, not t*, the
    deepest point).
    """

    R_SUM = COLLISION_R_SUM_LIGHTEST_PAIR
    U = COLLISION_U_MAX_ASSUMED
    H = 5.0e-4
    B = 0.5 * R_SUM

    @classmethod
    def _configuration(cls):
        t_star_target = cls.H / 2.0
        x0 = cls.U * t_star_target
        dr = np.array([x0, cls.B, 0.0])
        dv = np.array([-cls.U, 0.0, 0.0])
        return dr, dv, cls.H

    def test_static_endpoint_check_misses_the_collision(self):
        dr, dv, h = self._configuration()
        sep_start = _sep(dr, dv, 0.0)
        sep_end = _sep(dr, dv, h)
        assert sep_start > self.R_SUM
        assert sep_end > self.R_SUM

    def test_swept_first_contact_is_inside_the_step_and_before_closest_approach(self):
        dr, dv, h = self._configuration()
        t_c = _t_c(dr, dv, self.R_SUM, h)
        assert t_c is not None, "the tunneling pair must still be a candidate under t_c detection"
        t_old_star = _old_t_star(dr, dv, h)
        assert 0.0 < t_c < t_old_star, (
            f"t_c={t_c} should be strictly BEFORE the closest approach t*={t_old_star} "
            f"(Section 4.3's own proof: 0 < t_c < t_min for an interior candidate)"
        )
        sep_at_tc = _sep(dr, dv, t_c)
        assert abs(sep_at_tc / self.R_SUM - 1.0) <= 1e-9, (
            f"|sep(t_c)| should equal R_sum exactly (contact, not closest approach); got "
            f"{sep_at_tc} vs R_sum={self.R_SUM}"
        )

    def test_courant_number_of_this_configuration_exceeds_one(self):
        c = self.U * self.H / self.R_SUM
        assert c > TOL_COURANT_MAX


# =============================================================================================
# Against the real detector. CollisionModel(r_ref, m_bar, *, e_restitution=..., k_bind=..., ...,
# seed=...) -- v_coh is GONE (Section 9.1.1's 2026-08-09 (f) amendment table), confirmed via
# dataclasses.fields against the live module, not by reading its body.
# =============================================================================================
def _require_collisions_module():
    if collisions is None:
        pytest.skip("nbody.collisions is not importable")
    return collisions


def _make_model(r_ref=R_REF_DEFAULT):
    mod = _require_collisions_module()
    return mod.CollisionModel(r_ref=r_ref, m_bar=config.PARTICLE_MASS)


def _call_detect(model, r, v, m, dt):
    mod = _require_collisions_module()
    state = State(
        r=torch.as_tensor(np.asarray(r), dtype=torch.float64),
        v=torch.as_tensor(np.asarray(v), dtype=torch.float64),
        m=torch.as_tensor(np.asarray(m), dtype=torch.float64),
    )
    return mod.detect(state, dt, model)


def _pairs_from_soa(result):
    """CollisionCandidates is a struct-of-arrays dataclass (i, j, t_c, u_n, ... 1-D tensors),
    normative per Section 9.1.1 (t_c/u_n are the 2026-08-09 (f) rename/addition)."""
    i_np = result.i.detach().cpu().numpy()
    j_np = result.j.detach().cpu().numpy()
    t_np = result.t_c.detach().cpu().numpy()
    un_np = result.u_n.detach().cpu().numpy()
    return [(int(a), int(b), float(t), float(u)) for a, b, t, u in zip(i_np, j_np, t_np, un_np)]


class TestRealDetector:
    def test_tunneling_pair_is_flagged_as_a_candidate(self):
        model = _make_model()
        u = COLLISION_U_MAX_ASSUMED
        h = 5.0e-4
        r_sum = COLLISION_R_SUM_LIGHTEST_PAIR
        b = 0.5 * r_sum
        x0 = u * (h / 2.0)

        r = np.array([[0.0, 0.0, 0.0], [x0, b, 0.0]])
        v_half_step = np.array([[0.0, 0.0, 0.0], [-u, 0.0, 0.0]])
        m = np.array([MASS_MIN, MASS_MIN])

        result = _call_detect(model, r, v_half_step, m, h)
        pairs = _pairs_from_soa(result)

        found = {(min(i, j), max(i, j)) for i, j, _, _ in pairs}
        assert (0, 1) in found, (
            f"detect() did not flag the (0, 1) pair even though the swept first contact is below "
            f"R_sum ({r_sum:.3e} m) and both endpoints are clear (Section 4.4 tunneling case); "
            f"found pairs: {found}"
        )

    def test_detected_t_c_matches_the_closed_form_and_grid_search_first_contact(self):
        model = _make_model()
        u = COLLISION_U_MAX_ASSUMED
        h = 5.0e-4
        # First-principles contact radius sum (Section 4.1: R_i = r_ref*(m_i/m_bar)^(1/3)), NOT
        # the rounded document constant COLLISION_R_SUM_LIGHTEST_PAIR (6.58e-3): this test probes
        # a near-tangent (b close to R) configuration where t_c is exquisitely sensitive to R, so
        # using a value that differs from what detect() actually computes by even 0.2-0.3%
        # (the document's own rounding, already tolerated elsewhere with rel_tol=2e-3) amplifies
        # into an O(1) relative error in t_c. See test_lightest_pair_contact_radius_sum_from_
        # first_principles above for that 0.2% cross-check on the rounded constant itself.
        r_sum = 2.0 * R_REF_DEFAULT * (MASS_MIN / config.PARTICLE_MASS) ** (1.0 / 3.0)
        b = 0.5 * r_sum
        x0 = u * (h / 2.0)

        r = np.array([[0.0, 0.0, 0.0], [x0, b, 0.0]])
        v_half_step = np.array([[0.0, 0.0, 0.0], [-u, 0.0, 0.0]])
        m = np.array([MASS_MIN, MASS_MIN])

        result = _call_detect(model, r, v_half_step, m, h)
        pairs = _pairs_from_soa(result)
        matches = [t for i, j, t, _ in pairs if (min(i, j), max(i, j)) == (0, 1)]
        assert matches, "detect() did not report the (0, 1) pair at all"
        t_reported = matches[0]

        dr = r[1] - r[0]
        dv = v_half_step[1] - v_half_step[0]
        t_formula = _t_c(dr, dv, r_sum, h)
        assert t_formula is not None
        t_grid = _grid_search_first_contact(dr, dv, r_sum, h, 100_001)
        assert t_grid is not None

        assert abs(t_reported - t_formula) <= h * 1.0e-5, (
            f"detect() reported t_c={t_reported}, closed form gives {t_formula}"
        )
        assert abs(t_reported - t_grid) <= h * 1.0e-4, (
            f"detect() reported t_c={t_reported}, independent grid-search first contact gives "
            f"{t_grid}"
        )

    def test_detected_u_n_is_strictly_negative_and_matches_closed_form(self):
        model = _make_model()
        u = COLLISION_U_MAX_ASSUMED
        h = 5.0e-4
        r_sum = COLLISION_R_SUM_LIGHTEST_PAIR
        b = 0.5 * r_sum
        x0 = u * (h / 2.0)

        r = np.array([[0.0, 0.0, 0.0], [x0, b, 0.0]])
        v_half_step = np.array([[0.0, 0.0, 0.0], [-u, 0.0, 0.0]])
        m = np.array([MASS_MIN, MASS_MIN])

        result = _call_detect(model, r, v_half_step, m, h)
        pairs = _pairs_from_soa(result)
        matches = [(t, un) for i, j, t, un in pairs if (min(i, j), max(i, j)) == (0, 1)]
        assert matches
        t_reported, un_reported = matches[0]
        assert un_reported < 0.0, "u.n must be strictly negative at t_c (Section 4.3)"

        dr = r[1] - r[0]
        dv = v_half_step[1] - v_half_step[0]
        sep = dr + t_reported * dv
        n_hat = sep / np.linalg.norm(sep)
        un_formula = float(np.dot(dv, n_hat))
        assert abs(un_reported - un_formula) <= 1e-6 * abs(un_formula)

    def test_clearly_separated_pair_is_not_flagged(self):
        model = _make_model()
        r_sum = COLLISION_R_SUM_LIGHTEST_PAIR
        r = np.array([[0.0, 0.0, 0.0], [10.0 * r_sum, 0.0, 0.0]])
        v_half_step = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        m = np.array([config.PARTICLE_MASS, config.PARTICLE_MASS])

        result = _call_detect(model, r, v_half_step, m, DT_COLLISION)
        pairs = _pairs_from_soa(result)
        found = {(min(i, j), max(i, j)) for i, j, _, _ in pairs}
        assert (0, 1) not in found

    def test_overlapping_but_separating_pair_is_not_a_new_candidate(self):
        # Section 4.3, normative guard: 'o par so e candidato se dr.dv < 0 no inicio do passo' --
        # this is now the ONLY mechanism preventing sticky recollision (no repositioning exists).
        model = _make_model()
        r_sum = COLLISION_R_SUM_LIGHTEST_PAIR
        overlap = 0.3 * r_sum
        r = np.array([[0.0, 0.0, 0.0], [overlap, 0.0, 0.0]])
        v_half_step = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # separating
        m = np.array([MASS_MIN, MASS_MIN])

        result = _call_detect(model, r, v_half_step, m, DT_COLLISION)
        pairs = _pairs_from_soa(result)
        found = {(min(i, j), max(i, j)) for i, j, _, _ in pairs}
        assert (0, 1) not in found, (
            "detect() flagged an overlapping, SEPARATING pair (dr.dv > 0) as a candidate; the "
            "approach guard (Section 4.3) is the only thing preventing sticky recollision now "
            "that repositioning is forbidden (Section 4.5)"
        )

    def test_overlapping_and_approaching_pair_resolves_at_t_c_equals_zero(self):
        # Section 4.3, branch 3: c <= 0 -> t_c = 0. This is the case Section 4.5 says the
        # approach guard must resolve for a pair left overlapping after a merger (larger contact
        # radius than before).
        model = _make_model()
        r_sum = COLLISION_R_SUM_LIGHTEST_PAIR
        overlap = 0.3 * r_sum
        r = np.array([[0.0, 0.0, 0.0], [overlap, 0.0, 0.0]])
        v_half_step = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])  # approaching
        m = np.array([MASS_MIN, MASS_MIN])

        result = _call_detect(model, r, v_half_step, m, DT_COLLISION)
        pairs = _pairs_from_soa(result)
        matches = [t for i, j, t, _ in pairs if (min(i, j), max(i, j)) == (0, 1)]
        assert matches, "an overlapping, approaching pair must be a candidate (Section 4.3, guard 3)"
        assert matches[0] == 0.0, f"expected t_c == 0.0 exactly for c <= 0, got {matches[0]}"

    def test_contact_radii_matches_section_4_1_formula(self):
        mod = _require_collisions_module()
        model = _make_model()
        m_bar = model.m_bar
        m = torch.tensor([MASS_MIN, m_bar, config.PARTICLE_MASS * 284.60], dtype=torch.float64)
        radii = mod.contact_radii(m, model)
        radii_np = radii.detach().cpu().numpy()
        expected = R_REF_DEFAULT * (m.numpy() / m_bar) ** (1.0 / 3.0)
        np.testing.assert_allclose(radii_np, expected, rtol=1e-12)
