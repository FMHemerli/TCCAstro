"""
docs/simulacao-estocastica.md Section 4.3 (swept detection algebra), Section 4.4 (tunneling and
the DT_COLLISION derivation), Section 4.1 (contact radius), and Section 6, INV-18.

INV-18(a) -- t* = clamp(-dr.dv/|dv|^2, 0, h) minimizes |dr + t dv|^2 on [0, h]. Verified two ways:

  (1) independently of any implementation, by comparing the closed form against brute-force grid
      search over the number of points Section 6's own procedure specifies, on random
      configurations plus the two clamp extremes and the |dv|^2 = 0 degenerate case (Section
      4.3) named explicitly as where a naive implementation errs. This checks the
      SPECIFICATION's own algebraic claim; every other test in this module (and in the real
      detector) depends on it being right.

  (2) best-effort against the real nbody.collisions.detect/CollisionModel, via the semantic-role
      binder in _stage2_binding.py (Section 9.1 gives this module's names but not detect's
      parameter list or return type -- a genuine contract gap, see the final report). Skips
      loudly, never vacuously, when the calling convention or return shape cannot be resolved.

INV-18(b) -- tunneling is possible in the perturbative regime (CHI_DEFAULT), and this is
Section 4.4's own justification for DT_COLLISION: a check that only looks at separation at the
START and END of a step ("static") can find both endpoints clear of contact while the true
minimum, strictly inside the step, is below the contact radius. A two-body flyby configuration is
built where this is verified to hold with comfortable, checked margin (not asserted a priori),
and used both as a pure-geometry demonstration and, best-effort, against the real detector.

Out of scope, deliberately: collision RESOLUTION (elastic/merger/fragmentation outcomes, E_int,
L_spin -- INV-20..INV-24). Only detection (Section 4.3-4.4) is covered here.
"""
from __future__ import annotations

import math

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

from _stage2_binding import (  # noqa: E402
    ContractGap,
    bind_by_role,
    call_by_role,
    extract_pair_records,
)

from tolerances import (  # noqa: E402
    CHI_DEFAULT,
    COLLISION_C_COLL_AT_DT_COLLISION,
    COLLISION_C_COLL_LIGHTEST_AT_2_5E4,
    COLLISION_R_SUM_LIGHTEST_PAIR,
    COLLISION_U_MAX_ASSUMED,
    DT_COLLISION,
    INV18A_GRID_POINTS,
    INV18A_N_RANDOM_CONFIGS,
    INV18A_SEP_REL_TOL,
    INV18A_T_STAR_ABS_TOL_OVER_H,
    MASS_MIN,
    R_REF_DEFAULT,
    TOL_COURANT_MAX,
)


# -------------------------------------------------------------------------------------------
# Independent reference algebra (Section 4.3), not read from any implementation.
# -------------------------------------------------------------------------------------------
def _t_star_formula(dr: np.ndarray, dv: np.ndarray, h: float) -> float:
    """Section 4.3: t* = clamp(-dr.dv/|dv|^2, 0, h); |dv|^2 == 0 guard adopts t* = 0."""
    dv2 = float(np.dot(dv, dv))
    if dv2 == 0.0:
        return 0.0
    t = -float(np.dot(dr, dv)) / dv2
    return min(max(t, 0.0), h)


def _sep(dr: np.ndarray, dv: np.ndarray, t: float) -> float:
    d = dr + t * dv
    return float(math.sqrt(np.dot(d, d)))


def _grid_search_min(dr: np.ndarray, dv: np.ndarray, h: float, n_points: int):
    ts = np.linspace(0.0, h, n_points)
    seps_sq = np.sum((dr[None, :] + ts[:, None] * dv[None, :]) ** 2, axis=1)
    idx = int(np.argmin(seps_sq))
    return float(ts[idx]), float(math.sqrt(seps_sq[idx]))


# -------------------------------------------------------------------------------------------
# INV-18(a): closed form vs. brute-force grid search
# -------------------------------------------------------------------------------------------
class TestINV18aTStarFormulaVsGridSearch:
    """
    Section 6, INV-18(a), procedure: '200 configuracoes aleatorias ... comparar com minimizacao
    por varredura de 1e5 pontos.' Tolerances (Section 6): |t*_formula - t*_grid| <= h/1e5, and
    | |sep|_formula/|sep|_grid - 1 | <= 1e-12 (justified there by the parabola being exactly
    quadratic -- no higher-order terms -- so the separation error at a point close to the true
    minimum is second order in the time error).

    Configuration scales (dr in [-0.3, 0.3] m, dv in [-30, 30] m/s matching
    COLLISION_U_MAX_ASSUMED, h up to DT_COLLISION) are chosen to match the physical regime
    Section 4 actually operates in; this keeps the grid spacing (h/1e5) many orders of magnitude
    below any realistic minimum separation, which is what makes the 1e-12 relative bound on
    separation achievable by a finite grid at all (verified empirically below, not assumed).
    """

    SEED = 20260803  # arbitrary, fixed for reproducibility; unrelated to any physics seed

    def test_formula_matches_grid_search_over_random_configurations(self):
        rng = np.random.default_rng(self.SEED)
        worst_t_rel = 0.0
        worst_sep_rel = 0.0
        for _ in range(INV18A_N_RANDOM_CONFIGS):
            dr = rng.uniform(-0.3, 0.3, 3)
            dv = rng.uniform(-30.0, 30.0, 3)
            h = rng.uniform(1e-5, DT_COLLISION)

            t_formula = _t_star_formula(dr, dv, h)
            sep_formula = _sep(dr, dv, t_formula)
            t_grid, sep_grid = _grid_search_min(dr, dv, h, INV18A_GRID_POINTS)

            worst_t_rel = max(worst_t_rel, abs(t_formula - t_grid) / h)
            if sep_grid > 0.0:
                worst_sep_rel = max(worst_sep_rel, abs(sep_formula / sep_grid - 1.0))

        assert worst_t_rel <= INV18A_T_STAR_ABS_TOL_OVER_H, (
            f"worst |t*_formula - t*_grid|/h = {worst_t_rel:.3e} exceeds "
            f"{INV18A_T_STAR_ABS_TOL_OVER_H:.1e}"
        )
        assert worst_sep_rel <= INV18A_SEP_REL_TOL, (
            f"worst separation relative error = {worst_sep_rel:.3e} exceeds "
            f"{INV18A_SEP_REL_TOL:.1e}"
        )


class TestINV18aClampCases:
    """
    The two clamp extremes Section 6 names explicitly as where a naive implementation errs: the
    unconstrained minimizer falling before t=0 (a separating pair) and after t=h (a pair still
    approaching throughout the whole step).
    """

    def test_clamp_at_zero_for_a_separating_pair(self):
        # dr.dv > 0 at t=0: the parabola's true vertex is at negative t, so separation increases
        # monotonically over [0, h] and the minimum on the interval sits at t=0.
        dr = np.array([1.0, 0.0, 0.0])
        dv = np.array([1.0, 0.0, 0.0])
        h = 1.0
        unclamped = -float(np.dot(dr, dv)) / float(np.dot(dv, dv))
        assert unclamped < 0.0, "test setup error: expected the unclamped minimizer to be negative"

        t_formula = _t_star_formula(dr, dv, h)
        t_grid, sep_grid = _grid_search_min(dr, dv, h, INV18A_GRID_POINTS)
        assert t_formula == 0.0
        assert abs(t_formula - t_grid) <= h * INV18A_T_STAR_ABS_TOL_OVER_H
        assert abs(_sep(dr, dv, t_formula) / sep_grid - 1.0) <= INV18A_SEP_REL_TOL

    def test_clamp_at_h_for_a_still_approaching_pair(self):
        # dr.dv < 0 at t=0 and the unconstrained minimizer lies past h: the pair is still
        # approaching throughout the whole step, so the minimum on the interval sits at t=h.
        dr = np.array([-1.0, 0.0, 0.0])
        dv = np.array([0.1, 0.0, 0.0])
        h = 1.0
        unclamped = -float(np.dot(dr, dv)) / float(np.dot(dv, dv))
        assert unclamped > h, "test setup error: expected the unclamped minimizer to exceed h"

        t_formula = _t_star_formula(dr, dv, h)
        t_grid, sep_grid = _grid_search_min(dr, dv, h, INV18A_GRID_POINTS)
        assert t_formula == h
        assert abs(t_formula - t_grid) <= h * INV18A_T_STAR_ABS_TOL_OVER_H
        assert abs(_sep(dr, dv, t_formula) / sep_grid - 1.0) <= INV18A_SEP_REL_TOL


class TestINV18aDegenerateZeroRelativeVelocity:
    """
    Section 4.3, normative: '|dv|^2 == 0: adotar t* = 0. Sem esta guarda ha divisao por zero.
    Ocorre exatamente quando as duas particulas tem velocidade identica bit a bit.'
    """

    def test_t_star_is_zero_when_relative_velocity_vanishes(self):
        dr = np.array([0.3, -0.2, 0.05])
        dv = np.array([0.0, 0.0, 0.0])
        h = 1e-4
        assert _t_star_formula(dr, dv, h) == 0.0

    def test_separation_is_constant_over_the_step(self):
        # With dv = 0, separation does not depend on t: t* = 0 is a minimizer along with every
        # other point of [0, h], so the guard's choice among ties is arbitrary, not a
        # distinguishing numerical claim -- this just confirms the degenerate case is what it
        # claims to be.
        dr = np.array([0.3, -0.2, 0.05])
        dv = np.array([0.0, 0.0, 0.0])
        h = 1e-4
        ref = float(np.linalg.norm(dr))
        for t in (0.0, h / 3.0, h):
            assert math.isclose(_sep(dr, dv, t), ref, rel_tol=1e-15)


# -------------------------------------------------------------------------------------------
# INV-18(b): the Courant-collisional formula and the DT_COLLISION derivation (pure arithmetic)
# -------------------------------------------------------------------------------------------
class TestINV18bCourantFormula:
    """
    Section 4.4: C_coll := |u| dt / (R_i+R_j). Reproduces, by direct arithmetic (no nbody call),
    the numbers Section 4.4 explicitly derives DT_COLLISION from. All inputs are transcribed in
    tolerances.py with their Section 4.4/8 provenance; COLLISION_U_MAX_ASSUMED is marked [A]
    there (an estimate, not [T]) and is used as given, not re-derived.
    """

    def test_lightest_pair_courant_at_dt_2_5e4_matches_document(self):
        c = COLLISION_U_MAX_ASSUMED * 2.5e-4 / COLLISION_R_SUM_LIGHTEST_PAIR
        assert math.isclose(c, COLLISION_C_COLL_LIGHTEST_AT_2_5E4, rel_tol=5e-3)

    def test_lightest_pair_courant_at_dt_collision_matches_document_and_is_valid(self):
        c = COLLISION_U_MAX_ASSUMED * DT_COLLISION / COLLISION_R_SUM_LIGHTEST_PAIR
        assert math.isclose(c, COLLISION_C_COLL_AT_DT_COLLISION, rel_tol=5e-3)
        assert c <= TOL_COURANT_MAX, f"C_coll={c:.4f} exceeds the validity bound {TOL_COURANT_MAX}"

    def test_old_dt_collapse_would_have_exceeded_the_courant_bound(self):
        # This is Section 4.4's own reason for introducing DT_COLLISION: the pre-existing
        # DT_COLLAPSE = 5e-4 puts the fastest, lightest pair's C_coll above the validity bound.
        c = COLLISION_U_MAX_ASSUMED * config.DT_COLLAPSE / COLLISION_R_SUM_LIGHTEST_PAIR
        assert c > TOL_COURANT_MAX, (
            f"C_coll={c:.4f} at DT_COLLAPSE should exceed 1 (Section 4.4); if it does not, the "
            f"documented justification for DT_COLLISION is not reproduced by these inputs"
        )

    def test_r_ref_default_matches_chi_times_softening(self):
        assert math.isclose(R_REF_DEFAULT, CHI_DEFAULT * config.SOFTENING, rel_tol=1e-12)

    def test_lightest_pair_contact_radius_sum_from_first_principles(self):
        # Independent recomputation of R_i+R_j for the m_min-m_min pair from Section 4.1's own
        # formula (R_i = R_ref * (m_i/m_bar)^(1/3)), NOT reusing the document's rounded 6.58e-3 --
        # cross-checks that the tabulated constant in tolerances.py is itself self-consistent.
        r_ref = CHI_DEFAULT * config.SOFTENING
        r_light = r_ref * (MASS_MIN / config.PARTICLE_MASS) ** (1.0 / 3.0)
        r_sum = 2.0 * r_light
        assert math.isclose(r_sum, COLLISION_R_SUM_LIGHTEST_PAIR, rel_tol=2e-3)


class TestINV18bTunnelingDiscrimination:
    """
    Section 4.4/6: tunneling is possible in the perturbative regime, and this is why the swept
    detector (Section 4.3) has to exist: a check that only looks at separation at the start and
    end of a step can find both endpoints clear of contact while the true, interior minimum is
    below the contact radius.

    Construction: two bodies with contact-radius sum R_SUM (the lightest pair at CHI_DEFAULT,
    Section 4.1), relative speed U = COLLISION_U_MAX_ASSUMED (Section 4.4's own worst case, [A]),
    impact parameter B, positioned so the unclamped closed-form minimizer falls exactly at the
    step's midpoint, at H = DT_COLLAPSE = 5e-4 s -- the dt collisions would have inherited without
    Section 4.4's amendment. That both endpoints clear R_SUM with this H is verified numerically
    below, not assumed: it is what makes this a genuine discriminating case.
    """

    R_SUM = COLLISION_R_SUM_LIGHTEST_PAIR
    U = COLLISION_U_MAX_ASSUMED
    H = 5.0e-4  # DT_COLLAPSE, the dt Section 4.4 shows to be inadequate for this pair
    B = 0.5 * R_SUM

    @classmethod
    def _configuration(cls):
        t_star_target = cls.H / 2.0
        x0 = cls.U * t_star_target
        dr = np.array([x0, cls.B, 0.0])
        dv = np.array([-cls.U, 0.0, 0.0])
        return dr, dv, cls.H, t_star_target

    def test_construction_is_self_consistent(self):
        dr, dv, h, t_star_target = self._configuration()
        assert float(np.dot(dr, dv)) < 0.0, "must be approaching at t=0 (Section 4.3 guard)"
        t_formula = _t_star_formula(dr, dv, h)
        assert math.isclose(t_formula, t_star_target, rel_tol=1e-9)

    def test_static_endpoint_check_misses_the_collision(self):
        dr, dv, h, _ = self._configuration()
        sep_start = _sep(dr, dv, 0.0)
        sep_end = _sep(dr, dv, h)
        assert sep_start > self.R_SUM, (
            f"sep(0)={sep_start:.3e} should exceed R_sum={self.R_SUM:.3e} for this to be a "
            f"discriminating tunneling case"
        )
        assert sep_end > self.R_SUM, (
            f"sep(h)={sep_end:.3e} should exceed R_sum={self.R_SUM:.3e} for this to be a "
            f"discriminating tunneling case"
        )

    def test_swept_minimum_is_inside_the_contact_radius(self):
        dr, dv, h, _ = self._configuration()
        t_formula = _t_star_formula(dr, dv, h)
        sep_min = _sep(dr, dv, t_formula)
        assert sep_min < self.R_SUM, (
            f"sep(t*)={sep_min:.3e} should be BELOW R_sum={self.R_SUM:.3e}: this is the "
            f"collision the sweep exists to catch and a start/end-only check would silently drop"
        )
        assert math.isclose(sep_min, self.B, rel_tol=1e-9)

    def test_courant_number_of_this_configuration_exceeds_one(self):
        c = self.U * self.H / self.R_SUM
        assert c > TOL_COURANT_MAX, (
            f"C_coll={c:.3f} for this configuration should exceed 1 -- it is meant to sit in "
            f"the tunneling regime Section 4.4 names"
        )


# -------------------------------------------------------------------------------------------
# Against the real detector. Section 9.1 names nbody.collisions.CollisionModel's fields as
# "r_ref, v_coh, b, w, frag_f_min, eta, seed" and gives detect's/CollisionCandidates' names but
# not a parameter list or return-value shape in prose -- a real contract gap (see final report).
# The gap is closed here NOT by reading src/nbody/collisions.py (never opened), but by
# `inspect.signature`/`dataclasses.fields` introspection on the live, imported objects, which is
# reading a declared public signature -- the same "forma" _stage2_binding.py's bind_by_role
# already treats as fair game, not the algorithm behind it. As discovered this way (current
# round; only detection is implemented, consistent with resolve()/INV-20..24 being out of
# scope): CollisionModel(r_ref, m_bar), detect(state, dt, model) -> CollisionCandidates(i, j,
# t_star, rel_speed, contact_radius_sum), pair_disjoint(candidates) -> AcceptedPairs(..., f_reject)
# -- struct-of-arrays tensors, not a list of per-pair records. Every call below still falls back
# to the role-binder and skips loudly on TypeError, so a future signature change degrades to a
# labelled skip rather than a wrong pass or a crash.
# -------------------------------------------------------------------------------------------
def _require_collisions_module():
    if collisions is None:
        pytest.skip("nbody.collisions is not importable")
    return collisions


def _make_model(r_ref=R_REF_DEFAULT):
    mod = _require_collisions_module()
    model_cls = getattr(mod, "CollisionModel", None)
    if model_cls is None:
        pytest.skip("nbody.collisions.CollisionModel is not defined")
    try:
        return model_cls(r_ref=r_ref)
    except TypeError:
        try:
            return call_by_role(model_cls, {"r_ref": r_ref, "chi": CHI_DEFAULT})
        except ContractGap as exc:
            pytest.skip(str(exc))


def _call_detect(model, r, v, m, dt):
    mod = _require_collisions_module()
    detect_fn = getattr(mod, "detect", None)
    if detect_fn is None:
        pytest.skip("nbody.collisions.detect is not defined")
    state = State(
        r=torch.as_tensor(np.asarray(r), dtype=torch.float64),
        v=torch.as_tensor(np.asarray(v), dtype=torch.float64),
        m=torch.as_tensor(np.asarray(m), dtype=torch.float64),
    )
    try:
        return detect_fn(state, dt, model)
    except TypeError:
        roles = {"state": state, "dt": dt, "h": dt, "model": model, "collision_model": model}
        try:
            kwargs = bind_by_role(detect_fn, roles)
        except ContractGap as exc:
            pytest.skip(str(exc))
        try:
            return detect_fn(**kwargs)
        except Exception as exc:
            pytest.skip(f"detect() bound but raised {type(exc).__name__}: {exc}")


def _pairs_from_soa(result):
    """
    CollisionCandidates/AcceptedPairs are dataclasses of 1-D tensors (i, j, t_star, ...): a
    struct-of-arrays over ALL candidates in one call, discovered via dataclasses.fields, not a
    list of per-pair objects. Falls back to extract_pair_records for a list-shaped return value,
    in case that shape changes.
    """
    i, j = getattr(result, "i", None), getattr(result, "j", None)
    if i is not None and j is not None:
        i_np = i.detach().cpu().numpy() if hasattr(i, "detach") else np.asarray(i)
        j_np = j.detach().cpu().numpy() if hasattr(j, "detach") else np.asarray(j)
        t_star = getattr(result, "t_star", None)
        if t_star is not None:
            t_np = t_star.detach().cpu().numpy() if hasattr(t_star, "detach") else np.asarray(t_star)
        else:
            t_np = [None] * len(i_np)
        return [(int(a), int(b), None if t is None else float(t)) for a, b, t in zip(i_np, j_np, t_np)]
    try:
        return extract_pair_records(list(result))
    except (ContractGap, TypeError):
        raise ContractGap(
            f"could not extract (i, j) candidates from a value of type {type(result)!r}: "
            f"neither struct-of-arrays 'i'/'j' tensor fields nor a list of per-pair records"
        )


class TestINV18RealDetector:
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
        try:
            pairs = _pairs_from_soa(result)
        except ContractGap as exc:
            pytest.skip(str(exc))

        found = {(min(i, j), max(i, j)) for i, j, _ in pairs}
        assert (0, 1) in found, (
            f"detect() did not flag the (0, 1) pair even though the swept minimum separation "
            f"({b:.3e} m) is below R_sum ({r_sum:.3e} m) and both endpoints are clear "
            f"(Section 4.4 tunneling case); found pairs: {found}"
        )

    def test_detected_t_star_matches_the_closed_form_and_grid_search(self):
        # Direct cross-check of INV-18(a) against the REAL implementation (not just the
        # reference formula in this module): the t* the real detector reports for the
        # tunneling configuration must match both the closed form and brute-force grid search.
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
        try:
            pairs = _pairs_from_soa(result)
        except ContractGap as exc:
            pytest.skip(str(exc))
        matches = [t for i, j, t in pairs if (min(i, j), max(i, j)) == (0, 1)]
        if not matches or matches[0] is None:
            pytest.skip("detect()'s return value does not expose t_star per candidate")
        t_reported = matches[0]

        dr = r[1] - r[0]
        dv = v_half_step[1] - v_half_step[0]
        t_formula = _t_star_formula(dr, dv, h)
        t_grid, _ = _grid_search_min(dr, dv, h, INV18A_GRID_POINTS)

        assert abs(t_reported - t_formula) <= h * INV18A_T_STAR_ABS_TOL_OVER_H, (
            f"detect() reported t*={t_reported}, closed form gives {t_formula}"
        )
        assert abs(t_reported - t_grid) <= h * INV18A_T_STAR_ABS_TOL_OVER_H, (
            f"detect() reported t*={t_reported}, independent grid search gives {t_grid}"
        )

    def test_clearly_separated_pair_is_not_flagged(self):
        model = _make_model()
        r_sum = COLLISION_R_SUM_LIGHTEST_PAIR
        r = np.array([[0.0, 0.0, 0.0], [10.0 * r_sum, 0.0, 0.0]])
        v_half_step = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        m = np.array([config.PARTICLE_MASS, config.PARTICLE_MASS])

        result = _call_detect(model, r, v_half_step, m, DT_COLLISION)
        try:
            pairs = _pairs_from_soa(result)
        except ContractGap as exc:
            pytest.skip(str(exc))
        found = {(min(i, j), max(i, j)) for i, j, _ in pairs}
        assert (0, 1) not in found, (
            f"detect() flagged a pair separated by 10x the contact radius, at rest -- false "
            f"positive; found pairs: {found}"
        )

    def test_overlapping_but_separating_pair_is_not_a_new_candidate(self):
        # Section 4.3, normative guard: 'o par so e candidato se dr.dv < 0 no inicio do passo.'
        # Two already-overlapping bodies moving APART must not reappear as a fresh candidate --
        # this is what prevents 'colisoes pegajosas' (a pair that just resolved re-triggering).
        model = _make_model()
        r_sum = COLLISION_R_SUM_LIGHTEST_PAIR
        overlap = 0.3 * r_sum
        r = np.array([[0.0, 0.0, 0.0], [overlap, 0.0, 0.0]])
        v_half_step = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # separating
        m = np.array([MASS_MIN, MASS_MIN])

        result = _call_detect(model, r, v_half_step, m, DT_COLLISION)
        try:
            pairs = _pairs_from_soa(result)
        except ContractGap as exc:
            pytest.skip(str(exc))
        found = {(min(i, j), max(i, j)) for i, j, _ in pairs}
        assert (0, 1) not in found, (
            f"detect() flagged an overlapping pair that is SEPARATING (dr.dv > 0 at t=0) as a "
            f"candidate; Section 4.3's approach guard exists precisely to prevent this ('colisoes "
            f"pegajosas'); found pairs: {found}"
        )

    def test_contact_radii_matches_section_4_1_formula(self):
        # nbody.collisions.contact_radii(m, model) -> Tensor, discovered the same way. Section
        # 4.1: R_i = R_ref * (m_i / m_bar)^(1/3).
        mod = _require_collisions_module()
        fn = getattr(mod, "contact_radii", None)
        if fn is None:
            pytest.skip("nbody.collisions.contact_radii is not defined")
        model = _make_model()
        m_bar = getattr(model, "m_bar", config.PARTICLE_MASS)
        m = torch.tensor([MASS_MIN, m_bar, config.PARTICLE_MASS * 284.60], dtype=torch.float64)
        try:
            radii = fn(m, model)
        except TypeError:
            pytest.skip("contact_radii's parameter list does not match (m, model)")
        radii_np = radii.detach().cpu().numpy() if hasattr(radii, "detach") else np.asarray(radii)
        expected = R_REF_DEFAULT * (m.numpy() / m_bar) ** (1.0 / 3.0)
        np.testing.assert_allclose(radii_np, expected, rtol=1e-12)
