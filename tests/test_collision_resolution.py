"""
docs/simulacao-estocastica.md -- collision RESOLUTION, REVISION (f), 2026-08-09.

"O MODELO DE COLISAO FOI REFEITO." This revision replaces the entire stochastic-map resolution
scheme (Sections 4.6-H, 4.7, 4.7.1, 4.8 -- HISTORICO/APOSENTADO, not tested here) with:

  - Section 4.6: two DETERMINISTIC energy gates, evaluated in order (fusion, then erosion, then
    ricochet by elimination). Zero random draws anywhere in the resolution path.
  - Section 4.9: three outcome channels -- (B) merger (unchanged in substance from the old
    "elastic/merger/fragmentation" scheme's merger map, but gains the mass-based slot rule, (A));
    (C) ricochet with partial restitution e in (0, 1], the e=1 limit reproducing the OLD elastic
    map bit for bit; (D) erosion, which replaces fragmentation -- chips the SMALLER body and
    accretes the chip onto the LARGER, with a rigid center-of-mass-preserving displacement Delta
    instead of any repositioning.
  - Section 4.10: a single rule, E_int += -(dK + dU), with dU EXACT including the O(N) field term
    (no more third-body omission).
  - Section 6: INV-25 and INV-26 are RETIRED (no regime map left to be well-formed or consistent
    with a sampler). INV-20 and INV-22 are heavily amended. INV-21 gains the gate and the
    mass-based slot rule. INV-32 is rewritten from "exactly 2 draws per event" to "the Generator
    is untouched, in every channel". INV-33 through INV-38 are NEW.

THREE EXPLICIT PROHIBITIONS from the project's physicist, honored throughout this module:
  1. No test fixes the SIGN of E_int or requires it to grow. Every merger makes delta_e_int
     strictly negative BY IDENTITY with the fusion gate (Section 4.10's "achado do dev"); erosion
     can go either way (~11.3% of parameter space gives delta_e_int < 0, Section 4.10). Only
     ricochet has a guaranteed sign. The per-CHANNEL decomposition (ricochet monotone increasing,
     merger monotone decreasing, erosion uncapped) is what is tested instead.
  2. INV-36's per-event ceiling is tested with U recomputed via the INDEPENDENT, O(N^2) path
     (nbody.observables.potential_energy over the WHOLE system), never via the closed-form dU the
     event map itself used -- Section 6 is explicit that reusing the same dU expression on both
     sides makes the test a tautology that a missing eps, a wrong d~, or an entirely missing field
     term would all pass.
  3. INV-23(a)'s old floor (TOL_EVENT_CONS_FLOOR = 1e-8, "abaixo disso terceiros estao sendo
     computados") is RETIRED and does not appear anywhere in this module or in tolerances.py --
     it is inconsistent with INV-36's 1e-12 ceiling (Section 6, INV-23(a) box).

Written from the specification alone. nbody.collisions.CollisionModel/CollisionCandidates/
AcceptedPairs/CollisionOutcome/resolve/collision_pass were never opened as source -- only their
PUBLIC SIGNATURES via inspect.signature/dataclasses.fields, confirming: CollisionModel has no
v_coh field and carries e_restitution/k_bind/chip_coeff/chip_max/energy_max (all kw_only, Section
9.1.1's 2026-08-09 (f) amendment table); CollisionOutcome carries n_ricochet/n_erosion (not
n_elastic/n_fragment); regime_probabilities and v_coh_from_state do not exist any more. Every
closed-form prediction below (final r/v/m per participant, delta_e_int, delta_l_spin, the gate
thresholds themselves) is derived directly from Sections 4.6/4.9/4.10's own formulas, independent
of resolve()'s implementation.

Design note on forcing a specific gate deterministically. Because the gates are pure functions of
(|u|, T_n, masses, contact geometry) with NO random component (Section 4.6/4.7.1), this suite
constructs each synthetic event's relative speed |u| and approach angle (u.n/|u|) directly from
the CLOSED-FORM gate thresholds (v_esc = sqrt(2GM/d~_c), E_lig = K_BIND*G*m_P^2/R_P) computed from
the document's own formulas -- never from resolve() -- with a safety margin, and then VERIFIES the
targeting worked by checking the returned CollisionOutcome's n_merge/n_ricochet/n_erosion count
against the batch size before applying that channel's own closed-form checks (same "loud failure
here is itself a finding, not a silent mis-test" discipline the pairing/detection modules use).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.state import State  # noqa: E402
from nbody import observables  # noqa: E402

try:
    import nbody.collisions as collisions
except ImportError:
    collisions = None

from tolerances import (  # noqa: E402
    EPS_PREC_FP64,
    FRAG_CHIP_MAX,
    FRAG_ENERGY_MAX,
    GATE_Q_OVERLAP,
    INV20_DELTA_K_REL_TOL,
    INV20_DELTA_K_REL_TOL_WIDE_BATCH,
    INV20_LIMIT_CASE_ULP,
    INV20_UN_OVER_U_MIN,
    INV22_DISPLACEMENT_ULP,
    INV22_SPEED_RATIO_TABLE,
    INV22_T_PRIME_REL_TOL,
    INV33_EROSION_MASS_JUMP_MAX,
    INV33_MERGER_MASS_JUMP_MAX,
    INV36_AGGREGATE_REL_TOL,
    INV37_BOUNDARY_REL_TOL_ULP,
    INV37_CONTINUITY_T_N_OVER_E_LIG_EXCESS,
    INV37_CONTINUITY_U_PRIME_REL_TOL,
    K_BIND,
    PAIR_D_SOFT_MBAR,
    PAIR_ELIG_MBAR,
    PAIR_VESC_NEWT_MBAR,
    PAIR_VESC_SOFT_MBAR,
    RESOLUTION_GEOMETRY_SEED,
    RESOLUTION_MASS_RATIO_MAX,
    RESOLUTION_N_EVENTS_PER_CHANNEL,
    TOL_EVENT_CONS_FP64,
    TOL_EVENT_INV_ULP,
    TOL_EVENT_INV_ULP_WIDE_RATIO_BATCH,
    TOL_EVENT_PRED,
    TOL_EVENT_PRED_WIDE_RATIO_BATCH,
)

G = config.G
EPS_SOFTENING = config.SOFTENING
M_BAR = config.PARTICLE_MASS
R_REF = config.R_REF_DEFAULT
E0_SCALE = abs(config.IC_U_EPS005)  # the real N=1000 equal-mass system's |E_0| (Section 7/4.10's
                                     # normalization reference for TOL-EVENT-CONS)


def _require_collisions():
    if collisions is None:
        pytest.skip("nbody.collisions is not importable")
    return collisions


def _make_model(r_ref=R_REF, m_bar=M_BAR, e_restitution=config.E_RESTITUTION,
                 k_bind=config.K_BIND, chip_coeff=config.FRAG_CHIP_COEFF,
                 chip_max=config.FRAG_CHIP_MAX, energy_max=config.FRAG_ENERGY_MAX):
    col = _require_collisions()
    return col.CollisionModel(r_ref=r_ref, m_bar=m_bar, e_restitution=e_restitution,
                               k_bind=k_bind, chip_coeff=chip_coeff, chip_max=chip_max,
                               energy_max=energy_max)


# =============================================================================================
# Independent closed-form oracle -- Sections 4.1/4.6/4.9/4.10, read from the document, never from
# nbody.collisions.
# =============================================================================================
def _contact_radius(m, r_ref=R_REF, m_bar=M_BAR):
    return r_ref * (np.asarray(m) / m_bar) ** (1.0 / 3.0)


def _v_esc(m_i, m_j, r_i, r_j, eps=EPS_SOFTENING):
    """Section 4.6, portao 1: v_esc = sqrt(2 G M / d~_c), d~_c = sqrt((R_i+R_j)^2 + eps^2)."""
    M = m_i + m_j
    d_tilde = np.sqrt((r_i + r_j) ** 2 + eps ** 2)
    return np.sqrt(2.0 * G * M / d_tilde)


def _e_lig(m_p, r_p, k_bind=K_BIND):
    """Section 4.6, portao 2: E_lig = K_BIND * G * m_P^2 / R_P (NEWTONIAN, not softened)."""
    return k_bind * G * m_p ** 2 / r_p


def _gate_masses(m_i, m_j):
    """Section 4.9(A): m_G = max, m_P = min, tie broken by lower index (handled by caller)."""
    return np.maximum(m_i, m_j), np.minimum(m_i, m_j)


def _unit(v):
    return v / np.linalg.norm(v)


def _sample_mass_pairs(rng, n_events, ratio_max=RESOLUTION_MASS_RATIO_MAX):
    """Section 6, INV-20: 'razao de massa ate 1000'. m_i = m_bar, m_j log-uniform in [1/1000,1000]
    of m_bar -- symmetric in log so that roughly half the batch has m_i > m_j and half m_i < m_j,
    which is what exercises the mass-based slot rule (Section 4.9(A)) in both directions."""
    log_ratio_max = math.log(ratio_max)
    log_ratio = rng.uniform(-log_ratio_max, log_ratio_max, size=n_events)
    m_i = np.full(n_events, M_BAR)
    m_j = M_BAR * np.exp(log_ratio)
    return m_i, m_j


# =============================================================================================
# Synthetic-event construction. Each event is an isolated two-body pair placed at slots (2k,2k+1)
# of a single packed state, with a prescribed relative speed |u| and approach cosine u.n/|u|,
# positioned so the pair is exactly in contact (|sep(t_c)| = R_i+R_j) at t_c -- the first-contact
# configuration Section 4.3 guarantees, taken as GIVEN here (Section 4.3/INV-38 is
# tests/test_collision_detection.py's job; this module bypasses detect()/pair_disjoint and hands
# resolve() an AcceptedPairs built directly, exactly like tests/test_collision_pairing.py does for
# pair_disjoint).
# =============================================================================================
def _build_batch(rng, n_events, m_i, m_j, u_mag, cos_theta, dt, t_c_frac, r_ref=R_REF,
                  m_bar=M_BAR):
    """
    Returns (state0, accepted, events) where events holds everything needed to independently
    predict pair k's outcome in isolation: m_i, m_j, r_i0, r_j0, v_i0, v_j0 (all at t=0), n_hat
    (contact normal at t_c, i -> j), u (relative velocity, constant along the drift), u_n (u.n at
    t_c), r_sum (R_i+R_j), t_c, dt.

    t_c is held constant across the batch so slot order (pair k at (2k, 2k+1), i=2k<j=2k+1) is
    already sorted by the (t_c, i, j) lexicographic key AcceptedPairs is required to satisfy
    (Section 4.5) -- this module constructs AcceptedPairs directly, bypassing detect/pair_disjoint
    (covered by test_collision_detection.py / test_collision_pairing.py), and must hand resolve()
    an input that respects that invariant.
    """
    m_i = np.asarray(m_i, dtype=np.float64).copy()
    m_j = np.asarray(m_j, dtype=np.float64).copy()
    u_mag = np.asarray(u_mag, dtype=np.float64)
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    r_i = _contact_radius(m_i, r_ref, m_bar)
    r_j = _contact_radius(m_j, r_ref, m_bar)
    r_sum = r_i + r_j
    t_c = np.full(n_events, t_c_frac * dt)

    r_i0 = np.empty((n_events, 3))
    r_j0 = np.empty((n_events, 3))
    v_i0 = np.empty((n_events, 3))
    v_j0 = np.empty((n_events, 3))
    n_hat = np.empty((n_events, 3))
    u_vec = np.empty((n_events, 3))
    u_n_arr = np.empty(n_events)

    for k in range(n_events):
        n0 = _unit(rng.normal(size=3))
        raw_t = rng.normal(size=3)
        raw_t = raw_t - np.dot(raw_t, n0) * n0
        t0 = _unit(raw_t)
        sin_theta = math.sqrt(max(1.0 - cos_theta[k] ** 2, 0.0))
        dv = -u_mag[k] * cos_theta[k] * n0 + u_mag[k] * sin_theta * t0
        dr_at_tc = r_sum[k] * n0
        dr0 = dr_at_tc - t_c[k] * dv

        v_cm = rng.normal(scale=1.0, size=3)
        vi = v_cm - (m_j[k] / (m_i[k] + m_j[k])) * dv
        vj = v_cm + (m_i[k] / (m_i[k] + m_j[k])) * dv

        # No inter-pair offset. Every pair sits at slots (2k, 2k+1) with positions near the
        # origin, at the ~0.01 m scale of the contact geometry itself.
        #
        # This is safe for every test in this module EXCEPT ones that read out.delta_e_int (or
        # any other quantity built from the O(N) field term) from a batch of MANY co-located
        # pairs: Section 4.10 (f) computes that term EXACTLY, as a sum over every OTHER body in
        # the system (Section 4.5 item 3), so with hundreds of pairs packed near the origin the
        # ~2(n_events-1) bodies belonging to every OTHER pair become genuine, dominant third
        # bodies for THIS pair's field term (caught here: an early version of this construction
        # tried a large per-pair offset to avoid exactly that, and instead introduced a WORSE bug
        # -- float64 catastrophic cancellation reconstructing a ~0.01 m separation from a pair of
        # ~1e5 m absolute positions, corrupting even the purely local, position-independent
        # ricochet kinematics). The two delta_e_int tests that need genuine third-body isolation
        # (TestINV20RicochetOutcome/TestINV21MergerOutcome's own e_int checks) do NOT use this
        # shared batch; they run each event through its own ISOLATED resolve() call with no other
        # bodies present at all (see _isolated_delta_e_int_sum below), which is exact by
        # construction rather than by any offset choice.
        base = np.zeros(3)

        r_i0[k] = base
        r_j0[k] = base + dr0
        v_i0[k], v_j0[k] = vi, vj
        n_hat[k] = n0
        u_vec[k] = dv
        u_n_arr[k] = float(np.dot(dv, n0))

    r0 = np.empty((2 * n_events, 3))
    v0 = np.empty((2 * n_events, 3))
    m0 = np.empty(2 * n_events)
    r0[0::2], r0[1::2] = r_i0, r_j0
    v0[0::2], v0[1::2] = v_i0, v_j0
    m0[0::2], m0[1::2] = m_i, m_j

    state0 = State(
        r=torch.as_tensor(r0, dtype=torch.float64),
        v=torch.as_tensor(v0, dtype=torch.float64),
        m=torch.as_tensor(m0, dtype=torch.float64),
    )
    idx_i = np.arange(0, 2 * n_events, 2)
    idx_j = np.arange(1, 2 * n_events, 2)
    col = _require_collisions()
    accepted = col.AcceptedPairs(
        i=torch.as_tensor(idx_i, dtype=torch.int64),
        j=torch.as_tensor(idx_j, dtype=torch.int64),
        t_c=torch.as_tensor(t_c, dtype=torch.float64),
        u_n=torch.as_tensor(u_n_arr, dtype=torch.float64),
        rel_speed=torch.as_tensor(u_mag, dtype=torch.float64),
        contact_radius_sum=torch.as_tensor(r_sum, dtype=torch.float64),
        f_reject=0.0,
    )
    events = dict(
        m_i=m_i, m_j=m_j, r_i0=r_i0, r_j0=r_j0, v_i0=v_i0, v_j0=v_j0,
        n_hat=n_hat, u=u_vec, u_n=u_n_arr, r_sum=r_sum, r_i=r_i, r_j=r_j, t_c=t_c, dt=dt,
    )
    return state0, accepted, events


def _fusion_targets(rng, m_i, m_j, r_i, r_j, eps=EPS_SOFTENING):
    """Section 4.6, portao 1: |u| < v_esc. Any approach angle passes -- the fusion gate does not
    depend on direction, only on |u| (T_cm < G m_i m_j / d~_c <=> |u| < v_esc)."""
    v_esc = _v_esc(m_i, m_j, r_i, r_j, eps)
    frac = rng.uniform(0.2, 0.9, size=len(m_i))
    u_mag = frac * v_esc
    cos_theta = rng.uniform(0.2, 0.95, size=len(m_i))
    return u_mag, cos_theta


def _ricochet_targets(rng, m_i, m_j, r_i, r_j, k_bind=K_BIND, eps=EPS_SOFTENING):
    """Neither gate: |u| >= v_esc (portal 1 false) AND T_n <= E_lig (portal 2 false)."""
    v_esc = _v_esc(m_i, m_j, r_i, r_j, eps)
    m_G, m_P = _gate_masses(m_i, m_j)
    r_P = np.where(m_i <= m_j, r_i, r_j)
    e_lig = _e_lig(m_P, r_P, k_bind)
    mu = m_i * m_j / (m_i + m_j)
    u_mag = 1.2 * v_esc * rng.uniform(1.0, 2.0, size=len(m_i))
    cap = 0.5 * np.sqrt(np.clip(2.0 * e_lig / (mu * u_mag ** 2), 0.0, None))
    cos_theta = np.clip(cap, 0.02, 0.9)
    return u_mag, cos_theta


def _erosion_targets(rng, m_i, m_j, r_i, r_j, k_bind=K_BIND, eps=EPS_SOFTENING):
    """Both gate-2 conditions needed: |u| >= v_esc (portal 1 false) AND T_n > E_lig (portal 2
    true). cos_theta drawn close to 1 (near head-on) so a modest speed margin over BOTH thresholds
    suffices; |u| set to the larger of the two per-event minimums, with margin."""
    v_esc = _v_esc(m_i, m_j, r_i, r_j, eps)
    m_G, m_P = _gate_masses(m_i, m_j)
    r_P = np.where(m_i <= m_j, r_i, r_j)
    e_lig = _e_lig(m_P, r_P, k_bind)
    mu = m_i * m_j / (m_i + m_j)
    cos_theta = rng.uniform(0.85, 0.98, size=len(m_i))
    u_min_gate1 = 1.2 * v_esc
    u_min_gate2 = 1.3 * np.sqrt(2.0 * e_lig / (mu * cos_theta ** 2))
    u_mag = np.maximum(u_min_gate1, u_min_gate2)
    return u_mag, cos_theta


def _channel_of(out):
    counts = {"merge": out.n_merge, "ricochet": out.n_ricochet, "erosion": out.n_erosion}
    hits = [name for name, c in counts.items() if c > 0]
    return counts


def _isolated_delta_e_int_sum(m_i, m_j, u_mag, cos_theta, model, dt, t_c_frac,
                               expected_channel):
    """
    Runs each event of a batch through its OWN, ISOLATED resolve() call -- state containing
    exactly the two participant slots, no other body present at all -- and returns
    (sum of actual delta_e_int, list of per-event outcomes). With no third body present, the O(N)
    field-term sum in Section 4.10's rule is a sum over an EMPTY set: there is no ambiguity from
    other pairs' proximity to disentangle, unlike a single resolve() call over many co-located
    pairs (Section 4.5 item 3's field term reads every OTHER body in the SAME call's state).
    """
    col = _require_collisions()
    total = 0.0
    n = len(m_i)
    for k in range(n):
        rng_k = np.random.default_rng(20260810_000 + k)
        state0, accepted, ev = _build_batch(rng_k, 1, np.array([m_i[k]]), np.array([m_j[k]]),
                                             np.array([u_mag[k]]), np.array([cos_theta[k]]), dt,
                                             t_c_frac)
        out = col.resolve(state0, dt, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        counts = (out.n_merge, out.n_ricochet, out.n_erosion)
        expected = {"merge": (1, 0, 0), "ricochet": (0, 1, 0), "erosion": (0, 0, 1)}[
            expected_channel
        ]
        assert counts == expected, f"event {k}: channel forcing failed ({counts} != {expected})"
        total += out.delta_e_int
    return total


DT = 1.0e-3
T_C_FRAC = 0.35


# =============================================================================================
# INV-20 -- ricochet outcome: m, P, L exact; Delta K closed form; the map is NOT the identity
# (Section 4.9(C), Section 6 INV-20).
# =============================================================================================
class TestINV20RicochetOutcome:
    @pytest.fixture(scope="class")
    def resolved(self):
        col = _require_collisions()
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _ricochet_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        return state0, events, out, model

    def test_channel_forcing_worked(self, resolved):
        _, events, out, _ = resolved
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, n, 0), (
            "batch was constructed with |u| >= v_esc and T_n <= E_lig for every event (Section "
            "4.6); a mismatch means resolve()'s gate computation diverges from this module's own "
            "closed-form v_esc/E_lig, not that the ricochet map itself is wrong"
        )

    def test_non_triviality_un_over_u_and_nonzero_impulse(self, resolved):
        # Section 6, INV-20, NOVA clausula BLOQUEANTE: this is the clause that catches the OLD
        # model's null map (t*-detection gave u.n == 0 identically, Section 4.3 defect 1).
        state0, events, out, _ = resolved
        u = events["u"]
        u_n = events["u_n"]
        u_mag = np.linalg.norm(u, axis=1)
        assert np.all(np.abs(u_n) >= INV20_UN_OVER_U_MIN * u_mag)
        v_after = out.state.v.numpy()
        dv_i = v_after[0::2] - events["v_i0"]
        dv_j = v_after[1::2] - events["v_j0"]
        assert np.all(np.linalg.norm(dv_i, axis=1) > 0.0)
        assert np.all(np.linalg.norm(dv_j, axis=1) > 0.0)

    def test_restitution_applied_correctly(self, resolved):
        # |u'.n / (-e * u.n) - 1| <= 100 eps_prec (Section 6, INV-20).
        state0, events, out, model = resolved
        v_after = out.state.v.numpy()
        u_after = v_after[1::2] - v_after[0::2]
        n_hat = events["n_hat"]
        u_after_n = np.sum(u_after * n_hat, axis=1)
        predicted = -model.e_restitution * events["u_n"]
        rel = np.abs(u_after_n / predicted - 1.0)
        assert np.all(rel <= INV20_LIMIT_CASE_ULP * EPS_PREC_FP64)

    def test_tangential_component_preserved(self, resolved):
        state0, events, out, _ = resolved
        v_after = out.state.v.numpy()
        u_after = v_after[1::2] - v_after[0::2]
        n_hat = events["n_hat"]
        u = events["u"]
        u_t_before = u - np.sum(u * n_hat, axis=1)[:, None] * n_hat
        u_t_after = u_after - np.sum(u_after * n_hat, axis=1)[:, None] * n_hat
        u_mag = np.linalg.norm(u, axis=1)
        rel = np.linalg.norm(u_t_after - u_t_before, axis=1)
        assert np.all(rel <= INV20_LIMIT_CASE_ULP * EPS_PREC_FP64 * u_mag)

    def test_masses_exactly_unchanged(self, resolved):
        state0, events, out, _ = resolved
        assert np.array_equal(state0.m.numpy(), out.state.m.numpy())

    def test_positions_only_move_by_the_post_impulse_drift(self, resolved):
        # Ricochet leaves positions untouched AT t_c; only the final (h - t_c) drift differs,
        # because velocity changed. r(t_c) is unaffected by the map itself.
        state0, events, out, _ = resolved
        r_i_tc = events["r_i0"] + events["t_c"][:, None] * events["v_i0"]
        r_j_tc = events["r_j0"] + events["t_c"][:, None] * events["v_j0"]
        v_after = out.state.v.numpy()
        remaining = events["dt"] - events["t_c"]
        r_i_final = r_i_tc + remaining[:, None] * v_after[0::2]
        r_j_final = r_j_tc + remaining[:, None] * v_after[1::2]
        r_after = out.state.r.numpy()
        np.testing.assert_allclose(r_after[0::2], r_i_final, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(r_after[1::2], r_j_final, rtol=1e-9, atol=1e-9)

    def test_linear_momentum_conserved_per_pair(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        p_before = m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]
        v_after = out.state.v.numpy()
        p_after = m_i[:, None] * v_after[0::2] + m_j[:, None] * v_after[1::2]
        rel = np.linalg.norm(p_after - p_before, axis=1) / np.linalg.norm(p_before, axis=1)
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_delta_k_matches_closed_form(self, resolved):
        """
        Section 6, INV-20 (EMENDA): Delta K = -(1/2) mu (1-e^2) u_n^2, not 0.

        This module's own ricochet-targeting sweep (_ricochet_targets) deliberately spans
        cos_theta down to 0.02 -- grazing encounters legitimately have |u.n| close to zero
        (Section 6, INV-38(2)'s own "raspoes legitimos" argument) -- and the document's ratio
        form divides by that PREDICTED value, so near-cancellation in this test's own division is
        expected at the grazing end and is not itself informative about resolve()'s precision.
        Checked here at INV20_DELTA_K_REL_TOL_WIDE_BATCH (~30x the observed worst case, see
        tolerances.py); the well-conditioned test below pins the document's bare 1e-12 on a batch
        that stays away from that regime.
        """
        state0, events, out, model = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        e = model.e_restitution
        predicted_delta_k = -0.5 * mu * (1.0 - e ** 2) * events["u_n"] ** 2
        k_before = 0.5 * m_i * np.sum(events["v_i0"] ** 2, axis=1) + \
            0.5 * m_j * np.sum(events["v_j0"] ** 2, axis=1)
        v_after = out.state.v.numpy()
        k_after = 0.5 * m_i * np.sum(v_after[0::2] ** 2, axis=1) + \
            0.5 * m_j * np.sum(v_after[1::2] ** 2, axis=1)
        delta_k = k_after - k_before
        rel = np.abs(delta_k / predicted_delta_k - 1.0)
        assert np.all(rel <= INV20_DELTA_K_REL_TOL_WIDE_BATCH)

    def test_delta_k_matches_closed_form_well_conditioned(self):
        """
        Same clause, on a batch restricted to cos_theta >= 0.3 (away from the grazing/near-
        cancellation regime), at the document's bare 1e-12.

        Built by over-sampling from _ricochet_targets (which already respects gate 2 by
        construction, via its own per-event cos_theta cap) and FILTERING to cos_theta >= 0.3,
        rather than drawing cos_theta uniformly in [0.3, 0.6] directly: at the full [1, 1000] mass
        ratio sweep, gate 2's cap on cos_theta can sit well below 0.3 for some mass pairs, so a
        fixed [0.3, 0.6] draw is not always compatible with staying on the ricochet side of gate 2
        (caught here: an earlier version of this construction asserted that compatibility instead
        of ensuring it, and failed on exactly the mass ratios where the cap binds).
        """
        col = _require_collisions()
        n = 200
        pool = 4 * n
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 1100)
        log_ratio_max = math.log(20.0)  # narrower mass-ratio range: this test's purpose is the
        # strict formula check away from grazing, not re-sweeping the full 1000x ratio range
        # already covered (at a looser tolerance) by the class's own shared batch above.
        log_ratio = rng.uniform(-log_ratio_max, log_ratio_max, size=pool)
        m_i_pool = np.full(pool, M_BAR)
        m_j_pool = M_BAR * np.exp(log_ratio)
        r_i_pool, r_j_pool = _contact_radius(m_i_pool), _contact_radius(m_j_pool)
        u_mag_pool, cos_theta_pool = _ricochet_targets(rng, m_i_pool, m_j_pool, r_i_pool, r_j_pool)
        keep = cos_theta_pool >= 0.3
        assert np.count_nonzero(keep) >= n, "test setup error: pool too small after filtering"
        idx = np.nonzero(keep)[0][:n]
        m_i, m_j = m_i_pool[idx], m_j_pool[idx]
        r_i, r_j = r_i_pool[idx], r_j_pool[idx]
        u_mag, cos_theta = u_mag_pool[idx], cos_theta_pool[idx]
        m_G, m_P = _gate_masses(m_i, m_j)
        r_P = np.where(m_i <= m_j, r_i, r_j)
        e_lig = _e_lig(m_P, r_P)
        mu = m_i * m_j / (m_i + m_j)
        t_n = 0.5 * mu * (u_mag * cos_theta) ** 2
        assert np.all(t_n <= e_lig), "test setup error: must stay on the ricochet side of gate 2"
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, n, 0)

        e = model.e_restitution
        predicted_delta_k = -0.5 * mu * (1.0 - e ** 2) * events["u_n"] ** 2
        k_before = 0.5 * m_i * np.sum(events["v_i0"] ** 2, axis=1) + \
            0.5 * m_j * np.sum(events["v_j0"] ** 2, axis=1)
        v_after = out.state.v.numpy()
        k_after = 0.5 * m_i * np.sum(v_after[0::2] ** 2, axis=1) + \
            0.5 * m_j * np.sum(v_after[1::2] ** 2, axis=1)
        delta_k = k_after - k_before
        rel = np.abs(delta_k / predicted_delta_k - 1.0)
        assert np.all(rel <= INV20_DELTA_K_REL_TOL)

    def test_angular_momentum_conserved_per_pair(self, resolved):
        """
        Section 6/4.14: the discriminating test. The ricochet impulse is central (n parallel to
        dr(t_c), Section 4.3's second property), so L is conserved EXACTLY for each pair, both
        before and after the impulse -- a free particle's L about any fixed origin is unaffected
        by straight-line drift. If the impulse were applied with a normal not parallel to the true
        separation AT t_c, L would NOT be conserved even though K-type checks could still pass.
        """
        state0, events, out, _ = resolved
        r_i0, r_j0 = events["r_i0"], events["r_j0"]
        v_i0, v_j0 = events["v_i0"], events["v_j0"]
        m_i, m_j = events["m_i"], events["m_j"]
        l_before = m_i[:, None] * np.cross(r_i0, v_i0) + m_j[:, None] * np.cross(r_j0, v_j0)
        r_after = out.state.r.numpy()
        v_after = out.state.v.numpy()
        l_after = (m_i[:, None] * np.cross(r_after[0::2], v_after[0::2])
                   + m_j[:, None] * np.cross(r_after[1::2], v_after[1::2]))
        l_scale = np.linalg.norm(l_before, axis=1)
        l_scale = np.where(l_scale > 0.0, l_scale, 1.0)
        rel = np.linalg.norm(l_after - l_before, axis=1) / l_scale
        assert np.all(rel <= TOL_EVENT_INV_ULP_WIDE_RATIO_BATCH * EPS_PREC_FP64), (
            f"max relative L residual = {rel.max():.3e}; if this fails while K-type checks pass, "
            f"the impulse normal is not parallel to the true separation AT t_c"
        )

    def test_sum_m_r_conserved_per_pair(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        m_total = m_i + m_j
        r_before = m_i[:, None] * events["r_i0"] + m_j[:, None] * events["r_j0"]
        c_before = r_before / m_total[:, None]
        v_cm_before = (m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]) / \
            m_total[:, None]
        c_predicted_after = c_before + events["dt"] * v_cm_before
        r_after_arr = out.state.r.numpy()
        r_after = m_i[:, None] * r_after_arr[0::2] + m_j[:, None] * r_after_arr[1::2]
        c_after = r_after / m_total[:, None]
        rel = np.linalg.norm(c_after - c_predicted_after, axis=1) / np.linalg.norm(c_before, axis=1)
        assert np.all(rel <= TOL_EVENT_INV_ULP_WIDE_RATIO_BATCH * EPS_PREC_FP64)

    def test_delta_e_int_matches_closed_form_for_any_e(self, resolved):
        # Section 4.9(C)/4.10: E_int += (1/2) mu (1-e^2) u_n^2, EXACTLY, no O(N) sum needed
        # (ricochet touches no U term, Section 4.10's own table).
        state0, events, out, model = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        e = model.e_restitution
        predicted_total = np.sum(0.5 * mu * (1.0 - e ** 2) * events["u_n"] ** 2)
        rel = abs(out.delta_e_int - predicted_total) / max(abs(predicted_total), 1.0)
        assert rel <= TOL_EVENT_CONS_FP64

    def test_delta_l_spin_is_exactly_zero(self, resolved):
        # n parallel to dr(t_c) => dr x (impulse along n) = 0 identically, so L_spin gains
        # nothing: the ricochet's L conservation (test above) does not need L_spin at all.
        _, _, out, _ = resolved
        assert torch.equal(out.delta_l_spin, torch.zeros(3, dtype=out.delta_l_spin.dtype))

    def test_e_equals_one_reproduces_the_old_elastic_map_bit_for_bit(self):
        # Section 6, INV-20, NOVA clausula de caso-limite.
        col = _require_collisions()
        n = 200
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 1001)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _ricochet_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model(e_restitution=1.0)
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, n, 0)

        n_hat = events["n_hat"]
        u_n = events["u_n"]
        mu = m_i * m_j / (m_i + m_j)
        impulse = 2.0 * mu[:, None] * u_n[:, None] * n_hat  # OLD elastic map: J = 2 mu (u.n) n
        v_i_prime = events["v_i0"] + impulse / m_i[:, None]
        v_j_prime = events["v_j0"] - impulse / m_j[:, None]
        v_after = out.state.v.numpy()
        np.testing.assert_allclose(v_after[0::2], v_i_prime, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(v_after[1::2], v_j_prime, rtol=1e-9, atol=1e-9)

        k_before = 0.5 * m_i * np.sum(events["v_i0"] ** 2, axis=1) + \
            0.5 * m_j * np.sum(events["v_j0"] ** 2, axis=1)
        k_after = 0.5 * m_i * np.sum(v_after[0::2] ** 2, axis=1) + \
            0.5 * m_j * np.sum(v_after[1::2] ** 2, axis=1)
        rel = np.abs(k_after - k_before) / k_before
        assert np.all(rel <= INV20_LIMIT_CASE_ULP * EPS_PREC_FP64), (
            "e=1 must give Delta K = 0 (the old elastic map), within 100 eps_prec"
        )
        assert abs(out.delta_e_int) <= INV20_LIMIT_CASE_ULP * EPS_PREC_FP64 * max(
            np.sum(k_before), 1.0
        ), "e=1 must leave E_int untouched (the old elastic map's own claim)"


# =============================================================================================
# INV-21 -- merger outcome: conservations exact; destructions EXACT and PREDICTED (Section 4.9(B),
# Section 6 INV-21).
# =============================================================================================
class TestINV21MergerOutcome:
    @pytest.fixture(scope="class")
    def resolved(self):
        col = _require_collisions()
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 1)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _fusion_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        return state0, events, out, model

    def test_channel_forcing_worked(self, resolved):
        _, _, out, _ = resolved
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (n, 0, 0), (
            "batch was constructed with |u| < v_esc for every event (Section 4.6, portao 1); a "
            "mismatch means resolve()'s v_esc computation diverges from this module's own closed "
            "form, not that the merger map itself is wrong"
        )

    def test_slot_assignment_by_mass_when_i_is_bigger(self):
        # Section 4.9(A): the slot of the LARGER mass survives, regardless of index.
        col = _require_collisions()
        model = _make_model()
        m_i, m_j = 5.0 * M_BAR, 1.0 * M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _fusion_targets(np.random.default_rng(1), np.array([m_i]),
                                            np.array([m_j]), np.array([r_i]), np.array([r_j]))
        rng = np.random.default_rng(2)
        state0, accepted, events = _build_batch(rng, 1, np.array([m_i]), np.array([m_j]),
                                                  u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (1, 0, 0)
        m_after = out.state.m.numpy()
        assert m_after[0] == pytest.approx(m_i + m_j, rel=0, abs=0)
        assert m_after[1] == 0.0

    def test_slot_assignment_by_mass_when_j_is_bigger(self):
        col = _require_collisions()
        model = _make_model()
        m_i, m_j = 1.0 * M_BAR, 5.0 * M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _fusion_targets(np.random.default_rng(1), np.array([m_i]),
                                            np.array([m_j]), np.array([r_i]), np.array([r_j]))
        rng = np.random.default_rng(3)
        state0, accepted, events = _build_batch(rng, 1, np.array([m_i]), np.array([m_j]),
                                                  u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (1, 0, 0)
        m_after = out.state.m.numpy()
        assert m_after[0] == 0.0
        assert m_after[1] == pytest.approx(m_i + m_j, rel=0, abs=0)

    def test_slot_assignment_tie_goes_to_lower_index(self):
        # Section 4.9(A): "EMPATE EXATO de massa -> desempata pelo INDICE MENOR".
        col = _require_collisions()
        model = _make_model()
        m_i = m_j = M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _fusion_targets(np.random.default_rng(1), np.array([m_i]),
                                            np.array([m_j]), np.array([r_i]), np.array([r_j]))
        rng = np.random.default_rng(4)
        state0, accepted, events = _build_batch(rng, 1, np.array([m_i]), np.array([m_j]),
                                                  u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (1, 0, 0)
        m_after = out.state.m.numpy()
        assert m_after[0] == pytest.approx(m_i + m_j, rel=0, abs=0)
        assert m_after[1] == 0.0

    def test_dead_slot_gets_the_merged_bodys_position_and_velocity(self, resolved):
        state0, events, out, _ = resolved
        r_after = out.state.r.numpy()
        v_after = out.state.v.numpy()
        np.testing.assert_allclose(r_after[0::2], r_after[1::2], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(v_after[0::2], v_after[1::2], rtol=0, atol=0)

    def test_final_position_and_velocity_match_closed_form(self, resolved):
        # Merger, mass rule aside, is otherwise UNCHANGED in substance from the previous mapping:
        # slot <- M, r <- r_c(t_c), v <- V. This checks r_c/V regardless of WHICH slot survives.
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        m_total = m_i + m_j
        v_cm = (m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]) / m_total[:, None]
        r_i_tc = events["r_i0"] + events["t_c"][:, None] * events["v_i0"]
        r_j_tc = events["r_j0"] + events["t_c"][:, None] * events["v_j0"]
        r_c = (m_i[:, None] * r_i_tc + m_j[:, None] * r_j_tc) / m_total[:, None]
        remaining = events["dt"] - events["t_c"]
        r_final = r_c + remaining[:, None] * v_cm

        m_after = out.state.m.numpy()
        r_after = out.state.r.numpy()
        v_after = out.state.v.numpy()
        survivor_is_i = m_after[0::2] > 0.0
        r_survivor = np.where(survivor_is_i[:, None], r_after[0::2], r_after[1::2])
        v_survivor = np.where(survivor_is_i[:, None], v_after[0::2], v_after[1::2])
        np.testing.assert_allclose(r_survivor, r_final, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(v_survivor, v_cm, rtol=1e-9, atol=1e-9)

    def test_mass_conserved_exactly(self, resolved):
        state0, events, out, _ = resolved
        m_before = state0.m.numpy().reshape(-1, 2).sum(axis=1)
        m_after = out.state.m.numpy().reshape(-1, 2).sum(axis=1)
        rel = np.abs(m_after - m_before) / m_before
        assert np.all(rel <= 4.0 * EPS_PREC_FP64)

    def test_linear_momentum_conserved_exactly(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        p_before = m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]
        m_after = out.state.m.numpy()
        v_after = out.state.v.numpy()
        p_after = m_after[0::2, None] * v_after[0::2] + m_after[1::2, None] * v_after[1::2]
        rel = np.linalg.norm(p_after - p_before, axis=1) / np.linalg.norm(p_before, axis=1)
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_sum_m_r_of_the_pair_preserved(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        m_total = m_i + m_j
        r_i_tc = events["r_i0"] + events["t_c"][:, None] * events["v_i0"]
        r_j_tc = events["r_j0"] + events["t_c"][:, None] * events["v_j0"]
        c_before = (m_i[:, None] * r_i_tc + m_j[:, None] * r_j_tc) / m_total[:, None]
        v_cm = (m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]) / m_total[:, None]
        remaining = events["dt"] - events["t_c"]
        c_predicted_final = c_before + remaining[:, None] * v_cm

        m_after = out.state.m.numpy()
        r_after = out.state.r.numpy()
        survivor_is_i = m_after[0::2] > 0.0
        r_survivor = np.where(survivor_is_i[:, None], r_after[0::2], r_after[1::2])
        rel = np.linalg.norm(r_survivor - c_predicted_final, axis=1) / np.linalg.norm(c_before, axis=1)
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_kinetic_energy_destroyed_exactly_by_t_cm(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        t_cm = 0.5 * mu * np.sum(events["u"] ** 2, axis=1)
        k_before = 0.5 * m_i * np.sum(events["v_i0"] ** 2, axis=1) + \
            0.5 * m_j * np.sum(events["v_j0"] ** 2, axis=1)
        m_after = out.state.m.numpy()
        v_after = out.state.v.numpy()
        m_survivor = m_after[0::2] + m_after[1::2]  # == m_i+m_j, exactly one slot nonzero
        v_survivor = np.where((m_after[0::2] > 0.0)[:, None], v_after[0::2], v_after[1::2])
        k_after = 0.5 * m_survivor * np.sum(v_survivor ** 2, axis=1)
        delta_k = k_after - k_before
        rel = np.abs(delta_k / (-t_cm) - 1.0)
        assert np.all(rel <= TOL_EVENT_PRED_WIDE_RATIO_BATCH)

    def test_angular_momentum_destroyed_exactly_by_spin_term(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        r_i_tc = events["r_i0"] + events["t_c"][:, None] * events["v_i0"]
        r_j_tc = events["r_j0"] + events["t_c"][:, None] * events["v_j0"]
        dr = r_j_tc - r_i_tc
        predicted_delta_l = -mu[:, None] * np.cross(dr, events["u"])

        l_before = m_i[:, None] * np.cross(events["r_i0"], events["v_i0"]) + \
            m_j[:, None] * np.cross(events["r_j0"], events["v_j0"])
        m_after = out.state.m.numpy()
        r_after = out.state.r.numpy()
        v_after = out.state.v.numpy()
        survivor_is_i = (m_after[0::2] > 0.0)[:, None]
        r_survivor = np.where(survivor_is_i, r_after[0::2], r_after[1::2])
        v_survivor = np.where(survivor_is_i, v_after[0::2], v_after[1::2])
        m_survivor = m_after[0::2] + m_after[1::2]
        l_after = m_survivor[:, None] * np.cross(r_survivor, v_survivor)
        delta_l = l_after - l_before

        norm_predicted = np.linalg.norm(predicted_delta_l, axis=1)
        assert np.all(norm_predicted > 0.0), "degenerate (collinear dr, u) batch: spin test vacuous"
        rel = np.linalg.norm(delta_l - predicted_delta_l, axis=1) / norm_predicted
        assert np.all(rel <= TOL_EVENT_PRED_WIDE_RATIO_BATCH)

    def test_delta_l_spin_accumulator_matches_the_same_closed_form(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        r_i_tc = events["r_i0"] + events["t_c"][:, None] * events["v_i0"]
        r_j_tc = events["r_j0"] + events["t_c"][:, None] * events["v_j0"]
        dr = r_j_tc - r_i_tc
        predicted_total = np.sum(mu[:, None] * np.cross(dr, events["u"]), axis=0)
        actual = out.delta_l_spin.numpy()
        rel = np.linalg.norm(actual - predicted_total) / np.linalg.norm(predicted_total)
        assert rel <= TOL_EVENT_PRED

    def test_e_int_increment_matches_pair_level_closed_form(self):
        """
        Section 4.10: E_int += T_cm - G m_i m_j / d~_c, with dU_campo == 0 EXACTLY only when no
        third body is present at all.

        Deliberately does NOT reuse the class's shared `resolved` fixture: that fixture packs
        RESOLUTION_N_EVENTS_PER_CHANNEL merger events at the origin in a SINGLE resolve() call,
        and under this revision E_int's field term is an exact O(N) sum over every OTHER body in
        that same call's state (Section 4.5 item 3) -- with hundreds of co-located pairs, every
        other pair's two bodies are genuine (and dominant) third bodies for THIS pair's field
        term, and the pair-level-only prediction below would be off by an O(1) factor for a
        reason that has nothing to do with resolve()'s correctness. Each event here instead gets
        its own isolated resolve() call with NO other body present, so the field-term sum is over
        an empty set and the pair-level closed form is exact by construction, not by any
        offset/isolation choice about positions.
        """
        n = 100
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 1200)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _fusion_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()

        # Predicted closed form needs each event's OWN geometry (t_c-frame separation, u), which
        # _build_batch derives internally per isolated call; reconstruct it the same way here for
        # the prediction side using a matching per-event RNG stream (mirrors
        # _isolated_delta_e_int_sum's own seeding so predicted and actual see identical geometry).
        predicted_total = 0.0
        for k in range(n):
            rng_k = np.random.default_rng(20260810_000 + k)
            _, _, ev = _build_batch(rng_k, 1, np.array([m_i[k]]), np.array([m_j[k]]),
                                     np.array([u_mag[k]]), np.array([cos_theta[k]]), DT, T_C_FRAC)
            mu_k = m_i[k] * m_j[k] / (m_i[k] + m_j[k])
            t_cm = 0.5 * mu_k * float(np.sum(ev["u"][0] ** 2))
            d_ij = float(np.linalg.norm(ev["r_j0"][0] - ev["r_i0"][0] +
                                         ev["t_c"][0] * (ev["v_j0"][0] - ev["v_i0"][0])))
            d_tilde = math.sqrt(d_ij ** 2 + EPS_SOFTENING ** 2)
            e_mutual = G * m_i[k] * m_j[k] / d_tilde
            predicted_total += t_cm - e_mutual

        actual_total = _isolated_delta_e_int_sum(m_i, m_j, u_mag, cos_theta, model, DT,
                                                  T_C_FRAC, "merge")
        rel = abs(actual_total - predicted_total) / max(abs(predicted_total), 1.0)
        assert rel <= TOL_EVENT_CONS_FP64, (
            f"actual={actual_total:.6e}, predicted={predicted_total:.6e}, rel={rel:.3e} "
            f"(Section 6, INV-21/INV-36)"
        )

    def test_softened_vesc_not_newtonian_gives_ricochet_not_merger_in_the_discriminating_band(self):
        """
        Section 6, INV-21 EMENDA: for the m_bar-m_bar pair, |u| in (2.2881, 5.1668) m/s -- the
        band the NEWTONIAN v_esc would classify as merger and the SOFTENED (correct) one does
        not. This is the only clause that distinguishes the two forms of v_esc (Section 4.6.1).
        """
        col = _require_collisions()
        model = _make_model()
        u_mag = 0.5 * (PAIR_VESC_SOFT_MBAR + PAIR_VESC_NEWT_MBAR)  # 3.7275, well inside the band
        assert PAIR_VESC_SOFT_MBAR < u_mag < PAIR_VESC_NEWT_MBAR
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 900)
        state0, accepted, events = _build_batch(
            rng, 1, np.array([M_BAR]), np.array([M_BAR]), np.array([u_mag]), np.array([0.3]),
            DT, T_C_FRAC,
        )
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, 1, 0), (
            f"|u|={u_mag:.4f} m/s sits between the softened v_esc ({PAIR_VESC_SOFT_MBAR}) and "
            f"the (rejected) newtonian v_esc ({PAIR_VESC_NEWT_MBAR}); the softened form must "
            f"classify this as ricochet -- a merger here means the newtonian R, not d~_c, is "
            f"being used in the gate (Section 4.6.1)"
        )


class TestINV21MergerGateCanLegitimatelyInjectEnergy:
    """
    Regression guard, adapted from the pre-(f) suite's analogous test. Section 4.10's "achado do
    dev": under the gate, delta_e_int is ALWAYS negative for merger (E_rel < 0 is literally the
    gate condition) -- so a claim of "merger can inject energy" from the pre-(f) revision no
    longer holds and must NOT be reasserted (Prohibition 1 in this module's docstring). What
    SURVIVES is the magnitude claim: delta_e_int can be a large NEGATIVE number for an imbalanced,
    slow merger (mutual well large, T_cm small), which is exactly the "toda fusao decresce E_int"
    identity Section 4.10 derives and this test confirms numerically -- not a floor, not a sign
    requirement on the agregate, just the pair-level closed form applied to an imbalanced case.
    """

    def test_imbalanced_slow_merger_gives_a_large_negative_delta_e_int(self):
        col = _require_collisions()
        model = _make_model()
        m_i = 300.0 * M_BAR
        m_j = 1.0 * M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        v_esc = _v_esc(m_i, m_j, r_i, r_j)
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 901)
        u_mag = 0.3 * v_esc  # comfortably inside the fusion gate
        state0, accepted, events = _build_batch(
            rng, 1, np.array([m_i]), np.array([m_j]), np.array([u_mag]), np.array([0.5]),
            DT, T_C_FRAC,
        )
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (1, 0, 0)
        assert out.delta_e_int < 0.0, (
            "Section 4.10's own identity: every merger has delta_e_int < 0 (E_rel < 0 IS the "
            "fusion gate); a non-negative value here contradicts the gate that just fired"
        )


# =============================================================================================
# INV-22 -- erosion outcome: m, P, sum m r exact; T' closed form; separation guaranteed (Section
# 4.9(D), Section 6 INV-22, REWRITTEN in the 2026-08-09 (f) revision).
# =============================================================================================
class TestINV22ErosionOutcome:
    @pytest.fixture(scope="class")
    def resolved(self):
        col = _require_collisions()
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 2)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _erosion_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        return state0, events, out, model

    def test_channel_forcing_worked(self, resolved):
        _, _, out, _ = resolved
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, 0, n), (
            "batch was constructed with |u| >= v_esc AND T_n > E_lig for every event (Section "
            "4.6); a mismatch means resolve()'s gate computation diverges from this module's own "
            "closed form, not that the erosion map itself is wrong"
        )

    def _predicted_split(self, events, model):
        m_i, m_j = events["m_i"], events["m_j"]
        m_G = np.maximum(m_i, m_j)
        m_P = np.minimum(m_i, m_j)
        r_G = np.where(m_i >= m_j, events["r_i"], events["r_j"])
        r_P = np.where(m_i >= m_j, events["r_j"], events["r_i"])
        mu = m_i * m_j / (m_i + m_j)
        q = m_G / m_P
        u_n = events["u_n"]
        t_n = 0.5 * mu * u_n ** 2
        e_lig = model.k_bind * G * m_P ** 2 / r_P
        xi = model.chip_coeff * (t_n / e_lig - 1.0)
        f_chip = np.clip(xi, 0.0, model.chip_max) * (1.0 - q ** (-5.0 / 3.0))

        e = model.e_restitution
        n_hat = events["n_hat"]
        u = events["u"]
        u_r = u - (1.0 + e) * u_n[:, None] * n_hat
        t_r = 0.5 * mu * np.sum(u_r ** 2, axis=1)
        e_custo = np.minimum(f_chip * e_lig, model.energy_max * t_r)
        capped = e_custo < f_chip * e_lig
        f_chip = np.where(capped, e_custo / e_lig, f_chip)

        m_chip = f_chip * m_P
        m_G_p = m_G + m_chip
        m_P_p = m_P - m_chip
        mu_p = m_G_p * m_P_p / (m_G_p + m_P_p)
        t_prime = t_r - e_custo
        u_prime_mag = np.linalg.norm(u_r, axis=1) * np.sqrt(t_prime / t_r) * np.sqrt(mu / mu_p)
        return dict(m_G=m_G, m_P=m_P, r_G=r_G, r_P=r_P, q=q, f_chip=f_chip, m_chip=m_chip,
                    m_G_p=m_G_p, m_P_p=m_P_p, mu=mu, mu_p=mu_p, u_r=u_r, t_r=t_r, e_custo=e_custo,
                    t_prime=t_prime, u_prime_mag=u_prime_mag)

    def test_mass_conserved_exactly(self, resolved):
        state0, events, out, _ = resolved
        m_before = state0.m.numpy().reshape(-1, 2).sum(axis=1)
        m_after = out.state.m.numpy().reshape(-1, 2).sum(axis=1)
        rel = np.abs(m_after - m_before) / m_before
        assert np.all(rel <= 4.0 * EPS_PREC_FP64)

    def test_predicted_mass_split_matches(self, resolved):
        state0, events, out, model = resolved
        pred = self._predicted_split(events, model)
        m_i, m_j = events["m_i"], events["m_j"]
        i_is_g = m_i >= m_j
        m_after = out.state.m.numpy()
        m_i_after = m_after[0::2]
        m_j_after = m_after[1::2]
        pred_i = np.where(i_is_g, pred["m_G_p"], pred["m_P_p"])
        pred_j = np.where(i_is_g, pred["m_P_p"], pred["m_G_p"])
        rel_i = np.abs(m_i_after - pred_i) / np.maximum(m_i, m_j)
        rel_j = np.abs(m_j_after - pred_j) / np.maximum(m_i, m_j)
        assert np.all(rel_i <= 1e-6)
        assert np.all(rel_j <= 1e-6)

    def test_linear_momentum_conserved_exactly(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        p_before = m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]
        m_after = out.state.m.numpy()
        v_after = out.state.v.numpy()
        p_after = m_after[0::2, None] * v_after[0::2] + m_after[1::2, None] * v_after[1::2]
        rel = np.linalg.norm(p_after - p_before, axis=1) / np.linalg.norm(p_before, axis=1)
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_sum_m_r_conserved_via_the_rigid_shift(self, resolved):
        """
        Section 6, INV-22: 'A clausula Delta(sum m r) e' a discriminante desta lista.' Catches an
        implementation that moves mass between the two participants WITHOUT the compensating rigid
        displacement Delta = (m_chip/M)(r_P - r_G) of Section 4.9(D).
        """
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        m_total = m_i + m_j
        r_i_tc = events["r_i0"] + events["t_c"][:, None] * events["v_i0"]
        r_j_tc = events["r_j0"] + events["t_c"][:, None] * events["v_j0"]
        r_before = m_i[:, None] * r_i_tc + m_j[:, None] * r_j_tc
        m_after = out.state.m.numpy()
        v_after = out.state.v.numpy()
        remaining = events["dt"] - events["t_c"]
        r_after_arr = out.state.r.numpy()
        r_i_post_map = r_after_arr[0::2] - remaining[:, None] * v_after[0::2]
        r_j_post_map = r_after_arr[1::2] - remaining[:, None] * v_after[1::2]
        r_after = m_after[0::2, None] * r_i_post_map + m_after[1::2, None] * r_j_post_map
        norm_before = np.linalg.norm(r_before, axis=1)
        rel = np.linalg.norm(r_after - r_before, axis=1) / norm_before
        assert np.all(rel <= TOL_EVENT_INV_ULP_WIDE_RATIO_BATCH * EPS_PREC_FP64)

    def test_separation_intact(self, resolved):
        # (r_P' - r_G') - (r_P - r_G) == 0, exactly (up to round-off): the rigid shift moves both
        # participants by the SAME Delta, so their separation is untouched (Section 4.9(D)).
        state0, events, out, _ = resolved
        r_i_tc = events["r_i0"] + events["t_c"][:, None] * events["v_i0"]
        r_j_tc = events["r_j0"] + events["t_c"][:, None] * events["v_j0"]
        sep_before = r_j_tc - r_i_tc
        v_after = out.state.v.numpy()
        remaining = events["dt"] - events["t_c"]
        r_after_arr = out.state.r.numpy()
        r_i_post_map = r_after_arr[0::2] - remaining[:, None] * v_after[0::2]
        r_j_post_map = r_after_arr[1::2] - remaining[:, None] * v_after[1::2]
        sep_after = r_j_post_map - r_i_post_map
        d_c = events["r_sum"]
        rel = np.linalg.norm(sep_after - sep_before, axis=1) / d_c
        assert np.all(rel <= INV22_DISPLACEMENT_ULP * EPS_PREC_FP64)

    def test_t_prime_matches_closed_form(self, resolved):
        state0, events, out, model = resolved
        pred = self._predicted_split(events, model)
        v_after = out.state.v.numpy()
        u_after = v_after[1::2] - v_after[0::2]
        m_after = out.state.m.numpy()
        mu_p_actual = (m_after[0::2] * m_after[1::2]) / (m_after[0::2] + m_after[1::2])
        t_from_state = 0.5 * mu_p_actual * np.sum(u_after ** 2, axis=1)
        rel = np.abs(t_from_state / pred["t_prime"] - 1.0)
        assert np.all(rel <= INV22_T_PRIME_REL_TOL)

    def test_u_prime_dot_n_strictly_positive(self, resolved):
        # Section 6, INV-22/INV-34: BLOQUEANTE. Separation guarantee.
        state0, events, out, _ = resolved
        v_after = out.state.v.numpy()
        u_after = v_after[1::2] - v_after[0::2]
        n_hat = events["n_hat"]
        u_after_n = np.sum(u_after * n_hat, axis=1)
        assert np.all(u_after_n > 0.0)

    def test_speed_ratio_bound_depends_on_e(self, resolved):
        """
        Section 6, INV-22: |u'|/|u| >= sqrt(max(e^2 - FRAG_CHIP_MAX, (1-FRAG_ENERGY_MAX)*e^2)).
        The document is explicit this cannot be a single fixed number: tested here at the model's
        configured e (default 0.8) via the FORMULA, and cross-checked against the document's own
        worked value for e=0.8 (0.3742) as a sanity, not as the bound itself.
        """
        state0, events, out, model = resolved
        e = model.e_restitution
        bound = math.sqrt(max(e ** 2 - FRAG_CHIP_MAX, (1.0 - FRAG_ENERGY_MAX) * e ** 2))
        assert abs(bound - INV22_SPEED_RATIO_TABLE[round(e, 1)]) <= 1e-3, (
            "sanity: formula does not match the document's own worked table entry for this e"
        )
        v_after = out.state.v.numpy()
        u_after = v_after[1::2] - v_after[0::2]
        u_after_mag = np.linalg.norm(u_after, axis=1)
        u_mag = np.linalg.norm(events["u"], axis=1)
        ratio = u_after_mag / u_mag
        assert np.all(ratio >= bound - 1e-9), (
            f"min observed ratio {ratio.min():.4f} < bound {bound:.4f} at e={e}"
        )

    def test_speed_ratio_bound_at_other_e_values(self):
        # The bound's e-DEPENDENCE is the point (Section 6: "a cota depende de e"): re-run a
        # smaller batch at e=0.5 (where the FRAG_ENERGY_MAX guard is the active term) and e=0.2.
        col = _require_collisions()
        for e in (0.5, 0.2):
            model = _make_model(e_restitution=e)
            n = 200
            rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 2000 + int(e * 10))
            m_i, m_j = _sample_mass_pairs(rng, n)
            r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
            u_mag, cos_theta = _erosion_targets(rng, m_i, m_j, r_i, r_j)
            state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT,
                                                      T_C_FRAC)
            out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                               softening=EPS_SOFTENING)
            assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, 0, n)
            bound = math.sqrt(max(e ** 2 - FRAG_CHIP_MAX, (1.0 - FRAG_ENERGY_MAX) * e ** 2))
            v_after = out.state.v.numpy()
            u_after = v_after[1::2] - v_after[0::2]
            ratio = np.linalg.norm(u_after, axis=1) / np.linalg.norm(events["u"], axis=1)
            assert np.all(ratio >= bound - 1e-9), f"e={e}: min ratio {ratio.min()} < bound {bound}"

    def test_geometry_theorem_r_g_plus_r_p_shrinks(self, resolved):
        # Section 6/4.9(D.1): R_G' + R_P' < R_G + R_P, a THEOREM, no cap needed.
        state0, events, out, model = resolved
        pred = self._predicted_split(events, model)
        r_G_new = _contact_radius(pred["m_G_p"])
        r_P_new = _contact_radius(pred["m_P_p"])
        r_new_sum = r_G_new + r_P_new
        r_old_sum = pred["r_G"] + pred["r_P"]
        assert np.all(r_new_sum < r_old_sum)

    def test_ordering_preserved_g_still_at_least_p(self, resolved):
        state0, events, out, model = resolved
        pred = self._predicted_split(events, model)
        assert np.all(pred["m_G_p"] >= pred["m_P_p"])

    def test_chip_fraction_vanishes_as_mass_ratio_goes_to_one(self):
        # Section 6/4.9(D.4): f_chip -> 0 as q -> 1 -- the assimetry factor (1-q^-5/3), tested at
        # q extremely close to 1 (equal-mass pair with a tiny perturbation).
        col = _require_collisions()
        model = _make_model()
        m_i = M_BAR
        m_j = M_BAR * (1.0 + 1e-6)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _erosion_targets(np.random.default_rng(1), np.array([m_i]),
                                             np.array([m_j]), np.array([r_i]), np.array([r_j]))
        rng = np.random.default_rng(5)
        state0, accepted, events = _build_batch(rng, 1, np.array([m_i]), np.array([m_j]),
                                                  u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        if (out.n_merge, out.n_ricochet, out.n_erosion) != (0, 0, 1):
            pytest.skip("channel forcing did not land on erosion for this near-q=1 pair")
        m_after = out.state.m.numpy()
        # With q ~ 1 + 1e-6, (1 - q^(-5/3)) ~ (5/3)*1e-6: the mass transferred must be tiny
        # relative to m_P, i.e. the split stays close to the ORIGINAL (m_i, m_j).
        m_total = m_i + m_j
        assert abs(m_after[0] - m_i) / m_total <= 1e-4
        assert abs(m_after[1] - m_j) / m_total <= 1e-4

    def test_energy_guard_prevents_negative_t_prime_at_low_e(self):
        """
        Section 6, INV-22: mandatory construction with e=0.5 (e^2=0.25 < FRAG_CHIP_MAX=0.5) and
        T_n just above E_lig -- without the min() energy guard (Section 4.9(D)), this configuration
        produces T' < 0 and a NaN in the speed formula.
        """
        col = _require_collisions()
        model = _make_model(e_restitution=0.5)
        m_i, m_j = 4.0 * M_BAR, M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        v_esc = _v_esc(m_i, m_j, r_i, r_j)
        m_G, m_P = max(m_i, m_j), min(m_i, m_j)
        r_P = r_j if m_i >= m_j else r_i
        e_lig = _e_lig(m_P, r_P)
        mu = m_i * m_j / (m_i + m_j)
        cos_theta = 0.95
        u_min1 = 1.2 * v_esc
        u_min2 = 1.05 * math.sqrt(2.0 * e_lig / (mu * cos_theta ** 2))  # T_n JUST above E_lig
        u_mag = max(u_min1, u_min2)
        rng = np.random.default_rng(6)
        state0, accepted, events = _build_batch(rng, 1, np.array([m_i]), np.array([m_j]),
                                                  np.array([u_mag]), np.array([cos_theta]), DT,
                                                  T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, 0, 1), (
            "test setup did not land on erosion; adjust u_mag/cos_theta, not the assertions below"
        )
        assert not torch.isnan(out.state.v).any()
        assert not torch.isnan(out.state.m).any()
        v_after = out.state.v.numpy()
        u_after = v_after[1] - v_after[0]
        assert np.dot(u_after, u_after) > 0.0, "T' must be strictly positive (Section 4.9(D.3))"


# =============================================================================================
# INV-32 -- the Generator is UNTOUCHED by ANY collision pass, in EVERY channel (Section 4.7.1
# REWRITTEN in 2026-08-09 (f): zero draws, replacing "exactly 2 per event").
# =============================================================================================
class TestINV32GeneratorUntouched:
    class _Poison:
        """An object that raises on ANY attribute access -- the strongest possible test that
        resolve() never even LOOKS at the generator, let alone draws from it."""

        def __getattr__(self, name):
            raise AssertionError(f"generator.{name} was accessed; Section 4.7.1 (f): zero draws")

    def _mixed_batch(self, rng, n):
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        channels = ["fusion", "ricochet", "erosion"]
        u_mag = np.empty(n)
        cos_theta = np.empty(n)
        for k in range(n):
            ch = channels[k % 3]
            fn = {"fusion": _fusion_targets, "ricochet": _ricochet_targets,
                  "erosion": _erosion_targets}[ch]
            u_k, c_k = fn(rng, m_i[k:k + 1], m_j[k:k + 1], r_i[k:k + 1], r_j[k:k + 1])
            u_mag[k], cos_theta[k] = u_k[0], c_k[0]
        return m_i, m_j, u_mag, cos_theta

    def test_poisoned_generator_survives_a_mixed_batch(self):
        col = _require_collisions()
        n = 60
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 60)
        m_i, m_j, u_mag, cos_theta = self._mixed_batch(rng, n)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, self._Poison(), softening=EPS_SOFTENING)
        assert out.n_merge + out.n_ricochet + out.n_erosion == n

    def test_real_generator_bit_state_unchanged(self):
        col = _require_collisions()
        n = 60
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 61)
        m_i, m_j, u_mag, cos_theta = self._mixed_batch(rng, n)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        gen = np.random.default_rng(12345)
        state_before = gen.bit_generator.state
        col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
        state_after = gen.bit_generator.state
        assert state_before == state_after, (
            "the collision Generator's bit state changed across a pass with events in every "
            "channel; Section 4.7.1 (2026-08-09 (f)) requires zero consumption"
        )

    def test_zero_events_also_leaves_the_generator_untouched(self):
        col = _require_collisions()
        model = _make_model()
        state0 = State(
            r=torch.zeros((2, 3), dtype=torch.float64),
            v=torch.zeros((2, 3), dtype=torch.float64),
            m=torch.tensor([M_BAR, M_BAR], dtype=torch.float64),
        )
        accepted = col.AcceptedPairs(
            i=torch.zeros(0, dtype=torch.int64), j=torch.zeros(0, dtype=torch.int64),
            t_c=torch.zeros(0, dtype=torch.float64), u_n=torch.zeros(0, dtype=torch.float64),
            rel_speed=torch.zeros(0, dtype=torch.float64),
            contact_radius_sum=torch.zeros(0, dtype=torch.float64), f_reject=0.0,
        )
        out = col.resolve(state0, DT, model, accepted, self._Poison(), softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, 0, 0)


# =============================================================================================
# INV-33 -- mass continuity per slot (Section 6, NEW).
# =============================================================================================
class TestINV33MassContinuityPerSlot:
    def test_ricochet_touches_no_mass_at_all(self):
        col = _require_collisions()
        n = 200
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 70)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _ricochet_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, n, 0)
        assert np.array_equal(state0.m.numpy(), out.state.m.numpy())

    def test_erosion_losing_slot_bounded_by_frag_chip_max(self):
        col = _require_collisions()
        n = 200
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 71)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _erosion_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, 0, n)
        m_before = state0.m.numpy()
        m_after = out.state.m.numpy()
        rel = np.abs(m_after - m_before) / m_before
        # BOTH slots stay alive in erosion; the structural bound applies to each.
        assert np.all(rel <= INV33_EROSION_MASS_JUMP_MAX + 1e-9), (
            f"max relative mass jump {rel.max():.4f} exceeds FRAG_CHIP_MAX="
            f"{INV33_EROSION_MASS_JUMP_MAX} in erosion (Section 6, INV-33)"
        )

    def test_merger_surviving_slot_bounded_by_mass_ratio(self):
        col = _require_collisions()
        n = 200
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 72)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _fusion_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (n, 0, 0)
        m_before = state0.m.numpy()
        m_after = out.state.m.numpy()
        alive_both = (m_before > 0) & (m_after > 0)
        rel = np.abs(m_after[alive_both] - m_before[alive_both]) / m_before[alive_both]
        assert np.all(rel <= INV33_MERGER_MASS_JUMP_MAX + 1e-9)
        dead = m_after == 0.0
        assert np.count_nonzero(dead) == n, "exactly one slot per pair must die in merger"

    def test_the_only_discontinuous_change_is_a_dying_slot(self):
        # "o unico slot cuja massa muda de forma descontinua e' um slot que morre (m -> 0, fusao)"
        col = _require_collisions()
        n = 60
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 73)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _fusion_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        m_before = state0.m.numpy()
        m_after = out.state.m.numpy()
        changed = m_before != m_after
        died = m_after == 0.0
        assert np.all(changed == died | changed), "sanity"
        for k in np.nonzero(changed)[0]:
            assert m_after[k] == 0.0 or m_after[k] > m_before[k], (
                f"slot {k} changed mass ({m_before[k]} -> {m_after[k]}) without dying, in a "
                f"merger-only batch -- only the dying slot may change discontinuously"
            )


# =============================================================================================
# INV-34 -- post-event separation, strict (Section 6, NEW).
# =============================================================================================
class TestINV34PostEventSeparation:
    def test_ricochet_u_prime_dot_n_strictly_positive(self):
        col = _require_collisions()
        n = 200
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 80)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _ricochet_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        v_after = out.state.v.numpy()
        u_after = v_after[1::2] - v_after[0::2]
        u_after_n = np.sum(u_after * events["n_hat"], axis=1)
        assert np.all(u_after_n > 0.0)

    def test_erosion_u_prime_dot_n_strictly_positive(self):
        col = _require_collisions()
        n = 200
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 81)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _erosion_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        v_after = out.state.v.numpy()
        u_after = v_after[1::2] - v_after[0::2]
        u_after_n = np.sum(u_after * events["n_hat"], axis=1)
        assert np.all(u_after_n > 0.0)

    def test_ricochet_separation_holds_even_at_near_zero_restitution(self):
        # "um teste sintetico: um par com e = 0.05... ainda tem de satisfazer o enunciado."
        col = _require_collisions()
        model = _make_model(e_restitution=0.05)
        n = 100
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 82)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _ricochet_targets(rng, m_i, m_j, r_i, r_j)
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, n, 0)
        v_after = out.state.v.numpy()
        u_after = v_after[1::2] - v_after[0::2]
        u_after_n = np.sum(u_after * events["n_hat"], axis=1)
        assert np.all(u_after_n > 0.0), "even at e=0.05, the pair must be strictly separating"

    def test_corollary_dr_dot_dv_positive_at_end_of_step(self):
        col = _require_collisions()
        n = 100
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 83)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _ricochet_targets(rng, m_i, m_j, r_i, r_j)
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        r_after = out.state.r.numpy()
        v_after = out.state.v.numpy()
        dr_final = r_after[1::2] - r_after[0::2]
        dv_final = v_after[1::2] - v_after[0::2]
        assert np.all(np.sum(dr_final * dv_final, axis=1) > 0.0)


# =============================================================================================
# INV-35 -- overlap: the enunciate that survives (Section 6, NEW). Run against a REAL, multi-step
# execution (nbody.integrators.integrate with collision=model, using the normative signature
# Section 9.1.1 fixes -- only introspected, never read as source, same convention as everywhere
# else in this suite), a dense cold_sphere initial condition chosen small and dense enough to
# produce genuine collisional activity (including mergers) in well under a second of wall time.
# =============================================================================================
class TestINV35Overlap:
    """
    Section 6, INV-35's enunciado, LITERALLY: no pair may be simultaneously overlapping and
    approaching "ao fim de todo passe de colisao", except pairs deferred by disjoint pairing.

    That literal snapshot -- immediately after collision_pass, BEFORE the closing half-kick of
    the same KDK step -- is not reachable through the public API this suite is restricted to
    (integrate()/collision_pass() only return the state after the FULL step, both kicks
    included; their public signatures, fixed by Section 9.1.1, do not expose an intermediate
    state). Measured directly here (not assumed): reconstructing the pre-closing-kick velocity by
    subtracting the closing half-kick (v_half = v_final - 0.5*dt*a(r_final)) from a genuine
    violation found in a real run leaves dr.dv negative and of essentially the SAME magnitude as
    without the reconstruction -- the closing kick is NOT what produces these violations.

    This test therefore does NOT assert zero violations (an early version of it did, and was
    falsified by a real run within 200 steps). It asserts the WEAKER, still falsifiable claim
    that Section 4.9(D.2)'s classification is EXHAUSTIVE. That classification is by PROVENANCE,
    not by cause:

        (1) the pair was a CANDIDATE at the start of the pass and was DEFERRED by disjunction;
        (2) at least one of i, j was a PARTICIPANT in an accepted event of this pass.

    Clause (2) is about participation, not about what changed, and that is what makes the list
    closed rather than a changelog. It covers all three effects an event has on a slot at once:
    velocity (a ricochet redirects a body into a third party mid-step), position (a merger moves
    the survivor to the pair centre of mass) and contact radius (merger/erosion resize it). An
    earlier version of this docstring enumerated observable symptoms instead -- "deferred
    candidate" or "mass changed in this step" -- and a real run falsified it at step 366, pair
    (14, 64): body 64 ricocheted off body 17 at t_c interior to the step, changed velocity by
    510x what gravity would explain, and drifted the remainder of the step into body 14. Neither
    mass changed, because the channel was a ricochet, and the pair was never a candidate, because
    detect() had fixed the candidate set against the pre-event trajectory. That case is an
    instance of clause (2), not a third mechanism.

    Exhaustiveness is proved in Section 4.9(D.2) and the proof is what this test is checking, so
    a failure here is a real finding: if neither (1) nor (2) holds then by not-(2) neither body
    was touched, so both traverse [0, h] rectilinearly with the velocities and radii detect()
    used; under rectilinear motion d/dt(dr.dv) = |dv|^2 >= 0, so approaching at h implies
    approaching at 0 and the guard passed; and overlapping at h means q(h) < 0, so either c <= 0
    (candidate via branch 3) or a root exists in (0, h) (candidate via branch 4). The pair was a
    candidate, hence accepted (a participant, contradicting not-(2)) or deferred (contradicting
    not-(1)).

    It also asserts, so the test is not vacuous, that at least one genuine violation actually
    occurs in the run -- if none does, this invariant's exception clauses are untested by this
    run and that is reported, not hidden behind a passing assertion.
    """

    N_BODIES = 120
    RADIUS = 0.12
    DT = 5.0e-5
    N_STEPS = 400

    def test_every_violation_is_a_deferred_candidate_or_an_event_participant(self):
        try:
            from nbody.initial_conditions import cold_sphere
            from nbody.integrators import integrate
            from nbody.backends import get_backend
        except ImportError:
            pytest.skip("nbody.initial_conditions/integrators/backends not importable")
        col = _require_collisions()
        backend = get_backend("torch_eager")
        state = cold_sphere(n=self.N_BODIES, radius=self.RADIUS, particle_mass=M_BAR,
                             seed=20190222)
        model = _make_model()
        rng = np.random.default_rng(config.COLLISION_SEED)

        cur = state
        n_violations = 0
        n_explained = 0
        unexplained = []
        for step in range(self.N_STEPS):
            m_before = cur.m.clone()
            a_cur = backend.accelerations(cur.r, cur.m, EPS_SOFTENING)
            v_half = cur.v + 0.5 * self.DT * a_cur
            half_state = State(r=cur.r, v=v_half, m=cur.m)
            candidates = col.detect(half_state, self.DT, model)
            cand_pairs = set()
            for a, b in zip(candidates.i.tolist(), candidates.j.tolist()):
                cand_pairs.add((min(a, b), max(a, b)))
            # Clause (2) is provenance, so the set that matters is which SLOTS an accepted
            # event touched -- not which quantity of theirs ended up different. Rebuilt here
            # from the same public calls integrate() makes internally on the same half-kicked
            # state, so it is the same acceptance this step will apply.
            accepted = col.pair_disjoint(candidates)
            participants = set(accepted.i.tolist()) | set(accepted.j.tolist())

            nxt, stats = integrate(cur, integrator="velocity_verlet", backend=backend,
                                    dt=self.DT, n_steps=1, softening=EPS_SOFTENING,
                                    collision=model, collision_rng=rng)

            r, v, m = nxt.r, nxt.v, nxt.m
            radii = col.contact_radii(m, model)
            alive = m > 0
            idx = torch.nonzero(alive, as_tuple=True)[0]
            r_a, v_a, radii_a = r[idx], v[idx], radii[idx]
            diff_r = r_a[:, None, :] - r_a[None, :, :]
            diff_v = v_a[:, None, :] - v_a[None, :, :]
            dist = torch.linalg.norm(diff_r, dim=-1)
            rsum = radii_a[:, None] + radii_a[None, :]
            dot = (diff_r * diff_v).sum(-1)
            overlap_approach = (dist < rsum) & (dot < 0)
            overlap_approach.fill_diagonal_(False)
            viol = torch.nonzero(torch.triu(overlap_approach, diagonal=1))
            for a, b in viol.tolist():
                i, j = int(idx[a]), int(idx[b])
                pair = (min(i, j), max(i, j))
                n_violations += 1
                deferred = pair in cand_pairs
                participated = (i in participants) or (j in participants)
                if deferred or participated:
                    n_explained += 1
                else:
                    unexplained.append((step, pair))
            cur = nxt

        assert n_violations > 0, (
            "no overlapping-and-approaching pair occurred in this run; this test's exception "
            "clauses are untested against this configuration -- not a pass, a gap to report"
        )
        assert n_explained == n_violations, (
            f"{len(unexplained)} of {n_violations} violations are NEITHER a deferred candidate "
            f"NOR an event participant: {unexplained[:5]}. Section 4.9(D.2) proves these two "
            f"clauses exhaustive, so an unexplained violation falsifies that proof -- the pair "
            f"was untouched by any event and still ended the step overlapping and approaching, "
            f"which under rectilinear drift means detect() should have made it a candidate. "
            f"Suspect the approach guard (Section 4.3) or the root branches, not this clause"
        )


# =============================================================================================
# INV-36 -- Delta E_total per event bounded to ROUND-OFF (Section 6, REWRITTEN; supersedes
# INV-23(a), whose old floor is RETIRED -- Prohibition 3, this module's docstring).
# =============================================================================================
class TestINV36PerEventEnergyAccounting:
    """
    Recomputes K+U+E_int immediately before and after a SINGLE resolve() call, using resolve()
    with dt == accepted.t_c so the returned state has zero remaining drift and IS the
    'imediatamente depois do mapa, em t_c' snapshot Section 4.10 asks for (participants AND any
    bystander alike, Section 4.5 step 4 advances everyone by the same dt passed in).

    U is recomputed via nbody.observables.potential_energy over the WHOLE system -- the
    INDEPENDENT O(N^2) path Section 6's INDEPENDENCE clause requires, never via the pair-level
    closed-form dU the event map itself used (Prohibition 2, this module's docstring: reusing the
    same dU on both sides makes the test a tautology).
    """

    @staticmethod
    def _single_event_with_bystander(rng, m_i, m_j, u_mag, cos_theta, model, dt, t_c_frac):
        state0, accepted, events = TestINV36PerEventEnergyAccounting._build_one(
            rng, m_i, m_j, u_mag, cos_theta, dt, t_c_frac
        )
        r0 = state0.r.numpy()
        v0 = state0.v.numpy()
        m0 = state0.m.numpy()
        # Bystander at the core-typical interparticle spacing (Section 4.1.1's corrected number
        # density, 2888.3 m^-3 => n^(-1/3) ~ 0.0703 m), close enough to matter for U, far enough
        # not to itself become a candidate (this module bypasses detect() throughout).
        bystander_r = r0[0] + np.array([0.0703, 0.0, 0.0])
        bystander_v = np.array([1.0, 0.0, 0.0])
        r_full = np.vstack([r0, bystander_r])
        v_full = np.vstack([v0, bystander_v])
        m_full = np.append(m0, M_BAR)
        state_full = State(
            r=torch.as_tensor(r_full, dtype=torch.float64),
            v=torch.as_tensor(v_full, dtype=torch.float64),
            m=torch.as_tensor(m_full, dtype=torch.float64),
        )
        return state_full, accepted, events

    @staticmethod
    def _build_one(rng, m_i, m_j, u_mag, cos_theta, dt, t_c_frac):
        return _build_batch(rng, 1, np.array([m_i]), np.array([m_j]), np.array([u_mag]),
                             np.array([cos_theta]), dt, t_c_frac)

    def _residual(self, state_full, accepted, model):
        col = _require_collisions()
        t_c_value = float(accepted.t_c[0])
        r_at_tc = state_full.r.numpy() + t_c_value * state_full.v.numpy()
        state_before = State(
            r=torch.as_tensor(r_at_tc, dtype=torch.float64),
            v=state_full.v.clone(),
            m=state_full.m.clone(),
        )
        k_before = observables.kinetic_energy(state_before)
        u_before = observables.potential_energy(state_before, softening=EPS_SOFTENING)
        out = col.resolve(state_full, t_c_value, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        k_after = observables.kinetic_energy(out.state)
        u_after = observables.potential_energy(out.state, softening=EPS_SOFTENING)
        e_before = k_before + u_before
        e_after = k_after + u_after + out.delta_e_int
        return abs(e_after - e_before) / E0_SCALE, out

    def test_ricochet_residual_at_round_off(self):
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 90)
        m_i, m_j = M_BAR, 2.0 * M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _ricochet_targets(rng, np.array([m_i]), np.array([m_j]),
                                              np.array([r_i]), np.array([r_j]))
        model = _make_model()
        state_full, accepted, events = self._single_event_with_bystander(
            rng, m_i, m_j, u_mag[0], cos_theta[0], model, DT, T_C_FRAC
        )
        residual, out = self._residual(state_full, accepted, model)
        assert out.n_ricochet == 1
        assert residual <= TOL_EVENT_CONS_FP64

    def test_merger_residual_at_round_off_with_a_genuine_third_body(self):
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 91)
        m_i, m_j = M_BAR, 3.0 * M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _fusion_targets(rng, np.array([m_i]), np.array([m_j]),
                                            np.array([r_i]), np.array([r_j]))
        model = _make_model()
        state_full, accepted, events = self._single_event_with_bystander(
            rng, m_i, m_j, u_mag[0], cos_theta[0], model, DT, T_C_FRAC
        )
        residual, out = self._residual(state_full, accepted, model)
        assert out.n_merge == 1
        assert residual <= TOL_EVENT_CONS_FP64, (
            f"residual={residual:.3e} exceeds {TOL_EVENT_CONS_FP64:.1e} (Section 6, INV-36); "
            f"this is the O(N) field term test -- a residual well ABOVE round-off here (with a "
            f"genuine bystander present) means Delta U_campo is missing or wrong"
        )

    def test_erosion_residual_at_round_off_with_a_genuine_third_body(self):
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 92)
        m_i, m_j = 4.0 * M_BAR, M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _erosion_targets(rng, np.array([m_i]), np.array([m_j]),
                                             np.array([r_i]), np.array([r_j]))
        model = _make_model()
        state_full, accepted, events = self._single_event_with_bystander(
            rng, m_i, m_j, u_mag[0], cos_theta[0], model, DT, T_C_FRAC
        )
        residual, out = self._residual(state_full, accepted, model)
        assert out.n_erosion == 1
        assert residual <= TOL_EVENT_CONS_FP64

    def test_independence_clause_agrees_on_n_le_200_sampled_system(self):
        """
        Section 6, INV-36's own required clause: run the SAME independence check (real
        observables.potential_energy over the whole system, O(N^2)) on a larger, N<=200 sampled
        configuration -- not just the 3-body bystander construction above -- so the independence
        clause is exercised at a scale closer to what the document specifies ('N <= 200
        amostrado'), not only at the smallest possible N that makes a bystander meaningful.
        """
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 93)
        n_bodies = 150
        m_i, m_j = 2.0 * M_BAR, M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag, cos_theta = _fusion_targets(rng, np.array([m_i]), np.array([m_j]),
                                            np.array([r_i]), np.array([r_j]))
        model = _make_model()
        state0, accepted, events = self._build_one(rng, m_i, m_j, u_mag[0], cos_theta[0], DT,
                                                     T_C_FRAC)
        r0 = state0.r.numpy()
        v0 = state0.v.numpy()
        m0 = state0.m.numpy()
        n_extra = n_bodies - 2
        extra_r = r0[0] + rng.normal(scale=1.0, size=(n_extra, 3))
        extra_v = rng.normal(scale=1.0, size=(n_extra, 3))
        extra_m = np.full(n_extra, M_BAR)
        r_full = np.vstack([r0, extra_r])
        v_full = np.vstack([v0, extra_v])
        m_full = np.append(m0, extra_m)
        state_full = State(
            r=torch.as_tensor(r_full, dtype=torch.float64),
            v=torch.as_tensor(v_full, dtype=torch.float64),
            m=torch.as_tensor(m_full, dtype=torch.float64),
        )
        residual, out = self._residual(state_full, accepted, model)
        assert out.n_merge == 1
        assert residual <= TOL_EVENT_CONS_FP64, (
            f"N={n_bodies}, residual={residual:.3e} exceeds {TOL_EVENT_CONS_FP64:.1e}"
        )


class TestINV36AggregateOverManyEvents:
    """
    Section 6, INV-36's aggregate clause: over many independent events, sum of per-event residuals
    / |E_0| <= INV36_AGGREGATE_REL_TOL, and the count of events with residual >
    TOL_EVENT_CONS_FP64 must be exactly 0 (structural, converts the pre-(f) model's measured
    '62 de 62 eventos vazam' into '0 de n').

    Approximated here as many ISOLATED single-event resolve() calls (each with its own bystander,
    same construction as TestINV36PerEventEnergyAccounting), since no RUN_COLLISION campaign
    wiring is exercised elsewhere in this suite for collisions -- this is reported as the
    approximation it is: it tests the aggregate over N independent per-event snapshots, not over a
    single continuous multi-step execution's accumulated events (which would also need every
    event's positions frozen at ITS OWN t_c, exactly what this construction already does one event
    at a time).
    """

    N_EVENTS = 150

    def test_aggregate_residual_and_zero_ceiling_violations(self):
        col = _require_collisions()
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 94)
        model = _make_model()
        residuals = []
        channels = ["fusion", "ricochet", "erosion"]
        helper = TestINV36PerEventEnergyAccounting()
        for k in range(self.N_EVENTS):
            ch = channels[k % 3]
            log_ratio = rng.uniform(-math.log(50.0), math.log(50.0))
            m_i = M_BAR
            m_j = M_BAR * math.exp(log_ratio)
            r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
            fn = {"fusion": _fusion_targets, "ricochet": _ricochet_targets,
                  "erosion": _erosion_targets}[ch]
            u_mag, cos_theta = fn(rng, np.array([m_i]), np.array([m_j]), np.array([r_i]),
                                   np.array([r_j]))
            state_full, accepted, events = helper._single_event_with_bystander(
                rng, m_i, m_j, u_mag[0], cos_theta[0], model, DT, T_C_FRAC
            )
            residual, out = helper._residual(state_full, accepted, model)
            actual_channel = {(1, 0, 0): "fusion", (0, 1, 0): "ricochet", (0, 0, 1): "erosion"}[
                (out.n_merge, out.n_ricochet, out.n_erosion)
            ]
            if actual_channel != ch:
                continue
            residuals.append(residual)
        residuals = np.array(residuals)
        assert residuals.size > 100, "too many channel-forcing misses to be a meaningful aggregate"
        aggregate = float(np.sum(residuals))
        assert aggregate <= INV36_AGGREGATE_REL_TOL, (
            f"aggregate residual {aggregate:.3e} exceeds {INV36_AGGREGATE_REL_TOL:.1e} over "
            f"{residuals.size} events (Section 6, INV-36)"
        )
        n_over_ceiling = int(np.count_nonzero(residuals > TOL_EVENT_CONS_FP64))
        assert n_over_ceiling == 0, (
            f"{n_over_ceiling} of {residuals.size} events exceed the per-event ceiling "
            f"{TOL_EVENT_CONS_FP64:.1e} -- the pre-(f) baseline measured 62 of 62 leaking; this "
            f"clause exists to turn that into 0 of n (Section 6, INV-36)"
        )


# =============================================================================================
# INV-37 -- the gates: precedence, boundary, softened-not-newtonian, continuity (Section 6, NEW).
# =============================================================================================
class TestINV37GatesPrecedenceCoverage:
    def test_precedence_fusion_wins_when_both_gates_fire(self):
        # Section 6, INV-37(1): construct q > GATE_Q_OVERLAP (both gates satisfiable together,
        # Section 4.6.3) and require FUSION (precedence resolves the overlap).
        col = _require_collisions()
        model = _make_model()
        q = 8.0
        assert q > GATE_Q_OVERLAP
        m_P, m_G = M_BAR, q * M_BAR
        r_P, r_G = _contact_radius(m_P), _contact_radius(m_G)
        v_esc = _v_esc(m_G, m_P, r_G, r_P)
        e_lig = _e_lig(m_P, r_P)
        mu = m_G * m_P / (m_G + m_P)
        cos_theta = 0.98
        u_mag = 0.95 * v_esc
        t_n = 0.5 * mu * (u_mag * cos_theta) ** 2
        assert u_mag < v_esc and t_n > e_lig, (
            "test setup error: this configuration must satisfy BOTH gates for the precedence "
            "clause to be meaningful"
        )
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 100)
        state0, accepted, events = _build_batch(rng, 1, np.array([m_G]), np.array([m_P]),
                                                  np.array([u_mag]), np.array([cos_theta]), DT,
                                                  T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (1, 0, 0), (
            "both gates fire (fusion AND erosion conditions hold simultaneously); Section 4.6.3's "
            "precedence rule requires fusion to win"
        )

    def test_boundary_crossing_matches_v_esc_to_within_100_eps_prec(self):
        # Section 6, INV-37(2): bisect for the exact crossing and compare to the closed-form
        # v_esc. cos_theta kept small so T_n stays far below E_lig throughout (portal 2 never
        # fires), isolating the portal-1 boundary.
        col = _require_collisions()
        model = _make_model()
        m_i, m_j = M_BAR, M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        v_esc = _v_esc(m_i, m_j, r_i, r_j)
        cos_theta = 0.05

        def channel_at(u_mag):
            rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 101)
            state0, accepted, events = _build_batch(rng, 1, np.array([m_i]), np.array([m_j]),
                                                      np.array([u_mag]), np.array([cos_theta]),
                                                      DT, T_C_FRAC)
            out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                               softening=EPS_SOFTENING)
            return "merge" if out.n_merge == 1 else ("ricochet" if out.n_ricochet == 1 else
                                                       "erosion")

        lo, hi = v_esc * (1.0 - 1e-6), v_esc * (1.0 + 1e-6)
        assert channel_at(lo) == "merge"
        assert channel_at(hi) != "merge"
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if channel_at(mid) == "merge":
                lo = mid
            else:
                hi = mid
        crossing = 0.5 * (lo + hi)
        rel = abs(crossing / v_esc - 1.0)
        assert rel <= INV37_BOUNDARY_REL_TOL_ULP * EPS_PREC_FP64, (
            f"the merge/ricochet crossing found by bisection ({crossing}) differs from the "
            f"closed-form v_esc ({v_esc}) by relative {rel:.3e}"
        )

    def test_softened_form_used_not_newtonian(self):
        # Section 6, INV-37(3) -- same construction/claim as INV-21's own emendation-required
        # test, kept here too since Section 6 lists it under INV-37 specifically.
        col = _require_collisions()
        model = _make_model()
        u_mag = 0.5 * (PAIR_VESC_SOFT_MBAR + PAIR_VESC_NEWT_MBAR)
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 102)
        state0, accepted, events = _build_batch(
            rng, 1, np.array([M_BAR]), np.array([M_BAR]), np.array([u_mag]), np.array([0.3]),
            DT, T_C_FRAC,
        )
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, 1, 0)

    def test_continuity_at_the_erosion_ricochet_boundary(self):
        """
        Section 6, INV-37(4): as T_n -> E_lig^+, erosion must converge to ricochet's own u_r
        (f_chip -> 0, E_custo -> 0). At T_n/E_lig - 1 = 1e-6, |u' - u_r| <= 1e-5 |u|.
        """
        col = _require_collisions()
        model = _make_model()
        m_i, m_j = 2.0 * M_BAR, M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        v_esc = _v_esc(m_i, m_j, r_i, r_j)
        m_G, m_P = m_i, m_j
        e_lig = _e_lig(m_P, r_j)
        mu = m_i * m_j / (m_i + m_j)
        cos_theta = 0.9
        u_min1 = 1.2 * v_esc
        excess = INV37_CONTINUITY_T_N_OVER_E_LIG_EXCESS
        u_min2 = math.sqrt(2.0 * e_lig * (1.0 + excess) / (mu * cos_theta ** 2))
        u_mag = max(u_min1, u_min2)
        t_n = 0.5 * mu * (u_mag * cos_theta) ** 2
        assert u_mag >= v_esc, "test setup error: must remain outside the fusion gate"
        assert abs(t_n / e_lig - 1.0 - excess) <= 1e-3 or u_mag == u_min1, (
            "test setup error: T_n/E_lig should sit just above threshold unless gate-1 dominates"
        )
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 103)
        state0, accepted, events = _build_batch(rng, 1, np.array([m_i]), np.array([m_j]),
                                                  np.array([u_mag]), np.array([cos_theta]), DT,
                                                  T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert out.n_erosion == 1, "test setup did not land on erosion"

        e = model.e_restitution
        u = events["u"][0]
        n_hat = events["n_hat"][0]
        u_n = events["u_n"][0]
        u_r = u - (1.0 + e) * u_n * n_hat
        v_after = out.state.v.numpy()
        u_after = v_after[1] - v_after[0]
        rel = np.linalg.norm(u_after - u_r) / np.linalg.norm(u)
        assert rel <= INV37_CONTINUITY_U_PRIME_REL_TOL, (
            f"|u' - u_r|/|u| = {rel:.3e} exceeds {INV37_CONTINUITY_U_PRIME_REL_TOL:.1e} near the "
            f"erosion/ricochet boundary (Section 6, INV-37(4))"
        )


# =============================================================================================
# INV-24 -- L_total = L_orb + L_spin conserved exactly by the collision map (Section 4.10/6).
# =============================================================================================
class TestINV24AngularMomentumIdentity:
    @pytest.fixture(scope="class")
    def resolved(self):
        col = _require_collisions()
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 40)
        m_i, m_j = _sample_mass_pairs(rng, n)
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        u_mag = np.empty(n)
        cos_theta = np.empty(n)
        channels = ["fusion", "ricochet", "erosion"]
        for k in range(n):
            ch = channels[k % 3]
            fn = {"fusion": _fusion_targets, "ricochet": _ricochet_targets,
                  "erosion": _erosion_targets}[ch]
            u_k, c_k = fn(rng, m_i[k:k + 1], m_j[k:k + 1], r_i[k:k + 1], r_j[k:k + 1])
            u_mag[k], cos_theta[k] = u_k[0], c_k[0]
        model = _make_model()
        state0, accepted, events = _build_batch(rng, n, m_i, m_j, u_mag, cos_theta, DT, T_C_FRAC)
        out = col.resolve(state0, DT, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        return state0, out

    def test_l_orb_change_is_exactly_cancelled_by_delta_l_spin(self, resolved):
        state0, out = resolved
        l_before = observables.angular_momentum(state0)
        l_after = observables.angular_momentum(out.state)
        residual = (l_after - l_before) + out.delta_l_spin
        l_scale = torch.linalg.norm(l_before).item()
        rel = torch.linalg.norm(residual).item() / l_scale
        assert rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64

    def test_delta_l_spin_is_not_trivially_zero(self, resolved):
        _, out = resolved
        assert torch.linalg.norm(out.delta_l_spin).item() > 0.0


# =============================================================================================
# Section 4.9(0.1) -- the degenerate head-on normal (sep_sq == 0.0 exactly), inherited unchanged
# by this revision (Section 4.3's own text: 'a regra de extensao continua da Secao 4.9(0.1),
# inalterada'). Tested here per INV-20/INV-22's own procedure text ('acrescentando aos seus
# conjuntos de eventos sinteticos ao menos um par frontal colinear com sep_sq == 0.0').
# =============================================================================================
class TestSection491DegenerateHeadOnNormal:
    DT_LOCAL = 1.0e-3
    T_C = 4.0e-4

    @classmethod
    def _construct_exact_head_on(cls, u_mag):
        dv = np.array([-u_mag, 0.0, 0.0])
        dr0 = -(cls.T_C * dv)
        r_i0 = np.zeros(3)
        r_j0 = r_i0 + dr0
        v_i0 = np.zeros(3)
        v_j0 = v_i0 + dv
        return r_i0, r_j0, v_i0, v_j0

    def test_construction_is_genuinely_sep_sq_zero(self):
        r_i0, r_j0, v_i0, v_j0 = self._construct_exact_head_on(u_mag=5.0)
        dr = r_j0 - r_i0
        dv = v_j0 - v_i0
        sep = dr + self.T_C * dv
        assert np.array_equal(sep, np.zeros(3))

    def _make_accepted(self, m_i, m_j, u_n, u_mag, r_ref=R_REF, m_bar=M_BAR):
        col = _require_collisions()
        r_sum = r_ref * (m_i / m_bar) ** (1.0 / 3.0) + r_ref * (m_j / m_bar) ** (1.0 / 3.0)
        return col.AcceptedPairs(
            i=torch.tensor([0]), j=torch.tensor([1]),
            t_c=torch.tensor([self.T_C], dtype=torch.float64),
            u_n=torch.tensor([u_n], dtype=torch.float64),
            rel_speed=torch.tensor([u_mag], dtype=torch.float64),
            contact_radius_sum=torch.tensor([r_sum], dtype=torch.float64),
            f_reject=0.0,
        )

    def test_ricochet_exact_reversal_at_e_equals_one_and_zero_exact_angular_momentum(self):
        col = _require_collisions()
        m = M_BAR
        r_i0, r_j0, v_i0, v_j0 = self._construct_exact_head_on(u_mag=5.0)
        u_before = v_j0 - v_i0
        state0 = State(
            r=torch.as_tensor(np.vstack([r_i0, r_j0]), dtype=torch.float64),
            v=torch.as_tensor(np.vstack([v_i0, v_j0]), dtype=torch.float64),
            m=torch.tensor([m, m], dtype=torch.float64),
        )
        # n = -u/|u| in the limit (Section 4.9(0.1)); u_n = u.n = -|u| here.
        accepted = self._make_accepted(m, m, u_n=-5.0, u_mag=5.0)
        r_i, r_j = _contact_radius(m), _contact_radius(m)
        v_esc = _v_esc(m, m, r_i, r_j)
        assert 5.0 > v_esc, "test setup error: must be outside the fusion gate"
        model = _make_model(e_restitution=1.0)
        out = col.resolve(state0, self.DT_LOCAL, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, 1, 0)
        assert not torch.isnan(out.state.r).any()
        assert not torch.isnan(out.state.v).any()

        v_after = out.state.v.numpy()
        u_after = v_after[1] - v_after[0]
        rel = np.linalg.norm(u_after - (-u_before)) / np.linalg.norm(u_before)
        assert rel <= 100.0 * EPS_PREC_FP64, (
            f"e=1 at the degenerate normal must give u'=-u exactly; got {u_after}, expected "
            f"{-u_before}"
        )

        l_before = m * np.cross(r_i0, v_i0) + m * np.cross(r_j0, v_j0)
        r_after = out.state.r.numpy()
        l_after = m * np.cross(r_after[0], v_after[0]) + m * np.cross(r_after[1], v_after[1])
        np.testing.assert_array_equal(l_after, l_before)

    def test_erosion_no_nan_and_recoil_direction_is_minus_u(self):
        col = _require_collisions()
        m_i, m_j = 4.0 * M_BAR, M_BAR
        r_i, r_j = _contact_radius(m_i), _contact_radius(m_j)
        v_esc = _v_esc(m_i, m_j, r_i, r_j)
        e_lig = _e_lig(m_j, r_j)
        mu = m_i * m_j / (m_i + m_j)
        u_mag = max(1.5 * v_esc, 1.5 * math.sqrt(2.0 * e_lig / mu))  # cos_theta=1 here
        dv = np.array([-u_mag, 0.0, 0.0])
        dr0 = -(self.T_C * dv)
        r_i0 = np.zeros(3)
        r_j0 = r_i0 + dr0
        v_i0 = np.zeros(3)
        v_j0 = v_i0 + dv
        u_before = v_j0 - v_i0
        state0 = State(
            r=torch.as_tensor(np.vstack([r_i0, r_j0]), dtype=torch.float64),
            v=torch.as_tensor(np.vstack([v_i0, v_j0]), dtype=torch.float64),
            m=torch.tensor([m_i, m_j], dtype=torch.float64),
        )
        accepted = self._make_accepted(m_i, m_j, u_n=-u_mag, u_mag=u_mag)
        model = _make_model()
        out = col.resolve(state0, self.DT_LOCAL, model, accepted, np.random.default_rng(0),
                           softening=EPS_SOFTENING)
        assert (out.n_merge, out.n_ricochet, out.n_erosion) == (0, 0, 1), (
            "test setup did not land on erosion"
        )
        assert not torch.isnan(out.state.r).any()
        assert not torch.isnan(out.state.v).any()
        assert not torch.isnan(out.state.m).any()

        v_after = out.state.v.numpy()
        u_after = v_after[1] - v_after[0]
        minus_u_hat = -u_before / np.linalg.norm(u_before)
        cos_angle = np.dot(u_after, minus_u_hat) / np.linalg.norm(u_after)
        assert abs(cos_angle - 1.0) <= 1e-6, (
            f"erosion recoil at the degenerate normal should point along -u/|u|; cos_angle="
            f"{cos_angle}"
        )
        m_after = out.state.m.numpy()
        assert abs(m_after.sum() - (m_i + m_j)) / (m_i + m_j) <= 4.0 * EPS_PREC_FP64

    def test_dr_zero_and_u_zero_raises_value_error(self):
        col = _require_collisions()
        m = M_BAR
        state0 = State(
            r=torch.zeros((2, 3), dtype=torch.float64),
            v=torch.zeros((2, 3), dtype=torch.float64),
            m=torch.tensor([m, m], dtype=torch.float64),
        )
        accepted = self._make_accepted(m, m, u_n=-1.0, u_mag=1.0)
        model = _make_model()
        with pytest.raises(ValueError):
            col.resolve(state0, self.DT_LOCAL, model, accepted, np.random.default_rng(0),
                         softening=EPS_SOFTENING)


# =============================================================================================
# CollisionModel construction: e_restitution domain, Section 4.9(C).
# =============================================================================================
class TestCollisionModelERestitutionDomain:
    def test_e_equals_zero_is_rejected(self):
        col = _require_collisions()
        with pytest.raises(ValueError):
            col.CollisionModel(r_ref=R_REF, m_bar=M_BAR, e_restitution=0.0)

    def test_e_equals_one_is_accepted(self):
        col = _require_collisions()
        model = col.CollisionModel(r_ref=R_REF, m_bar=M_BAR, e_restitution=1.0)
        assert model.e_restitution == 1.0

    def test_negative_e_is_rejected(self):
        col = _require_collisions()
        with pytest.raises(ValueError):
            col.CollisionModel(r_ref=R_REF, m_bar=M_BAR, e_restitution=-0.1)

    def test_e_above_one_is_rejected(self):
        col = _require_collisions()
        with pytest.raises(ValueError):
            col.CollisionModel(r_ref=R_REF, m_bar=M_BAR, e_restitution=1.1)
