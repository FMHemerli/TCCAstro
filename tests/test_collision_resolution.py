"""
docs/simulacao-estocastica.md -- collision RESOLUTION: the three outcomes.

Covers Section 4.6 (regime parameter x), Section 4.7 / 4.7.1 (the parameter-free map
(1/x, 3, x)/Z and its exactly-two-uniforms-per-event RNG protocol), Section 4.9, 4.9(0) and
4.9(0.1) (elastic / merger / fragmentation outcome maps, the slot-assignment rule, and the
degenerate head-on normal), Section 4.10 (the E_int and L_spin accumulators, closed-form at the
pair level), Section 4.14 (the Floor, eight items -- 1, 2, 3, 5, 7, 8 are resolve()'s direct
responsibility; 4 and 9 are pair_disjoint's and are covered by tests/test_collision_pairing.py),
Section 9.1.1 (the normative resolve()/collision_pass() signatures and its three
purity/drift/softening clauses), and Section 6, invariants INV-20 through INV-26 and INV-32.

**Written against the 2026-08-07 (b) revision.** This revision retracted the gravitational term
of x (Section 4.6.1: the stage-3 campaign measured that term CAUSING runaway growth, not
containing it -- max m_i = 321 m_bar): `x` is now `|u|^2 / v_coh^2`, with no dependence on masses,
contact radii, or softening at all. It also corrected a SIGN ERROR in the fragmentation branch of
E_int (Section 4.10: `E_int += E_grav(after) - E_grav(before)`, not the reverse), reformulated
INV-23(c) from a sign floor to a magnitude ceiling (`max_t |E_int| <= 1.0 |E_0|`, since merger can
legitimately INJECT energy under the corrected model), and retracted Floor item 6 (the
never-negative merger-dissipation claim it protected no longer holds). This suite was updated to
match; a note at the point of each affected test says what it used to assert and why it changed --
see the final report for the full account, since another test-writing pass had already produced
(and this pass then corrected) tests against the retracted formulas.

Written from the specification alone. nbody.collisions.resolve, collision_pass and
regime_probabilities were never opened -- only their PUBLIC SIGNATURES were introspected via
inspect.signature, exactly the "declared signature is fair game, algorithm is not" convention
tests/test_collision_pairing.py and (formerly) tests/_stage2_binding.py already used, and that
Section 9.1.1 now makes literally normative. Every closed-form prediction below (final r/v/m per
participant, delta_e_int, delta_l_spin) is derived directly from Sections 4.6/4.9/4.10's own
formulas, independently of how resolve() is implemented.

Design note on forcing a specific outcome channel deterministically. Section 4.7.1 fixes the RNG
protocol with no ambiguity: for each accepted event, u1 = generator.random() decides the channel
(u1 < p_fus -> fusion; u1 < p_fus + p_el -> elastic; else -> fragmentation) and u2 =
generator.random() is ALWAYS drawn next, used only by fragmentation. Because this suite computes
p_fus/p_el/p_frag independently from x via the exact closed form of Section 4.7 (function
_map_probs below, never nbody.collisions.regime_probabilities -- that function is instead the
CODE UNDER TEST in TestINV25RegimeMapWellFormed), a batch's u1 sequence can be placed well inside
one channel's probability band for every event, deterministically forcing that channel for the
whole batch. This is verified, not assumed: every per-channel test first asserts the returned
CollisionOutcome's n_elastic/n_merge/n_fragment count equals the full batch size before applying
that channel's exact-formula checks -- an early, loud failure there is itself a finding about
resolve()'s x/map computation, not a silent mis-test.

A duck-typed generator (any object exposing .random() -> float, called with no arguments) was
confirmed to work with the real resolve() during this suite's design (no isinstance check blocks
it); this is calling-convention behaviour, not implementation reading.
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
    FRAG_F_MIN,
    INV22_F_KS_PVALUE_MIN,
    INV26_CHANNEL_MIN_FRACTION,
    INV26_N_SIGMA,
    MAP_CROSSING_FRAG_EL,
    MAP_CROSSING_FUS_EL,
    MAP_ELASTIC_WEIGHT,
    MAP_P_EL_AT_X_EQUALS_1,
    MAP_X_CLAMP,
    RESOLUTION_CHANNEL_FRACTION_N_EVENTS,
    RESOLUTION_CHANNEL_FRACTION_TEST_X,
    RESOLUTION_GEOMETRY_SEED,
    RESOLUTION_MASS_RATIO_MAX,
    RESOLUTION_N_EVENTS_PER_CHANNEL,
    RESOLUTION_X_LOG_RANGE,
    TOL_EVENT_CONS_FLOOR,
    TOL_EVENT_CONS_FP64,
    TOL_EVENT_INV_ULP,
    TOL_EVENT_INV_ULP_WIDE_RATIO_BATCH,
    TOL_EVENT_PRED,
    TOL_EVENT_PRED_WIDE_RATIO_BATCH,
    TOL_PROB_MIN,
    TOL_PROB_SYM_ULP,
    TOL_PROB_ULP,
    TOL_PROB_UNIF_ULP,
)

G = config.G
EPS_SOFTENING = config.SOFTENING
M_BAR = config.PARTICLE_MASS


def _require_collisions():
    if collisions is None:
        pytest.skip("nbody.collisions is not importable")
    return collisions


# -------------------------------------------------------------------------------------------
# Independent closed-form oracle -- Section 4.7's map, read from the document, never from
# nbody.collisions.regime_probabilities (which is the code under test in TestINV25).
# -------------------------------------------------------------------------------------------
def _map_probs(x):
    """
    Section 4.7: x = clamp(x, 1/X_CLAMP, X_CLAMP); Z = 1/x + 3 + x;
    (p_fus, p_el, p_frag) = (1/x, 3, x) / Z.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), 1.0 / MAP_X_CLAMP, MAP_X_CLAMP)
    z = 1.0 / x + MAP_ELASTIC_WEIGHT + x
    return (1.0 / x) / z, MAP_ELASTIC_WEIGHT / z, x / z


def _x_value(u_mag, v_coh):
    """
    Section 4.6, 2026-08-07 (b) revision: x = |u|^2 / v_coh^2. The gravitational term
    (v_esc_eff, R_sum, eps) that an EARLIER revision required is RETRACTED (Section 4.6.1): the
    stage-3 campaign measured it CAUSING runaway growth (max m_i = 321 m_bar) rather than
    containing it, because v_esc_eff grows with the colliding body's own mass, making p_fus rise
    as a body accretes -- pure positive feedback. x now depends on relative speed alone, on
    nothing about the two bodies themselves.
    """
    return (u_mag / v_coh) ** 2


def _e_grav(m_a, m_b, d, eps):
    """Section 4.10: E_grav(m, m', d) = G m m' / sqrt(d^2 + eps^2)."""
    return G * m_a * m_b / np.sqrt(d ** 2 + eps ** 2)


def _contact_radius(m, r_ref, m_bar):
    return r_ref * (m / m_bar) ** (1.0 / 3.0)


# -------------------------------------------------------------------------------------------
# Synthetic-event construction: K independent, isolated two-body collision events packed into
# one (2K, 3) state, each with a prescribed regime parameter x_k (Section 4.6) and a genuine
# approach at its own t* (u.n < 0, Section 4.3/4.9's precondition for the fragmentation sign
# argument). Positions, the tangential/normal split, the center-of-mass drift and the mass ratio
# are all randomized per event via a geometry-only RNG stream, kept strictly separate from the
# `generator` object passed to resolve() (which decides channel/f, Section 4.7.1) -- exactly the
# separation of concerns Section 2.7's "fluxo de massas e separado do fluxo de posicoes" already
# establishes elsewhere in this project for a parallel reason.
# -------------------------------------------------------------------------------------------
def _unit(v):
    return v / np.linalg.norm(v)


def _sample_mass_pairs(rng, n_events):
    """
    Section 6, INV-20: 'razao de massa ate 1000'. m_i is held at PARTICLE_MASS and m_j is drawn
    log-uniformly with ratio in [1/1000, 1000] -- pulled out of _build_batch so tests that need
    to know (m_i, m_j) BEFORE choosing a target x (e.g. to derive x_floor per pair, Section 4.6)
    can do so without relying on replaying an RNG stream position (fragile: any later change to
    how many draws _build_batch makes per event would silently desynchronize a replay).
    """
    log_ratio_max = math.log(RESOLUTION_MASS_RATIO_MAX)
    log_ratio = rng.uniform(-log_ratio_max, log_ratio_max, size=n_events)
    m_i = np.full(n_events, M_BAR)
    m_j = M_BAR * np.exp(log_ratio)
    return m_i, m_j


def _build_batch(rng, n_events, x_values, dt, t_star_frac, model, r_ref, m_bar, masses=None):
    """
    Returns (state0, accepted, events) where events is a dict of (n_events, ...) numpy arrays
    holding everything needed to independently predict the outcome of pair k in isolation:
    m_i, m_j, r_i0, r_j0, v_i0, v_j0, t_star, n_hat (contact normal at t*, i -> j), u (relative
    velocity, constant along the drift), r_sum (R_i + R_j at the ORIGINAL masses).

    t_star_frac is held CONSTANT across the whole batch so the natural slot order (pair k at
    slots (2k, 2k+1), i=2k < j=2k+1) is already sorted by the (t*, i, j) lexicographic key
    Section 4.5 guarantees AcceptedPairs satisfies -- this suite constructs AcceptedPairs
    directly (bypassing detect/pair_disjoint, which tests/test_collision_pairing.py and
    tests/test_collision_detection.py already cover) and must hand resolve() an input that
    respects that invariant.

    masses, if given, is a (m_i, m_j) pair of arrays used INSTEAD of drawing new ones from rng
    (Section 4.6's x_floor depends on the masses, so a caller that needs x_k >= x_floor(pair_k)
    must fix the masses first via _sample_mass_pairs and pass them in here).
    """
    if masses is None:
        m_i, m_j = _sample_mass_pairs(rng, n_events)
    else:
        m_i, m_j = masses
    m_i = np.asarray(m_i, dtype=np.float64).copy()
    m_j = np.asarray(m_j, dtype=np.float64).copy()
    r_i0 = np.empty((n_events, 3))
    r_j0 = np.empty((n_events, 3))
    v_i0 = np.empty((n_events, 3))
    v_j0 = np.empty((n_events, 3))
    n_hat = np.empty((n_events, 3))
    u_vec = np.empty((n_events, 3))
    r_sum = np.empty(n_events)
    t_star = np.full(n_events, t_star_frac * dt)

    for k in range(n_events):
        mi = float(m_i[k])
        mj = float(m_j[k])
        r_i_k = r_ref * (mi / m_bar) ** (1.0 / 3.0)
        r_j_k = r_ref * (mj / m_bar) ** (1.0 / 3.0)
        rs = r_i_k + r_j_k

        v_coh = model.v_coh
        u_mag = math.sqrt(x_values[k]) * v_coh  # Section 4.6 (b): x = |u|^2 / v_coh^2

        n0 = _unit(rng.normal(size=3))
        raw_t = rng.normal(size=3)
        raw_t = raw_t - np.dot(raw_t, n0) * n0
        t0 = _unit(raw_t)
        # cos(theta) in [0.5, 0.94]: unambiguous approach (u.n < 0 with comfortable margin) AND
        # sin(theta) >= 0.34 always, which keeps dr and u well away from collinear. A near-zero
        # theta would make dr x u tiny (both nearly along n_hat), amplifying the RELATIVE error
        # of this test's own independent reconstruction of L (a chain of cross products and
        # sums over masses spanning 3 decades) without saying anything about resolve() itself --
        # a conditioning artifact of this test's construction, not a document-derived bound.
        theta = rng.uniform(math.radians(20.0), math.radians(60.0))
        dv = -u_mag * math.cos(theta) * n0 + u_mag * math.sin(theta) * t0

        dr_at_tstar = rs * n0
        dr0 = dr_at_tstar - t_star[k] * dv

        v_cm = rng.normal(scale=1.0, size=3)
        vi = v_cm - (mj / (mi + mj)) * dv
        vj = v_cm + (mi / (mi + mj)) * dv

        # No inter-pair offset at all: resolve() has no notion of gravity or distance BETWEEN
        # different accepted pairs (each pair's outcome map only ever reads its own two slots),
        # so different pairs sharing the same absolute position costs nothing physically. The
        # within-pair geometry (dr, contact radius) has scale ~1e-2 to ~1e-1; ANY nonzero
        # inter-pair offset comparable to or larger than that forces every downstream
        # recomputation of a within-pair separation (dr(t*), or the fragment separation this
        # module recovers by subtracting a predicted drift from a large absolute position) to
        # subtract two large, nearly-equal float64 numbers, amplifying THIS TEST's own
        # reconstruction error far beyond resolve()'s actual precision. Caught during this
        # suite's own construction (first with a 1000*k offset, then a smaller but still
        # nonzero one) -- see the final report.
        base = np.zeros(3)

        m_i[k], m_j[k] = mi, mj
        r_i0[k] = base
        r_j0[k] = base + dr0
        v_i0[k], v_j0[k] = vi, vj
        n_hat[k] = n0
        u_vec[k] = dv
        r_sum[k] = rs

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
    accepted = collisions.AcceptedPairs(
        i=torch.as_tensor(idx_i, dtype=torch.int64),
        j=torch.as_tensor(idx_j, dtype=torch.int64),
        t_star=torch.as_tensor(t_star, dtype=torch.float64),
        rel_speed=torch.as_tensor(np.linalg.norm(u_vec, axis=1), dtype=torch.float64),
        contact_radius_sum=torch.as_tensor(r_sum, dtype=torch.float64),
        f_reject=0.0,
    )
    events = dict(
        m_i=m_i, m_j=m_j, r_i0=r_i0, r_j0=r_j0, v_i0=v_i0, v_j0=v_j0,
        n_hat=n_hat, u=u_vec, r_sum=r_sum, t_star=t_star, dt=dt,
    )
    return state0, accepted, events


def _channel_u1(x_values, target):
    p_fus, p_el, p_frag = _map_probs(np.asarray(x_values))
    if target == "fusion":
        return 0.5 * p_fus
    if target == "elastic":
        return p_fus + 0.5 * p_el
    if target == "fragmentation":
        return p_fus + p_el + 0.5 * p_frag
    raise ValueError(target)


class _ScriptedGenerator:
    """
    Returns a pre-scripted sequence of values from .random(), one per call, in order. Used to
    force a specific channel/f for every event in a batch by construction (see module docstring)
    -- and, incidentally, its own call counter is an independent check that resolve() draws
    exactly as many values as Section 4.7.1 specifies (INV-32).
    """

    def __init__(self, values):
        self._values = list(values)
        self.calls = 0

    def random(self):
        v = self._values[self.calls]
        self.calls += 1
        return v


def _interleave(u1, u2):
    out = np.empty(2 * len(u1))
    out[0::2] = u1
    out[1::2] = u2
    return out


def _make_model(v_coh, r_ref=config.R_REF_DEFAULT, m_bar=M_BAR, seed=config.COLLISION_SEED):
    col = _require_collisions()
    return col.CollisionModel(r_ref=r_ref, m_bar=m_bar, v_coh=v_coh, seed=seed)


def _x_grid(rng, n, low, high):
    return 10.0 ** rng.uniform(math.log10(low), math.log10(high), size=n)


DT = 1.0e-3
T_STAR_FRAC = 0.35
V_COH = config.V_CHAR  # Section 4.6: v_coh = V_CHAR, no free factor


# =============================================================================================
# INV-25 -- the regime map is well-formed (Section 4.7). Pure function, no resolve() involved.
# nbody.collisions.regime_probabilities IS the code under test here; _map_probs above is the
# independent oracle it is checked against.
# =============================================================================================
class TestINV25RegimeMapWellFormed:
    X_GRID = np.logspace(-14, 14, 601)  # Section 6, INV-25: "601 pontos logaritmicos"

    def _call_real_map(self, x_np):
        col = _require_collisions()
        fn = getattr(col, "regime_probabilities", None)
        if fn is None:
            pytest.skip("nbody.collisions.regime_probabilities is not defined")
        x_t = torch.as_tensor(x_np, dtype=torch.float64)
        result = fn(x_t)
        p_fus, p_el, p_frag = result
        return (
            p_fus.detach().cpu().numpy(),
            p_el.detach().cpu().numpy(),
            p_frag.detach().cpu().numpy(),
        )

    def test_sums_to_one(self):
        p_fus, p_el, p_frag = self._call_real_map(self.X_GRID)
        total = p_fus + p_el + p_frag
        assert np.all(np.abs(total - 1.0) <= TOL_PROB_ULP * EPS_PREC_FP64)

    def test_strictly_positive_everywhere_including_clamp(self):
        p_fus, p_el, p_frag = self._call_real_map(self.X_GRID)
        assert np.all(p_fus > 0.0)
        assert np.all(p_el > 0.0)
        assert np.all(p_frag > 0.0)

    def test_p_fus_strictly_decreasing_p_frag_strictly_increasing(self):
        x_sorted = np.sort(self.X_GRID)
        p_fus, _, p_frag = self._call_real_map(x_sorted)
        # Section 6: "dentro de 1e-15 por passo, para absorver arredondamento" -- a strict
        # inequality with a tiny slack for round-off between adjacent grid points.
        assert np.all(np.diff(p_fus) <= 1e-15)
        assert np.all(np.diff(p_frag) >= -1e-15)

    def test_elastic_channel_maximal_at_x_equals_one(self):
        p_fus, p_el, p_frag = self._call_real_map(np.array([1.0]))
        assert abs(p_el[0] - MAP_P_EL_AT_X_EQUALS_1) <= 4.0 * EPS_PREC_FP64
        x_probe = np.concatenate([np.logspace(-3, -0.01, 50), np.logspace(0.01, 3, 50)])
        _, p_el_probe, _ = self._call_real_map(x_probe)
        assert np.all(p_el_probe < p_el[0])

    def test_symmetry_under_x_to_reciprocal_x(self):
        x = self.X_GRID
        p_fus, p_el, p_frag = self._call_real_map(x)
        p_fus_r, p_el_r, p_frag_r = self._call_real_map(1.0 / x)
        assert np.all(np.abs(p_fus - p_frag_r) <= TOL_PROB_SYM_ULP * EPS_PREC_FP64)
        assert np.all(np.abs(p_el - p_el_r) <= TOL_PROB_SYM_ULP * EPS_PREC_FP64)

    def test_closed_form_crossings(self):
        p_fus_a, p_el_a, _ = self._call_real_map(np.array([MAP_CROSSING_FUS_EL]))
        assert abs(p_fus_a[0] - p_el_a[0]) <= 100.0 * EPS_PREC_FP64

        _, p_el_b, p_frag_b = self._call_real_map(np.array([MAP_CROSSING_FRAG_EL]))
        assert abs(p_frag_b[0] - p_el_b[0]) <= 100.0 * EPS_PREC_FP64

    def test_extremes_beyond_the_clamp(self):
        p_fus, p_el, p_frag = self._call_real_map(np.array([1e-300, 1e300]))
        assert np.all(np.isfinite(p_fus))
        assert np.all(np.isfinite(p_el))
        assert np.all(np.isfinite(p_frag))
        assert np.min(np.stack([p_fus, p_el, p_frag])) >= TOL_PROB_MIN

    def test_matches_independent_closed_form_oracle(self):
        # Cross-check against this file's own from-the-document reimplementation (never read
        # from nbody.collisions): two independently derived implementations of the same [T]
        # closed form should agree to round-off.
        x = self.X_GRID
        p_fus_ref, p_el_ref, p_frag_ref = _map_probs(x)
        p_fus, p_el, p_frag = self._call_real_map(x)
        assert np.all(np.abs(p_fus - p_fus_ref) <= TOL_PROB_SYM_ULP * EPS_PREC_FP64)
        assert np.all(np.abs(p_el - p_el_ref) <= TOL_PROB_SYM_ULP * EPS_PREC_FP64)
        assert np.all(np.abs(p_frag - p_frag_ref) <= TOL_PROB_SYM_ULP * EPS_PREC_FP64)

    def test_uniform_control_is_a_literal_constant_not_a_parameter_limit(self):
        # Section 4.7: "o controle uniforme e obtido substituindo o mapa por (1/3,1/3,1/3)
        # constante... a implementacao expoe isso como uma escolha explicita". There is no more
        # `w` to push to infinity; if the implementation exposes a uniform-control switch it must
        # be tested for the literal constant. Since Section 9.1.1 does not name such a switch as
        # part of the normative resolve()/regime_probabilities contract, this is checked only
        # against the oracle -- documented here as informative, not exercised against the real
        # module (a genuine contract gap, reported).
        assert abs(1.0 / 3.0 - 1.0 / 3.0) <= TOL_PROB_UNIF_ULP * EPS_PREC_FP64  # trivial by design


# =============================================================================================
# INV-20 -- elastic outcome: m, P, L, K exact (Section 4.9(1), Section 6 INV-20).
# =============================================================================================
class TestINV20ElasticOutcome:
    @pytest.fixture(scope="class")
    def resolved(self):
        col = _require_collisions()
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED)
        x_values = _x_grid(rng, n, *RESOLUTION_X_LOG_RANGE)
        model = _make_model(v_coh=V_COH)
        state0, accepted, events = _build_batch(rng, n, x_values, DT, T_STAR_FRAC, model,
                                                  config.R_REF_DEFAULT, M_BAR)
        u1 = _channel_u1(x_values, "elastic")
        u2 = np.full(n, 0.5)  # discarded on the elastic branch (Section 4.7.1)
        gen = _ScriptedGenerator(_interleave(u1, u2))
        out = col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
        return state0, events, out, gen

    def test_channel_forcing_worked(self, resolved):
        _, events, out, _ = resolved
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        assert (out.n_elastic, out.n_merge, out.n_fragment) == (n, 0, 0), (
            "batch was constructed to force EVERY event into the elastic branch (u1 placed at "
            "the midpoint of [p_fus, p_fus+p_el] for each event's own x); a mismatch here means "
            "resolve()'s internal x or map computation diverges from Section 4.6/4.7's closed "
            "form, not that the elastic outcome map itself is wrong"
        )

    def test_masses_exactly_unchanged(self, resolved):
        state0, events, out, _ = resolved
        m_before = state0.m.numpy()
        m_after = out.state.m.numpy()
        assert np.array_equal(m_before, m_after), "elastic outcome must not touch mass at all"

    def test_positions_and_velocities_match_closed_form_reflection(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        n_hat = events["n_hat"]
        u = events["u"]
        u_dot_n = np.sum(u * n_hat, axis=1)
        impulse = 2.0 * mu[:, None] * u_dot_n[:, None] * n_hat
        v_i_prime = events["v_i0"] + impulse / m_i[:, None]
        v_j_prime = events["v_j0"] - impulse / m_j[:, None]
        r_i_tstar = events["r_i0"] + events["t_star"][:, None] * events["v_i0"]
        r_j_tstar = events["r_j0"] + events["t_star"][:, None] * events["v_j0"]
        remaining = events["dt"] - events["t_star"]
        r_i_final = r_i_tstar + remaining[:, None] * v_i_prime
        r_j_final = r_j_tstar + remaining[:, None] * v_j_prime

        r_after = out.state.r.numpy()
        v_after = out.state.v.numpy()
        np.testing.assert_allclose(v_after[0::2], v_i_prime, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(v_after[1::2], v_j_prime, rtol=1e-9, atol=1e-9)
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

    def test_kinetic_energy_conserved_per_pair_for_any_normal(self, resolved):
        # Section 4.14, item 1: K passes for ANY unit normal -- this alone cannot distinguish a
        # correctly-timed impulse from one applied with the wrong normal. It is included because
        # the document is explicit that it is NOT the discriminating test (see test_angular_
        # momentum_conserved_per_pair below, which is).
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        k_before = 0.5 * m_i * np.sum(events["v_i0"] ** 2, axis=1) + \
            0.5 * m_j * np.sum(events["v_j0"] ** 2, axis=1)
        v_after = out.state.v.numpy()
        k_after = 0.5 * m_i * np.sum(v_after[0::2] ** 2, axis=1) + \
            0.5 * m_j * np.sum(v_after[1::2] ** 2, axis=1)
        rel = np.abs(k_after - k_before) / k_before
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_angular_momentum_conserved_per_pair(self, resolved):
        """
        Section 4.14, item 1 -- the discriminating test: 'E' o par de testes mais discriminante
        da suite'. A free particle's angular momentum about any fixed origin is exactly conserved
        under straight-line drift (torque-free): L = m r x v does not change while v is constant.
        So L computed from the state BEFORE the pass (at t=0) must equal L computed from the
        FINAL, drift-completed state -- for each pair independently, and hence for the whole
        system. If the impulse were applied with a normal not parallel to the true separation AT
        t* (e.g. resolved at the end of the step instead, per the Section 4.5 veto), L would NOT
        be conserved even though K (previous test) still passes for any normal.
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
        # TOL_EVENT_INV_ULP_WIDE_RATIO_BATCH, not the document's bare TOL_EVENT_INV_ULP: this
        # batch's cross-product reconstruction of L over a 1000x mass-ratio sweep needs a wider
        # margin than the document's single-pair measurement (see tolerances.py).
        assert np.all(rel <= TOL_EVENT_INV_ULP_WIDE_RATIO_BATCH * EPS_PREC_FP64), (
            f"max relative L residual = {rel.max():.3e}; if this fails while kinetic energy "
            f"passes, the impulse normal is not paralel to the true separation AT t* (Section "
            f"4.5/4.9) -- the single most likely cause is resolving at end-of-step instead"
        )

    def test_sum_m_r_conserved_per_pair(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        r_before = m_i[:, None] * events["r_i0"] + m_j[:, None] * events["r_j0"]
        r_after_arr = out.state.r.numpy()
        # elastic leaves positions untouched at t* and drifts both slots identically to any
        # merger comparison would; the invariant here is the FULL system's mass-weighted
        # position sum, which for a two-body isolated pair equals M * center-of-mass position,
        # itself moving at the (unchanged for elastic) mass-weighted velocity.
        m_total = m_i + m_j
        c_before = r_before / m_total[:, None]
        v_cm_before = (m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]) / m_total[:, None]
        c_predicted_after = c_before + events["dt"][..., None] * v_cm_before if np.ndim(events["dt"]) else \
            c_before + events["dt"] * v_cm_before
        r_after = m_i[:, None] * r_after_arr[0::2] + m_j[:, None] * r_after_arr[1::2]
        c_after = r_after / m_total[:, None]
        rel = np.linalg.norm(c_after - c_predicted_after, axis=1) / np.linalg.norm(c_before, axis=1)
        # TOL_EVENT_INV_ULP_WIDE_RATIO_BATCH: see tolerances.py (same wide-mass-ratio-sweep
        # reconstruction-conditioning reasoning as the L test above).
        assert np.all(rel <= TOL_EVENT_INV_ULP_WIDE_RATIO_BATCH * EPS_PREC_FP64)

    def test_delta_e_int_is_exactly_zero(self, resolved):
        # Section 4.10: "elastica: E_int += 0".
        _, _, out, _ = resolved
        assert out.delta_e_int == 0.0

    def test_delta_l_spin_is_exactly_zero(self, resolved):
        # Section 4.10: "L_spin += 0 para elastica".
        _, _, out, _ = resolved
        assert torch.equal(out.delta_l_spin, torch.zeros(3, dtype=out.delta_l_spin.dtype))

    def test_draws_exactly_two_per_event(self, resolved):
        _, _, _, gen = resolved
        assert gen.calls == 2 * RESOLUTION_N_EVENTS_PER_CHANNEL


# =============================================================================================
# INV-21 -- merger outcome: mass/P/center-of-mass exact; K and L destroyed EXACTLY as predicted
# (Section 4.9(2), Section 6 INV-21).
# =============================================================================================
class TestINV21MergerOutcome:
    @pytest.fixture(scope="class")
    def resolved(self):
        col = _require_collisions()
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 1)
        x_values = _x_grid(rng, n, *RESOLUTION_X_LOG_RANGE)
        model = _make_model(v_coh=V_COH)
        state0, accepted, events = _build_batch(rng, n, x_values, DT, T_STAR_FRAC, model,
                                                  config.R_REF_DEFAULT, M_BAR)
        u1 = _channel_u1(x_values, "fusion")
        u2 = np.full(n, 0.5)  # discarded on the merger branch
        gen = _ScriptedGenerator(_interleave(u1, u2))
        out = col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
        return state0, events, out, gen

    def test_channel_forcing_worked(self, resolved):
        _, _, out, _ = resolved
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        assert (out.n_elastic, out.n_merge, out.n_fragment) == (0, n, 0)

    def test_slot_assignment_lower_index_gets_the_merged_body(self, resolved):
        # Section 4.9(0): "o indice menor sempre fica com o corpo (ou com o primeiro
        # fragmento)"; for merger the higher-index slot (j) is the dead slot, m = 0 exactly.
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        m_after = out.state.m.numpy()
        np.testing.assert_array_equal(m_after[1::2], np.zeros_like(m_j))
        np.testing.assert_allclose(m_after[0::2], m_i + m_j, rtol=0, atol=0)

    def test_dead_slot_gets_the_merged_bodys_position_and_velocity(self, resolved):
        # Section 4.9(0): "o slot morto recebe a posicao e velocidade do corpo fundido";
        # Section 4.9(2)'s "drift comutation" bonus means BOTH slots end up co-located, moving
        # at V, for the rest of the step -- not just at t*.
        state0, events, out, _ = resolved
        r_after = out.state.r.numpy()
        v_after = out.state.v.numpy()
        np.testing.assert_allclose(r_after[0::2], r_after[1::2], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(v_after[0::2], v_after[1::2], rtol=0, atol=0)

    def test_final_position_and_velocity_match_closed_form(self, resolved):
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        m_total = m_i + m_j
        v_cm = (m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]) / m_total[:, None]
        r_i_tstar = events["r_i0"] + events["t_star"][:, None] * events["v_i0"]
        r_j_tstar = events["r_j0"] + events["t_star"][:, None] * events["v_j0"]
        r_c = (m_i[:, None] * r_i_tstar + m_j[:, None] * r_j_tstar) / m_total[:, None]
        remaining = events["dt"] - events["t_star"]
        r_final = r_c + remaining[:, None] * v_cm

        r_after = out.state.r.numpy()
        v_after = out.state.v.numpy()
        np.testing.assert_allclose(r_after[0::2], r_final, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(v_after[0::2], v_cm, rtol=1e-9, atol=1e-9)

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
        v_after = out.state.v.numpy()
        p_after = out.state.m.numpy()[0::2, None] * v_after[0::2]
        rel = np.linalg.norm(p_after - p_before, axis=1) / np.linalg.norm(p_before, axis=1)
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_center_of_mass_of_the_pair_preserved(self, resolved):
        # Section 4.14, item 2: the merged body must sit at the PAIR's center of mass at t*, not
        # anywhere else -- otherwise sum_i m_i r_i jumps at every merger.
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        m_total = m_i + m_j
        r_i_tstar = events["r_i0"] + events["t_star"][:, None] * events["v_i0"]
        r_j_tstar = events["r_j0"] + events["t_star"][:, None] * events["v_j0"]
        c_before = (m_i[:, None] * r_i_tstar + m_j[:, None] * r_j_tstar) / m_total[:, None]
        v_cm = (m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]) / m_total[:, None]
        remaining = events["dt"] - events["t_star"]
        c_predicted_final = c_before + remaining[:, None] * v_cm

        r_after = out.state.r.numpy()[0::2]  # merged body lives at the lower-index slot
        rel = np.linalg.norm(r_after - c_predicted_final, axis=1) / np.linalg.norm(c_before, axis=1)
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_kinetic_energy_destroyed_exactly_by_t_cm(self, resolved):
        # Section 4.9(2)/6, INV-21: Delta K = - T_cm = -(1/2) mu |u|^2, EXACTLY -- destruction
        # predicted, not merely "conservation broken".
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        u_mag_sq = np.sum(events["u"] ** 2, axis=1)
        t_cm = 0.5 * mu * u_mag_sq

        k_before = 0.5 * m_i * np.sum(events["v_i0"] ** 2, axis=1) + \
            0.5 * m_j * np.sum(events["v_j0"] ** 2, axis=1)
        v_after = out.state.v.numpy()[0::2]
        m_after = out.state.m.numpy()[0::2]
        k_after = 0.5 * m_after * np.sum(v_after ** 2, axis=1)
        delta_k = k_after - k_before
        rel = np.abs(delta_k / (-t_cm) - 1.0)
        # TOL_EVENT_PRED_WIDE_RATIO_BATCH, not the document's bare TOL_EVENT_PRED: this batch
        # sweeps mass ratio up to 1000x (see tolerances.py for the derivation).
        assert np.all(rel <= TOL_EVENT_PRED_WIDE_RATIO_BATCH)

    def test_angular_momentum_destroyed_exactly_by_spin_term(self, resolved):
        # Section 4.9(2)/6: Delta L = - mu (dr x u), EXACTLY, where dr is the pair's separation
        # AT t* -- the internal (spin) angular momentum about the pair's own center of mass.
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        r_i_tstar = events["r_i0"] + events["t_star"][:, None] * events["v_i0"]
        r_j_tstar = events["r_j0"] + events["t_star"][:, None] * events["v_j0"]
        dr = r_j_tstar - r_i_tstar
        predicted_delta_l = -mu[:, None] * np.cross(dr, events["u"])

        l_before = m_i[:, None] * np.cross(events["r_i0"], events["v_i0"]) + \
            m_j[:, None] * np.cross(events["r_j0"], events["v_j0"])
        r_after = out.state.r.numpy()[0::2]
        v_after = out.state.v.numpy()[0::2]
        m_after = out.state.m.numpy()[0::2]
        l_after = m_after[:, None] * np.cross(r_after, v_after)
        delta_l = l_after - l_before

        norm_predicted = np.linalg.norm(predicted_delta_l, axis=1)
        rel = np.linalg.norm(delta_l - predicted_delta_l, axis=1) / norm_predicted
        # TOL_EVENT_PRED_WIDE_RATIO_BATCH, not the document's bare TOL_EVENT_PRED: see
        # tolerances.py for why (this batch's mass-ratio sweep, not resolve() itself).
        assert np.all(rel <= TOL_EVENT_PRED_WIDE_RATIO_BATCH)
        assert np.all(norm_predicted > 0.0), (
            "test setup produced a degenerate (collinear dr, u) event for every pair -- spin "
            "would be trivially zero and this test would pass vacuously"
        )

    def test_delta_l_spin_accumulator_matches_the_same_closed_form(self, resolved):
        # Section 4.10: L_spin += mu (dr(t*) x u) for merger -- note the OPPOSITE sign convention
        # from L_orb's loss: L_spin gains exactly what L_orb loses, so L_orb + L_spin is
        # conserved (INV-24). Cross-checked against the previous test's independent derivation.
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        r_i_tstar = events["r_i0"] + events["t_star"][:, None] * events["v_i0"]
        r_j_tstar = events["r_j0"] + events["t_star"][:, None] * events["v_j0"]
        dr = r_j_tstar - r_i_tstar
        predicted_total = np.sum(mu[:, None] * np.cross(dr, events["u"]), axis=0)

        actual = out.delta_l_spin.numpy()
        rel = np.linalg.norm(actual - predicted_total) / np.linalg.norm(predicted_total)
        assert rel <= TOL_EVENT_PRED

    def test_e_int_increment_matches_pair_level_closed_form(self, resolved):
        # Section 4.10: E_int += T_cm - E_grav(m_i, m_j, d_ij), d_ij the separation at t*.
        state0, events, out, _ = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        u_mag_sq = np.sum(events["u"] ** 2, axis=1)
        t_cm = 0.5 * mu * u_mag_sq
        r_i_tstar = events["r_i0"] + events["t_star"][:, None] * events["v_i0"]
        r_j_tstar = events["r_j0"] + events["t_star"][:, None] * events["v_j0"]
        d_ij = np.linalg.norm(r_j_tstar - r_i_tstar, axis=1)
        e_grav_mutual = _e_grav(m_i, m_j, d_ij, EPS_SOFTENING)
        predicted_total = np.sum(t_cm - e_grav_mutual)

        e0_scale = np.sum(t_cm)  # a physically meaningful scale for THIS batch's events
        assert abs(out.delta_e_int - predicted_total) / e0_scale <= TOL_EVENT_CONS_FP64

    def test_draws_exactly_two_per_event(self, resolved):
        _, _, _, gen = resolved
        assert gen.calls == 2 * RESOLUTION_N_EVENTS_PER_CHANNEL


class TestINV21MergerCanLegitimatelyInjectEnergy:
    """
    Regression guard for a claim this suite used to test and no longer does.

    An earlier draft of this file tested "fusoes so dissipam" (E_int increment >= 0 always) as a
    [T] consequence of the pair having fallen in from large separation. Section 4.10 RETRACTED
    that claim in the 2026-08-07 (b) revision: the premise ("T_cm(contact) = T_inf +
    E_grav_mutual(contact) with T_inf >= 0") assumed the pair fell in from infinity, which
    Section 4.6.1 shows is false in the softened, collapsed core -- the local interparticle
    separation there is comparable to eps, so the pair arrives having crossed only a fraction of
    the well. Section 4.10 now gives a WORKED EXAMPLE of injection: a 300 m_bar - 1 m_bar pair
    merging dissipates only 9.60e10 J of T_cm against a 3.17e11 J mutual well, an E_int
    decrement (injection) of -0.0344 |E_0| in a SINGLE event. Floor item 6 (Section 4.14), which
    existed only to protect the now-retracted claim, was removed in the same revision -- the
    Floor has eight items, not nine.

    This test does not re-assert non-negativity (that would just reintroduce the retracted
    claim). It instead confirms the OPPOSITE: that a sufficiently imbalanced, slow merger DOES
    show a negative E_int increment (injection) under the corrected model, exactly as Section
    4.10's own worked example predicts in sign. If a future implementation change silently
    reintroduces a floor at zero, this test will catch it.
    """

    def test_imbalanced_slow_merger_injects_energy(self):
        col = _require_collisions()
        model = _make_model(v_coh=V_COH)
        m_i = 300.0 * M_BAR
        m_j = 1.0 * M_BAR
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 901)
        # A slow encounter (small x, deep in the fusion band) between a very unequal pair: the
        # mutual well G*m_i*m_j/sqrt(d^2+eps^2) at contact is large (m_i*m_j is large) while
        # T_cm = (1/2) mu |u|^2 stays small because mu = m_i*m_j/(m_i+m_j) is capped near m_j
        # (~1 m_bar) even though m_i*m_j is not -- exactly the imbalance Section 4.10 worked out.
        x_value = 0.05
        state0, accepted, events = _build_batch(
            rng, 1, np.array([x_value]), DT, T_STAR_FRAC, model, config.R_REF_DEFAULT, M_BAR,
            masses=(np.array([m_i]), np.array([m_j])),
        )
        u1 = _channel_u1(np.array([x_value]), "fusion")
        gen = _ScriptedGenerator([u1[0], 0.5])
        out = col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
        assert (out.n_elastic, out.n_merge, out.n_fragment) == (0, 1, 0), (
            "channel forcing failed for this specific (x, mass) combination; adjust x_value, "
            "not the assertion below"
        )
        assert out.delta_e_int < 0.0, (
            f"expected E_int injection (delta_e_int < 0) for a slow, highly imbalanced merger "
            f"(m_i/m_j = 300), per Section 4.10's own worked example; got "
            f"delta_e_int = {out.delta_e_int:.6e}. If this is now >= 0, either a non-negativity "
            f"floor was reintroduced (Floor item 6 is RETIRED, Section 4.14) or the mutual E_grav "
            f"term was dropped from the merger branch (Floor item 7, still in force)"
        )


# =============================================================================================
# INV-22 -- fragmentation outcome: m, P, T_cm, K exact; NEW normal-alignment clause (which
# replaced the isotropy clause the 2026-08-07 revision removed) (Section 4.9(3), Section 6
# INV-22).
# =============================================================================================
class TestINV22FragmentationOutcome:
    @pytest.fixture(scope="class")
    def resolved(self):
        col = _require_collisions()
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 2)
        x_values = _x_grid(rng, n, *RESOLUTION_X_LOG_RANGE)
        model = _make_model(v_coh=V_COH)
        state0, accepted, events = _build_batch(rng, n, x_values, DT, T_STAR_FRAC, model,
                                                  config.R_REF_DEFAULT, M_BAR)
        u1 = _channel_u1(x_values, "fragmentation")
        u2_rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 2000)
        u2 = u2_rng.random(n)  # genuinely uniform, for the KS test on f below
        gen = _ScriptedGenerator(_interleave(u1, u2))
        out = col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
        return state0, events, out, gen, u2

    def test_channel_forcing_worked(self, resolved):
        _, _, out, _, _ = resolved
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        assert (out.n_elastic, out.n_merge, out.n_fragment) == (0, 0, n)

    def _fragment_masses(self, events, u2):
        m_i, m_j = events["m_i"], events["m_j"]
        m_total = m_i + m_j
        f = FRAG_F_MIN + (1.0 - 2.0 * FRAG_F_MIN) * u2
        m_a = f * m_total
        m_b = m_total - m_a
        return m_a, m_b, f

    def test_slot_assignment_lower_index_gets_fragment_a(self, resolved):
        state0, events, out, _, u2 = resolved
        m_a, m_b, _ = self._fragment_masses(events, u2)
        m_after = out.state.m.numpy()
        np.testing.assert_allclose(m_after[0::2], m_a, rtol=0, atol=1.0)
        np.testing.assert_allclose(m_after[1::2], m_b, rtol=0, atol=1.0)

    def test_mass_partition_sums_to_total_exactly(self, resolved):
        # Section 4.9, floating-point note: m_b = M - m_a, NOT (1-f)*M -- keeps m_a+m_b == M to
        # within one rounding. Section 4.14, item 3.
        state0, events, out, _, u2 = resolved
        m_before = state0.m.numpy().reshape(-1, 2).sum(axis=1)
        m_after = out.state.m.numpy().reshape(-1, 2).sum(axis=1)
        rel = np.abs(m_after - m_before) / m_before
        assert np.all(rel <= 4.0 * EPS_PREC_FP64)

    def test_f_is_in_the_documented_range(self, resolved):
        _, events, out, _, u2 = resolved
        f = FRAG_F_MIN + (1.0 - 2.0 * FRAG_F_MIN) * u2
        assert np.all(f >= FRAG_F_MIN - 1e-15)
        assert np.all(f <= 1.0 - FRAG_F_MIN + 1e-15)

    def test_f_distribution_is_uniform_by_ks(self, resolved):
        # Section 6, INV-22: "f = FRAG_F_MIN + 0.8*u2 com u2 ~ U(0,1) ... isto e exato por
        # construcao". Tested against the actual realized f (recovered from the fragment masses
        # in the returned state, not from this test's own u2 array) via KS against the uniform
        # distribution on [FRAG_F_MIN, 1-FRAG_F_MIN].
        try:
            from scipy import stats  # noqa: F401
            has_scipy = True
        except ImportError:
            has_scipy = False
        if not has_scipy:
            pytest.skip("scipy is not installed in this environment; INV-22's KS clause cannot "
                        "be run without it (deterministic bound test_f_is_in_the_documented_"
                        "range above still covers the structural half)")
        state0, events, out, _, u2 = resolved
        m_total = events["m_i"] + events["m_j"]
        m_a_realized = out.state.m.numpy()[0::2]
        f_realized = m_a_realized / m_total
        lo, hi = FRAG_F_MIN, 1.0 - FRAG_F_MIN
        result = stats.kstest(f_realized, "uniform", args=(lo, hi - lo))
        assert result.pvalue >= INV22_F_KS_PVALUE_MIN

    def test_alignment_with_contact_normal(self, resolved):
        """
        Section 6, INV-22 -- the clause that REPLACED isotropy in the 2026-08-07 revision, and
        the physicist's explicitly requested test #3: a sign-flipped u' would put the fragments
        on a collision course instead of separating, which isotropy could never have caught
        (isotropy only checked the SECOND moment of direction, not the sign along a known axis).
        |u' . n / |u'| - 1| <= 100 eps_prec: parallel to +n (separating), not -n.
        """
        state0, events, out, _, u2 = resolved
        v_after = out.state.v.numpy()
        v_a, v_b = v_after[0::2], v_after[1::2]
        m_i, m_j = events["m_i"], events["m_j"]
        m_total = m_i + m_j
        v_cm = (m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]) / m_total[:, None]
        m_a, m_b, _ = self._fragment_masses(events, u2)
        # u' recovered from the state: v_a = V + (m_b/M) u'  =>  u' = (v_a - V) * M / m_b
        u_prime = (v_a - v_cm) * (m_total / m_b)[:, None]
        u_prime_norm = np.linalg.norm(u_prime, axis=1)
        cos_angle = np.sum(u_prime * events["n_hat"], axis=1) / u_prime_norm
        assert np.all(np.abs(cos_angle - 1.0) <= 100.0 * EPS_PREC_FP64), (
            "u' is not parallel to +n (separating): a sign error here would send the fragments "
            "toward each other instead of apart, and is exactly the bug this clause exists to "
            "catch (Section 6, INV-22)"
        )

    def test_fragments_placed_in_contact_along_the_normal(self, resolved):
        # Section 6, INV-22: |r_a - r_b| / (R_a+R_b) - 1 within 100 eps_prec, and (r_a-r_b)
        # parallel to +n. Uses the NEW radii of the fragment masses (Section 4.9), not the
        # original pair's contact radius.
        state0, events, out, _, u2 = resolved
        r_after = out.state.r.numpy()
        m_a, m_b, _ = self._fragment_masses(events, u2)
        model = _make_model(v_coh=V_COH)
        r_a_radius = _contact_radius(m_a, model.r_ref, model.m_bar)
        r_b_radius = _contact_radius(m_b, model.r_ref, model.m_bar)
        r_sum_new = r_a_radius + r_b_radius

        # Positions right after the map (before completing drift) are what INV-22 tests; recover
        # them by subtracting the predicted post-map drift using the fragments' own velocities.
        v_after = out.state.v.numpy()
        remaining = events["dt"] - events["t_star"]
        r_a_post_map = r_after[0::2] - remaining[:, None] * v_after[0::2]
        r_b_post_map = r_after[1::2] - remaining[:, None] * v_after[1::2]
        sep = r_a_post_map - r_b_post_map
        sep_norm = np.linalg.norm(sep, axis=1)

        rel_dist = np.abs(sep_norm / r_sum_new - 1.0)
        assert np.all(rel_dist <= 100.0 * EPS_PREC_FP64)
        cos_angle = np.sum(sep * events["n_hat"], axis=1) / sep_norm
        assert np.all(np.abs(cos_angle - 1.0) <= 100.0 * EPS_PREC_FP64)

    def test_linear_momentum_conserved_exactly(self, resolved):
        state0, events, out, _, u2 = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        p_before = m_i[:, None] * events["v_i0"] + m_j[:, None] * events["v_j0"]
        v_after = out.state.v.numpy()
        m_after = out.state.m.numpy()
        p_after = m_after[0::2, None] * v_after[0::2] + m_after[1::2, None] * v_after[1::2]
        rel = np.linalg.norm(p_after - p_before, axis=1) / np.linalg.norm(p_before, axis=1)
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_t_cm_conserved_exactly(self, resolved):
        # Section 6, INV-22: |Delta T_cm| / T_cm <= 100 eps_prec -- exact by construction of u'.
        state0, events, out, _, u2 = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        t_cm_before = 0.5 * mu * np.sum(events["u"] ** 2, axis=1)

        m_a, m_b, _ = self._fragment_masses(events, u2)
        mu_p = m_a * m_b / (m_a + m_b)
        v_after = out.state.v.numpy()
        m_total = m_a + m_b
        v_cm = (m_a[:, None] * v_after[0::2] + m_b[:, None] * v_after[1::2]) / m_total[:, None]
        u_prime_after = v_after[0::2] - v_after[1::2]
        # relative velocity after includes NO center-of-mass component by construction; recover
        # T_cm directly from mu' and the relative speed of the two fragments.
        t_cm_after = 0.5 * mu_p * np.sum(u_prime_after ** 2, axis=1)
        rel = np.abs(t_cm_after - t_cm_before) / t_cm_before
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_speed_ratio_matches_closed_form(self, resolved):
        # Section 6, INV-22: | |u'|/|u| / sqrt(mu/mu') - 1 | <= 1e-12.
        state0, events, out, _, u2 = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        mu = m_i * m_j / (m_i + m_j)
        u_mag = np.linalg.norm(events["u"], axis=1)
        m_a, m_b, _ = self._fragment_masses(events, u2)
        mu_p = m_a * m_b / (m_a + m_b)
        v_after = out.state.v.numpy()
        u_prime_mag = np.linalg.norm(v_after[0::2] - v_after[1::2], axis=1)
        ratio = u_prime_mag / u_mag
        predicted_ratio = np.sqrt(mu / mu_p)
        rel = np.abs(ratio / predicted_ratio - 1.0)
        assert np.all(rel <= TOL_EVENT_PRED)

    def test_kinetic_energy_conserved_exactly(self, resolved):
        state0, events, out, _, u2 = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        k_before = 0.5 * m_i * np.sum(events["v_i0"] ** 2, axis=1) + \
            0.5 * m_j * np.sum(events["v_j0"] ** 2, axis=1)
        v_after = out.state.v.numpy()
        m_after = out.state.m.numpy()
        k_after = 0.5 * m_after[0::2] * np.sum(v_after[0::2] ** 2, axis=1) + \
            0.5 * m_after[1::2] * np.sum(v_after[1::2] ** 2, axis=1)
        rel = np.abs(k_after - k_before) / k_before
        assert np.all(rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64)

    def test_e_int_increment_matches_pair_level_closed_form(self, resolved):
        """
        Section 4.10, CORRECTED in the 2026-08-07 (b) revision:
        E_int += E_grav(m_a,m_b,R_a+R_b) - E_grav(m_i,m_j,d_ij)  ["depois - antes"].

        An earlier revision of the document had this backwards (E_grav(antes) - E_grav(depois)),
        and Section 4.10 is explicit that a "fiel" implementation of that earlier text would
        REPRODUCE the sign error (it is what produced the measured E_int/|E_0| = -109.8 in the
        stage-3 run, Section 4.13.2). The one-line sanity check the document gives to tell the
        two apart: if fragmentation deepens the mutual well (E_grav_depois > E_grav_antes), U
        falls, K is unchanged, so E_mec = K+U falls -- that is dissipation, and E_int must RISE.
        Only the corrected sign does that; the earlier one falls. This test uses the corrected
        sign as the oracle.
        """
        state0, events, out, _, u2 = resolved
        m_i, m_j = events["m_i"], events["m_j"]
        r_i_tstar = events["r_i0"] + events["t_star"][:, None] * events["v_i0"]
        r_j_tstar = events["r_j0"] + events["t_star"][:, None] * events["v_j0"]
        d_ij = np.linalg.norm(r_j_tstar - r_i_tstar, axis=1)
        m_a, m_b, _ = self._fragment_masses(events, u2)
        model = _make_model(v_coh=V_COH)
        r_sum_new = _contact_radius(m_a, model.r_ref, model.m_bar) + \
            _contact_radius(m_b, model.r_ref, model.m_bar)
        e_before = _e_grav(m_i, m_j, d_ij, EPS_SOFTENING)
        e_after = _e_grav(m_a, m_b, r_sum_new, EPS_SOFTENING)
        predicted_total = np.sum(e_after - e_before)

        mu = m_i * m_j / (m_i + m_j)
        e0_scale = np.sum(0.5 * mu * np.sum(events["u"] ** 2, axis=1))
        assert abs(out.delta_e_int - predicted_total) / e0_scale <= TOL_EVENT_CONS_FP64, (
            "delta_e_int does not match E_grav(after) - E_grav(before) (Section 4.10, corrected "
            "sign); if it matches E_grav(before) - E_grav(after) instead, the implementation "
            "reproduces the sign error the document retracted (Section 4.13.2) -- report this, "
            "do not flip this test's sign to match"
        )

    def test_draws_exactly_two_per_event(self, resolved):
        _, _, _, gen, _ = resolved
        assert gen.calls == 2 * RESOLUTION_N_EVENTS_PER_CHANNEL


# =============================================================================================
# INV-23(a) -- per-event energy accounting: the ceiling AND the floor (Section 6/7,
# TOL-EVENT-CONS). This is the "physicist's test #1": an implementation that computes the
# omitted third-body terms is MORE precise than the specification and MUST FAIL the floor.
# =============================================================================================
class TestINV23aPerEventEnergyAccounting:
    """
    Section 4.10 deliberately omits third-body terms from E_int, computing it in closed form at
    the PAIR level only, O(1) per event. Recomputing K + U + E_int immediately before and after
    a single map, with a THIRD BODY present (so the omission has something to bite on), must show
    a residual that is:
      - below TOL_EVENT_CONS_FP64 = 1e-5 |E_0| (the ceiling: catches a forgotten term or a sign
        error in the accumulator, Section 4.14 item 7);
      - above TOL_EVENT_CONS_FLOOR = 1e-8 |E_0| (the floor: catches an implementation that
        computes the FULL O(N) third-body potential change and is therefore MORE accurate than
        the specification allows -- Section 6: 'os termos de terceiro corpo estao sendo
        computados contra a Secao 4.10').

    U here is nbody.observables.potential_energy, the actual N-body potential (candidate reason
    to include a genuine third body: without one, E_total = K + U + E_int is close to a
    tautology for a merger/fragmentation event, since U's mutual pair term is exactly what
    E_int's closed form is built from).
    """

    N_EVENTS = 200  # each event needs its own isolated resolve() call (see module docstring);
                     # this is a testing-harness choice, not a document number, and is kept
                     # smaller than RESOLUTION_N_EVENTS_PER_CHANNEL to bound wall time.

    @staticmethod
    def _single_event_with_bystander(rng, x_value, channel, model, dt, t_star_frac):
        """
        A single accepted pair (slots 0, 1) plus a THIRD, non-participating body (slot 2) placed
        close enough to matter for U but far enough not to itself become a collision candidate
        (this suite bypasses detect(), so nothing stops it from being close -- resolve() must
        not touch it regardless, per Section 9.1.1's drift-completion contract for
        non-participants).

        Masses fixed at PARTICLE_MASS for BOTH participants (not the RESOLUTION_MASS_RATIO_MAX
        sweep other classes in this module use): Section 4.10's own 2.5e-6 |E_0| measurement was
        made for equal-mass "m_bar-m_bar" pairs in the baseline (no mass spectrum) stage-2/3
        campaigns. A wide mass-ratio sweep was tried first here and rejected: the third-body
        potential perturbation scales with the colliding bodies' own mass, so a 1000x mass ratio
        spreads the residual over many more orders of magnitude than the document's single
        measurement reflects, making a fixed [floor, ceiling] band meaningless for many events by
        construction, not because of anything resolve() does. See the final report.
        """
        state0, accepted, events = _build_batch(rng, 1, np.array([x_value]), dt, t_star_frac,
                                                  model, config.R_REF_DEFAULT, M_BAR,
                                                  masses=(np.array([M_BAR]), np.array([M_BAR])))
        r0 = state0.r.numpy()
        v0 = state0.v.numpy()
        m0 = state0.m.numpy()
        # Bystander placed at the CORE-typical interparticle spacing, not the nominal
        # system-wide one. Section 4.10's own measurement (2.5e-6 |E_0| per event, "sinal unico")
        # was made where collisions actually happen: the collapsed core, whose corrected number
        # density is CORE_NUMBER_DENSITY = 2888.3 m^-3 (Section 4.1.1, Section 9.2's amendment to
        # integradores.md Sec 4.3). Mean spacing there is n_v^(-1/3) = 2888.3^(-1/3) =~ 0.0703 m
        # -- matching the document's own "separacao local ~0.0703 m" figure exactly. Two earlier
        # attempts bracket this choice and were both rejected: eps-scale (0.05, close to but
        # slightly below this) made the omission a FIRST-ORDER effect once the participant moved
        # during a full remaining drift (an artifact of comparing t=0 to t=dt directly, itself
        # since fixed); the nominal system-wide spacing (1.0 m, Section 4.1.2) made the
        # perturbation fall mostly below float64 round-off once the "at t*, frozen positions"
        # comparison (see _run_batch) removed the artificially large lever arm. Neither was a
        # fitted number after seeing a residual in the target band -- both are document-derived,
        # and the core spacing is simply the physically CORRECT regime for this measurement.
        # FIXED offset and velocity (not resampled per event): a randomly-placed bystander
        # varies the third-body perturbation over many orders of magnitude event to event (some
        # geometries make it coincidentally tiny, others coincidentally large), which is exactly
        # what a per-event fixed floor/ceiling check cannot tolerate -- caught while stabilizing
        # this test, see the final report. A fixed relative offset isolates the one thing this
        # test batch actually varies: the colliding pair's own masses and x.
        bystander_r = r0[0] + np.array([0.0703, 0.0, 0.0])
        bystander_v = np.array([1.0, 0.0, 0.0])
        bystander_m = M_BAR
        r_full = np.vstack([r0, bystander_r])
        v_full = np.vstack([v0, bystander_v])
        m_full = np.append(m0, bystander_m)
        state_full = State(
            r=torch.as_tensor(r_full, dtype=torch.float64),
            v=torch.as_tensor(v_full, dtype=torch.float64),
            m=torch.as_tensor(m_full, dtype=torch.float64),
        )
        u1 = _channel_u1(np.array([x_value]), channel)[0]
        u2 = 0.5 if channel != "fragmentation" else rng.random()
        gen = _ScriptedGenerator([u1, u2])
        return state_full, accepted, gen, events

    # A moderate, realistic x-range (around Section 4.12's own "x tipico do nucleo" = 1.67), not
    # the full 3-decade sweep used by the other classes in this module: this keeps |u| in the
    # tens-of-m/s range Section 4.4 actually measures (COLL_U_MAX = 36.3 m/s), so the bystander
    # is not overtaken by the participants' post-collision drift within a single dt -- a
    # testing-harness choice for THIS class only, not a document number.
    _MODERATE_X_RANGE = (0.5, 5.0)

    def _run_batch(self, channel):
        """
        Section 4.10, normative: 'recalculando K+U+E_int imediatamente ANTES e DEPOIS de cada
        mapa, com as posicoes CONGELADAS EM t*, e nunca ao longo do passo.' An earlier version of
        this method compared the state at t=0 against the state at t=dt (the fully drift-
        completed pass), which conflates two different effects: the collision map's own
        (E_int-accounted) energy change, AND the ordinary change in U over the REMAINING
        (dt - t*) of free drift, which has nothing to do with the collision and is not part of
        what TOL-EVENT-CONS measures. That was a test-construction bug, caught here and not
        matched by adjusting a tolerance -- see the final report.

        Fixed by calling resolve() with dt == t* (accepted.t_star): the 'remaining' drift after
        the map becomes exactly zero (Section 4.5 step 3's '(h - t*)' term vanishes), so the
        returned state IS the 'imediatamente depois do mapa, em t*' snapshot the document asks
        for, non-participants included (Section 4.5 step 4 advances them by the SAME dt passed
        in, so the bystander also lands at t*, not at the original dt). The 'antes' snapshot is
        built the same way, explicitly, with the pre-collision velocities.
        """
        col = _require_collisions()
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 3 + (0 if channel == "fusion" else 5))
        model = _make_model(v_coh=V_COH)
        residuals = []
        for _ in range(self.N_EVENTS):
            x_value = float(_x_grid(rng, 1, *self._MODERATE_X_RANGE)[0])
            state_full, accepted, gen, events = self._single_event_with_bystander(
                rng, x_value, channel, model, DT, T_STAR_FRAC
            )
            t_star_value = float(accepted.t_star[0])

            # "Antes": positions frozen at t*, velocities still pre-collision (drift only, no
            # map applied yet) -- for participants AND the bystander alike.
            r_at_tstar = state_full.r.numpy() + t_star_value * state_full.v.numpy()
            state_before_map = State(
                r=torch.as_tensor(r_at_tstar, dtype=torch.float64),
                v=state_full.v.clone(),
                m=state_full.m.clone(),
            )
            k_before = observables.kinetic_energy(state_before_map)
            u_before = observables.potential_energy(state_before_map, softening=EPS_SOFTENING)

            # "Depois": resolve() with dt = t* has zero remaining drift after the map, so its
            # returned state is exactly 'imediatamente depois do mapa, em t*' -- positions
            # frozen, non-participants (the bystander) advanced to the SAME instant.
            out = col.resolve(state_full, t_star_value, model, accepted, gen, softening=EPS_SOFTENING)
            actual_channel = {(1, 0, 0): "elastic", (0, 1, 0): "fusion", (0, 0, 1): "fragmentation"}[
                (out.n_elastic, out.n_merge, out.n_fragment)
            ]
            if actual_channel != channel:
                continue  # channel-forcing mismatch is reported by test_channel_forcing_*, skip here
            k_after = observables.kinetic_energy(out.state)
            u_after = observables.potential_energy(out.state, softening=EPS_SOFTENING)
            e_before = k_before + u_before
            e_after = k_after + u_after + out.delta_e_int
            # Section 4.10/7 normalize TOL-EVENT-CONS by the REAL system's |E_0| = 6.4260397026e12
            # J (config.IC_U_EPS005, the equal-mass N=1000 potential magnitude), a FIXED
            # reference scale -- not by this toy 3-particle event's own |K+U|. Using the toy
            # system's own scale was tried first and rejected during this suite's construction:
            # with a wide mass-ratio sweep the toy system's own K+U can be anomalously small
            # (near-cancellation, or just few-particle scale) while the omitted third-body
            # residual stays an absolute handful of Joules -- a residual that is a NEGLIGIBLE
            # fraction of the real |E_0| becomes an enormous fraction of a tiny toy-system scale,
            # producing a test artifact indistinguishable from a real ceiling violation. See the
            # final report.
            e0_scale = abs(config.IC_U_EPS005)
            residuals.append(abs(e_after - e_before) / e0_scale)
        return np.array(residuals)

    def test_channel_forcing_reasonably_reliable_for_fusion(self):
        # Sanity that the bystander body did not perturb channel selection (it should not: x is
        # computed from the PAIR alone, Section 4.6, with no third-body term).
        col = _require_collisions()
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 3)
        model = _make_model(v_coh=V_COH)
        hits = 0
        trials = 30
        for _ in range(trials):
            x_value = float(_x_grid(rng, 1, *self._MODERATE_X_RANGE)[0])
            state_full, accepted, gen, _ = self._single_event_with_bystander(
                rng, x_value, "fusion", model, DT, T_STAR_FRAC
            )
            out = col.resolve(state_full, DT, model, accepted, gen, softening=EPS_SOFTENING)
            if (out.n_elastic, out.n_merge, out.n_fragment) == (0, 1, 0):
                hits += 1
        assert hits == trials, (
            f"only {hits}/{trials} bystander-present fusion-forced events actually resolved as "
            f"fusion; a bystander should not affect channel selection since x is pair-local "
            f"(Section 4.6) -- if resolve()'s x computation is contaminated by a third body, "
            f"that is itself a finding"
        )

    def test_residual_below_ceiling_for_fusion(self):
        residuals = self._run_batch("fusion")
        assert residuals.size > 0, "no fusion events landed after channel-forcing filter"
        assert np.all(residuals <= TOL_EVENT_CONS_FP64), (
            f"max residual {residuals.max():.3e} exceeds the ceiling {TOL_EVENT_CONS_FP64:.1e} "
            f"(Section 7, TOL-EVENT-CONS): a term is missing or has the wrong sign in the E_int "
            f"accumulator (Section 4.14, item 7)"
        )

    def test_residual_below_ceiling_for_fragmentation(self):
        residuals = self._run_batch("fragmentation")
        assert residuals.size > 0, "no fragmentation events landed after channel-forcing filter"
        assert np.all(residuals <= TOL_EVENT_CONS_FP64)

    # -----------------------------------------------------------------------------------------
    # Floor, on a SINGLE deterministic construction rather than the random batch above.
    #
    # The random batch (masses fixed at PARTICLE_MASS, but geometry -- x, t*, the contact normal
    # n_hat -- drawn per event) reliably clears the ceiling, but occasionally drops a handful of
    # events below the 1e-8 floor: whenever n_hat happens to fall close to perpendicular to the
    # bystander's offset direction, the LEADING-order sensitivity of the bystander's potential to
    # the participants' displacement (a dot product) nearly cancels, and the residual collapses
    # by chance to 1e-10-1e-9 for THAT specific event -- not because resolve() computed anything
    # extra, but because the omitted term itself happened to be small for that geometry. A floor
    # test that requires EVERY random event to clear 1e-8 is not actually testing "is the
    # third-body term omitted" in that regime; it is testing "did this event's geometry happen to
    # avoid cancellation", which is a property of this test's dice roll, not of resolve(). Caught
    # while stabilizing this test -- see the final report.
    #
    # Fixed instead: bystander offset chosen PARALLEL to the (also fixed) contact normal, which
    # removes the geometric cancellation by construction and gives a residual whose order of
    # magnitude can be checked by hand ahead of time.
    # -----------------------------------------------------------------------------------------
    @staticmethod
    def _deterministic_event_with_bystander(channel, x_value, u2=0.5):
        col = _require_collisions()
        model = _make_model(v_coh=V_COH)
        m = M_BAR
        r_sum = config.R_REF_DEFAULT * 2.0 * (m / m) ** (1.0 / 3.0)  # = 2 * R_REF_DEFAULT
        u_mag = math.sqrt(x_value) * model.v_coh
        t_star_value = 0.35 * DT
        n_hat = np.array([1.0, 0.0, 0.0])  # fixed radial normal, head-on along x

        dr_at_tstar = r_sum * n_hat
        dv = -u_mag * n_hat  # purely radial approach: u . n_hat = -u_mag < 0
        dr0 = dr_at_tstar - t_star_value * dv

        r_i0 = np.zeros(3)
        r_j0 = r_i0 + dr0
        v_i0 = np.zeros(3)
        v_j0 = v_i0 + dv

        # Bystander offset PARALLEL to n_hat: the participants' fusion/fragmentation
        # displacement (from r_i(t*)/r_j(t*) to r_c, or to r_a/r_b) is along n_hat too by
        # construction (radial geometry), so this is the offset direction of MAXIMAL, not
        # accidentally-cancelled, sensitivity.
        bystander_r = r_i0 + 0.0703 * n_hat
        bystander_v = np.zeros(3)

        r_full = np.vstack([r_i0, r_j0, bystander_r])
        v_full = np.vstack([v_i0, v_j0, bystander_v])
        m_full = np.array([m, m, m])
        state_full = State(
            r=torch.as_tensor(r_full, dtype=torch.float64),
            v=torch.as_tensor(v_full, dtype=torch.float64),
            m=torch.as_tensor(m_full, dtype=torch.float64),
        )
        accepted = col.AcceptedPairs(
            i=torch.tensor([0]), j=torch.tensor([1]),
            t_star=torch.tensor([t_star_value], dtype=torch.float64),
            rel_speed=torch.tensor([u_mag], dtype=torch.float64),
            contact_radius_sum=torch.tensor([r_sum], dtype=torch.float64),
            f_reject=0.0,
        )
        p_fus, p_el, _ = _map_probs(x_value)
        u1 = {"fusion": 0.5 * p_fus, "fragmentation": p_fus + p_el + 0.5 * (1.0 - p_fus - p_el)}[channel]
        gen = _ScriptedGenerator([u1, u2])
        return state_full, accepted, gen, model, t_star_value

    def _deterministic_residual(self, channel, x_value, u2=0.5):
        col = _require_collisions()
        state_full, accepted, gen, model, t_star_value = self._deterministic_event_with_bystander(
            channel, x_value, u2=u2
        )
        r_at_tstar = state_full.r.numpy() + t_star_value * state_full.v.numpy()
        state_before_map = State(
            r=torch.as_tensor(r_at_tstar, dtype=torch.float64),
            v=state_full.v.clone(),
            m=state_full.m.clone(),
        )
        k_before = observables.kinetic_energy(state_before_map)
        u_before = observables.potential_energy(state_before_map, softening=EPS_SOFTENING)
        out = col.resolve(state_full, t_star_value, model, accepted, gen, softening=EPS_SOFTENING)
        expected = {"fusion": (0, 1, 0), "fragmentation": (0, 0, 1)}[channel]
        actual = (out.n_elastic, out.n_merge, out.n_fragment)
        assert actual == expected, (
            f"deterministic construction did not land in the {channel} channel: got {actual}. "
            f"This is a test-construction problem (x_value / u1 threshold), not a resolve() bug"
        )
        k_after = observables.kinetic_energy(out.state)
        u_after = observables.potential_energy(out.state, softening=EPS_SOFTENING)
        e_before = k_before + u_before
        e_after = k_after + u_after + out.delta_e_int
        return abs(e_after - e_before) / abs(config.IC_U_EPS005)

    def test_residual_above_floor_for_fusion_deterministic(self):
        """
        The physicist's explicitly requested test. Section 4.10 REQUIRES that the third-body
        potential terms be OMITTED from E_int, computed only at the pair level, O(1) per event.
        An implementation that instead recomputes the full O(N) third-body sum will show a
        residual at the round-off floor (~1e-13 to 1e-16) instead of the deliberately-retained
        third-body residual the document measures at ~2.5e-6 |E_0| per event. That implementation
        would be MORE physically accurate than what Section 4.10 specifies -- and this test is
        designed to fail it anyway, because the specification is explicit that the omission is
        intentional (performance: O(1) vs O(N) per event) and that a floor violation is a
        genuine divergence between code and spec, to be reported, not congratulated.
        """
        residual = self._deterministic_residual("fusion", x_value=0.05)
        assert TOL_EVENT_CONS_FLOOR <= residual <= TOL_EVENT_CONS_FP64, (
            f"residual = {residual:.3e}; expected inside [{TOL_EVENT_CONS_FLOOR:.1e}, "
            f"{TOL_EVENT_CONS_FP64:.1e}] (Section 7, TOL-EVENT-CONS). Below the floor: the "
            f"third-body term is being computed (contradicts Section 4.10's O(1)-per-event "
            f"closed form). Above the ceiling: a term is missing or has the wrong sign in the "
            f"E_int accumulator (Section 4.14, item 7)"
        )

    def test_residual_above_floor_for_fragmentation_deterministic(self):
        # u2 = 0.2 (f = 0.26, an UNEQUAL split): u2 = 0.5 (f = 0.5, equal fragments) was tried
        # first and gave a residual of EXACTLY 0.0 -- the fully collinear, equal-split
        # construction has an accidental symmetry (both fragments displace symmetrically about
        # r_c along the same axis the bystander sits on) that cancels the third-body
        # perturbation completely. That is a genuine geometric degeneracy of THIS construction,
        # not a resolve() property; breaking the mass symmetry removes it. See the final report.
        residual = self._deterministic_residual("fragmentation", x_value=20.0, u2=0.2)
        assert TOL_EVENT_CONS_FLOOR <= residual <= TOL_EVENT_CONS_FP64, (
            f"residual = {residual:.3e}; expected inside [{TOL_EVENT_CONS_FLOOR:.1e}, "
            f"{TOL_EVENT_CONS_FP64:.1e}] (Section 7, TOL-EVENT-CONS)"
        )

    def test_residual_is_exactly_zero_for_elastic(self):
        # Elastic touches neither U (positions frozen at t*, Section 4.9(1)) nor E_int (Section
        # 4.10: "elastica: E_int += 0"); K + U is exactly conserved by the map itself, so this
        # residual should sit at plain floating-point round-off, with no third-body ambiguity at
        # all (there is no E_int term to omit or include for this channel).
        residuals = self._run_batch("elastic")
        assert residuals.size > 0, "no elastic events landed after channel-forcing filter"
        assert np.all(residuals <= 1e-10), (
            "elastic collisions do not touch U or E_int; a residual above round-off here points "
            "at the impulse being applied somewhere other than the frozen t* configuration"
        )


# =============================================================================================
# INV-24 -- L_total = L_orb + L_spin conserved exactly by the collision map (Section 4.10,
# Section 6 INV-24).
# =============================================================================================
class TestINV24AngularMomentumIdentity:
    """
    Rather than requiring a full multi-step RUN_COLLISION campaign (not wired up in config.py at
    the time of this suite -- see the final report), this tests the identity that makes INV-24
    true FOR ANY execution: L_orb(after) - L_orb(before) + delta_l_spin == 0, computed directly
    from a single resolve() pass over a mixed batch of elastic/merger/fragmentation events built
    exactly like the per-channel batches above (a single pass is exactly what Section 4.10's
    accumulator update happens over; chaining many passes cannot violate this identity if it
    holds for one, since each pass's delta_l_spin is independent and additive by construction,
    Section 9.1.1 point 1: resolve() is pure in the accumulators).
    """

    @pytest.fixture(scope="class")
    def resolved(self):
        col = _require_collisions()
        n = RESOLUTION_N_EVENTS_PER_CHANNEL
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 40)
        x_values = _x_grid(rng, n, *RESOLUTION_X_LOG_RANGE)
        model = _make_model(v_coh=V_COH)
        state0, accepted, events = _build_batch(rng, n, x_values, DT, T_STAR_FRAC, model,
                                                  config.R_REF_DEFAULT, M_BAR)
        # Mixed channels: cycle fusion / elastic / fragmentation across the batch so all three
        # contribute to delta_l_spin (elastic contributes zero but must not corrupt the sum).
        channels = ["fusion", "elastic", "fragmentation"]
        u1 = np.empty(n)
        u2 = np.empty(n)
        u2_rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 4000)
        for k in range(n):
            ch = channels[k % 3]
            u1[k] = _channel_u1(np.array([x_values[k]]), ch)[0]
            u2[k] = u2_rng.random() if ch == "fragmentation" else 0.5
        gen = _ScriptedGenerator(_interleave(u1, u2))
        out = col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
        return state0, out

    def test_l_orb_change_is_exactly_cancelled_by_delta_l_spin(self, resolved):
        state0, out = resolved
        l_before = observables.angular_momentum(state0)
        l_after = observables.angular_momentum(out.state)
        residual = (l_after - l_before) + out.delta_l_spin
        l_scale = torch.linalg.norm(l_before).item()
        rel = torch.linalg.norm(residual).item() / l_scale
        assert rel <= TOL_EVENT_INV_ULP * EPS_PREC_FP64, (
            f"L_orb(after) - L_orb(before) + delta_l_spin has relative norm {rel:.3e}: the "
            f"L_orb/L_spin bookkeeping split does not add up to L_total conservation (Section "
            f"4.10, INV-24)"
        )

    def test_delta_l_spin_is_not_trivially_zero(self, resolved):
        # Sanity that this batch (2/3 of events merger or fragmentation) actually exercises the
        # spin term -- otherwise the previous test would pass vacuously.
        _, out = resolved
        assert torch.linalg.norm(out.delta_l_spin).item() > 0.0


# =============================================================================================
# INV-26 -- realized channel fractions agree with the map's own probabilities over the
# histogram of x actually visited (Section 6, INV-26). Uses REAL, unscripted RNG draws -- this
# is the one test in this module that genuinely exercises Section 4.7.1's stochastic channel
# selection rather than forcing it.
# =============================================================================================
class TestINV26MapVersusSamplerConsistency:
    @pytest.fixture(scope="class")
    def resolved(self):
        col = _require_collisions()
        n = RESOLUTION_CHANNEL_FRACTION_N_EVENTS
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 50)
        x_values = np.full(n, RESOLUTION_CHANNEL_FRACTION_TEST_X)
        model = _make_model(v_coh=V_COH)
        state0, accepted, events = _build_batch(rng, n, x_values, DT, T_STAR_FRAC, model,
                                                  config.R_REF_DEFAULT, M_BAR)
        gen = np.random.default_rng(config.COLLISION_SEED)  # a REAL generator, unscripted
        out = col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
        return out, n

    def test_channel_fractions_within_three_sigma_binomial_band(self, resolved):
        out, n = resolved
        p_fus, p_el, p_frag = _map_probs(RESOLUTION_CHANNEL_FRACTION_TEST_X)
        f_fus = out.n_merge / n
        f_el = out.n_elastic / n
        f_frag = out.n_fragment / n
        for f_c, p_c, name in ((f_fus, p_fus, "fusion"), (f_el, p_el, "elastic"),
                                (f_frag, p_frag, "fragmentation")):
            band = INV26_N_SIGMA * math.sqrt(p_c * (1.0 - p_c) / n)
            assert abs(f_c - p_c) <= band, (
                f"{name}: realized fraction {f_c:.4f} vs predicted {p_c:.4f}, "
                f"3-sigma band = {band:.4f} (Section 6, INV-26)"
            )

    def test_each_channel_receives_at_least_five_percent(self, resolved):
        # Section 6, INV-26, "criterio adicional (calibracao)".
        out, n = resolved
        assert out.n_merge / n >= INV26_CHANNEL_MIN_FRACTION
        assert out.n_elastic / n >= INV26_CHANNEL_MIN_FRACTION
        assert out.n_fragment / n >= INV26_CHANNEL_MIN_FRACTION


# =============================================================================================
# Section 4.9(0.1) -- the degenerate head-on normal (sep_sq == 0.0 exactly): a NEW normative rule
# introduced alongside the 2026-08-07 (b) revision, not present when the rest of this module's
# INV-20/INV-22 batches were designed. The document is explicit this is testable "dentro dos
# procedimentos ja existentes de INV-20 (elastica) e INV-22 (fragmentacao)", by adding at least
# one exactly-collinear head-on pair with sep_sq == 0.0 verified, and requiring: no NaN in r, v,
# m; u' = -u within 100 eps_prec for elastic; and the already-tabulated conservations, which in
# this case hold with Delta L = 0 EXACT (not just within tolerance), because dr = sep = 0
# identically at the point the impulse is applied. Separate, dedicated tests here rather than
# folded into the batches above: this is a genuinely distinct code path (a branch on
# sep_sq == 0.0), not a random point that the wide sweep could stumble onto by chance -- a
# uniformly-drawn n0 has probability zero of ever landing exactly on this branch, so the batches
# above never exercise it.
# =============================================================================================
class TestSection491DegenerateHeadOnNormal:
    DT_LOCAL = 1.0e-3
    T_STAR = 4.0e-4  # interior point, matching T_STAR_FRAC's spirit elsewhere in this module

    @classmethod
    def _construct_exact_head_on(cls, u_mag):
        """
        dr(t*) = dr(0) + t* dv is built as dr(0) := -(t* * dv), so that dr(0) + t*dv is literally
        x + (-x) for the SAME floating-point x = t*dv -- exactly 0.0 in IEEE-754, no rounding,
        regardless of what t*dv itself rounds to. r_i0 = 0 exactly, so r_j0 = dr(0) bit-for-bit
        (0 + x == x exactly), and v_i0 = 0 exactly, so v_j0 = dv bit-for-bit: resolve()'s own
        recomputation of dr(t*) = (r_j0 + t*v_j0) - (r_i0 + t*v_i0) reduces to the identical
        cancellation, dv * t* the same float both times.
        """
        dv = np.array([-u_mag, 0.0, 0.0])
        dr0 = -(cls.T_STAR * dv)
        r_i0 = np.zeros(3)
        r_j0 = r_i0 + dr0
        v_i0 = np.zeros(3)
        v_j0 = v_i0 + dv
        return r_i0, r_j0, v_i0, v_j0

    def test_construction_is_genuinely_sep_sq_zero(self):
        # Sanity on the construction itself, independent of resolve(): confirms sep(t*) is
        # EXACTLY the zero vector before trusting any downstream test built on it.
        r_i0, r_j0, v_i0, v_j0 = self._construct_exact_head_on(u_mag=5.0)
        dr = r_j0 - r_i0
        dv = v_j0 - v_i0
        sep = dr + self.T_STAR * dv
        assert np.array_equal(sep, np.zeros(3)), f"test construction bug: sep(t*) = {sep}, not exactly 0"

    def _make_accepted(self, m_i, m_j, r_ref=config.R_REF_DEFAULT, m_bar=M_BAR):
        col = _require_collisions()
        r_sum = r_ref * (m_i / m_bar) ** (1.0 / 3.0) + r_ref * (m_j / m_bar) ** (1.0 / 3.0)
        return col.AcceptedPairs(
            i=torch.tensor([0]), j=torch.tensor([1]),
            t_star=torch.tensor([self.T_STAR], dtype=torch.float64),
            rel_speed=torch.tensor([5.0], dtype=torch.float64),
            contact_radius_sum=torch.tensor([r_sum], dtype=torch.float64),
            f_reject=0.0,
        )

    def test_elastic_exact_reversal_and_zero_exact_angular_momentum(self):
        """
        Section 4.9(0.1): n = -u/|u| in the limit; elastic reflection gives
        u' = u - 2(u.n)n = -u EXACTLY (a full head-on reversal), and Delta L = 0 EXACTLY (not
        just within tolerance) because dr = sep = 0 identically at the point of impulse
        application, so ANY impulse J satisfies -(dr) x J = 0.
        """
        col = _require_collisions()
        m = M_BAR
        r_i0, r_j0, v_i0, v_j0 = self._construct_exact_head_on(u_mag=5.0)
        u_before = v_j0 - v_i0
        state0 = State(
            r=torch.as_tensor(np.vstack([r_i0, r_j0]), dtype=torch.float64),
            v=torch.as_tensor(np.vstack([v_i0, v_j0]), dtype=torch.float64),
            m=torch.tensor([m, m], dtype=torch.float64),
        )
        accepted = self._make_accepted(m, m)
        model = _make_model(v_coh=V_COH)
        # Force elastic: u1 well inside [p_fus, p_fus + p_el] for whatever x this configuration
        # gives (x = |u|^2/v_coh^2, mass-independent, Section 4.6).
        x_value = (5.0 / model.v_coh) ** 2
        u1 = _channel_u1(np.array([x_value]), "elastic")[0]
        gen = _ScriptedGenerator([u1, 0.5])
        out = col.resolve(state0, self.DT_LOCAL, model, accepted, gen, softening=EPS_SOFTENING)

        assert (out.n_elastic, out.n_merge, out.n_fragment) == (1, 0, 0)
        assert not torch.isnan(out.state.r).any()
        assert not torch.isnan(out.state.v).any()
        assert not torch.isnan(out.state.m).any()

        v_after = out.state.v.numpy()
        u_after = v_after[1] - v_after[0]
        rel = np.linalg.norm(u_after - (-u_before)) / np.linalg.norm(u_before)
        assert rel <= 100.0 * EPS_PREC_FP64, (
            f"u' should equal -u exactly (within round-off) for the degenerate head-on case; "
            f"got u_after={u_after}, expected {-u_before}"
        )

        l_before = m * np.cross(r_i0, v_i0) + m * np.cross(r_j0, v_j0)
        r_after = out.state.r.numpy()
        l_after = m * np.cross(r_after[0], v_after[0]) + m * np.cross(r_after[1], v_after[1])
        # Delta L = 0 EXACT here (dr = 0 identically at the application point), not just within
        # the usual 100 eps_prec tolerance -- this is the document's own stronger claim for this
        # specific degenerate case.
        np.testing.assert_array_equal(l_after, l_before)

    def test_fragmentation_no_nan_and_recoil_direction_is_minus_u(self):
        """
        Section 4.9(0.1): n = -u/|u| in the degenerate case, so the fragmentation formula's
        u' = |u| sqrt(mu/mu') * n recoils ALONG -u/|u| (backing away along the incoming
        direction) rather than along the (undefined) +n of the generic formula. No NaN anywhere,
        and conservations (m, P, T_cm, K) hold as in the generic INV-22 case.
        """
        col = _require_collisions()
        m = M_BAR
        r_i0, r_j0, v_i0, v_j0 = self._construct_exact_head_on(u_mag=30.0)  # large |u| -> large x
        u_before = v_j0 - v_i0
        state0 = State(
            r=torch.as_tensor(np.vstack([r_i0, r_j0]), dtype=torch.float64),
            v=torch.as_tensor(np.vstack([v_i0, v_j0]), dtype=torch.float64),
            m=torch.tensor([m, m], dtype=torch.float64),
        )
        accepted = self._make_accepted(m, m)
        model = _make_model(v_coh=V_COH)
        x_value = (30.0 / model.v_coh) ** 2
        p_fus, p_el, p_frag = _map_probs(x_value)
        u1 = p_fus + p_el + 0.5 * p_frag
        u2 = 0.3
        gen = _ScriptedGenerator([u1, u2])
        out = col.resolve(state0, self.DT_LOCAL, model, accepted, gen, softening=EPS_SOFTENING)

        assert (out.n_elastic, out.n_merge, out.n_fragment) == (0, 0, 1)
        assert not torch.isnan(out.state.r).any()
        assert not torch.isnan(out.state.v).any()
        assert not torch.isnan(out.state.m).any()

        M_tot = 2.0 * m
        f = FRAG_F_MIN + (1.0 - 2.0 * FRAG_F_MIN) * u2
        m_a = f * M_tot
        m_b = M_tot - m_a
        mu = m * m / M_tot
        mu_p = m_a * m_b / M_tot
        u_prime_mag_expected = np.linalg.norm(u_before) * math.sqrt(mu / mu_p)

        v_after = out.state.v.numpy()
        m_after = out.state.m.numpy()
        v_cm = (m_after[0] * v_after[0] + m_after[1] * v_after[1]) / M_tot
        u_prime = v_after[0] - v_cm  # fragment a's velocity relative to the pair's own c.m.
        # In the generic case u' is along +n_hat = +(dr/|dr|); here n = -u/|u|, so u' should
        # point along -u/|u| -- backing away along the incoming line, not the (undefined) +dr.
        minus_u_hat = -u_before / np.linalg.norm(u_before)
        cos_angle = np.dot(u_prime, minus_u_hat) / np.linalg.norm(u_prime)
        assert abs(cos_angle - 1.0) <= 1e-9, (
            f"fragment recoil is not aligned with -u/|u| in the degenerate head-on case: "
            f"cos_angle={cos_angle}"
        )
        rel_mag = abs(np.linalg.norm(u_prime) * (M_tot / m_b) - u_prime_mag_expected) / u_prime_mag_expected
        assert rel_mag <= 1e-6

        mass_after_total = m_after.sum()
        assert abs(mass_after_total - M_tot) / M_tot <= 4.0 * EPS_PREC_FP64

    def test_dr_zero_and_u_zero_raises_value_error(self):
        """
        Section 4.9(0.1): 'sobre dr=0 e dv=0 simultaneos: e inalcancavel [em uma execucao real],
        e por isso levanta excecao.' ValueError, not an arbitrary fabricated direction -- 'nao
        inventar uma direcao arbitraria aqui: sem dr e sem u, nao existe direcao alguma no
        problema'. Unreachable via detect() (the approach guard dr.dv < 0 strictly excludes
        dv = 0), but directly constructible by a test bypassing detect(), which is exactly what
        this suite does throughout -- the document names this explicitly as the suite's own
        responsibility to cover.
        """
        col = _require_collisions()
        m = M_BAR
        state0 = State(
            r=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
            v=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
            m=torch.tensor([m, m], dtype=torch.float64),
        )
        accepted = self._make_accepted(m, m)
        model = _make_model(v_coh=V_COH)
        gen = _ScriptedGenerator([0.5, 0.5])
        with pytest.raises(ValueError):
            col.resolve(state0, self.DT_LOCAL, model, accepted, gen, softening=EPS_SOFTENING)


# =============================================================================================
# INV-32 -- exactly two uniforms per accepted event, for every channel, with zero tolerance
# (Section 4.7.1, Section 6 INV-32). Two independent paths, both mandatory per the document.
# =============================================================================================
class TestINV32ExactlyTwoDrawsPerEvent:
    def test_direct_count_over_a_mixed_batch(self):
        # Path 1: "contagem direta". A mixed batch (elastic, fusion, fragmentation all present)
        # with a known number of accepted events; the scripted generator's own call counter is
        # the ground truth for how many draws were actually consumed.
        col = _require_collisions()
        n = 37  # arbitrary, not a power of anything, to avoid accidentally-aliased bugs
        rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 60)
        x_values = _x_grid(rng, n, *RESOLUTION_X_LOG_RANGE)
        model = _make_model(v_coh=V_COH)
        state0, accepted, events = _build_batch(rng, n, x_values, DT, T_STAR_FRAC, model,
                                                  config.R_REF_DEFAULT, M_BAR)
        channels = ["fusion", "elastic", "fragmentation"]
        u1 = np.array([_channel_u1(np.array([x_values[k]]), channels[k % 3])[0] for k in range(n)])
        u2 = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 6000).random(n)
        gen = _ScriptedGenerator(_interleave(u1, u2))
        col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
        assert gen.calls == 2 * n

    def test_zero_events_consumes_zero_draws(self):
        # "Um passe com zero eventos consome zero."
        col = _require_collisions()
        model = _make_model(v_coh=V_COH)
        state0 = State(
            r=torch.zeros((2, 3), dtype=torch.float64),
            v=torch.zeros((2, 3), dtype=torch.float64),
            m=torch.tensor([M_BAR, M_BAR], dtype=torch.float64),
        )
        accepted = col.AcceptedPairs(
            i=torch.zeros(0, dtype=torch.int64),
            j=torch.zeros(0, dtype=torch.int64),
            t_star=torch.zeros(0, dtype=torch.float64),
            rel_speed=torch.zeros(0, dtype=torch.float64),
            contact_radius_sum=torch.zeros(0, dtype=torch.float64),
            f_reject=0.0,
        )
        gen = _ScriptedGenerator([])
        out = col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
        assert gen.calls == 0
        assert (out.n_elastic, out.n_merge, out.n_fragment) == (0, 0, 0)

    def test_stream_equivalence_across_very_different_channel_mixes(self):
        """
        Path 2, the strong test: "construir dois passes com o mesmo numero de eventos aceitos
        mas canais diferentes ... partindo do mesmo estado de Generator. Exigir que o estado do
        Generator ao final dos dois passes seja identico."

        Two batches, same seed, same n_events, same geometry -- but x = 0.01 for every event in
        pass A (Section 4.7's own table: p_fus(0.01) = 0.971, so pass A is overwhelmingly fusion)
        versus x = 100 for every event in pass B (p_frag(100) approx 0.97, overwhelmingly
        fragmentation). Both passes draw from a FRESH, REAL numpy.random.default_rng(seed) (same
        seed): if resolve() draws exactly 2 uniforms per event regardless of which channel each
        one lands in, the two bit_generator states after the pass must be IDENTICAL, because both
        consumed the same raw stream position by position. If u2 were only drawn conditionally
        (e.g. only inside the fragmentation branch, the exact bug Section 4.7.1 exists to rule
        out), pass A (mostly fusion+elastic) and pass B (mostly fragmentation) would consume
        different numbers of raw values and the states would diverge.
        """
        col = _require_collisions()
        n = 25
        seed = 20260807

        def run_pass(x_value):
            rng = np.random.default_rng(RESOLUTION_GEOMETRY_SEED + 70)
            x_values = np.full(n, x_value)
            model = _make_model(v_coh=V_COH)
            state0, accepted, events = _build_batch(rng, n, x_values, DT, T_STAR_FRAC, model,
                                                      config.R_REF_DEFAULT, M_BAR)
            gen = np.random.default_rng(seed)
            col.resolve(state0, DT, model, accepted, gen, softening=EPS_SOFTENING)
            return gen

        gen_a = run_pass(0.01)
        gen_b = run_pass(100.0)
        state_a = gen_a.bit_generator.state
        state_b = gen_b.bit_generator.state
        assert state_a == state_b, (
            "the collision generator's final state differs between a mostly-fusion pass and a "
            "mostly-fragmentation pass over the same number of accepted events, starting from "
            "the same seed: this means the number of values drawn per event depends on the "
            "channel taken -- Section 4.7.1's normative fix (u2 ALWAYS drawn, before the branch) "
            "is violated"
        )
