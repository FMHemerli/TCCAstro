"""
nbody.observables.half_mass_radius against docs/simulacao-estocastica.md Section 5.3 and
Section 6, INV-29: half_mass_radius must be the MASS median distance from the center of mass,
not the count median (sorted_d[N//2 - 1]), which is only correct for equal masses and even N.

This file does not touch tests/test_observables.py, which already exercises half_mass_radius
under the OLD (pre-Section-5.3) equal-mass reference values; those remain valid (Section 5.3:
"Nenhum valor publicado muda"). This file adds the mass-median-specific checks.

Reference implementation (independent of any closed form, "ordenar, acumular, procurar" per
Section 5.3):
    c    = center of mass
    d_i  = |r_i - c|
    pi   = permutation sorting d ascending
    C_k  = sum_{l<=k} m_pi(l)   (cumulative mass)
    k*   = min{k : C_k >= M_real/2}
    r_half = d_pi(k*)
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody.observables import half_mass_radius  # noqa: E402
from nbody.state import State  # noqa: E402

from tolerances import (  # noqa: E402
    INV29A_EQUAL_MASS_REL_TOL,
    INV29B_MASS_RATIO_FOR_SEPARATION,
    INV29B_REL_TOL,
)


def reference_half_mass_radius(r: np.ndarray, m: np.ndarray) -> float:
    total_m = m.sum()
    com = (m[:, None] * r).sum(axis=0) / total_m
    d = np.linalg.norm(r - com, axis=1)
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(m[order])
    k_star = int(np.searchsorted(cum, total_m / 2.0, side="left"))
    k_star = min(k_star, len(d) - 1)
    return float(d[order[k_star]])


def reference_count_median_radius(r: np.ndarray, m: np.ndarray) -> float:
    """The OLD (pre-Section-5.3) formula: count median of distances, index N//2 - 1 (0-based),
    valid only for equal masses and even N.

    The centre is the MASS-WEIGHTED centre of mass, not r.mean(axis=0). Section 5.3's normative
    pseudocode opens with `c = centro de massa`, and property 1 claims only that the INDEX rule
    reduces exactly (k* = N/2 in 1-based indexing) -- it claims nothing about reproducing a
    different centroid arithmetic. For equal masses the two centres are mathematically identical
    but numerically differ: sum(r)/n rounds differently from sum(m*r)/sum(m). Using the unweighted
    centroid here made this test demand bit-identity across that unrelated difference, and it
    failed at N=100 by 2 ULP while the index rule was exactly right.
    """
    total_m = m.sum()
    com = (m[:, None] * r).sum(axis=0) / total_m
    d = np.sort(np.linalg.norm(r - com, axis=1))
    n = len(d)
    return float(d[n // 2 - 1])


def _random_positions(n, seed, scale=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=(n, 3))


def _state(r, m, v=None):
    r_t = torch.tensor(r, dtype=torch.float64)
    m_t = torch.tensor(m, dtype=torch.float64)
    v_t = torch.zeros_like(r_t) if v is None else torch.tensor(v, dtype=torch.float64)
    return State(r=r_t, v=v_t, m=m_t)


class TestINV29aEqualMassCompatibility:
    """For equal masses and even N the mass median reduces to the count median (Section 5.3,
    property 1: C_k = k*m, so C_k >= N*m/2 <=> k >= N/2, giving index N//2 - 1 0-based)."""

    def test_equal_mass_index_rule_is_exact(self):
        # The claim of property 1 is about INDEX SELECTION, and here it is tested exactly, with
        # no floating-point slack anywhere: 8 particles on the x axis at +-1, +-2, +-3, +-4 with
        # equal masses. The configuration is symmetric so the centre of mass is exactly the
        # origin, and every distance is an exact binary fraction, so both formulas are computed
        # in exact arithmetic and equality is meaningful rather than lucky.
        #
        #   sorted distances: 1, 1, 2, 2, 3, 3, 4, 4      index N//2 - 1 = 3  =>  2.0
        #   equal masses m: C_k = (k+1)m, C_3 = 4m = M/2  =>  k* = 3          =>  2.0
        x = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
        r = np.zeros((8, 3))
        r[:, 0] = x
        m = np.full(8, 3.0)

        got = half_mass_radius(_state(r, m))
        assert got == 2.0, (
            f"half_mass_radius={got!r}, expected exactly 2.0: with equal masses the half-mass "
            f"crossing must land on index N//2 - 1 = 3 of the sorted distances "
            f"[1, 1, 2, 2, 3, 3, 4, 4] (Section 5.3, property 1)"
        )

    @pytest.mark.parametrize("n", [6, 100, 1000])
    def test_equal_mass_even_n_agrees_with_count_median_reference(self, n):
        # Same claim on generic random positions, against the independent numpy reference. Here
        # equality is only up to INV29A_EQUAL_MASS_REL_TOL, and deliberately so: the two sides
        # sum N terms in different orders. See tolerances.py for the bound and why demanding
        # bit-identity across two implementations was the wrong assertion -- it failed at N=100
        # by 2 ULP while the index rule under test was exactly right.
        r = _random_positions(n, seed=1000 + n)
        m = np.full(n, 3.0)
        state = _state(r, m)

        got = half_mass_radius(state)
        expected = reference_count_median_radius(r, m)
        rel = abs(got / expected - 1.0)
        assert rel <= INV29A_EQUAL_MASS_REL_TOL, (
            f"N={n}, equal masses: half_mass_radius={got!r} vs count-median reference "
            f"{expected!r}, rel dev {rel:.3e} > {INV29A_EQUAL_MASS_REL_TOL} -- the index rule of "
            f"Section 5.3 property 1 did not reduce to N//2 - 1"
        )


class TestINV29bMassMedianForUnequalMasses:
    def test_unequal_masses_matches_independent_reference(self):
        n = 200
        rng = np.random.default_rng(4242)
        r = _random_positions(n, seed=4242)
        # Mass ratio matching Section 5.3's worked example (m_max/m_min = 223).
        m = rng.uniform(1.0, float(INV29B_MASS_RATIO_FOR_SEPARATION), size=n)
        state = _state(r, m)

        got = half_mass_radius(state)
        expected = reference_half_mass_radius(r, m)
        rel = abs(got / expected - 1.0)
        assert rel <= INV29B_REL_TOL, (
            f"half_mass_radius={got:.15e} vs reference={expected:.15e}, "
            f"rel dev {rel:.3e} > {INV29B_REL_TOL}"
        )

    def test_mass_median_differs_materially_from_count_median(self):
        # Discriminative power check: this is what proves the test above would have FAILED
        # against the pre-Section-5.3 (count-median) formula.
        #
        # Built analytically rather than sampled. With masses drawn independently of position the
        # two medians converge as N grows -- measured separations over random draws range from
        # 0.06% to 12% depending only on the seed, so any threshold on a single random draw is a
        # seed lottery, and picking one from measurement is exactly the tolerance-fitting that
        # tests/tolerances.py forbids. The configuration below has an exact hand-computable
        # answer and no free parameter.
        #
        #   8 particles on the x axis at +-1, +-2, +-3, +-4; m(+-1) = 100, all others 1.
        #   The mass distribution is symmetric, so the centre of mass is exactly the origin.
        #   distances sorted: 1, 1, 2, 2, 3, 3, 4, 4     masses in that order: 100, 100, 1, ...
        #   M_total = 206, M/2 = 103. C_0 = 100 < 103, C_1 = 200 >= 103  =>  k* = 1  =>  r = 1.
        #   Count median takes index N//2 - 1 = 3        =>  r = 2.
        #
        # The two formulas differ by a factor of exactly 2 here: the mass median follows the mass,
        # the count median follows the head count.
        x = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
        r = np.zeros((8, 3))
        r[:, 0] = x
        m = np.where(np.abs(x) == 1.0, 100.0, 1.0)
        state = _state(r, m)

        mass_median = half_mass_radius(state)
        count_median = reference_count_median_radius(r, m)

        assert mass_median == 1.0, (
            f"mass median is {mass_median!r}, expected exactly 1.0: the two particles at |x| = 1 "
            f"carry 200 of 206 total mass, so the half-mass crossing happens at the second one"
        )
        assert count_median == 2.0, (
            f"count median is {count_median!r}, expected exactly 2.0 (index N//2 - 1 = 3 of the "
            f"sorted distances [1, 1, 2, 2, 3, 3, 4, 4])"
        )

    def test_reference_implementation_reduces_to_count_median_for_equal_masses(self):
        # Self-check of THIS file's reference implementation, independent of nbody: with equal
        # masses and even N it must reduce to the same count-median formula (Section 5.3,
        # property 1), verified here purely in numpy with no nbody call at all.
        n = 40
        r = _random_positions(n, seed=99)
        m = np.full(n, 7.0)
        assert reference_half_mass_radius(r, m) == reference_count_median_radius(r, m)


class TestINV29cDeadSlotsExcludedFromMedian:
    """
    Section 5.3, properties 2-3: a slot with m=0 contributes 0 to the cumulative mass (falls
    out of the median "on its own"), and k* can never land on a dead slot, because C_{k*} =
    C_{k*-1} would then already satisfy the threshold, contradicting k*'s minimality. Full
    m=0-inertia coverage across all observables is INV-27's job (out of this file's scope, per
    the collision-invariant boundary); this test covers the half_mass_radius-specific claim
    that Section 5.3 makes and INV-29(c) names explicitly.
    """

    def test_dead_slot_coincident_with_a_live_particle_does_not_change_the_result(self):
        n = 60
        r = _random_positions(n, seed=7)
        rng = np.random.default_rng(7)
        m = rng.uniform(1.0, 50.0, size=n)
        state_before = _state(r, m)
        rh_before = half_mass_radius(state_before)

        # Most adverse placement per Section 5.1: the dead slot exactly coincides with a live
        # particle (here, particle 0).
        r_with_dead = np.concatenate([r, r[0:1]], axis=0)
        m_with_dead = np.concatenate([m, [0.0]])
        state_after = _state(r_with_dead, m_with_dead)
        rh_after = half_mass_radius(state_after)

        assert rh_before == rh_after, (
            f"inserting a dead (m=0) slot changed half_mass_radius: {rh_before!r} -> "
            f"{rh_after!r}; Section 5.3 requires this to be exact, not approximate"
        )

    def test_multiple_dead_slots_at_arbitrary_positions_do_not_change_the_result(self):
        n = 60
        r = _random_positions(n, seed=11)
        rng = np.random.default_rng(11)
        m = rng.uniform(1.0, 50.0, size=n)
        state_before = _state(r, m)
        rh_before = half_mass_radius(state_before)

        extra_r = _random_positions(5, seed=12, scale=100.0)  # far away, not coincident
        r_with_dead = np.concatenate([r, extra_r], axis=0)
        m_with_dead = np.concatenate([m, np.zeros(5)])
        state_after = _state(r_with_dead, m_with_dead)
        rh_after = half_mass_radius(state_after)

        assert rh_before == rh_after
