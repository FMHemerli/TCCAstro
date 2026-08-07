"""
nbody.populations mass sampling against docs/simulacao-estocastica.md Section 2 (spectrum,
CDF/inverse-CDF, conditioned construction, mean-mass closed form) and Section 6, INV-11 through
INV-14. docs/simulacao-estocastica.md Section 9.1 names sample_masses and mass_min_from_mean but
does not fix their parameter lists; see tests/_stage2_binding.py's module docstring for how this
suite calls them without reading src/nbody/populations.py.

INV-11: the pooled sample follows the truncated power law, tested separately in the body sector
[m_min, m_big] and the tail sector [m_big, m_max] (Section 2.6: the two sectors, conditioned on
K, are each i.i.d. truncated power law -- the UNCONDITIONED mixture is a different distribution
and must not be tested against).

INV-12: (a) structural -- every realization has 1 <= k <= 3 massive bodies; (b) statistical --
the empirical distribution of k matches the renormalized binomial.

INV-13: the single invariant the specification says can catch a missing Section 2.7 step-4
permutation while every other invariant in this document still passes. Written so that a
"tail masses always land in slots 0..k-1" implementation is caught hard: with k <= 3, the
max-mass particle would then always sit in slot 0, 1 or 2, producing massive concentration in
the first of ten equal-width bins over {0..999} and an enormous chi-square statistic.

INV-14: the closed-form m_min reproduces <m> = PARTICLE_MASS, verified by an independently
implemented Simpson quadrature over the Section 2.1 density -- NOT by re-evaluating the
document's own closed-form g(alpha, R) -- at alpha in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, which
includes both the alpha=1 and alpha=2 degenerate branches.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402

populations = pytest.importorskip(
    "nbody.populations", reason="nbody.populations does not exist yet (Section 9.1 amendment)"
)
sample_masses = getattr(populations, "sample_masses", None)
mass_min_from_mean = getattr(populations, "mass_min_from_mean", None)
if sample_masses is None:
    pytest.skip("nbody.populations.sample_masses is not exposed", allow_module_level=True)

from _stage2_binding import (  # noqa: E402
    ContractGap,
    call_mass_min_from_mean,
    call_sample_masses,
    chi2_sf,
    chi_square_gof,
    ks_pvalue,
    ks_statistic,
)

from tolerances import (  # noqa: E402
    INV11_KS_PVALUE_MIN,
    INV11_MEAN_REL_TOL,
    INV12A_N_REALIZATIONS,
    INV12B_CHI2_PVALUE_MIN,
    INV12B_PER_K_3SIGMA,
    INV13_CHI2_PVALUE_MIN,
    INV13_N_BINS,
    INV13_N_REALIZATIONS,
    INV14_ALPHA_SWEEP,
    INV14_MEAN_REL_TOL,
    INV14_SIMPSON_NODES,
    MASS_ALPHA,
    MASS_BIG,
    MASS_COND_MEAN,
    MASS_G_FACTOR,
    MASS_K_WEIGHTS,
    MASS_MAX,
    MASS_MIN,
    MASS_RATIO,
)

N = config.N_PARTICLES  # 1000, throughout Section 2's worked numbers
PARTICLE_MASS = config.PARTICLE_MASS


# -------------------------------------------------------------------------------------------
# Section 2.1/2.2 density and CDF, implemented independently of the document's closed-form
# g(alpha, R) mean -- used both for the KS reference CDF (INV-11) and the quadrature integrand
# (INV-14). The |alpha - 1| < 1e-12 guard mirrors the normative guard of Section 2.2 itself
# (an exact `== 1.0` compare is explicitly forbidden there), applied here independently.
# -------------------------------------------------------------------------------------------
def mass_density(m, alpha, m_min, m_max):
    m = np.asarray(m, dtype=np.float64)
    if abs(alpha - 1.0) < 1e-12:
        c = 1.0 / math.log(m_max / m_min)
        return c / m
    c = (1.0 - alpha) / (m_max ** (1.0 - alpha) - m_min ** (1.0 - alpha))
    return c * m ** (-alpha)


def mass_cdf(m, alpha, m_min, m_max):
    m = np.asarray(m, dtype=np.float64)
    if abs(alpha - 1.0) < 1e-12:
        return np.log(m / m_min) / math.log(m_max / m_min)
    return (m ** (1.0 - alpha) - m_min ** (1.0 - alpha)) / (
        m_max ** (1.0 - alpha) - m_min ** (1.0 - alpha)
    )


def _simpson(y, x):
    n_intervals = len(x) - 1
    if n_intervals % 2 != 0:
        raise ValueError("composite Simpson requires an even number of intervals")
    h = (x[-1] - x[0]) / n_intervals
    return h / 3.0 * (
        y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
    )


def mean_mass_via_quadrature(alpha, m_min, m_max, n_nodes=INV14_SIMPSON_NODES):
    """
    Independent numerical <m> = int m p(m) dm, via a log-substitution (t = ln m, dm = m dt) so
    that a UNIFORM grid in t -- required by composite Simpson -- still resolves the power-law
    integrand well across three decades of m. Uses only the Section 2.1 density formula, never
    the Section 2.3/2.4 closed-form mean.
    """
    t = np.linspace(math.log(m_min), math.log(m_max), n_nodes)
    m = np.exp(t)
    integrand = m * mass_density(m, alpha, m_min, m_max) * m  # extra m from dm = m dt
    return float(_simpson(integrand, t))


# -------------------------------------------------------------------------------------------
# INV-11
# -------------------------------------------------------------------------------------------
class TestINV11PowerLawShape:
    N_REALIZATIONS = 200
    SEED_BASE = 3_100_000  # arbitrary, fixed test-local seed scheme (spec requires fixed, not
    # a specific value: "Semente do teste fixa, de modo que a execucao e deterministica").

    @pytest.fixture(scope="class")
    def pooled(self):
        body, tail = [], []
        for i in range(self.N_REALIZATIONS):
            try:
                m = call_sample_masses(
                    sample_masses, N, self.SEED_BASE + i,
                    alpha=MASS_ALPHA, mass_ratio=MASS_RATIO, particle_mass=PARTICLE_MASS,
                )
            except ContractGap as exc:
                pytest.skip(str(exc))
            assert m.shape == (N,), f"realization {i}: expected {N} masses, got {m.shape}"
            is_tail = m > MASS_BIG
            tail.append(m[is_tail])
            body.append(m[~is_tail])
        return np.concatenate(body), np.concatenate(tail)

    def test_body_sector_ks(self, pooled):
        body, _ = pooled
        body_sorted = np.sort(body)
        f_a = mass_cdf(MASS_MIN, MASS_ALPHA, MASS_MIN, MASS_MAX)
        f_b = mass_cdf(MASS_BIG, MASS_ALPHA, MASS_MIN, MASS_MAX)

        def g(m):
            return (mass_cdf(m, MASS_ALPHA, MASS_MIN, MASS_MAX) - f_a) / (f_b - f_a)

        d = ks_statistic(body_sorted, g)
        p = ks_pvalue(d, len(body_sorted))
        assert p >= INV11_KS_PVALUE_MIN, (
            f"INV-11 body sector: KS p={p:.4f} (D={d:.5f}, n={len(body_sorted)}) below "
            f"{INV11_KS_PVALUE_MIN}"
        )

    def test_tail_sector_ks(self, pooled):
        _, tail = pooled
        tail_sorted = np.sort(tail)
        f_a = mass_cdf(MASS_BIG, MASS_ALPHA, MASS_MIN, MASS_MAX)
        f_b = mass_cdf(MASS_MAX, MASS_ALPHA, MASS_MIN, MASS_MAX)

        def g(m):
            return (mass_cdf(m, MASS_ALPHA, MASS_MIN, MASS_MAX) - f_a) / (f_b - f_a)

        d = ks_statistic(tail_sorted, g)
        p = ks_pvalue(d, len(tail_sorted))
        assert p >= INV11_KS_PVALUE_MIN, (
            f"INV-11 tail sector: KS p={p:.4f} (D={d:.5f}, n={len(tail_sorted)}) below "
            f"{INV11_KS_PVALUE_MIN}"
        )

    def test_bounds_respected(self, pooled):
        body, tail = pooled
        all_m = np.concatenate([body, tail])
        assert all_m.min() >= MASS_MIN * (1.0 - 1e-12)
        assert all_m.max() <= MASS_MAX * (1.0 + 1e-12)

    def test_pooled_mean_matches_conditioned_theory(self, pooled):
        body, tail = pooled
        all_m = np.concatenate([body, tail])
        theoretical_mean = MASS_COND_MEAN / N
        rel = abs(all_m.mean() / theoretical_mean - 1.0)
        assert rel <= INV11_MEAN_REL_TOL, (
            f"pooled sample mean {all_m.mean():.6e} vs conditioned theory "
            f"{theoretical_mean:.6e}: rel dev {rel:.4f} > {INV11_MEAN_REL_TOL}"
        )


# -------------------------------------------------------------------------------------------
# INV-12
# -------------------------------------------------------------------------------------------
class TestINV12MassiveBodyCount:
    SEED_BASE = 3_200_000

    @pytest.fixture(scope="class")
    def k_values(self):
        ks = []
        for i in range(INV12A_N_REALIZATIONS):
            try:
                m = call_sample_masses(
                    sample_masses, N, self.SEED_BASE + i,
                    alpha=MASS_ALPHA, mass_ratio=MASS_RATIO, particle_mass=PARTICLE_MASS,
                )
            except ContractGap as exc:
                pytest.skip(str(exc))
            ks.append(int(np.sum(m > MASS_BIG)))
        return np.asarray(ks)

    def test_a_every_realization_has_one_to_three_massive_bodies(self, k_values):
        violations = np.sum((k_values < 1) | (k_values > 3))
        assert violations == 0, (
            f"{violations}/{len(k_values)} realizations violate 1 <= k <= 3 "
            f"(k distribution: {np.bincount(k_values)})"
        )

    def test_b_k_distribution_matches_renormalized_binomial_chi2(self, k_values):
        counts = np.array([np.sum(k_values == k) for k in (1, 2, 3)], dtype=np.float64)
        stat, p = chi_square_gof(counts, MASS_K_WEIGHTS, len(k_values))
        assert p >= INV12B_CHI2_PVALUE_MIN, (
            f"chi2={stat:.3f} (2 dof) p={p:.4f} < {INV12B_CHI2_PVALUE_MIN}; "
            f"observed counts={counts}, expected={np.array(MASS_K_WEIGHTS) * len(k_values)}"
        )

    def test_b_per_k_three_sigma_band(self, k_values):
        n = len(k_values)
        freqs = np.array([np.sum(k_values == k) / n for k in (1, 2, 3)])
        for f_k, p_k, band in zip(freqs, MASS_K_WEIGHTS, INV12B_PER_K_3SIGMA):
            assert abs(f_k - p_k) <= band, (
                f"|f_k - P_k| = {abs(f_k - p_k):.4f} exceeds the 3-sigma band {band} "
                f"(f_k={f_k:.4f}, P_k={p_k:.4f})"
            )


# -------------------------------------------------------------------------------------------
# INV-13 -- the load-bearing test of this file (Section 6: "e um modo de falha que nenhum outro
# invariante deste documento pega").
# -------------------------------------------------------------------------------------------
class TestINV13SlotPermutationUniformity:
    N_REALIZATIONS = INV13_N_REALIZATIONS
    SEED_BASE = 3_300_000

    @pytest.fixture(scope="class")
    def max_mass_slots(self):
        slots = []
        for i in range(self.N_REALIZATIONS):
            try:
                m = call_sample_masses(
                    sample_masses, N, self.SEED_BASE + i,
                    alpha=MASS_ALPHA, mass_ratio=MASS_RATIO, particle_mass=PARTICLE_MASS,
                )
            except ContractGap as exc:
                pytest.skip(str(exc))
            slots.append(int(np.argmax(m)))
        return np.asarray(slots)

    def test_max_mass_slot_index_is_uniform_over_10_bins(self, max_mass_slots):
        bin_width = N // INV13_N_BINS
        bins = max_mass_slots // bin_width
        counts = np.array([np.sum(bins == b) for b in range(INV13_N_BINS)], dtype=np.float64)
        expected_probs = np.full(INV13_N_BINS, 1.0 / INV13_N_BINS)
        stat, p = chi_square_gof(counts, expected_probs, self.N_REALIZATIONS)
        assert p >= INV13_CHI2_PVALUE_MIN, (
            f"INV-13: chi2={stat:.2f} (9 dof) p={p:.5f} < {INV13_CHI2_PVALUE_MIN}. "
            f"Bin counts (expected {self.N_REALIZATIONS / INV13_N_BINS:.0f} each): {counts}. "
            f"This is the failure signature of an omitted Section 2.7 step-4 permutation: "
            f"with k<=3 massive bodies always written to slots 0..k-1, the max-mass slot would "
            f"concentrate almost entirely in bin 0."
        )

    def test_max_mass_slot_not_trivially_concentrated_in_first_three_slots(self, max_mass_slots):
        # Direct, human-legible version of the same check: an omitted permutation puts the
        # max-mass particle in slot 0, 1 or 2 on EVERY realization; a correct permutation should
        # do so on roughly 3/1000 of them.
        frac_in_first_three = np.mean(max_mass_slots < 3)
        assert frac_in_first_three < 0.10, (
            f"{frac_in_first_three:.1%} of realizations have the max-mass particle in slot "
            f"0, 1 or 2 (expected ~0.3%); this is the omitted-permutation failure mode."
        )


# -------------------------------------------------------------------------------------------
# INV-14
# -------------------------------------------------------------------------------------------
class TestINV14MeanMassClosedForm:
    @pytest.mark.parametrize("alpha", INV14_ALPHA_SWEEP)
    def test_mean_mass_matches_particle_mass_by_independent_quadrature(self, alpha):
        if mass_min_from_mean is None:
            pytest.skip("nbody.populations.mass_min_from_mean is not exposed")
        try:
            m_min = call_mass_min_from_mean(
                mass_min_from_mean, alpha, MASS_RATIO, PARTICLE_MASS
            )
        except ContractGap as exc:
            pytest.skip(str(exc))
        assert m_min > 0.0
        m_max = MASS_RATIO * m_min  # Section 2.4: m_max = R * m_min, R fixed at 1000
        mean_quad = mean_mass_via_quadrature(alpha, m_min, m_max)
        rel = abs(mean_quad / PARTICLE_MASS - 1.0)
        assert rel <= INV14_MEAN_REL_TOL, (
            f"alpha={alpha}: quadrature <m>={mean_quad:.10e} vs PARTICLE_MASS="
            f"{PARTICLE_MASS:.10e}, rel dev {rel:.3e} > {INV14_MEAN_REL_TOL}"
        )

    def test_reference_alpha_matches_documented_m_min(self):
        # Section 2.4's own worked example: alpha=2.35, R=1000 -> m_min = 2.8460126741e8 kg.
        # This does not test the closed form against itself (INV-14's own text forbids that);
        # it is a sanity check that the DOCUMENT's transcribed reference value is internally
        # consistent with this test's independent quadrature, decoupled from the implementation.
        m_min = MASS_MIN
        m_max = MASS_RATIO * m_min
        mean_quad = mean_mass_via_quadrature(MASS_ALPHA, m_min, m_max)
        rel = abs(mean_quad / PARTICLE_MASS - 1.0)
        assert rel <= INV14_MEAN_REL_TOL, (
            f"quadrature check of the document's own MASS_MIN={MASS_MIN:.10e} gives "
            f"<m>={mean_quad:.10e}, rel dev {rel:.3e} -- if this fails, the quadrature "
            f"routine itself (not the implementation) is at fault"
        )

    def test_g_factor_matches_documented_value(self):
        # <m>/m_min = g(alpha, R); with m_min = PARTICLE_MASS / g by construction, this is a
        # closed-form identity check of the document's own transcribed numbers, independent of
        # any implementation call.
        g = PARTICLE_MASS / MASS_MIN
        assert math.isclose(g, MASS_G_FACTOR, rel_tol=1e-9)
