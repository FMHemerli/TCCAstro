"""
nbody.config against docs/integradores.md Section 9 (constants) and Sections 4.5-4.8 (parameter
sets), and docs/api-contract.md's requirement that these be exposed as module-level uppercase
constants and frozen parameter objects.

Every numeric literal below is transcribed from Section 9 of the specification, not from any
implementation. Relative tolerance 1e-9 is used for floats that the document prints to 7-10
significant digits, to allow for the last printed digit being a rounding of the true constant
without masking a materially wrong value.
"""
import math

import pytest

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402


REL = 1e-9


def assert_close(actual, expected, rel=REL, msg=""):
    assert math.isclose(actual, expected, rel_tol=rel), (
        f"{msg}: expected {expected}, got {actual} (rel_tol={rel})"
    )


class TestPhysicalConstants:
    def test_G(self):
        assert_close(config.G, 6.67408e-11, msg="G")

    def test_particle_mass(self):
        assert_close(config.PARTICLE_MASS, 1.0e9, msg="PARTICLE_MASS")

    def test_n_particles(self):
        assert config.N_PARTICLES == 1000

    def test_sphere_radius(self):
        # R_0 = (3N/(4 pi))^(1/3), Section 4.2.
        assert_close(config.SPHERE_RADIUS, 6.2035049090, msg="SPHERE_RADIUS")

    def test_softening(self):
        assert_close(config.SOFTENING, 5.0e-2, msg="SOFTENING")

    def test_seed(self):
        assert config.SEED == 20190222

    def test_t_ff(self):
        assert_close(config.T_FF, 2.1007035, msg="T_FF")

    def test_t_conv_is_half_t_ff(self):
        assert_close(config.T_CONV, 1.05035175, msg="T_CONV")
        assert_close(config.T_CONV, 0.5 * config.T_FF, rel=1e-12, msg="T_CONV vs 0.5*T_FF")


class TestDerivedScales:
    def test_total_mass(self):
        assert_close(config.TOTAL_MASS, 1.0e12, msg="TOTAL_MASS")
        assert_close(
            config.TOTAL_MASS, config.N_PARTICLES * config.PARTICLE_MASS, rel=1e-12
        )

    def test_density(self):
        assert_close(config.DENSITY, 1.0e9, msg="DENSITY")

    def test_v_char(self):
        assert_close(config.V_CHAR, 3.2799885, msg="V_CHAR")

    def test_l_scale(self):
        assert_close(config.L_SCALE, 2.0347400e13, msg="L_SCALE")

    def test_omega_pair_max(self):
        assert_close(config.OMEGA_PAIR_MAX, 32.678, rel=1e-4, msg="OMEGA_PAIR_MAX")

    def test_p_eps(self):
        assert_close(config.P_EPS, 0.192276, rel=1e-4, msg="P_EPS")
        assert_close(
            config.P_EPS, 2 * math.pi / config.OMEGA_PAIR_MAX, rel=1e-4,
            msg="P_EPS vs 2*pi/OMEGA_PAIR_MAX",
        )

    def test_omega_max_design(self):
        assert_close(config.OMEGA_MAX_DESIGN, 42.0, msg="OMEGA_MAX_DESIGN")

    def test_u_min_bound(self):
        assert_close(config.U_MIN_BOUND, -6.6674e14, rel=1e-4, msg="U_MIN_BOUND")


class TestCollapseParameters:
    def test_dt_collapse(self):
        assert_close(config.DT_COLLAPSE, 5.0e-4, msg="DT_COLLAPSE")

    def test_n_steps_collapse(self):
        assert config.N_STEPS_COLLAPSE == 12600

    def test_out_dt(self):
        assert_close(config.OUT_DT, 1.0e-2, msg="OUT_DT")

    def test_dt_times_n_steps_equals_3_tff(self):
        t_end = config.DT_COLLAPSE * config.N_STEPS_COLLAPSE
        assert_close(t_end, 6.30, rel=1e-6, msg="T_END = DT * N_STEPS")
        assert_close(t_end / config.T_FF, 2.99899, rel=1e-4, msg="T_END / T_FF")

    def test_dt_divides_out_dt(self):
        ratio = config.OUT_DT / config.DT_COLLAPSE
        assert math.isclose(ratio, round(ratio), rel_tol=1e-9), (
            "DT_COLLAPSE must divide OUT_DT exactly (Section 7, INV-9 sampling grid)"
        )


class TestConvergenceParameters:
    def test_conv_n_steps_ladder(self):
        assert tuple(config.CONV_N_STEPS) == (64, 128, 256, 512, 1024, 2048)

    def test_ref_n_steps(self):
        assert config.REF_N_STEPS == 16384

    def test_ref_check_n_steps(self):
        assert config.REF_CHECK_N_STEPS == 8192


class TestBenchParameters:
    def test_dt_bench(self):
        assert_close(config.DT_BENCH, 1.0e-2, msg="DT_BENCH")

    def test_n_steps_bench(self):
        assert config.N_STEPS_BENCH == 1


class TestTwoBodyConstants:
    def test_mu(self):
        assert_close(config.TWOBODY_MU, 0.1334816, rel=1e-6, msg="TWOBODY_MU")
        expected_mu = config.G * 2 * config.PARTICLE_MASS
        assert_close(config.TWOBODY_MU, expected_mu, rel=1e-6, msg="TWOBODY_MU vs G*(m1+m2)")

    def test_kepler_a(self):
        assert_close(config.KEPLER_A, 1.0, msg="KEPLER_A")

    def test_kepler_e(self):
        assert_close(config.KEPLER_E, 0.5, msg="KEPLER_E")

    def test_kepler_period(self):
        assert_close(config.KEPLER_PERIOD, 17.19765239, rel=1e-8, msg="KEPLER_PERIOD")
        expected = 2 * math.pi * math.sqrt(config.KEPLER_A**3 / config.TWOBODY_MU)
        assert_close(config.KEPLER_PERIOD, expected, rel=1e-6, msg="KEPLER_PERIOD vs formula")

    def test_circ_separation(self):
        assert_close(config.CIRC_SEPARATION, 1.0, msg="CIRC_SEPARATION")

    def test_circ_omega(self):
        assert_close(config.CIRC_OMEGA, 0.3646677991, rel=1e-8, msg="CIRC_OMEGA")
        d, eps, mu = config.CIRC_SEPARATION, config.SOFTENING, config.TWOBODY_MU
        expected = math.sqrt(mu / (d**2 + eps**2) ** 1.5)
        assert_close(config.CIRC_OMEGA, expected, rel=1e-6, msg="CIRC_OMEGA vs formula")

    def test_circ_period(self):
        assert_close(config.CIRC_PERIOD, 17.22988792, rel=1e-8, msg="CIRC_PERIOD")


class TestReferenceMeasurements:
    """Section 5.4, INV-4, INV-8, INV-9 reference values, as exposed constants (Section 9)."""

    def test_ic_r_half_0(self):
        assert_close(config.IC_R_HALF_0, 4.881251, rel=1e-6, msg="IC_R_HALF_0")

    def test_ic_r_max(self):
        assert_close(config.IC_R_MAX, 6.323302, rel=1e-6, msg="IC_R_MAX")

    def test_ic_u_eps005(self):
        assert_close(config.IC_U_EPS005, -6.4260397026e12, rel=1e-8, msg="IC_U_EPS005")

    def test_ic_a_max(self):
        assert_close(config.IC_A_MAX, 9.075891, rel=1e-6, msg="IC_A_MAX")

    def test_ic_a_rms(self):
        assert_close(config.IC_A_RMS, 1.544392, rel=1e-6, msg="IC_A_RMS")

    def test_collapse_r_half_min(self):
        assert_close(config.COLLAPSE_R_HALF_MIN, 0.3472, rel=1e-4, msg="COLLAPSE_R_HALF_MIN")

    def test_collapse_t_over_tff(self):
        assert_close(config.COLLAPSE_T_OVER_TFF, 1.0361, rel=1e-4, msg="COLLAPSE_T_OVER_TFF")


class TestParameterSets:
    """
    Section 4.5's table gives explicit identifiers (N, PARTICLE_MASS, SPHERE_RADIUS, SOFTENING,
    DT, N_STEPS, T_END, SEED, OUT_DT) for RUN_COLLAPSE's *fields*, but neither source document
    fixes the letter case of the corresponding Python attribute name on the frozen parameter
    object (api-contract.md only says "atributos nomeados conforme o documento de fisica").
    This implementation uses lower_snake_case instance attributes (n, particle_mass, ...),
    which is the idiomatic convention for dataclass fields as opposed to module-level
    constants; the tests below were written against the upper-case guess first, failed, and
    were corrected here after confirming (Section 4.5's values, not just names) that this is a
    naming-convention gap in the contract rather than a numeric discrepancy. See the final
    report for this being flagged as underspecified in docs/api-contract.md.
    """

    def test_run_collapse_is_frozen_with_documented_fields(self):
        rc = config.RUN_COLLAPSE
        assert rc.n == 1000
        assert_close(rc.particle_mass, 1.0e9)
        assert_close(rc.sphere_radius, 6.2035049090)
        assert_close(rc.softening, 5.0e-2)
        assert_close(rc.dt, 5.0e-4)
        assert rc.n_steps == 12600
        assert_close(rc.t_end, 6.30, rel=1e-6)
        assert rc.seed == 20190222
        assert_close(rc.out_dt, 1.0e-2)

    def test_run_collapse_frozen(self):
        rc = config.RUN_COLLAPSE
        with pytest.raises((AttributeError, TypeError, Exception)):
            rc.n = 5  # type: ignore[misc]

    def test_run_convergence_matches_run_collapse_ic(self):
        # Section 4.6: "condicao inicial identica a RUN_COLLAPSE (mesma semente)".
        assert config.RUN_CONVERGENCE.seed == config.RUN_COLLAPSE.seed
        assert_close(config.RUN_CONVERGENCE.softening, 5.0e-2)
        assert_close(config.RUN_CONVERGENCE.t_end, config.T_CONV, rel=1e-6)

    def test_twobody_kepler_eps_is_zero(self):
        # Section 4.8: TWOBODY_KEPLER has eps = 0.0 exactly; this is load-bearing (INV-5
        # requires eps = 0 exactly, not merely small).
        assert config.TWOBODY_KEPLER.softening == 0.0

    def test_twobody_circ_eps_is_softening(self):
        assert_close(config.TWOBODY_CIRC.softening, config.SOFTENING)

    def test_twobody_ecc_eps_is_softening(self):
        assert_close(config.TWOBODY_ECC.softening, config.SOFTENING)
