"""
nbody.observables.scales_from_state against docs/simulacao-estocastica.md Section 2.10 and
Section 6, INV-15.

Section 2.10 is explicit that t_ff stops being a config constant once the mass spectrum is on:
    rho_real = M_real / ((4/3) pi R_0^3)
    t_ff     = sqrt(3 pi / (32 G rho_real))
with R_0 = SPHERE_RADIUS held fixed (Section 2.10: "e a esfera que e reutilizada, nao a
densidade") -- t_ff depends on the realized total mass ONLY, not on particle positions, which
is what makes test (a) below constructible without any real IC generator: doubling every mass
in an arbitrary State by an exact float factor must halve t_ff's square, independent of geometry.

Section 9.1 names scales_from_state but does not fix its return type (dict? dataclass?); see
tests/_stage2_binding.py's get_field for the best-effort, name-matching field extraction used
here (never the implementation's algorithm, only the value it returns).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.state import State  # noqa: E402

scales_from_state = getattr(
    pytest.importorskip("nbody.observables"), "scales_from_state", None
)
if scales_from_state is None:
    pytest.skip(
        "nbody.observables.scales_from_state is not exposed (Section 9.1 amendment)",
        allow_module_level=True,
    )

populations = None
try:
    import nbody.populations as populations  # noqa: E402
except ImportError:
    pass

from _stage2_binding import ContractGap, call_sample_masses, get_field  # noqa: E402

from tolerances import (  # noqa: E402
    INV15A_REL_TOL,
    INV15B_N_REALIZATIONS,
    INV15B_TFF_CV_HIGH,
    INV15B_TFF_CV_LOW,
    MASS_ALPHA,
    MASS_RATIO,
)

N = config.N_PARTICLES
PARTICLE_MASS = config.PARTICLE_MASS


def _t_ff(state):
    return get_field(scales_from_state(state), "t_ff", "T_FF", "tff", "t_free_fall")


@pytest.fixture(scope="module")
def base_ic():
    # Positions are irrelevant to t_ff by Section 2.10's own construction (R_0 is a fixed
    # config constant, not measured from the point cloud); cold_sphere is used only as a
    # convenient source of a well-formed (N, 3) position/velocity array.
    return cold_sphere(n=N, seed=config.SEED, dtype=torch.float64, device="cpu")


class TestINV15aScalingWithRealizedMass:
    def test_t_ff_scales_as_inverse_sqrt_of_realized_mass(self, base_ic):
        state_m = State(r=base_ic.r, v=base_ic.v, m=base_ic.m)
        state_2m = State(r=base_ic.r, v=base_ic.v, m=base_ic.m * 2.0)

        t_ff_m = float(_t_ff(state_m))
        t_ff_2m = float(_t_ff(state_2m))

        ratio = t_ff_2m / t_ff_m
        expected = 2.0 ** (-0.5)
        rel = abs(ratio / expected - 1.0)
        assert rel <= INV15A_REL_TOL, (
            f"t_ff(2M)/t_ff(M) = {ratio:.15e}, expected {expected:.15e} "
            f"(2^-1/2), rel dev {rel:.3e} > {INV15A_REL_TOL}. If this fails while other tests "
            f"pass, the likely cause (Section 6, INV-15) is a module-level T_FF constant being "
            f"read instead of the realized mass."
        )

    def test_t_ff_at_nominal_mass_matches_config_t_ff(self, base_ic):
        # Sanity anchor: with the unmodified cold_sphere masses (M_real = TOTAL_MASS exactly),
        # scales_from_state's t_ff must reduce to the documented config value -- this is the
        # one point where the "function of the realization" and "project constant" curves must
        # agree, since M_real = TOTAL_MASS here by construction.
        state = State(r=base_ic.r, v=base_ic.v, m=base_ic.m)
        t_ff = float(_t_ff(state))
        assert math.isclose(t_ff, config.T_FF, rel_tol=1e-6), (
            f"t_ff at nominal mass = {t_ff}, expected config.T_FF = {config.T_FF}"
        )

    def test_t_ff_does_not_depend_on_particle_positions(self, base_ic):
        # Section 2.10: R_0 is fixed, not measured from the point cloud, so scaling ALL
        # positions by a constant factor must leave t_ff unchanged.
        state_a = State(r=base_ic.r, v=base_ic.v, m=base_ic.m)
        state_b = State(r=base_ic.r * 3.0, v=base_ic.v, m=base_ic.m)
        assert math.isclose(float(_t_ff(state_a)), float(_t_ff(state_b)), rel_tol=1e-12)


class TestINV15bEnsembleDispersion:
    """
    Requires an actual mass sampler (nbody.populations.sample_masses) to build the 200
    realizations Section 6/INV-15(b) calls for; skips loudly if that module is not available or
    its signature cannot be matched (see module docstring).
    """

    SEED_BASE = 3_400_000

    @pytest.fixture(scope="class")
    def t_ff_ensemble(self, base_ic):
        if populations is None or not hasattr(populations, "sample_masses"):
            pytest.skip("nbody.populations.sample_masses is not available")
        t_ffs = []
        for i in range(INV15B_N_REALIZATIONS):
            try:
                m = call_sample_masses(
                    populations.sample_masses, N, self.SEED_BASE + i,
                    alpha=MASS_ALPHA, mass_ratio=MASS_RATIO, particle_mass=PARTICLE_MASS,
                )
            except ContractGap as exc:
                pytest.skip(str(exc))
            m_t = torch.tensor(m, dtype=torch.float64)
            state = State(r=base_ic.r, v=base_ic.v, m=m_t)
            t_ffs.append(float(_t_ff(state)))
        return np.asarray(t_ffs)

    def test_t_ff_ensemble_cv_within_predicted_band(self, t_ff_ensemble):
        mean = t_ff_ensemble.mean()
        sd = t_ff_ensemble.std(ddof=1)
        cv = sd / mean
        assert INV15B_TFF_CV_LOW <= cv <= INV15B_TFF_CV_HIGH, (
            f"sd(t_ff)/mean(t_ff) = {cv:.4f} over {INV15B_N_REALIZATIONS} realizations, "
            f"outside the predicted [{INV15B_TFF_CV_LOW}, {INV15B_TFF_CV_HIGH}] band "
            f"(Section 2.10 prediction: 4.62%)"
        )
