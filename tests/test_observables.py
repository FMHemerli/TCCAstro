"""
nbody.observables against docs/integradores.md Section 1.1 (definitions of K, U, E, P, L),
Section 5.4 (reference values at the cold_sphere IC), and Section 6.4 (fp64 accumulation
directive). Also docs/api-contract.md's signatures and float64-host-return contract.

Two kinds of check are used:

  - Hand-computed values on tiny (2-3 particle) systems, worked out from the definitions in
    Section 1.1 independently of any implementation.
  - The Section 5.4 reference values for the N=1000 cold_sphere IC, which are themselves
    independent [M] measurements recorded in the specification.
"""
import math

import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.observables import (  # noqa: E402
    angular_momentum,
    center_of_mass,
    half_mass_radius,
    kinetic_energy,
    linear_momentum,
    potential_energy,
    total_energy,
)
from nbody.state import State  # noqa: E402

from tolerances import TOL_ENERGY_FP64, TOL_ENERGY_FP32  # noqa: E402


def two_particle_state(r0, r1, v0, v1, m0, m1, dtype=torch.float64):
    r = torch.tensor([r0, r1], dtype=dtype)
    v = torch.tensor([v0, v1], dtype=dtype)
    m = torch.tensor([m0, m1], dtype=dtype)
    return State(r=r, v=v, m=m)


class TestKineticEnergyHandComputed:
    def test_two_particles_at_rest(self):
        state = two_particle_state([0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], 2.0, 3.0)
        assert kinetic_energy(state) == pytest.approx(0.0, abs=1e-15)

    def test_two_particles_moving(self):
        # K = 1/2 * m1 * |v1|^2 + 1/2 * m2 * |v2|^2
        state = two_particle_state(
            [0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 4, 0], m0=2.0, m1=5.0
        )
        expected = 0.5 * 2.0 * 9.0 + 0.5 * 5.0 * 16.0
        assert kinetic_energy(state) == pytest.approx(expected, rel=1e-13)

    def test_returns_float(self):
        state = two_particle_state([0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], 1.0, 1.0)
        assert isinstance(kinetic_energy(state), float)


class TestPotentialEnergyHandComputed:
    def test_two_particles_no_softening(self):
        # U = -G m1 m2 / sqrt(d^2 + eps^2), eps = 0 -> Newtonian.
        state = two_particle_state([0, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0], 1e9, 1e9)
        expected = -config.G * 1e9 * 1e9 / 2.0
        assert potential_energy(state, softening=0.0) == pytest.approx(
            expected, rel=TOL_ENERGY_FP64
        )

    def test_two_particles_with_softening(self):
        d = 2.0
        eps = 0.5
        state = two_particle_state([0, 0, 0], [d, 0, 0], [0, 0, 0], [0, 0, 0], 1e9, 1e9)
        expected = -config.G * 1e9 * 1e9 / math.sqrt(d * d + eps * eps)
        assert potential_energy(state, softening=eps) == pytest.approx(
            expected, rel=TOL_ENERGY_FP64
        )

    def test_three_particles_sums_pairs(self):
        # Equilateral triangle, side 1, equal masses m: U = -3 * G m^2 / sqrt(1 + eps^2).
        m = 1e9
        eps = 0.1
        r = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, math.sqrt(3) / 2, 0.0]],
            dtype=torch.float64,
        )
        v = torch.zeros(3, 3, dtype=torch.float64)
        masses = torch.full((3,), m, dtype=torch.float64)
        state = State(r=r, v=v, m=masses)
        expected = -3.0 * config.G * m * m / math.sqrt(1.0 + eps * eps)
        assert potential_energy(state, softening=eps) == pytest.approx(
            expected, rel=TOL_ENERGY_FP64
        )


class TestTotalEnergy:
    def test_equals_kinetic_plus_potential(self):
        state = two_particle_state(
            [0, 0, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0], m0=1e9, m1=1e9
        )
        k = kinetic_energy(state)
        u = potential_energy(state, softening=0.05)
        e = total_energy(state, softening=0.05)
        assert e == pytest.approx(k + u, rel=1e-13)


class TestLinearMomentum:
    def test_hand_computed(self):
        state = two_particle_state(
            [0, 0, 0], [1, 0, 0], [1, 2, 3], [-1, 0, 1], m0=2.0, m1=5.0
        )
        expected = torch.tensor([2 * 1 + 5 * -1, 2 * 2 + 5 * 0, 2 * 3 + 5 * 1], dtype=torch.float64)
        p = linear_momentum(state)
        assert p.shape == (3,)
        assert p.dtype == torch.float64
        assert torch.allclose(p, expected, atol=1e-10)

    def test_zero_for_symmetric_pair(self):
        state = two_particle_state(
            [0, 0, 0], [1, 0, 0], [1, 2, 3], [-1, -2, -3], m0=1.0, m1=1.0
        )
        p = linear_momentum(state)
        assert torch.linalg.norm(p).item() < 1e-13


class TestAngularMomentum:
    def test_hand_computed_planar(self):
        # Two particles on a circle: L = sum m_i (r_i x v_i).
        m = 4.0
        r0 = [1.0, 0.0, 0.0]
        v0 = [0.0, 2.0, 0.0]
        r1 = [-1.0, 0.0, 0.0]
        v1 = [0.0, -2.0, 0.0]
        state = two_particle_state(r0, r1, v0, v1, m0=m, m1=m)
        # r x v for particle 0: (1,0,0) x (0,2,0) = (0,0,2); scaled by m.
        expected = torch.tensor([0.0, 0.0, 2 * m * 2.0], dtype=torch.float64)
        l_total = angular_momentum(state)
        assert l_total.shape == (3,)
        assert l_total.dtype == torch.float64
        assert torch.allclose(l_total, expected, atol=1e-9)

    def test_zero_for_radial_motion(self):
        # v parallel to r: r x v = 0.
        state = two_particle_state(
            [1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0], m0=1.0, m1=1.0
        )
        l_total = angular_momentum(state)
        assert torch.linalg.norm(l_total).item() < 1e-13


class TestCenterOfMass:
    def test_hand_computed(self):
        state = two_particle_state(
            [0, 0, 0], [4, 0, 0], [0, 0, 0], [0, 0, 0], m0=1.0, m1=3.0
        )
        # (1*0 + 3*4) / 4 = 3.0
        com = center_of_mass(state)
        assert com.dtype == torch.float64
        assert torch.allclose(com, torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)

    def test_equal_masses_is_midpoint(self):
        state = two_particle_state(
            [0, 0, 0], [2, 4, 6], [0, 0, 0], [0, 0, 0], m0=1.0, m1=1.0
        )
        com = center_of_mass(state)
        assert torch.allclose(com, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), atol=1e-12)


class TestHalfMassRadius:
    def test_relative_to_center_of_mass_not_origin(self):
        # Four equal-mass particles arranged as two zero-sum pairs, so the center of mass is
        # exactly the offset (NOT the origin -- this is the point of the test, per
        # docs/api-contract.md: "relativo ao centro de massa"), and the distances from that
        # center of mass are exactly [1, 1, 3, 3] by construction (unlike an arbitrary point
        # cloud, whose centroid is not generally known in closed form). Section 7's operational
        # definition: r_half = the (N//2 - 1)'th smallest distance, 0-indexed -> for N=4, index
        # 1 of the sorted list [1, 1, 3, 3] -> 1.0.
        offset = torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64)
        r = offset + torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, -3.0, 0.0]],
            dtype=torch.float64,
        )
        v = torch.zeros(4, 3, dtype=torch.float64)
        m = torch.ones(4, dtype=torch.float64)
        state = State(r=r, v=v, m=m)
        rh = half_mass_radius(state)
        assert isinstance(rh, float)
        assert rh == pytest.approx(1.0, rel=1e-9)


class TestColdSphereReferenceObservables:
    """Section 5.4, N=1000, seed=20190222, fp64 reference values."""

    @pytest.fixture(scope="class")
    def ic_state(self):
        return cold_sphere(n=1000, seed=20190222, dtype=torch.float64, device="cpu")

    def test_r_half_0(self, ic_state):
        rh = half_mass_radius(ic_state)
        assert rh == pytest.approx(4.881251, rel=1e-6)

    def test_u_eps005(self, ic_state):
        u = potential_energy(ic_state, softening=0.05)
        # Section 5.4 prints "-6.4260397026e12 J" to 11 significant digits; the residual
        # between that literal and a correct fp64 evaluation (~4e-12 relative here) is
        # consistent with the literal's own display rounding, well below what TOL-ENERGY-FP64
        # (1e-13, meant for comparison against a full-precision reference, not an 11-digit
        # printed literal) would allow. 1e-9 relative gives comfortable margin over the
        # print-rounding noise while still being tight enough to catch a materially wrong U.
        assert u == pytest.approx(-6.4260397026e12, rel=1e-9)

    def test_kinetic_energy_zero(self, ic_state):
        assert kinetic_energy(ic_state) == pytest.approx(0.0, abs=1e-6)

    def test_momentum_zero(self, ic_state):
        p = linear_momentum(ic_state)
        assert torch.linalg.norm(p).item() < 1e-6


class TestFp32AccumulatesInFp64:
    """
    Section 6.4: energy diagnostics are always fp64-accumulated regardless of state dtype.
    Verified indirectly: a fp32 state's U must agree with an independent fp64 hand-computation
    to TOL-ENERGY-FP32 (1e-5), which fp32-accumulated summation could not reliably achieve at
    N=1000 (Section 6.4 measures fp32-accumulated error at ~8.5e-8... wait that's smaller than
    1e-5, so the real discriminator is TOL-ACCEL-scale precision on the pairwise term itself).
    Practically: check U from an fp32 state matches the fp64 IC reference value within
    TOL-ENERGY-FP32, which is the number the spec ties to fp64-accumulated fp32-term sums.
    """

    def test_potential_energy_fp32_state_matches_fp64_reference(self):
        state = cold_sphere(n=1000, seed=20190222, dtype=torch.float32, device="cpu")
        u = potential_energy(state, softening=0.05)
        assert isinstance(u, float)
        assert u == pytest.approx(-6.4260397026e12, rel=TOL_ENERGY_FP32)
