"""
Force/potential consistency and softening semantics: docs/integradores.md Sections 1.3, 2.2,
2.5, and invariants INV-1, INV-2(a,b), INV-10.

Uses nbody.backends.get_backend("torch_eager") as the reference backend for accelerations,
since torch_eager is guaranteed available on CPU by docs/api-contract.md's ladder table
(degrau 2) and this module does not itself test backend equivalence (that's test_backends.py).
"""
import math

import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.observables import potential_energy  # noqa: E402
from nbody.state import State  # noqa: E402

from tolerances import (  # noqa: E402
    EPS_PREC_FP32,
    EPS_PREC_FP64,
    INV10_SLACK,
    TOL_GRAD_H1E4,
    TOL_GRAD_H1E5,
    TOL_MOM_RATIO_MAX,
)

BACKEND = get_backend("torch_eager")
CPU = torch.device("cpu")


def central_difference_accelerations(r, v, m, softening, h):
    """
    a_i,c = -(1/m_i) * dU/dr_i,c, by central difference on U (Section 1.1, 1.3).
    r: (N, 3) fp64 tensor. Returns (N, 3) fp64 tensor of finite-difference accelerations.
    """
    n = r.shape[0]
    grad_u = torch.zeros_like(r)
    for i in range(n):
        for c in range(3):
            r_plus = r.clone()
            r_plus[i, c] += h
            r_minus = r.clone()
            r_minus[i, c] -= h
            u_plus = potential_energy(State(r=r_plus, v=v, m=m), softening=softening)
            u_minus = potential_energy(State(r=r_minus, v=v, m=m), softening=softening)
            grad_u[i, c] = (u_plus - u_minus) / (2 * h)
    return -grad_u / m.unsqueeze(-1)


@pytest.fixture(scope="module")
def inv1_config():
    rng = np.random.default_rng(7)
    n = 12
    positions = rng.normal(scale=2.0, size=(n, 3))
    r = torch.tensor(positions, dtype=torch.float64)
    v = torch.zeros(n, 3, dtype=torch.float64)
    m = torch.full((n,), config.PARTICLE_MASS, dtype=torch.float64)
    return r, v, m


class TestINV1ForcePotentialConsistency:
    """INV-1: a_i = -(1/m_i) dU/dr_i. Blocking if it fails (Section 1.3)."""

    def _max_relative_error(self, a_fd, a_direct):
        # Per-particle vector relative error, to avoid spurious blowup from near-zero scalar
        # components (a component of a well-separated random 3D acceleration is essentially
        # never near zero at the vector-norm level).
        diff_norm = torch.linalg.norm(a_fd - a_direct, dim=1)
        ref_norm = torch.linalg.norm(a_direct, dim=1)
        return (diff_norm / ref_norm).max().item()

    def test_gradient_matches_at_h_1e5(self, inv1_config):
        r, v, m = inv1_config
        h = 1e-5
        a_direct = BACKEND.accelerations(r, m, softening=0.05)
        a_fd = central_difference_accelerations(r, v, m, softening=0.05, h=h)
        rel_err = self._max_relative_error(a_fd, a_direct)
        assert rel_err <= TOL_GRAD_H1E5, (
            f"INV-1 / TOL-GRAD (h=1e-5): max relative error {rel_err:.3e} exceeds "
            f"{TOL_GRAD_H1E5:.1e}"
        )

    def test_gradient_matches_at_h_1e4(self, inv1_config):
        r, v, m = inv1_config
        h = 1e-4
        a_direct = BACKEND.accelerations(r, m, softening=0.05)
        a_fd = central_difference_accelerations(r, v, m, softening=0.05, h=h)
        rel_err = self._max_relative_error(a_fd, a_direct)
        assert rel_err <= TOL_GRAD_H1E4, (
            f"INV-1 / TOL-GRAD (h=1e-4): max relative error {rel_err:.3e} exceeds "
            f"{TOL_GRAD_H1E4:.1e}"
        )


class TestINV2LinearMomentumOfAccelerationField:
    """
    INV-2(a,b): sum_i m_i a_i == 0 in exact arithmetic; bounded by round-off in floating point.
    Bound: ||sum_i m_i a_i|| <= N * eps_prec * max_i(m_i ||a_i||). Equal masses cancel to
    ||sum_i a_i|| <= N * eps_prec * max_i ||a_i||.
    """

    def _check(self, r, m, softening, eps_prec):
        a = BACKEND.accelerations(r, m, softening=softening)
        residual = torch.linalg.norm(a.sum(dim=0).to(torch.float64)).item()
        max_a = torch.linalg.norm(a.to(torch.float64), dim=1).max().item()
        n = r.shape[0]
        bound = n * eps_prec * max_a
        ratio = residual / bound if bound > 0 else 0.0
        assert ratio <= TOL_MOM_RATIO_MAX, (
            f"INV-2 / TOL-MOM: residual {residual:.3e} exceeds bound {bound:.3e} "
            f"(ratio {ratio:.3f})"
        )

    def test_fp64(self):
        state = cold_sphere(n=1000, seed=20190222, dtype=torch.float64, device="cpu")
        self._check(state.r, state.m, softening=0.05, eps_prec=EPS_PREC_FP64)

    def test_fp32(self):
        state = cold_sphere(n=1000, seed=20190222, dtype=torch.float32, device="cpu")
        self._check(state.r, state.m, softening=0.05, eps_prec=EPS_PREC_FP32)

    def test_eps_zero_still_conserves(self):
        # Section 2.5 / INV-2: exclusion of i=j must not depend on eps != 0.
        state = cold_sphere(n=500, seed=1, dtype=torch.float64, device="cpu")
        self._check(state.r, state.m, softening=0.0, eps_prec=EPS_PREC_FP64)


class TestINV10PotentialEnergyLowerBound:
    """INV-10: U(t) >= -G m^2 N(N-1) / (2 eps), for all t, by construction. [T]"""

    def test_at_ic_fp64(self):
        state = cold_sphere(n=1000, seed=20190222, dtype=torch.float64, device="cpu")
        u = potential_energy(state, softening=0.05)
        n = config.N_PARTICLES
        bound = -config.G * config.PARTICLE_MASS**2 * n * (n - 1) / (2 * 0.05)
        assert math.isclose(bound, config.U_MIN_BOUND, rel_tol=1e-4)
        assert u >= bound * (1 + INV10_SLACK), f"U={u:.6e} violates lower bound {bound:.6e}"

    def test_at_ic_fp32(self):
        state = cold_sphere(n=1000, seed=20190222, dtype=torch.float32, device="cpu")
        u = potential_energy(state, softening=0.05)
        n = config.N_PARTICLES
        bound = -config.G * config.PARTICLE_MASS**2 * n * (n - 1) / (2 * 0.05)
        assert u >= bound * (1 + 1e-3), f"U={u:.6e} violates lower bound {bound:.6e}"

    def test_random_dense_configuration(self):
        # A more adversarial configuration: many particles crammed close together, where the
        # bound is closer to saturating than at the diffuse cold_sphere IC.
        rng = np.random.default_rng(3)
        n = 100
        r = torch.tensor(rng.normal(scale=0.05, size=(n, 3)), dtype=torch.float64)
        v = torch.zeros(n, 3, dtype=torch.float64)
        m = torch.full((n,), config.PARTICLE_MASS, dtype=torch.float64)
        state = State(r=r, v=v, m=m)
        u = potential_energy(state, softening=0.05)
        bound = -config.G * config.PARTICLE_MASS**2 * n * (n - 1) / (2 * 0.05)
        assert u >= bound * (1 + INV10_SLACK)


class TestSofteningSemantics:
    """Section 2.5: the i=j diagonal must be excluded before any multiplication by distance,
    never merely relying on eps != 0. Section 2.2: pairwise force is bounded, with a known
    closed-form maximum."""

    def test_eps_zero_produces_no_nan_or_inf(self):
        rng = np.random.default_rng(11)
        n = 50
        r = torch.tensor(rng.normal(scale=2.0, size=(n, 3)), dtype=torch.float64)
        m = torch.full((n,), config.PARTICLE_MASS, dtype=torch.float64)
        a = BACKEND.accelerations(r, m, softening=0.0)
        assert torch.isfinite(a).all(), "eps=0 must not produce nan/inf (Section 2.5)"

    def test_eps_zero_two_body_matches_newtonian_formula(self):
        d = 3.0
        m = config.PARTICLE_MASS
        r = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float64)
        masses = torch.tensor([m, m], dtype=torch.float64)
        a = BACKEND.accelerations(r, masses, softening=0.0)
        expected_mag = config.G * m / d**2
        assert math.isclose(torch.linalg.norm(a[0]).item(), expected_mag, rel_tol=1e-12)
        assert math.isclose(torch.linalg.norm(a[1]).item(), expected_mag, rel_tol=1e-12)
        # Newton's third law: equal and opposite, equal masses -> equal magnitude, opposite sign.
        assert torch.allclose(a[0], -a[1], rtol=1e-12)

    def test_softened_two_body_matches_plummer_formula(self):
        d = 1.0
        eps = 0.05
        m = config.PARTICLE_MASS
        r = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]], dtype=torch.float64)
        masses = torch.tensor([m, m], dtype=torch.float64)
        a = BACKEND.accelerations(r, masses, softening=eps)
        expected_mag = config.G * m * d / (d * d + eps * eps) ** 1.5
        assert math.isclose(torch.linalg.norm(a[0]).item(), expected_mag, rel_tol=1e-12)

    def test_pair_acceleration_never_exceeds_analytic_maximum(self):
        # a_pair_max = (2 / (3 sqrt(3))) * G m / eps^2, attained at d = eps/sqrt(2)
        # (Section 2.2, item 1). Scan a range of separations and verify the bound holds for all
        # of them, with the closed-form maximum attained (within numerical tolerance) at
        # d = eps/sqrt(2).
        eps = 0.05
        m = config.PARTICLE_MASS
        a_pair_max = (2.0 / (3.0 * math.sqrt(3.0))) * config.G * m / eps**2
        assert math.isclose(a_pair_max, 10.27, rel_tol=1e-3)

        separations = np.concatenate(
            [np.linspace(1e-4, 2.0, 200), [eps / math.sqrt(2)]]
        )
        max_seen = 0.0
        for d in separations:
            r = torch.tensor([[0.0, 0.0, 0.0], [float(d), 0.0, 0.0]], dtype=torch.float64)
            masses = torch.tensor([m, m], dtype=torch.float64)
            a = BACKEND.accelerations(r, masses, softening=eps)
            mag = torch.linalg.norm(a[0]).item()
            max_seen = max(max_seen, mag)
            assert mag <= a_pair_max * (1 + 1e-9), (
                f"pairwise acceleration {mag:.6e} at d={d:.6e} exceeds analytic max "
                f"{a_pair_max:.6e}"
            )
        assert math.isclose(max_seen, a_pair_max, rel_tol=1e-3), (
            f"maximum observed over the scan ({max_seen:.6e}) should approach the analytic "
            f"maximum ({a_pair_max:.6e}) near d=eps/sqrt(2)"
        )
