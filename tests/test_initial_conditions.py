"""
nbody.initial_conditions against docs/integradores.md Section 5 (cold_sphere) and Section 4.8 /
INV-5 / INV-6 (two_body_kepler, two_body_circular), and docs/api-contract.md's signatures.

cold_sphere is checked two different ways, deliberately kept apart:

  1. Against an independently written re-implementation of the normative algorithm (Section 5.2)
     in this file, at a numeric tolerance (not bit-exact -- the spec fixes the RNG draw order,
     which is what makes the *stream* reproducible, but does not mandate one particular
     vectorized arithmetic order for combining radius/sin/cos, so a correct implementation may
     round the last bit differently than this test's). This is the test that can catch a wrong
     formula (uniform-r instead of cbrt, theta-uniform instead of cos(theta)-uniform, velocities
     recentred instead of positions, etc).

  2. Bit-for-bit self-reproducibility: same seed/n/dtype called twice, and (when a CUDA device is
     available) called again on "cuda". This is the specific bit-exact claim the API contract
     makes ("reproduzivel bit a bit ... independentemente do device"), so it is tested with
     torch.equal, not a tolerance.
"""
import math

import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.initial_conditions import cold_sphere, two_body_circular, two_body_kepler  # noqa: E402

from conftest import require_cuda  # noqa: E402


# -------------------------------------------------------------------------------------------
# Independent re-implementation of Section 5.2's normative algorithm.
# -------------------------------------------------------------------------------------------
def reference_cold_sphere(n, radius, particle_mass, seed):
    rng = np.random.default_rng(seed)

    u = rng.random(n)
    r = radius * np.cbrt(u)

    cos_th = 2.0 * rng.random(n) - 1.0
    phi = 2.0 * np.pi * rng.random(n)
    sin_th = np.sqrt(1.0 - cos_th**2)

    positions = np.empty((n, 3), dtype=np.float64)
    positions[:, 0] = r * sin_th * np.cos(phi)
    positions[:, 1] = r * sin_th * np.sin(phi)
    positions[:, 2] = r * cos_th

    positions -= positions.mean(axis=0)
    velocities = np.zeros((n, 3), dtype=np.float64)
    masses = np.full(n, particle_mass, dtype=np.float64)
    return positions, velocities, masses


class TestColdSphereAgainstIndependentReference:
    # Numeric tolerance, not bit-exact: see module docstring. 1e-10 relative comfortably
    # separates "same formula, different summation order" from "wrong formula".
    REL_TOL = 1e-10

    def test_positions_match_reference_formula(self):
        n = 200
        state = cold_sphere(n=n, seed=12345, dtype=torch.float64, device="cpu")
        ref_positions, _, _ = reference_cold_sphere(
            n, config.SPHERE_RADIUS, config.PARTICLE_MASS, 12345
        )
        got = state.r.numpy()
        assert np.allclose(got, ref_positions, rtol=self.REL_TOL, atol=1e-12)

    def test_velocities_are_exactly_zero(self):
        state = cold_sphere(n=50, seed=1, dtype=torch.float64, device="cpu")
        assert torch.equal(state.v, torch.zeros_like(state.v))

    def test_masses_equal_particle_mass(self):
        n = 50
        state = cold_sphere(n=n, particle_mass=7.0e8, seed=1, dtype=torch.float64, device="cpu")
        assert torch.allclose(state.m, torch.full((n,), 7.0e8, dtype=torch.float64))

    def test_recentring_is_on_positions_not_a_correction_to_radius(self):
        # Section 5.2: "apos a recentragem, |r_i| pode exceder ligeiramente R_0 ... isso e
        # esperado e correto; nao 'corrigir'." A correct implementation must NOT clip |r_i|
        # back under R_0 after recentring.
        n = 500
        state = cold_sphere(n=n, seed=99, dtype=torch.float64, device="cpu")
        radii = torch.linalg.norm(state.r, dim=1)
        # With N=500 samples from this distribution it is expected (not a bug) that at least
        # one particle ends up slightly beyond R_0 after recentring.
        assert radii.max().item() > config.SPHERE_RADIUS * (1.0 - 1e-6), (
            "no particle exceeds R_0 after recentring -- suspicious, spec says this is expected"
        )


class TestColdSphereReferenceValues:
    """Section 5.4: SEED=20190222, N=1000, fp64 -- transcribed reference values."""

    @pytest.fixture(scope="class")
    def ic_state(self):
        return cold_sphere(n=1000, seed=20190222, dtype=torch.float64, device="cpu")

    def test_max_radius(self, ic_state):
        r = torch.linalg.norm(ic_state.r, dim=1)
        assert math.isclose(r.max().item(), 6.323302, rel_tol=1e-6)

    def test_min_pair_separation(self, ic_state):
        diff = ic_state.r.unsqueeze(0) - ic_state.r.unsqueeze(1)
        dist = torch.linalg.norm(diff, dim=-1)
        dist.fill_diagonal_(float("inf"))
        # Section 5.4 prints "0.0574 m" to 4 decimal places (rel_tol 1e-4 as stated in the
        # table cannot be honoured against a display value rounded at the ~5e-5 absolute /
        # ~9e-4 relative level); compared here at that display precision instead.
        assert math.isclose(dist.min().item(), 0.0574, abs_tol=5e-5)

    def test_median_nearest_neighbor_separation(self, ic_state):
        diff = ic_state.r.unsqueeze(0) - ic_state.r.unsqueeze(1)
        dist = torch.linalg.norm(diff, dim=-1)
        dist.fill_diagonal_(float("inf"))
        nearest = dist.min(dim=1).values
        # numpy's median (average of the two middle values for even N=1000) is the standard
        # statistical definition and the one Section 5.4's reference value was almost certainly
        # computed with; torch.Tensor.median() instead returns the lower of the two middle
        # values for even-sized input, which is a different (and non-standard) convention and
        # must not be used for this comparison.
        median = float(np.median(nearest.numpy()))
        assert math.isclose(median, 0.5695, rel_tol=1e-3)

    # Section 5.5 (docs/integradores.md), TOL-CENTER -- normative correction superseding the
    # old absolute "< 1e-13 m" bound from Section 5.4. sum_i r_i == 0 in exact arithmetic
    # (Section 5.2 subtracts the sample mean) but not in floating point: the residual comes
    # from rounding in positions.mean() (an error that reappears multiplied by N) and from
    # rounding of each individual subtraction. Both contributions scale as
    # eps_prec * sum_i |r_i|, i.e. linearly in N and in a length scale -- hence the
    # dimensionless bound below, which the physicist confirmed is data-dependent (reproduced by
    # math.fsum, not a summation-order artifact) and must hold identically in fp32 and fp64:
    #
    #     TOL-CENTER : || sum_i r_i || / (N * eps_prec * R_0) <= 1
    #
    # with R_0 = SPHERE_RADIUS (a configuration constant, not a measured quantity) and eps_prec
    # the unit roundoff of the working dtype. Reference measurements in Section 5.5 give ratios
    # of ~0.12-0.14 (fp64) and ~0.03-0.11 (fp32) for this exact seed/N, comfortably under 1, with
    # a documented sweep-of-ten-seeds worst case of 0.32 -- so the bound has real headroom while
    # still separating a working recentring step from a broken one (a missing/misapplied
    # recentring produces a ratio around 1e14, per Section 5.5's "poder discriminante").
    EPS_FP64 = 2.220446049250313e-16
    EPS_FP32 = 1.1920928955078125e-07

    @staticmethod
    def _tol_center_ratio(state, eps_prec):
        n = state.r.shape[0]
        total_norm = torch.linalg.norm(state.r.sum(dim=0)).item()
        denom = n * eps_prec * config.SPHERE_RADIUS
        return total_norm / denom

    def test_sum_of_positions_after_recentring_fp64(self, ic_state):
        ratio = self._tol_center_ratio(ic_state, self.EPS_FP64)
        assert ratio <= 1.0, f"TOL-CENTER violated (fp64): ratio={ratio}"

    def test_sum_of_positions_after_recentring_fp32(self):
        state = cold_sphere(n=1000, seed=20190222, dtype=torch.float32, device="cpu")
        ratio = self._tol_center_ratio(state, self.EPS_FP32)
        assert ratio <= 1.0, f"TOL-CENTER violated (fp32): ratio={ratio}"

    def test_kinetic_energy_zero_exactly(self, ic_state):
        assert torch.equal(ic_state.v, torch.zeros_like(ic_state.v))

    def test_momentum_zero_exactly(self):
        # P(t=0) = 0.0 exactly, per Section 5.4 ("binario"): m*v with v == 0 exactly.
        state = cold_sphere(n=1000, seed=20190222, dtype=torch.float64, device="cpu")
        p = (state.m.unsqueeze(-1) * state.v).sum(dim=0)
        assert torch.equal(p, torch.zeros_like(p))


class TestColdSphereBitReproducibility:
    """
    docs/api-contract.md: "cold_sphere ... e reproduzivel bit a bit para a mesma semente, o
    mesmo n e o mesmo dtype, independentemente do device."
    """

    def test_repeated_call_same_dtype_cpu_is_bit_identical(self):
        s1 = cold_sphere(n=300, seed=555, dtype=torch.float64, device="cpu")
        s2 = cold_sphere(n=300, seed=555, dtype=torch.float64, device="cpu")
        assert torch.equal(s1.r, s2.r)
        assert torch.equal(s1.v, s2.v)
        assert torch.equal(s1.m, s2.m)

    def test_repeated_call_fp32_is_bit_identical(self):
        s1 = cold_sphere(n=300, seed=555, dtype=torch.float32, device="cpu")
        s2 = cold_sphere(n=300, seed=555, dtype=torch.float32, device="cpu")
        assert torch.equal(s1.r, s2.r)

    def test_different_seed_gives_different_positions(self):
        s1 = cold_sphere(n=300, seed=1, dtype=torch.float64, device="cpu")
        s2 = cold_sphere(n=300, seed=2, dtype=torch.float64, device="cpu")
        assert not torch.equal(s1.r, s2.r)

    @pytest.mark.gpu
    def test_cpu_and_cuda_are_bit_identical(self):
        require_cuda()
        s_cpu = cold_sphere(n=300, seed=555, dtype=torch.float64, device="cpu")
        s_cuda = cold_sphere(n=300, seed=555, dtype=torch.float64, device="cuda")
        assert torch.equal(s_cpu.r, s_cuda.r.cpu())
        assert torch.equal(s_cpu.v, s_cuda.v.cpu())
        assert torch.equal(s_cpu.m, s_cuda.m.cpu())

    @pytest.mark.gpu
    def test_cpu_and_cuda_are_bit_identical_fp32(self):
        require_cuda()
        s_cpu = cold_sphere(n=300, seed=555, dtype=torch.float32, device="cpu")
        s_cuda = cold_sphere(n=300, seed=555, dtype=torch.float32, device="cuda")
        assert torch.equal(s_cpu.r, s_cuda.r.cpu())


@pytest.mark.golden
class TestColdSphereGoldenFile:
    """
    Section 5.3: the N=1000, seed=20190222 IC must be committed once to
    data/ic_sphere_N1000_seed20190222.npz. A separate, explicitly marked test verifies the
    sampler reproduces the file bit for bit. This is that test.

    The document also requires a SHA-256 of the file "registrado no repositorio", but does not
    say where that registration lives (a checksum file, config.py, a CI secret?), so that half
    of the requirement is not testable against the two source documents alone -- see final
    report.
    """

    GOLDEN_PATH = "data/ic_sphere_N1000_seed20190222.npz"

    def test_matches_committed_golden_file(self):
        import os

        if not os.path.exists(self.GOLDEN_PATH):
            pytest.skip(f"golden IC file not present at {self.GOLDEN_PATH}; nothing to check yet")

        data = np.load(self.GOLDEN_PATH)
        state = cold_sphere(n=1000, seed=20190222, dtype=torch.float64, device="cpu")
        assert np.array_equal(state.r.numpy(), data["positions"])
        assert np.array_equal(state.v.numpy(), data["velocities"])
        assert np.array_equal(state.m.numpy(), data["masses"])


class TestTwoBodyKepler:
    """INV-5: TWOBODY_KEPLER, eps = 0.0, a = 1.0, e = 0.5, start at apoapsis."""

    def test_default_configuration(self):
        state = two_body_kepler()
        assert state.n == 2
        r_apo = config.KEPLER_A * (1 + config.KEPLER_E)
        assert math.isclose(r_apo, 1.5, rel_tol=1e-9)

        expected_r0 = torch.tensor([+0.75, 0.0, 0.0], dtype=torch.float64)
        expected_r1 = torch.tensor([-0.75, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(state.r[0], expected_r0, atol=1e-9)
        assert torch.allclose(state.r[1], expected_r1, atol=1e-9)

        v_half = 0.1054678466
        expected_v0 = torch.tensor([0.0, +v_half, 0.0], dtype=torch.float64)
        expected_v1 = torch.tensor([0.0, -v_half, 0.0], dtype=torch.float64)
        assert torch.allclose(state.v[0], expected_v0, atol=1e-9)
        assert torch.allclose(state.v[1], expected_v1, atol=1e-9)

    def test_masses_equal_and_default_particle_mass(self):
        state = two_body_kepler()
        assert torch.allclose(state.m, torch.full((2,), config.PARTICLE_MASS, dtype=torch.float64))

    def test_center_of_mass_at_rest_at_origin(self):
        state = two_body_kepler()
        com_r = (state.m.unsqueeze(-1) * state.r).sum(dim=0) / state.m.sum()
        com_v = (state.m.unsqueeze(-1) * state.v).sum(dim=0) / state.m.sum()
        assert torch.linalg.norm(com_r).item() < 1e-9
        assert torch.linalg.norm(com_v).item() < 1e-9


class TestTwoBodyCircular:
    """INV-6: TWOBODY_CIRC, eps = 0.05, separation d = 1.0."""

    def test_default_configuration(self):
        state = two_body_circular()
        expected_r0 = torch.tensor([+0.5, 0.0, 0.0], dtype=torch.float64)
        expected_r1 = torch.tensor([-0.5, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(state.r[0], expected_r0, atol=1e-9)
        assert torch.allclose(state.r[1], expected_r1, atol=1e-9)

        v_half = 0.1823338995
        expected_v0 = torch.tensor([0.0, +v_half, 0.0], dtype=torch.float64)
        expected_v1 = torch.tensor([0.0, -v_half, 0.0], dtype=torch.float64)
        assert torch.allclose(state.v[0], expected_v0, atol=1e-9)
        assert torch.allclose(state.v[1], expected_v1, atol=1e-9)

    def test_separation_equals_requested_value(self):
        state = two_body_circular(separation=1.0)
        sep = torch.linalg.norm(state.r[0] - state.r[1]).item()
        assert math.isclose(sep, 1.0, rel_tol=1e-9)
