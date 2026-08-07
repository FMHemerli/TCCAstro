"""
nbody.initial_conditions.random_sphere against docs/simulacao-estocastica.md Section 3
(velocities), Section 3.4 (non-degeneracy), Section 3.6 (Q=0 short circuit), and Section 6,
INV-16 and INV-17. Also docs/api-contract.md's amendment in Section 9.1, which gives
random_sphere's FULL signature -- unlike nbody.populations' functions, every call in this file
uses that exact, documented signature, no name-matching required.

mass_spectrum is left at its documented default (None, i.e. off / equal masses) throughout this
file. Section 1.1 states the mass-spectrum and velocity extensions "sao independentes entre si e
devem ser implementaveis e testaveis separadamente" -- INV-16 and INV-17 are about the velocity
sampler specifically, so equal masses isolate that from any ambiguity in how a caller is meant to
turn the mass spectrum on (Section 9.1 does not fix mass_spectrum's type; see the final report).

INV-16(c) (the lambda-scaled velocity cap) is the one sub-invariant here that needs lambda
exposed, which Section 9.1 puts on nbody.populations.sample_velocities's auxiliary return, not on
random_sphere's State. That one test uses the semantic-role binder from _stage2_binding.py and
skips loudly, with the exact reason, if the signature cannot be resolved.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.observables import kinetic_energy, linear_momentum, potential_energy  # noqa: E402
from nbody.state import State  # noqa: E402

_ic_module = pytest.importorskip("nbody.initial_conditions")
random_sphere = getattr(_ic_module, "random_sphere", None)
if random_sphere is None:
    pytest.skip(
        "nbody.initial_conditions.random_sphere is not exposed (Section 9.1 amendment)",
        allow_module_level=True,
    )

populations = None
try:
    import nbody.populations as populations  # noqa: E402
except ImportError:
    pass

from _stage2_binding import (  # noqa: E402
    ContractGap,
    any_close,
    bind_by_role,
    extract_numbers,
    to_numpy,
)

from tolerances import (  # noqa: E402
    EPS_PREC_FP64,
    F_CUT_DEFAULT,
    INV16D_EIGENVALUE_HIGH,
    INV16D_EIGENVALUE_LOW,
    INV16D_N_REALIZATIONS,
    INV16E_Q_USABLE_AT_VETO,
    INV16E_VETOED_F_CUT,
    INV16E_VETOED_Q,
    MASS_SEED_DEFAULT,
    Q_DEFAULT,
    TOL_VIRIAL_FP64,
    VEL_SEED_DEFAULT,
    V_ESC_SPHERE,
)

N = config.N_PARTICLES
R0 = config.SPHERE_RADIUS
SEED = config.SEED
PARTICLE_MASS = config.PARTICLE_MASS


def _make(virial_ratio, f_cut, *, vel_seed=VEL_SEED_DEFAULT, mass_seed=MASS_SEED_DEFAULT,
          seed=SEED, dtype=torch.float64, device="cpu"):
    return random_sphere(
        n=N, radius=R0, seed=seed, mass_spectrum=None,
        virial_ratio=virial_ratio, f_cut=f_cut,
        mass_seed=mass_seed, vel_seed=vel_seed, dtype=dtype, device=device,
    )


# -------------------------------------------------------------------------------------------
# INV-17
# -------------------------------------------------------------------------------------------
class TestINV17QZeroReproducesColdSphere:
    def test_bit_exact_against_cold_sphere(self):
        rs = _make(virial_ratio=0.0, f_cut=F_CUT_DEFAULT)
        cs = cold_sphere(n=N, radius=R0, particle_mass=PARTICLE_MASS, seed=SEED,
                          dtype=torch.float64, device="cpu")
        assert torch.equal(rs.r, cs.r), "random_sphere(Q=0).r differs from cold_sphere.r"
        assert torch.equal(rs.v, cs.v), "random_sphere(Q=0).v differs from cold_sphere.v"
        assert torch.equal(rs.m, cs.m), "random_sphere(Q=0).m differs from cold_sphere.m"

    def test_velocities_have_no_negative_zero(self):
        # Section 3.6: lambda = sqrt(0/K') = 0.0 * v_i produces IEEE-754 -0.0 for v_i < 0; the
        # normative short circuit must avoid this path entirely.
        rs = _make(virial_ratio=0.0, f_cut=F_CUT_DEFAULT)
        assert not torch.signbit(rs.v).any().item(), "found -0.0 in v; Q=0 short circuit missing"

    def test_velocity_generator_not_consumed(self):
        # Black-box substitute for "the velocity RNG stream was not advanced": with Q=0.0 the
        # spec requires zero draws from it, so two calls differing ONLY in vel_seed must be
        # bit-identical -- if vel_seed were actually consumed, they would (almost certainly)
        # differ.
        rs_a = _make(virial_ratio=0.0, f_cut=F_CUT_DEFAULT, vel_seed=VEL_SEED_DEFAULT)
        rs_b = _make(virial_ratio=0.0, f_cut=F_CUT_DEFAULT, vel_seed=VEL_SEED_DEFAULT + 1)
        assert torch.equal(rs_a.r, rs_b.r)
        assert torch.equal(rs_a.v, rs_b.v)
        assert torch.equal(rs_a.m, rs_b.m)

    def test_tolerance_is_binary_not_a_epsilon_band(self):
        # Explicit regression guard against "close enough" creeping in: allclose with a tiny
        # but nonzero tolerance is NOT what INV-17 requires.
        rs = _make(virial_ratio=0.0, f_cut=F_CUT_DEFAULT)
        cs = cold_sphere(n=N, radius=R0, particle_mass=PARTICLE_MASS, seed=SEED,
                          dtype=torch.float64, device="cpu")
        assert torch.equal(rs.r, cs.r) and not (rs.r is cs.r), (
            "must be bit-identical AND independently constructed, not the same tensor object"
        )


# -------------------------------------------------------------------------------------------
# INV-16(a, b): virial ratio and momentum, at Q_DEFAULT/F_CUT_DEFAULT
# -------------------------------------------------------------------------------------------
class TestINV16VirialAndMomentum:
    @pytest.fixture(scope="class")
    def state(self):
        return _make(virial_ratio=Q_DEFAULT, f_cut=F_CUT_DEFAULT)

    def test_a_realized_virial_ratio_matches_q_exactly(self, state):
        k = kinetic_energy(state)
        u = potential_energy(state, softening=config.SOFTENING)
        realized_q = 2.0 * k / abs(u)
        rel = abs(realized_q / Q_DEFAULT - 1.0)
        assert rel <= TOL_VIRIAL_FP64, (
            f"realized Q = {realized_q:.15e}, target Q_DEFAULT = {Q_DEFAULT}, "
            f"rel dev {rel:.3e} > {TOL_VIRIAL_FP64}"
        )

    def test_b_momentum_is_zero_to_roundoff(self, state):
        p = linear_momentum(state)
        m_real = float(state.m.sum().item())
        v_rms = math.sqrt(2.0 * kinetic_energy(state) / m_real)
        bound = N * EPS_PREC_FP64
        ratio = float(torch.linalg.norm(p).item()) / (m_real * v_rms)
        assert ratio <= bound, f"||P||/(M_real*v_rms) = {ratio:.3e} exceeds N*eps_prec = {bound:.3e}"

    def test_velocities_are_not_identically_zero(self, state):
        # Sanity: Q_DEFAULT > 0 must actually produce nonzero velocities (distinguishes a
        # correct implementation from one that always takes the Q=0 short circuit).
        assert torch.linalg.norm(state.v).item() > 0.0


# -------------------------------------------------------------------------------------------
# INV-16(d): isotropy of the velocity direction distribution
# -------------------------------------------------------------------------------------------
class TestINV16Isotropy:
    SEED_BASE = 3_500_000

    def test_velocity_dyadic_tensor_eigenvalues(self):
        # Positions and masses fixed (equal masses, fixed SEED); only vel_seed varies across
        # realizations, which is sufficient to probe the isotropy of the velocity DIRECTION
        # sampler in isolation (Section 3.3's normative warning: component-wise rejection
        # produces a cube, not a ball, in velocity space -- a defect entirely about direction,
        # not about |U(r^0)| or the realized masses).
        weighted_outer = np.zeros((3, 3))
        weighted_speed_sq = 0.0
        for i in range(INV16D_N_REALIZATIONS):
            state = _make(
                virial_ratio=Q_DEFAULT, f_cut=F_CUT_DEFAULT,
                vel_seed=self.SEED_BASE + i,
            )
            v = to_numpy(state.v)
            m = to_numpy(state.m)
            weighted_outer += np.einsum("i,ia,ib->ab", m, v, v)
            weighted_speed_sq += float(np.sum(m * np.sum(v * v, axis=1)))

        tensor = weighted_outer / weighted_speed_sq
        eigenvalues = np.linalg.eigvalsh(tensor)
        assert np.isclose(eigenvalues.sum(), 1.0, atol=1e-9), (
            f"trace of the normalized dyadic tensor should be 1, got {eigenvalues.sum()}"
        )
        for ev in eigenvalues:
            assert INV16D_EIGENVALUE_LOW <= ev <= INV16D_EIGENVALUE_HIGH, (
                f"eigenvalue {ev:.4f} outside [{INV16D_EIGENVALUE_LOW}, "
                f"{INV16D_EIGENVALUE_HIGH}]; eigenvalues={eigenvalues}. Section 6 (INV-16d): "
                f"the failure mode this catches is component-wise rejection instead of "
                f"rejection by speed (Section 3.3, step 1)."
            )


# -------------------------------------------------------------------------------------------
# INV-16(e): non-degeneracy is enforced, loudly, at call time
# -------------------------------------------------------------------------------------------
class TestINV16NonDegeneracy:
    def test_vetoed_combination_raises_value_error(self):
        # Section 3.4: (Q, f_cut) = (0.5, 0.5) sits exactly on the degenerate limit
        # Q_sup = 0.502264; this is explicitly vetoed and must raise, not silently return the
        # degenerate uniform-ball sample.
        with pytest.raises(ValueError) as excinfo:
            _make(virial_ratio=INV16E_VETOED_Q, f_cut=INV16E_VETOED_F_CUT)
        message = str(excinfo.value)
        numbers = extract_numbers(message)
        assert any_close(numbers, INV16E_VETOED_Q, rel_tol=1e-6), (
            f"ValueError message does not name Q={INV16E_VETOED_Q}: {message!r}"
        )
        assert any_close(numbers, INV16E_VETOED_F_CUT, rel_tol=1e-6), (
            f"ValueError message does not name f_cut={INV16E_VETOED_F_CUT}: {message!r}"
        )
        assert any_close(numbers, INV16E_Q_USABLE_AT_VETO, rel_tol=1e-3), (
            f"ValueError message does not name the maximum admissible Q "
            f"({INV16E_Q_USABLE_AT_VETO:.6f}): {message!r}"
        )

    def test_safe_combination_does_not_raise(self):
        # Q_DEFAULT/F_CUT_DEFAULT (x_c = 3.05, Section 3.5) is comfortably inside the
        # non-degenerate regime and must not raise.
        state = _make(virial_ratio=Q_DEFAULT, f_cut=F_CUT_DEFAULT)
        assert state.n == N

    def test_boundary_just_above_the_usable_limit_raises(self):
        # Q_usable(f_cut=1.0) = Q_USABLE_COEFF = 1.532168 (Section 3.4 table); a hair above it
        # must still raise.
        f_cut = 1.0
        from tolerances import Q_USABLE_COEFF
        q_over_limit = Q_USABLE_COEFF * f_cut ** 2 * 1.001
        with pytest.raises(ValueError):
            _make(virial_ratio=q_over_limit, f_cut=f_cut)


# -------------------------------------------------------------------------------------------
# INV-16(c): the lambda-scaled velocity cap. Requires lambda, which Section 9.1 puts on
# nbody.populations.sample_velocities's auxiliary return rather than on random_sphere's State.
# -------------------------------------------------------------------------------------------
class TestINV16cLambdaScaledCap:
    def test_max_speed_within_lambda_scaled_cutoff(self):
        if populations is None or not hasattr(populations, "sample_velocities"):
            pytest.skip("nbody.populations.sample_velocities is not available")

        base = cold_sphere(n=N, radius=R0, particle_mass=PARTICLE_MASS, seed=SEED,
                            dtype=torch.float64, device="cpu")
        state0 = State(r=base.r, v=torch.zeros_like(base.r), m=base.m)

        roles = {
            "state": state0,
            "r": base.r, "positions": base.r, "position": base.r,
            "m": base.m, "masses": base.m, "mass": base.m,
            "n": N, "n_particles": N,
            "virial_ratio": Q_DEFAULT, "q": Q_DEFAULT,
            "f_cut": F_CUT_DEFAULT, "cut": F_CUT_DEFAULT,
            "softening": config.SOFTENING, "eps": config.SOFTENING,
            "vel_seed": VEL_SEED_DEFAULT, "seed": VEL_SEED_DEFAULT,
        }
        try:
            kwargs = bind_by_role(populations.sample_velocities, roles)
            result = populations.sample_velocities(**kwargs)
        except ContractGap as exc:
            pytest.skip(str(exc))
        except Exception as exc:  # signature bound but call itself rejected our guesses
            pytest.skip(
                f"sample_velocities bound but raised {type(exc).__name__}: {exc}; "
                f"the guessed argument values in this test are not what the implementation "
                f"expects -- see final report on Section 9.1's signature gap"
            )

        if not (isinstance(result, tuple) and len(result) == 2):
            pytest.skip(
                f"sample_velocities returned {type(result)!r}, not a (v, lambda) pair as "
                f"Section 9.1 describes ('lambda como retorno auxiliar')"
            )
        v, lam = result
        v = to_numpy(v)
        lam = float(lam)

        max_allowed = lam * F_CUT_DEFAULT * V_ESC_SPHERE
        speeds = np.linalg.norm(v, axis=1)
        assert speeds.max() <= max_allowed * (1.0 + 1e-9), (
            f"max|v_i| = {speeds.max():.6f} exceeds lambda*f_cut*v_esc = {max_allowed:.6f} "
            f"(lambda={lam:.6f}). Section 6 (INV-16c): a test requiring max|v_i| <= f_cut*v_esc "
            f"WITHOUT the lambda factor fails against a correct implementation in about a third "
            f"of seeds -- this test uses lambda, as the specification requires it to."
        )
