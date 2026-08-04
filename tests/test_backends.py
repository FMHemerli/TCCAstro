"""
nbody.backends against docs/api-contract.md: the registry, the Backend protocol, the six-rung
comparison ladder, tiling equivalence, and the "no silent fallback" error contract.

    BACKEND_NAMES = ("python_pure", "torch_eager", "torch_compiled", "triton")
    get_backend(name) -> Backend
    available_backends(device) -> tuple[str, ...]

    | rung | backend         | device |
    |------|-----------------|--------|
    | 1    | python_pure     | cpu    |
    | 2    | torch_eager     | cpu    |
    | 3    | torch_compiled  | cpu    |
    | 4    | torch_eager     | cuda   |
    | 5    | torch_compiled  | cuda   |
    | 6    | triton          | cuda   |

Equivalence between rungs, and between tile sizes, is judged at TOL-IMPL-SHORT (docs Section
6.3): 1e-13 relative for fp64, 1e-4 for fp32, on err(r) over a short trajectory (t << 0.5 t_ff,
well inside the window Section 6.2 declares valid for cross-implementation trajectory
comparison).
"""
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import BackendUnavailable, config  # noqa: E402
from nbody.backends import BACKEND_NAMES, available_backends, get_backend  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.integrators import FORCE_EVALS_PER_STEP, INTEGRATOR_NAMES, integrate  # noqa: E402

from conftest import require_cuda  # noqa: E402
from tolerances import (  # noqa: E402
    TOL_IMPL_SHORT_FP32,
    TOL_IMPL_SHORT_FP64,
)

CPU = torch.device("cpu")


def cpu_backends_that_work(dtype):
    """python_pure and torch_eager are always expected to work on CPU. torch_compiled may or
    may not (Section: this sandbox lacks g++); if it raises BackendUnavailable at call time
    that is itself a pass for the error-contract test elsewhere, so here it is simply left out
    of the equivalence set when unavailable -- excluding an unavailable rung from an
    equivalence comparison is not the same claim as "it is correct", and no such claim is made."""
    working = []
    n = 20
    r = torch.zeros(n, 3, dtype=dtype)
    r[:, 0] = torch.arange(n, dtype=dtype)
    m = torch.full((n,), config.PARTICLE_MASS, dtype=dtype)
    for name in ("python_pure", "torch_eager", "torch_compiled"):
        backend = get_backend(name)
        try:
            backend.accelerations(r, m, softening=0.05)
        except BackendUnavailable:
            continue
        working.append(name)
    return working


class TestRegistry:
    def test_backend_names_match_contract(self):
        assert BACKEND_NAMES == ("python_pure", "torch_eager", "torch_compiled", "triton")

    def test_get_backend_returns_named_backend(self):
        for name in BACKEND_NAMES:
            backend = get_backend(name)
            assert backend.name == name

    def test_get_backend_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError):
            get_backend("not_a_real_backend")

    def test_integrator_names_match_contract(self):
        assert INTEGRATOR_NAMES == ("euler", "symplectic_euler", "velocity_verlet", "rk4")

    def test_force_evals_per_step_table(self):
        # Section 3.5.
        assert FORCE_EVALS_PER_STEP["euler"] == 1
        assert FORCE_EVALS_PER_STEP["symplectic_euler"] == 1
        assert FORCE_EVALS_PER_STEP["velocity_verlet"] == 1
        assert FORCE_EVALS_PER_STEP["rk4"] == 4


class TestSupportsIsHonest:
    """docs/api-contract.md: 'python_pure nao suporta GPU; triton nao suporta CPU.' These two
    facts are the only ones the contract states explicitly, and hold regardless of whether a
    CUDA device physically exists (they are structural, not a hardware probe)."""

    def test_python_pure_does_not_support_cuda(self):
        backend = get_backend("python_pure")
        assert backend.supports(torch.device("cuda")) is False

    def test_triton_does_not_support_cpu(self):
        backend = get_backend("triton")
        assert backend.supports(torch.device("cpu")) is False

    def test_python_pure_supports_cpu(self):
        backend = get_backend("python_pure")
        assert backend.supports(CPU) is True

    def test_torch_eager_supports_cpu(self):
        backend = get_backend("torch_eager")
        assert backend.supports(CPU) is True

    def test_available_backends_is_self_consistent(self):
        # available_backends(device) must agree with each backend's own supports(device),
        # for any device -- this is the only property the contract actually specifies about
        # available_backends beyond it being a subset of BACKEND_NAMES.
        reported = set(available_backends(CPU))
        expected = {name for name in BACKEND_NAMES if get_backend(name).supports(CPU)}
        assert reported == expected
        assert reported.issubset(set(BACKEND_NAMES))

    def test_available_backends_excludes_triton_on_cpu(self):
        assert "triton" not in available_backends(CPU)

    def test_available_backends_excludes_python_pure_on_cuda(self):
        assert "python_pure" not in available_backends(torch.device("cuda"))


class TestNoSilentFallback:
    """docs/api-contract.md: an unavailable backend raises BackendUnavailable naming what is
    missing; it never falls back to a different device/precision/code path silently."""

    def test_triton_on_cpu_tensors_raises_backend_unavailable(self):
        backend = get_backend("triton")
        r = torch.zeros(5, 3, dtype=torch.float64)
        m = torch.ones(5, dtype=torch.float64)
        with pytest.raises(BackendUnavailable):
            backend.accelerations(r, m, softening=0.05)

    @pytest.mark.gpu
    def test_python_pure_on_cuda_tensors_raises_backend_unavailable(self):
        require_cuda()
        backend = get_backend("python_pure")
        r = torch.zeros(5, 3, dtype=torch.float64, device="cuda")
        m = torch.ones(5, dtype=torch.float64, device="cuda")
        with pytest.raises(BackendUnavailable):
            backend.accelerations(r, m, softening=0.05)

    def test_unavailable_backend_error_names_something(self):
        backend = get_backend("triton")
        r = torch.zeros(5, 3, dtype=torch.float64)
        m = torch.ones(5, dtype=torch.float64)
        with pytest.raises(BackendUnavailable) as excinfo:
            backend.accelerations(r, m, softening=0.05)
        assert len(str(excinfo.value)) > 0, "BackendUnavailable must name what is missing"

    def test_torch_compiled_cpu_either_works_or_names_unavailability(self):
        """
        This sandbox lacks g++, so torch.compile is expected to be unavailable on CPU here
        (per task instructions). The contract requires that unavailability surface as
        BackendUnavailable, never as a raw torch/inductor exception and never as a silent
        fallback to eager. Both branches below are a legitimate pass; what is NOT a pass is
        any exception type other than BackendUnavailable escaping, or the two backends
        disagreeing once torch_compiled *does* produce output.
        """
        backend = get_backend("torch_compiled")
        eager = get_backend("torch_eager")
        r = torch.zeros(20, 3, dtype=torch.float64)
        r[:, 0] = torch.arange(20, dtype=torch.float64)
        m = torch.full((20,), config.PARTICLE_MASS, dtype=torch.float64)

        try:
            a_compiled = backend.accelerations(r, m, softening=0.05)
        except BackendUnavailable as e:
            assert len(str(e)) > 0
            return

        a_eager = eager.accelerations(r, m, softening=0.05)
        rel_err = (
            torch.linalg.norm(a_compiled - a_eager) / torch.linalg.norm(a_eager)
        ).item()
        assert rel_err <= TOL_IMPL_SHORT_FP64


class TestAccelerationsContract:
    def test_shape_dtype_device_preserved(self):
        backend = get_backend("torch_eager")
        n = 30
        r = torch.randn(n, 3, dtype=torch.float64)
        m = torch.full((n,), config.PARTICLE_MASS, dtype=torch.float64)
        a = backend.accelerations(r, m, softening=0.05)
        assert a.shape == (n, 3)
        assert a.dtype == torch.float64
        assert a.device == r.device

    def test_accepts_softening_zero(self):
        backend = get_backend("torch_eager")
        n = 10
        r = torch.randn(n, 3, dtype=torch.float64)
        m = torch.full((n,), config.PARTICLE_MASS, dtype=torch.float64)
        a = backend.accelerations(r, m, softening=0.0)
        assert torch.isfinite(a).all()


class TestTileSizeEquivalence:
    """docs/api-contract.md: any valid tile_size must produce the same result as any other,
    within TOL-IMPL-SHORT -- the reduction order changes, the value must not."""

    @pytest.mark.parametrize("dtype,tol", [(torch.float64, TOL_IMPL_SHORT_FP64),
                                            (torch.float32, TOL_IMPL_SHORT_FP32)])
    def test_tile_sizes_agree_on_torch_eager(self, dtype, tol):
        backend = get_backend("torch_eager")
        state = cold_sphere(n=500, seed=20190222, dtype=dtype, device="cpu")
        reference = backend.accelerations(state.r, state.m, softening=0.05, tile_size=None)
        for tile_size in (1, 7, 64, 500):
            a = backend.accelerations(state.r, state.m, softening=0.05, tile_size=tile_size)
            rel_err = (
                torch.linalg.norm((a - reference).double())
                / torch.linalg.norm(reference.double())
            ).item()
            assert rel_err <= tol, (
                f"tile_size={tile_size} disagrees with tile_size=None by {rel_err:.3e} "
                f"(dtype={dtype}, tol={tol:.1e})"
            )


class TestLadderEquivalenceSingleEvaluation:
    """Single-evaluation equivalence between CPU rungs of the ladder (degraus 1-3), at the
    cold_sphere IC. GPU rungs (4-6) are covered separately and skip loudly without a CUDA
    device."""

    @pytest.mark.parametrize("dtype,tol", [(torch.float64, TOL_IMPL_SHORT_FP64),
                                            (torch.float32, TOL_IMPL_SHORT_FP32)])
    def test_cpu_backends_agree_at_ic(self, dtype, tol):
        state = cold_sphere(n=500, seed=20190222, dtype=dtype, device="cpu")
        names = cpu_backends_that_work(dtype)
        assert "torch_eager" in names, "torch_eager must always work on CPU"
        results = {name: get_backend(name).accelerations(state.r, state.m, softening=0.05)
                   for name in names}
        reference = results["torch_eager"]
        for name, a in results.items():
            rel_err = (
                torch.linalg.norm((a - reference).double())
                / torch.linalg.norm(reference.double())
            ).item()
            assert rel_err <= tol, f"{name} disagrees with torch_eager by {rel_err:.3e}"

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype,tol", [(torch.float64, TOL_IMPL_SHORT_FP64),
                                            (torch.float32, TOL_IMPL_SHORT_FP32)])
    def test_cuda_backends_agree_with_cpu_torch_eager(self, dtype, tol):
        require_cuda()
        state_cpu = cold_sphere(n=500, seed=20190222, dtype=dtype, device="cpu")
        state_cuda = state_cpu.to(device=torch.device("cuda"))
        reference = get_backend("torch_eager").accelerations(
            state_cpu.r, state_cpu.m, softening=0.05
        )
        for name in ("torch_eager", "torch_compiled", "triton"):
            backend = get_backend(name)
            if not backend.supports(torch.device("cuda")):
                continue
            try:
                a = backend.accelerations(state_cuda.r, state_cuda.m, softening=0.05)
            except BackendUnavailable:
                continue
            rel_err = (
                torch.linalg.norm((a.cpu() - reference).double())
                / torch.linalg.norm(reference.double())
            ).item()
            assert rel_err <= tol, f"{name}@cuda disagrees with torch_eager@cpu by {rel_err:.3e}"


class TestLadderEquivalenceShortTrajectory:
    """A short integrated trajectory (well inside the t <= 0.5 t_ff validity window of Section
    6.2) must agree between backends within TOL-IMPL-SHORT, measured with the err(r) metric of
    Section 6.1."""

    def test_cpu_backends_agree_over_short_trajectory(self, err_metric_fn):
        dtype = torch.float64
        n = 300
        state = cold_sphere(n=n, seed=20190222, dtype=dtype, device="cpu")
        names = [nm for nm in cpu_backends_that_work(dtype) if nm != "python_pure"]
        # python_pure excluded here purely for wall-clock cost (pure-Python O(N^2) loop over
        # many steps); it is already checked at a single evaluation above.
        assert "torch_eager" in names

        dt = config.DT_COLLAPSE
        n_steps = 20  # t = 0.01 s = 0.0048 t_ff, deep inside the t <= 0.5 t_ff window.
        results = {}
        for name in names:
            final = integrate(
                state, integrator="velocity_verlet", backend=name, dt=dt, n_steps=n_steps,
                softening=0.05,
            )
            results[name] = final.r
        reference = results["torch_eager"]
        for name, r_final in results.items():
            err = err_metric_fn(r_final, reference, config.SPHERE_RADIUS, n)
            assert err <= TOL_IMPL_SHORT_FP64, (
                f"{name} disagrees with torch_eager over short trajectory: err={err:.3e}"
            )


class TestFusedSymplecticEulerStep:
    """docs/api-contract.md: fused_symplectic_euler_step replicates the 2019 kernel with force
    and state update fused, existing only for the comparative benchmark; it is not used by the
    integrators, but it must compute the same recurrence as one step of symplectic_euler."""

    def test_matches_one_step_of_symplectic_euler(self):
        backend = get_backend("torch_eager")
        state = cold_sphere(n=200, seed=20190222, dtype=torch.float64, device="cpu")
        dt = config.DT_COLLAPSE

        via_integrate = integrate(
            state, integrator="symplectic_euler", backend=backend, dt=dt, n_steps=1,
            softening=0.05,
        )
        via_fused = backend.fused_symplectic_euler_step(state, dt, softening=0.05)

        rel_err_r = (
            torch.linalg.norm(via_fused.r - via_integrate.r)
            / torch.linalg.norm(via_integrate.r)
        ).item()
        rel_err_v = (
            torch.linalg.norm(via_fused.v - via_integrate.v)
            / torch.linalg.norm(via_integrate.v)
        ).item()
        assert rel_err_r <= TOL_IMPL_SHORT_FP64
        assert rel_err_v <= TOL_IMPL_SHORT_FP64
