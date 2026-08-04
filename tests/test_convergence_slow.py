"""
INV-8: empirical convergence order, docs/integradores.md Section 7 and Section 6.1.

Loads the committed reference trajectory (Section 6.1: RK4 fp64 at dt_ref = T_conv/16384,
regeneration is "responsabilidade de um script versionado, nao do teste" -- this test only
consumes data/reference_conv_N1000.npz, it does not build it). If that file is not present,
this test skips loudly with a clear reason rather than silently reporting nothing.

The acceptance windows below are the MEASURED windows from Section 7 (INV-8), not textbook
order: symplectic_euler is order 1 in position but order 2 in velocity; euler's exponent trends
toward 1 from below without reaching it; rk4's measured exponent is ~4.95, not 4, in both
position and velocity, and the document explicitly says not to write a test requiring p ~= 4 for
rk4. See docs/integradores.md Section 7, INV-8, ressalvas 1-4.

Marked slow: N=1000, O(N^2) force evaluations, full CONV_N_STEPS ladder (64..2048, plus 4096 for
euler) for four integrators. On the order of 3*10^10 pairwise-interaction-equivalent operations
in total; expect this to take from tens of seconds to a few minutes on CPU. Deselect with
`-m "not slow"` for per-commit CI.
"""
import math
import os

import numpy as np
import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.initial_conditions import cold_sphere  # noqa: E402
from nbody.integrators import integrate  # noqa: E402

pytestmark = pytest.mark.slow

BACKEND = get_backend("torch_eager")
REFERENCE_PATH = "data/reference_conv_N1000.npz"


def load_reference():
    if not os.path.exists(REFERENCE_PATH):
        pytest.skip(
            f"reference trajectory not present at {REFERENCE_PATH}; INV-8 cannot be checked "
            "without it (Section 6.1: regenerating it is a script's job, not this test's)"
        )
    data = np.load(REFERENCE_PATH)
    # Key names are not specified by either source document beyond "posicoes e velocidades";
    # tried against the naming convention used for the golden IC file (Section 5.3).
    if "positions" in data and "velocities" in data:
        return data["positions"], data["velocities"]
    if "r" in data and "v" in data:
        return data["r"], data["v"]
    raise KeyError(
        f"{REFERENCE_PATH} has keys {list(data.keys())}; expected "
        "('positions', 'velocities') or ('r', 'v')"
    )


def err_r_metric(r, r_ref, n):
    diff = r.to(torch.float64).numpy() - r_ref
    return math.sqrt(np.sum(diff * diff)) / (math.sqrt(n) * config.SPHERE_RADIUS)


def err_v_metric(v, v_ref, n):
    diff = v.to(torch.float64).numpy() - v_ref
    return math.sqrt(np.sum(diff * diff)) / (math.sqrt(n) * config.V_CHAR)


def run_ladder(integrator, n_steps_list, r_ref, v_ref, n):
    err_r, err_v = {}, {}
    for n_steps in n_steps_list:
        state = cold_sphere(n=n, seed=config.SEED, dtype=torch.float64, device="cpu")
        dt = config.T_CONV / n_steps
        result = integrate(state, integrator=integrator, backend=BACKEND, dt=dt,
                            n_steps=n_steps, softening=0.05)
        err_r[n_steps] = err_r_metric(result.r, r_ref, n)
        err_v[n_steps] = err_v_metric(result.v, v_ref, n)
    return err_r, err_v


def orders(err_dict, n_steps_list):
    p = {}
    for i in range(1, len(n_steps_list)):
        n = n_steps_list[i]
        n_prev = n_steps_list[i - 1]
        p[n] = math.log2(err_dict[n_prev] / err_dict[n])
    return p


LADDER = list(config.CONV_N_STEPS)  # (64, 128, 256, 512, 1024, 2048)
N = config.N_PARTICLES


@pytest.fixture(scope="module")
def reference():
    r_ref, v_ref = load_reference()
    return r_ref, v_ref


class TestSymplecticEulerConvergence:
    @pytest.fixture(scope="class")
    def data(self, reference):
        r_ref, v_ref = reference
        err_r, err_v = run_ladder("symplectic_euler", LADDER, r_ref, v_ref, N)
        return orders(err_r, LADDER), orders(err_v, LADDER)

    def test_position_order_from_n_256(self, data):
        p_r, _ = data
        for n in (256, 512, 1024, 2048):
            assert 0.95 <= p_r[n] <= 1.05, f"n={n}: p_r={p_r[n]:.3f} outside [0.95, 1.05]"

    def test_velocity_order_is_two_not_one(self, data):
        # The single most dangerous mistake this suite could make: requiring p_v ~= 1 here
        # would fail against a CORRECT implementation. Measured p_v = 2.000 (Section 7, INV-8).
        _, p_v = data
        for n in (256, 512, 1024, 2048):
            assert 1.95 <= p_v[n] <= 2.10, f"n={n}: p_v={p_v[n]:.3f} outside [1.95, 2.10]"


class TestVelocityVerletConvergence:
    @pytest.fixture(scope="class")
    def data(self, reference):
        r_ref, v_ref = reference
        err_r, err_v = run_ladder("velocity_verlet", LADDER, r_ref, v_ref, N)
        return orders(err_r, LADDER), orders(err_v, LADDER)

    def test_position_order_from_n_256(self, data):
        p_r, _ = data
        for n in (256, 512, 1024, 2048):
            assert 1.95 <= p_r[n] <= 2.05, f"n={n}: p_r={p_r[n]:.3f} outside [1.95, 2.05]"

    def test_velocity_order_from_n_256(self, data):
        _, p_v = data
        for n in (256, 512, 1024, 2048):
            assert 1.95 <= p_v[n] <= 2.05, f"n={n}: p_v={p_v[n]:.3f} outside [1.95, 2.05]"

    def test_first_ratio_is_preasymptotic_and_excluded_from_acceptance(self, data):
        # Not a correctness requirement -- documents the expected pre-asymptotic behaviour so a
        # future maintainer does not "fix" this test by including n=128 in the strict windows.
        p_r, p_v = data
        assert p_r[128] > 2.05 or p_v[128] > 2.10, (
            "n=128 is expected to be pre-asymptotic (measured p_r=4.30, p_v=3.56); if this "
            "assumption stops holding, the acceptance windows above may need revisiting -- but "
            "the windows themselves must not be loosened to accommodate a real regression"
        )


class TestRK4Convergence:
    LADDER_RK4 = LADDER  # "primeira razao utilizavel: n=128" -- i.e. all available ratios.

    @pytest.fixture(scope="class")
    def data(self, reference):
        r_ref, v_ref = reference
        err_r, err_v = run_ladder("rk4", self.LADDER_RK4, r_ref, v_ref, N)
        return orders(err_r, self.LADDER_RK4), orders(err_v, self.LADDER_RK4)

    def test_position_order_is_measured_4_95_not_textbook_4(self, data):
        p_r, _ = data
        for n in (128, 256, 512, 1024, 2048):
            assert 4.5 <= p_r[n] <= 5.3, (
                f"n={n}: p_r={p_r[n]:.3f} outside [4.5, 5.3] -- note this window is centered "
                f"on the document's measured 4.95, NOT on the textbook order 4 (Section 7, "
                f"INV-8, ressalva 4, the project's one open physics question)"
            )

    def test_velocity_order_is_measured_4_95_not_textbook_4(self, data):
        _, p_v = data
        for n in (128, 256, 512, 1024, 2048):
            assert 4.5 <= p_v[n] <= 5.3, f"n={n}: p_v={p_v[n]:.3f} outside [4.5, 5.3]"


class TestEulerConvergence:
    LADDER_EULER = LADDER + [4096]

    @pytest.fixture(scope="class")
    def data(self, reference):
        r_ref, v_ref = reference
        err_r, err_v = run_ladder("euler", self.LADDER_EULER, r_ref, v_ref, N)
        return orders(err_r, self.LADDER_EULER), orders(err_v, self.LADDER_EULER)

    def test_position_order_window(self, data):
        p_r, _ = data
        for n in self.LADDER_EULER[1:]:
            assert 0.55 <= p_r[n] <= 1.20, f"n={n}: p_r={p_r[n]:.3f} outside [0.55, 1.20]"

    def test_velocity_order_window(self, data):
        _, p_v = data
        for n in self.LADDER_EULER[1:]:
            assert 0.20 <= p_v[n] <= 1.10, f"n={n}: p_v={p_v[n]:.3f} outside [0.20, 1.10]"

    def test_position_order_trends_toward_one_from_below(self, data):
        # Section 7, INV-8, ressalva 3: the measured exponent grows toward 1, never reaching or
        # exceeding it cleanly within this ladder. Not a strict monotonic requirement (the
        # document's own measurements wobble slightly), but the last value must be closer to 1
        # than the first.
        p_r, _ = data
        ns = self.LADDER_EULER[1:]
        first, last = p_r[ns[0]], p_r[ns[-1]]
        assert abs(last - 1.0) < abs(first - 1.0), (
            f"p_r should trend toward 1 from below across the ladder: first={first:.3f}, "
            f"last={last:.3f}"
        )
