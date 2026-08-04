"""
nbody.state.State against docs/api-contract.md:

    @dataclass(frozen=True)
    class State:
        r: torch.Tensor   # (N, 3)
        v: torch.Tensor   # (N, 3)
        m: torch.Tensor   # (N,)
        n, dtype, device properties
        to(*, dtype=None, device=None) -> State

    "r, v e m compartilham dtype e device. A construcao valida formas e consistencia e levanta
    ValueError quando violadas."

This module tests shape/consistency validation and the immutability contract implied by
"frozen=True". It does not test any physics.
"""
import dataclasses

import pytest
import torch

nbody = pytest.importorskip("nbody")
from nbody.state import State  # noqa: E402


def make_state(n=5, dtype=torch.float64, device="cpu"):
    r = torch.randn(n, 3, dtype=dtype, device=device)
    v = torch.randn(n, 3, dtype=dtype, device=device)
    m = torch.ones(n, dtype=dtype, device=device)
    return State(r=r, v=v, m=m)


class TestConstruction:
    def test_valid_construction_succeeds(self):
        state = make_state(n=7)
        assert state.n == 7

    def test_n_property(self):
        state = make_state(n=13)
        assert state.n == 13

    def test_dtype_property(self):
        state = make_state(dtype=torch.float32)
        assert state.dtype == torch.float32

    def test_device_property(self):
        state = make_state(device="cpu")
        assert state.device == torch.device("cpu")


class TestShapeValidation:
    def test_r_wrong_second_dimension_raises(self):
        r = torch.zeros(5, 2)  # must be (N, 3)
        v = torch.zeros(5, 3)
        m = torch.ones(5)
        with pytest.raises(ValueError):
            State(r=r, v=v, m=m)

    def test_r_wrong_rank_raises(self):
        r = torch.zeros(5 * 3)  # flat, not (N, 3)
        v = torch.zeros(5, 3)
        m = torch.ones(5)
        with pytest.raises(ValueError):
            State(r=r, v=v, m=m)

    def test_v_wrong_second_dimension_raises(self):
        r = torch.zeros(5, 3)
        v = torch.zeros(5, 4)
        m = torch.ones(5)
        with pytest.raises(ValueError):
            State(r=r, v=v, m=m)

    def test_m_wrong_rank_raises(self):
        r = torch.zeros(5, 3)
        v = torch.zeros(5, 3)
        m = torch.ones(5, 1)  # must be (N,), not (N, 1)
        with pytest.raises(ValueError):
            State(r=r, v=v, m=m)

    def test_mismatched_n_between_r_and_v_raises(self):
        r = torch.zeros(5, 3)
        v = torch.zeros(6, 3)
        m = torch.ones(5)
        with pytest.raises(ValueError):
            State(r=r, v=v, m=m)

    def test_mismatched_n_between_r_and_m_raises(self):
        r = torch.zeros(5, 3)
        v = torch.zeros(5, 3)
        m = torch.ones(6)
        with pytest.raises(ValueError):
            State(r=r, v=v, m=m)


class TestDtypeDeviceConsistency:
    def test_mismatched_dtype_between_r_and_v_raises(self):
        r = torch.zeros(5, 3, dtype=torch.float64)
        v = torch.zeros(5, 3, dtype=torch.float32)
        m = torch.ones(5, dtype=torch.float64)
        with pytest.raises(ValueError):
            State(r=r, v=v, m=m)

    def test_mismatched_dtype_between_r_and_m_raises(self):
        r = torch.zeros(5, 3, dtype=torch.float64)
        v = torch.zeros(5, 3, dtype=torch.float64)
        m = torch.ones(5, dtype=torch.float32)
        with pytest.raises(ValueError):
            State(r=r, v=v, m=m)


class TestTo:
    def test_to_dtype_converts_all_three_tensors(self):
        state = make_state(dtype=torch.float64)
        converted = state.to(dtype=torch.float32)
        assert converted.dtype == torch.float32
        assert converted.r.dtype == torch.float32
        assert converted.v.dtype == torch.float32
        assert converted.m.dtype == torch.float32

    def test_to_preserves_values_within_precision_change(self):
        state = make_state(dtype=torch.float64)
        converted = state.to(dtype=torch.float32)
        assert torch.allclose(
            converted.r.to(torch.float64), state.r, rtol=1e-6, atol=1e-12
        )

    def test_to_does_not_mutate_original(self):
        state = make_state(dtype=torch.float64)
        original_dtype = state.r.dtype
        _ = state.to(dtype=torch.float32)
        assert state.r.dtype == original_dtype
        assert state.dtype == torch.float64

    def test_to_no_args_returns_equivalent_state(self):
        state = make_state()
        same = state.to()
        assert torch.equal(same.r, state.r)
        assert torch.equal(same.v, state.v)
        assert torch.equal(same.m, state.m)


class TestImmutability:
    def test_is_dataclass_frozen(self):
        assert dataclasses.is_dataclass(State)
        params = getattr(State, "__dataclass_params__", None)
        assert params is not None and params.frozen is True

    def test_assigning_to_field_raises(self):
        state = make_state()
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.r = torch.zeros(5, 3)  # type: ignore[misc]
