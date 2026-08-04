"""
Shared fixtures for the TCCAstro test suite.

All tests import against the public contract in docs/api-contract.md, derived independently
from docs/integradores.md without reading src/nbody. See the module docstrings in each test
file for which specification clause each test targets.
"""
import math

import numpy as np
import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

GPU_SKIP_REASON = "requires an actual CUDA device; this suite's CI runs CPU-only by design"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test, deselect with -m 'not slow'")
    config.addinivalue_line("markers", "gpu: requires a real CUDA device")
    config.addinivalue_line("markers", "golden: depends on a committed data/ reference file")


@pytest.fixture(scope="session")
def cuda_available():
    return CUDA_AVAILABLE


def require_cuda():
    """Call at the top of a GPU test body; skips loudly (not vacuously) when no CUDA device."""
    if not CUDA_AVAILABLE:
        pytest.skip(GPU_SKIP_REASON)


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


def err_metric(r, r_ref, sphere_radius, n):
    """
    err(r) = || r - r_ref ||_F / ( sqrt(N) * R_0 )   -- docs/integradores.md Section 6.1.

    The only normalization the document accepts. r, r_ref are (N, 3) tensors or arrays.
    Returns a Python float.
    """
    if isinstance(r, torch.Tensor):
        r = r.detach().to("cpu", dtype=torch.float64).numpy()
    if isinstance(r_ref, torch.Tensor):
        r_ref = r_ref.detach().to("cpu", dtype=torch.float64).numpy()
    diff = np.asarray(r, dtype=np.float64) - np.asarray(r_ref, dtype=np.float64)
    frob = math.sqrt(np.sum(diff * diff))
    return frob / (math.sqrt(n) * sphere_radius)


@pytest.fixture
def err_metric_fn():
    return err_metric
