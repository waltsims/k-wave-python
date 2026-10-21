
import numpy as np
import pytest

from kwave.bioheatExact import bioheat_exact


def test_zero_time_preserves_initial_temperature():
    T0 = np.arange(12, dtype=float).reshape(3, 4) + 20.0

    result = bioheat_exact(
        T0=T0,
        S=0,
        material=[1.2e-7, 0.02, 37.0],
        dx=(1e-3, 2e-3),
        t=0,
    )

    np.testing.assert_allclose(result, T0, rtol=1e-12, atol=1e-12)


def test_zero_diffusion_and_perfusion():
    T0 = np.arange(12, dtype=float).reshape(3, 4)
    S = np.full_like(T0, 2.5)
    t = 3.0

    result = bioheat_exact(
        T0=T0,
        S=S,
        material=[0.0, 0.0, 37.0],
        dx=(1e-3, 2e-3),
        t=t,
    )

    np.testing.assert_allclose(result, T0 + t * S, rtol=1e-12, atol=1e-12)


def test_single_fourier_mode_on_uneven_grid():
    nx, ny = 16, 12
    dx = (0.2, 0.07)
    D = 0.3
    P = 0.15
    Ta = 37.0
    t = 0.4

    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]

    # A single periodic Fourier mode with nonzero wavenumbers
    # along both axes.
    mode = np.cos(2 * np.pi * x / nx + 2 * np.pi * y / ny)
    T0 = Ta + mode

    kx = 2 * np.pi / (nx * dx[0])
    ky = 2 * np.pi / (ny * dx[1])
    decay = np.exp(-(D * (kx**2 + ky**2) + P) * t)

    expected = Ta + decay * mode

    result = bioheat_exact(
        T0=T0,
        S=0,
        material=[D, P, Ta],
        dx=dx,
        t=t,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-11)


def test_gpu_result_can_be_reused():
    cp = pytest.importorskip("cupy")

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA GPU available")
        cp.zeros(1).sum().item()  # Check that GPU execution works.
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No usable CUDA GPU available")

    T0 = cp.full((8, 8), 37.0)
    S = cp.ones((8, 8), dtype=cp.float64)

    kwargs = {
        "S": S,
        "material": [0.0, 0.0, 37.0],
        "dx": 1e-3,
        "t": 0.1,
        "device": "gpu",
    }

    T1 = bioheat_exact(T0=T0, **kwargs)
    T2 = bioheat_exact(T0=T1, **kwargs)

    assert isinstance(T1, cp.ndarray)
    assert isinstance(T2, cp.ndarray)

    cp.testing.assert_allclose(T1, 37.1)
    cp.testing.assert_allclose(T2, 37.2)
