"""Step-by-step parity test: Python solver vs MATLAB k-Wave.

Uses a 64x64 grid, single-point p0 source, no smoothing.
Pinpoints the divergence source between the two implementations.

Key finding: Python velocity = -MATLAB velocity (exact sign flip).
The gradient operator sign convention differs.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from kwave.solvers.kspace_solver import Simulation

Nx, Ny = 64, 64
dx, dy = 0.1e-3, 0.1e-3


def _load_ref(load_matlab_ref):
    return load_matlab_ref("example_parity_2D")


def _build_sim():
    """Build simulation directly with SimpleNamespace."""
    kgrid = SimpleNamespace(
        Nx=Nx,
        Ny=Ny,
        dx=dx,
        dy=dy,
        Nt=302,
        dt=2e-8,
        pml_size_x=20,
        pml_size_y=20,
        pml_alpha_x=2.0,
        pml_alpha_y=2.0,
    )
    medium = SimpleNamespace(
        sound_speed=np.float64(1500),
        density=np.float64(1000),
        alpha_coeff=None,
        alpha_power=None,
        BonA=None,
    )
    source = SimpleNamespace(
        p0=np.zeros((Nx, Ny)),
        p=None,
        p_mask=None,
        p_mode="additive",
        ux=None,
        uy=None,
        uz=None,
        u_mask=None,
        u_mode="additive",
    )
    source.p0[Nx // 2, Ny // 2] = 1.0
    sensor = SimpleNamespace(
        mask=np.ones((Nx, Ny), dtype=bool),
        record=("p",),
        record_start_index=1,
    )
    sim = Simulation(kgrid, medium, source, sensor, use_kspace=True, use_sg=True, smooth_p0=False)
    sim.setup()
    return sim


@pytest.mark.integration
def test_kvectors_same_values(load_matlab_ref):
    """k-vectors contain the same values (different ordering: fftfreq vs fftshift)."""
    ref = _load_ref(load_matlab_ref)
    sim = _build_sim()

    mat_kx = ref["kx"][:, 0]
    py_kx = sim.k_list[0].flatten()

    # MATLAB returns fftshift'd k-vectors; Python uses fftfreq order
    np.testing.assert_allclose(sorted(py_kx), sorted(mat_kx), rtol=1e-14)


@pytest.mark.integration
def test_p0_matches(load_matlab_ref):
    """Initial pressure matches exactly."""
    ref = _load_ref(load_matlab_ref)
    sim = _build_sim()
    np.testing.assert_allclose(sim._p0_initial, ref["p0"].astype(float), rtol=1e-14)


@pytest.mark.integration
def test_p_at_t0_matches(load_matlab_ref):
    """After step 0, pressure matches (p0 override)."""
    ref = _load_ref(load_matlab_ref)
    sim = _build_sim()
    sim.step()  # t=0: p is overridden with p0

    py_p = sim.p.flatten(order="F")
    mat_p = ref["sensor_data_p"][:, 0]
    np.testing.assert_allclose(py_p, mat_p, rtol=1e-14, atol=1e-14, err_msg="Pressure at t=0 should match exactly")


@pytest.mark.integration
def test_velocity_sign_flip(load_matlab_ref):
    """ROOT CAUSE: Python velocity = -MATLAB velocity after t=0.

    The gradient operator in the Python solver uses the opposite sign
    convention from MATLAB k-Wave, causing all velocity components
    to be negated. This propagates into pressure via the divergence
    at t=1, creating the ~30% divergence seen in end-to-end tests.
    """
    ref = _load_ref(load_matlab_ref)
    sim = _build_sim()
    sim.step()  # t=0

    mat_ux = ref["sensor_data_ux"][:, 0].reshape(Nx, Ny, order="F")
    mat_uy = ref["sensor_data_uy"][:, 0].reshape(Nx, Ny, order="F")

    # Python velocity is exactly -MATLAB velocity
    np.testing.assert_allclose(sim.u[0], -mat_ux, rtol=1e-14, atol=1e-20, err_msg="ux should be exactly -MATLAB ux")
    np.testing.assert_allclose(sim.u[1], -mat_uy, rtol=1e-14, atol=1e-20, err_msg="uy should be exactly -MATLAB uy")


@pytest.mark.integration
def test_p_at_t1_diverges(load_matlab_ref):
    """Pressure at t=1 diverges due to velocity sign flip."""
    ref = _load_ref(load_matlab_ref)
    sim = _build_sim()
    sim.step()  # t=0
    sim.step()  # t=1

    py_p = sim.p.flatten(order="F")
    mat_p = ref["sensor_data_p"][:, 1]

    max_diff = np.max(np.abs(py_p - mat_p))
    # This should be large (~0.5) due to the sign-flipped velocity
    assert max_diff > 0.1, f"Expected large divergence at t=1, got {max_diff:.6e}"
