"""Step-by-step parity test: Python solver vs MATLAB k-Wave.

Uses a 64x64 grid, single-point p0 source, no smoothing.
Validates machine-precision agreement at each stage of the solver.
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
def test_velocity_matches_matlab(load_matlab_ref):
    """Velocity at t=0 matches MATLAB exactly (sign bug was fixed)."""
    ref = _load_ref(load_matlab_ref)
    sim = _build_sim()
    sim.step()  # t=0

    mat_ux = ref["sensor_data_ux"][:, 0].reshape(Nx, Ny, order="F")
    mat_uy = ref["sensor_data_uy"][:, 0].reshape(Nx, Ny, order="F")

    np.testing.assert_allclose(sim.u[0], mat_ux, rtol=1e-14, atol=1e-20, err_msg="ux should match MATLAB")
    np.testing.assert_allclose(sim.u[1], mat_uy, rtol=1e-14, atol=1e-20, err_msg="uy should match MATLAB")


@pytest.mark.integration
def test_pressure_parity_10_steps(load_matlab_ref):
    """Pressure matches MATLAB for first 10 timesteps at machine precision."""
    ref = _load_ref(load_matlab_ref)
    sim = _build_sim()
    mat_p_all = ref["sensor_data_p"]

    for t in range(10):
        sim.step()
        py_p = sim.p.flatten(order="F")
        mat_p = mat_p_all[:, t]
        np.testing.assert_allclose(py_p, mat_p, rtol=1e-12, atol=1e-14, err_msg=f"Pressure diverged at timestep {t}")
