"""Source scaling with a heterogeneous sound speed uses c0 at each source voxel."""

from types import SimpleNamespace

import numpy as np
import pytest

from kwave.kWaveSimulation_helper.scale_source_terms_func import (
    scale_pressure_source_dirichlet,
    scale_pressure_source_uniform_grid,
    scale_stress_source,
    scale_velocity_source,
    scale_velocity_source_x,
)
from kwave.utils.matlab import matlab_find

N_DIM, DX, DT = 3, 1e-4, 2e-8


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    c0 = rng.uniform(1400.0, 1600.0, (7, 6, 5))  # distinct sound speed in every voxel
    mask = np.zeros(c0.shape, dtype=bool)
    mask[[1, 4, 2, 5], [2, 0, 5, 3], [0, 1, 4, 2]] = True
    pos_index = matlab_find(mask)  # 1-based, Fortran order (as kWaveSimulation)
    c_src = c0.ravel(order="F")[mask.ravel(order="F")][:, None]  # c0 at the source voxels, same order
    signals = rng.standard_normal((mask.sum(), 11))
    return c0, pos_index, c_src, signals


def test_pressure_uniform_grid_uses_c0_at_source(setup):
    c0, pos_index, c_src, signals = setup
    out = scale_pressure_source_uniform_grid(signals.copy(), c0, N_DIM, DX, DT, pos_index)
    np.testing.assert_allclose(out, signals * 2 * DT / (N_DIM * c_src * DX), rtol=1e-12)


def test_pressure_dirichlet_uses_c0_at_source(setup):
    c0, pos_index, c_src, signals = setup
    out = scale_pressure_source_dirichlet(signals.copy(), c0, N_DIM, pos_index)
    np.testing.assert_allclose(out, signals / (N_DIM * c_src**2), rtol=1e-12)


def test_velocity_per_voxel_signals(setup):
    c0, pos_index, c_src, signals = setup
    out = scale_velocity_source(True, "additive", signals.copy(), c0, DT, pos_index, DX)
    np.testing.assert_allclose(out, signals * 2 * c_src * DT / DX, rtol=1e-12)


def test_stress_per_voxel_signals(setup):
    c0, pos_index, c_src, signals = setup
    source = SimpleNamespace(s_mode="additive")
    out = scale_stress_source(source, c0, True, False, signals.copy(), DT, N_DIM, DX, pos_index)
    np.testing.assert_allclose(out, signals * 2 * DT * c_src / (N_DIM * DX), rtol=1e-12)


def test_single_signal_uses_first_source_voxel(setup):
    c0, pos_index, c_src, signals = setup
    out = scale_pressure_source_uniform_grid(signals[:1].copy(), c0, N_DIM, DX, DT, pos_index)
    np.testing.assert_allclose(out, signals[:1] * 2 * DT / (N_DIM * c_src[0] * DX), rtol=1e-12)


def test_velocity_x_dirichlet_returns_signal(setup):
    _, pos_index, _, signals = setup
    out = scale_velocity_source_x(True, "dirichlet", signals, None, 1500.0, DT, DX, pos_index, False)
    assert out is signals
