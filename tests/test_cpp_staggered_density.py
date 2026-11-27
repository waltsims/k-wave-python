"""CppSimulation staggered density: no wrap-around at the last plane, same as the legacy path."""

import numpy as np
import pytest

from kwave.kgrid import kWaveGrid
from kwave.solvers.cpp_simulation import CppSimulation
from kwave.utils.interp import interpolate2d, interpolate3d


def _sim(ndim):
    sim = CppSimulation.__new__(CppSimulation)
    sim.ndim = ndim
    return sim


@pytest.mark.parametrize("shape", [(9, 8, 7), (9, 8)])
def test_staggered_density_matches_legacy_interpolation(shape):
    rng = np.random.default_rng(0)
    rho0 = rng.uniform(900.0, 2000.0, shape)
    kgrid = kWaveGrid(list(shape), [1e-4] * len(shape))
    axes = [kgrid.x_vec.ravel(), kgrid.y_vec.ravel()] + ([kgrid.z_vec.ravel()] if len(shape) == 3 else [])
    interp = interpolate3d if len(shape) == 3 else interpolate2d
    grid = [kgrid.x, kgrid.y] + ([kgrid.z] if len(shape) == 3 else [])
    spacing = [kgrid.dx, kgrid.dy] + ([kgrid.dz] if len(shape) == 3 else [])

    staggered = _sim(len(shape))._compute_staggered_density(rho0)

    for axis, rho0_sg in enumerate(staggered):
        query = [g + (spacing[i] / 2 if i == axis else 0) for i, g in enumerate(grid)]
        expected = interp(axes, rho0, query)  # legacy kspaceFirstOrder3D: copy_nans keeps rho0 outside the grid
        np.testing.assert_allclose(rho0_sg, expected, rtol=1e-12)
        last = [slice(None)] * rho0.ndim
        last[axis] = -1
        np.testing.assert_array_equal(rho0_sg[tuple(last)], rho0[tuple(last)])
