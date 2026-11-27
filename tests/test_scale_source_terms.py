from types import SimpleNamespace

import numpy as np
import pytest

from kwave.kWaveSimulation_helper.scale_source_terms_func import scale_stress_source, scale_velocity_source
from kwave.utils.matlab import matlab_find


@pytest.mark.parametrize("kind", ["velocity", "stress"])
def test_per_voxel_source_scaled_with_c0_at_own_voxel(kind):
    # Before fix: TypeError ('int' object is not subscriptable) from ndarray.size[0] for heterogeneous c0.
    c0 = 1500.0 + np.arange(24.0).reshape(2, 3, 4)
    mask, dt, dx = c0 % 5 == 0, 1e-8, 1e-4
    signal, pos = np.ones((mask.sum(), 6)), matlab_find(mask)  # one row per source voxel; 1-based column-major indices
    if kind == "velocity":
        scaled = scale_velocity_source(True, "additive", signal, c0, dt, pos, dx)
    else:
        scaled = scale_stress_source(SimpleNamespace(s_mode="additive"), c0, True, False, signal, dt, 1, dx, pos)
    assert np.allclose(scaled, 2 * c0.T[mask.T][:, None] * dt / dx, rtol=1e-12)
