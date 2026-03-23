"""Integration test: 2D IVP disc source vs MATLAB reference.

Mirrors examples/new_api_ivp_2D.py — homogeneous medium, disc p0, full-grid sensor.
"""
import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.mapgen import make_disc

from .conftest import assert_fields_close


@pytest.mark.integration
def test_ivp_2D_vs_matlab(load_matlab_ref):
    ref = load_matlab_ref("example_ivp_2D")

    # Same setup as examples/new_api_ivp_2D.py and collect_example_ivp_2D.m
    grid_size = Vector([128, 128])
    kgrid = kWaveGrid(grid_size, Vector([0.1e-3, 0.1e-3]))
    kgrid.makeTime(1500)

    medium = kWaveMedium(sound_speed=1500, density=1000)

    source = kSource()
    source.p0 = make_disc(grid_size, Vector([64, 64]), 5).astype(float)

    sensor = kSensor(mask=np.ones((128, 128), dtype=bool))

    result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python")

    # Verify time stepping matches
    assert int(kgrid.Nt) == int(ref["Nt"]), f"Nt mismatch: Python {kgrid.Nt} vs MATLAB {ref['Nt']}"
    np.testing.assert_allclose(float(kgrid.dt), float(ref["dt"]), rtol=1e-12)

    assert_fields_close(
        result,
        ref,
        [("p", "sensor_data_p")],
    )
