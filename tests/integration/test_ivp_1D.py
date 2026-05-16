"""Integration test: 1D IVP homogeneous medium vs MATLAB reference.

Simple 1D test — homogeneous medium, Gaussian pulse, two-point sensor.
"""
import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder

from .conftest import assert_fields_close

Nx = 256
dx = 0.1e-3


@pytest.mark.integration
def test_ivp_1D_vs_matlab(load_matlab_ref):
    ref = load_matlab_ref("example_ivp_1D")

    kgrid = kWaveGrid(Vector([Nx]), Vector([dx]))
    kgrid.makeTime(1500)

    medium = kWaveMedium(sound_speed=1500, density=1000)

    source = kSource()
    source.p0 = np.zeros(Nx)
    source.p0[Nx // 2] = 1.0  # single-point impulse

    sensor_mask = np.zeros(Nx)
    sensor_mask[Nx // 4] = 1
    sensor_mask[3 * Nx // 4] = 1
    sensor = kSensor(mask=sensor_mask)

    result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", smooth_p0=False, pml_inside=True)

    assert int(kgrid.Nt) == int(ref["Nt"])
    np.testing.assert_allclose(float(kgrid.dt), float(ref["dt"]), rtol=1e-12)

    assert_fields_close(result, ref, [("p", "sensor_data_p")])
