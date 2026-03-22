"""Integration test: 1D IVP heterogeneous medium vs MATLAB reference.

Mirrors examples/ivp_1D_simulation.py — heterogeneous c0/rho0, smooth pulse, two-point sensor.
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

Nx = 512
dx = 0.05e-3


@pytest.mark.integration
def test_ivp_1D_vs_matlab(load_matlab_ref):
    ref = load_matlab_ref("example_ivp_1D")

    # Same setup as examples/ivp_1D_simulation.py and collect_example_ivp_1D.m
    kgrid = kWaveGrid(Vector([Nx]), Vector([dx]))

    sound_speed = 1500 * np.ones(Nx)
    sound_speed[: Nx // 3] = 2000

    density = 1000 * np.ones(Nx)
    density[4 * Nx // 5 :] = 1500

    medium = kWaveMedium(sound_speed=sound_speed, density=density)
    kgrid.makeTime(sound_speed, cfl=0.3)

    source = kSource()
    source.p0 = np.zeros(Nx)
    x0, width = 280, 100
    pulse = 0.5 * (np.sin(np.arange(width + 1) * np.pi / width - np.pi / 2) + 1)
    source.p0[x0 : x0 + width + 1] = pulse

    sensor_mask = np.zeros(Nx)
    sensor_mask[Nx // 4] = 1
    sensor_mask[3 * Nx // 4] = 1
    sensor = kSensor(mask=sensor_mask)

    result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python")

    # Verify time stepping matches
    assert int(kgrid.Nt) == int(ref["Nt"])
    np.testing.assert_allclose(float(kgrid.dt), float(ref["dt"]), rtol=1e-12)

    assert_fields_close(result, ref, [("p", "sensor_data_p")])
