"""Integration test: photoacoustic waveforms (2D + 3D) vs MATLAB reference.

Mirrors examples/ivp_photoacoustic_waveforms/ivp_photoacoustic_waveforms.py.
Single-point sensor at fixed offset from disc/ball source.
"""
import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.mapgen import make_ball, make_disc

from .conftest import assert_fields_close

Nx = 64
dx = 1e-3 / Nx
source_radius = 2
source_sensor_distance = 10
dt = 2e-9
t_end = 300e-9


@pytest.mark.integration
def test_photoacoustic_2D_vs_matlab(load_matlab_ref):
    ref = load_matlab_ref("example_photoacoustic")

    kgrid = kWaveGrid([Nx, Nx], [dx, dx])
    Nt = int(np.round(t_end / dt))
    kgrid.setTime(Nt, dt)

    medium = kWaveMedium(sound_speed=1500, density=1000)

    source = kSource()
    source.p0 = make_disc(Vector([Nx, Nx]), Vector([Nx // 2, Nx // 2]), source_radius)

    sensor = kSensor()
    sensor.mask = np.zeros((Nx, Nx), dtype=bool)
    # MATLAB 1-indexed: mask(Nx/2 + dist, Nx/2) → Python 0-indexed: mask[Nx/2 + dist - 1, Nx/2 - 1]
    sensor.mask[Nx // 2 + source_sensor_distance - 1, Nx // 2 - 1] = True
    sensor.record = ["p"]

    result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", pml_inside=True)

    assert_fields_close(result, ref, [("p", "sensor_data_2D_p")])


@pytest.mark.integration
def test_photoacoustic_3D_vs_matlab(load_matlab_ref):
    ref = load_matlab_ref("example_photoacoustic")

    kgrid = kWaveGrid([Nx, Nx, Nx], [dx, dx, dx])
    Nt = int(np.round(t_end / dt))
    kgrid.setTime(Nt, dt)

    medium = kWaveMedium(sound_speed=1500, density=1000)

    source = kSource()
    source.p0 = make_ball(Vector([Nx, Nx, Nx]), Vector([Nx // 2, Nx // 2, Nx // 2]), source_radius)

    sensor = kSensor()
    sensor.mask = np.zeros((Nx, Nx, Nx), dtype=bool)
    sensor.mask[Nx // 2 + source_sensor_distance - 1, Nx // 2 - 1, Nx // 2 - 1] = True
    sensor.record = ["p"]

    result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", pml_inside=True)

    assert_fields_close(result, ref, [("p", "sensor_data_3D_p")])
