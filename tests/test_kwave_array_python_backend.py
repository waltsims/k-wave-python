"""Tests for kWaveArray.combine_sensor_data with the Python backend.

The Python backend returns sensor_data["p"] with shape (n_sensor_points, n_time).
kWaveArray.combine_sensor_data expects this shape and uses boolean indexing to
extract per-element data. This test verifies the shapes are compatible.
"""
import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.signals import create_cw_signals


@pytest.fixture
def array_sim_2d():
    """Minimal 2D simulation with a kWaveArray sensor."""
    grid_size = Vector([64, 64])
    grid_spacing = Vector([0.1e-3, 0.1e-3])
    kgrid = kWaveGrid(grid_size, grid_spacing)

    medium = kWaveMedium(sound_speed=1500)
    kgrid.makeTime(medium.sound_speed)

    # Simple p0 source
    source = kSource()
    source.p0 = np.zeros(grid_size)
    source.p0[32, 32] = 1.0

    # kWaveArray with 2 line elements
    karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)
    karray.add_line_element([-2e-3, -2e-3], [-2e-3, 2e-3])
    karray.add_line_element([2e-3, -2e-3], [2e-3, 2e-3])

    sensor = kSensor()
    sensor.mask = karray.get_array_binary_mask(kgrid)

    return kgrid, medium, source, sensor, karray


class TestKWaveArrayPythonBackend:
    def test_combine_sensor_data_shape(self, array_sim_2d):
        """combine_sensor_data should work with python backend sensor data."""
        kgrid, medium, source, sensor, karray = array_sim_2d

        sensor_data = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python")

        # This is the call that fails — boolean index mismatch
        combined = karray.combine_sensor_data(kgrid, sensor_data["p"])

        assert combined.shape == (karray.number_elements, int(kgrid.Nt))

    def test_sensor_data_shape_matches_mask(self, array_sim_2d):
        """Python backend sensor data rows should equal number of True points in mask."""
        kgrid, medium, source, sensor, karray = array_sim_2d

        sensor_data = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python")
        n_mask_points = int(sensor.mask.sum())

        assert (
            sensor_data["p"].shape[0] == n_mask_points
        ), f"sensor_data has {sensor_data['p'].shape[0]} rows but mask has {n_mask_points} True points"

    def test_array_as_sensor_pattern(self):
        """Reproduce the at_array_as_sensor example pattern: kWaveArray for both source and sensor."""
        grid_size = Vector([64, 64])
        grid_spacing = Vector([0.5e-3, 0.5e-3])
        kgrid = kWaveGrid(grid_size, grid_spacing)
        medium = kWaveMedium(sound_speed=1500)
        kgrid.makeTime(medium.sound_speed)

        # kWaveArray used as sensor (like at_array_as_sensor example)
        karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)
        karray.add_line_element([-10e-3, -10e-3], [-10e-3, 10e-3])
        karray.add_line_element([10e-3, -10e-3], [10e-3, 10e-3])

        sensor = kSensor()
        sensor.mask = karray.get_array_binary_mask(kgrid)

        # Time-varying pressure source (not p0)
        source = kSource()
        source.p_mask = np.zeros(grid_size)
        source.p_mask[32, 32] = 1
        source.p = np.sin(2 * np.pi * 1e6 * kgrid.t_array.squeeze())

        sensor_data = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python")

        # This is the call that fails in the example
        combined = karray.combine_sensor_data(kgrid, sensor_data["p"])
        assert combined.shape == (karray.number_elements, int(kgrid.Nt))
