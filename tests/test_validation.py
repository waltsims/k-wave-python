"""Tests for kspaceFirstOrder validation."""
import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.solvers.validation import (
    validate_cfl,
    validate_medium,
    validate_pml,
    validate_sensor,
    validate_simulation,
    validate_source,
    validate_time_stepping,
)


@pytest.fixture
def kgrid_2d():
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
    kgrid.setTime(50, 1e-8)
    return kgrid


class TestTimeStepValidation:
    def test_auto_nt(self):
        kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
        with pytest.raises(ValueError, match="kgrid.Nt and kgrid.dt must be set"):
            validate_time_stepping(kgrid)

    def test_valid_time(self, kgrid_2d):
        validate_time_stepping(kgrid_2d)  # should not raise


class TestMediumValidation:
    def test_valid_homogeneous(self, kgrid_2d):
        medium = kWaveMedium(sound_speed=1500)
        validate_medium(medium, kgrid_2d)

    def test_wrong_size_sound_speed(self, kgrid_2d):
        medium = kWaveMedium(sound_speed=np.ones(100))
        with pytest.raises(ValueError, match="sound_speed"):
            validate_medium(medium, kgrid_2d)

    def test_negative_sound_speed(self, kgrid_2d):
        medium = kWaveMedium(sound_speed=-1500)
        with pytest.raises(ValueError, match="positive"):
            validate_medium(medium, kgrid_2d)

    def test_wrong_size_density(self, kgrid_2d):
        medium = kWaveMedium(sound_speed=1500, density=np.ones(100))
        with pytest.raises(ValueError, match="density"):
            validate_medium(medium, kgrid_2d)


class TestPmlValidation:
    def test_pml_too_large(self, kgrid_2d):
        with pytest.raises(ValueError, match="too large"):
            validate_pml((40, 20), kgrid_2d)

    def test_valid_pml(self, kgrid_2d):
        validate_pml((20, 20), kgrid_2d)

    def test_negative_pml(self, kgrid_2d):
        with pytest.raises(ValueError, match="non-negative"):
            validate_pml((-5, 20), kgrid_2d)


class TestCflValidation:
    def test_unstable_cfl_warns(self, kgrid_2d):
        medium = kWaveMedium(sound_speed=1500)
        # With dt=1e-8, dx=0.1e-3: CFL = 1500*1e-8/1e-4 = 0.15, stable
        validate_cfl(kgrid_2d, medium)

    def test_high_cfl_warns(self):
        kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
        kgrid.setTime(50, 1e-6)  # very large dt
        medium = kWaveMedium(sound_speed=1500)
        # CFL = 1500*1e-6/1e-4 = 15, unstable
        with pytest.warns(match="CFL"):
            validate_cfl(kgrid, medium)


class TestSourceValidation:
    def test_p0_wrong_size(self, kgrid_2d):
        source = kSource()
        source.p0 = np.ones(100)
        with pytest.raises(ValueError, match="source.p0"):
            validate_source(source, kgrid_2d)

    def test_p_without_mask(self, kgrid_2d):
        source = kSource()
        source.p = np.ones(50)
        with pytest.raises(ValueError, match="p_mask"):
            validate_source(source, kgrid_2d)


class TestSensorValidation:
    def test_sensor_mask_wrong_size(self, kgrid_2d):
        sensor = kSensor(mask=np.ones(100, dtype=bool))
        with pytest.raises(ValueError, match="sensor.mask"):
            validate_sensor(sensor, kgrid_2d)

    def test_valid_sensor(self, kgrid_2d):
        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        validate_sensor(sensor, kgrid_2d)

    def test_none_sensor(self, kgrid_2d):
        validate_sensor(None, kgrid_2d)


class TestIntegration:
    def test_full_validation_passes(self, kgrid_2d):
        medium = kWaveMedium(sound_speed=1500)
        source = kSource()
        source.p0 = np.zeros((64, 64))
        source.p0[32, 32] = 1.0
        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        validate_simulation(kgrid_2d, medium, source, sensor, pml_size=(20, 20))
