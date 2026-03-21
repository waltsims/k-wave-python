"""Tests for the unified kspaceFirstOrder entry point."""
import os

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import _normalize_pml, kspaceFirstOrder

# -- Fixtures --


@pytest.fixture
def sim_2d():
    """Basic 2D simulation setup."""
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
    kgrid.setTime(50, 1e-8)
    medium = kWaveMedium(sound_speed=1500)
    source = kSource()
    source.p0 = np.zeros((64, 64))
    source.p0[32, 32] = 1.0
    sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
    return kgrid, medium, source, sensor


@pytest.fixture
def sim_1d():
    """Basic 1D simulation setup."""
    kgrid = kWaveGrid(Vector([128]), Vector([0.1e-3]))
    kgrid.setTime(50, 1e-8)
    medium = kWaveMedium(sound_speed=1500)
    source = kSource()
    source.p0 = np.zeros(128)
    source.p0[64] = 1.0
    sensor = kSensor(mask=np.ones(128, dtype=bool))
    return kgrid, medium, source, sensor


# -- PML normalization --


class TestNormalizePml:
    def test_scalar_2d(self):
        assert _normalize_pml(10, 2) == (10, 10)

    def test_scalar_3d(self):
        assert _normalize_pml(10, 3) == (10, 10, 10)

    def test_tuple_matching(self):
        assert _normalize_pml((10, 20), 2) == (10, 20)

    def test_tuple_wrong_length(self):
        with pytest.raises(ValueError, match="must be a scalar or 2-element"):
            _normalize_pml((10, 20, 30), 2)

    def test_single_element_tuple(self):
        assert _normalize_pml((15,), 3) == (15, 15, 15)


# -- Native backend --


class TestNativeBackend:
    def test_basic_2d(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="native")
        assert "p" in result
        assert "p_final" in result
        assert result["p"].shape[0] == 64 * 64  # all sensor points
        assert result["p"].shape[1] == 50  # all time steps

    def test_basic_1d(self, sim_1d):
        kgrid, medium, source, sensor = sim_1d
        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="native")
        assert result["p"].shape[0] == 128
        assert result["p"].shape[1] == 50

    def test_custom_pml_size(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size=10, backend="native")
        assert result["p"].shape[1] == 50

    def test_per_dim_pml(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size=(10, 15), backend="native")
        assert result["p"].shape[1] == 50

    def test_auto_pml(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        result = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size="auto", backend="native")
        assert "p" in result

    def test_pressure_nonzero(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="native")
        assert np.max(np.abs(result["p"])) > 0


# -- Error handling --


class TestErrors:
    def test_time_not_set(self):
        kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
        medium = kWaveMedium(sound_speed=1500)
        source = kSource()
        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        with pytest.raises(ValueError, match="kgrid.Nt and kgrid.dt must be set"):
            kspaceFirstOrder(kgrid, medium, source, sensor)

    def test_unknown_backend(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        with pytest.raises(ValueError, match="Unknown backend"):
            kspaceFirstOrder(kgrid, medium, source, sensor, backend="unknown")

    def test_cpp_save_only(self, sim_2d):
        """C++ save_only should write HDF5 and return file paths."""
        import tempfile

        kgrid, medium, source, sensor = sim_2d
        data_path = tempfile.mkdtemp()
        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", save_only=True, data_path=data_path)
        assert "input_file" in result
        assert "output_file" in result
        assert os.path.exists(result["input_file"])
