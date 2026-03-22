"""Tests for the unified kspaceFirstOrder entry point — error handling and dispatch."""
import os
import tempfile

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import _normalize_pml, kspaceFirstOrder


@pytest.fixture
def sim_2d():
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
    kgrid.setTime(50, 1e-8)
    source = kSource()
    source.p0 = np.zeros((64, 64))
    source.p0[32, 32] = 1.0
    return kgrid, kWaveMedium(sound_speed=1500), source, kSensor(mask=np.ones((64, 64), dtype=bool))


class TestNormalizePml:
    def test_scalar(self):
        assert _normalize_pml(10, 2) == (10, 10)
        assert _normalize_pml(10, 3) == (10, 10, 10)

    def test_tuple(self):
        assert _normalize_pml((10, 20), 2) == (10, 20)
        assert _normalize_pml((15,), 3) == (15, 15, 15)

    def test_short_tuple_padded(self):
        # 2-element tuple in 3-D pads z with last value
        assert _normalize_pml((20, 15), 3) == (20, 15, 15)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="got 0 elements"):
            _normalize_pml((), 3)

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="got 3 elements"):
            _normalize_pml((10, 20, 30), 2)


class TestErrors:
    def test_time_not_set(self):
        kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
        source = kSource()
        with pytest.raises(ValueError, match="kgrid.Nt and kgrid.dt must be set"):
            kspaceFirstOrder(kgrid, kWaveMedium(sound_speed=1500), source, kSensor(mask=np.ones((64, 64), dtype=bool)))

    def test_unknown_backend(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        with pytest.raises(ValueError, match="Unknown backend"):
            kspaceFirstOrder(kgrid, medium, source, sensor, backend="unknown")

    def test_cpp_save_only(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", save_only=True, data_path=tempfile.mkdtemp())
        assert "input_file" in result and "output_file" in result
        assert os.path.exists(result["input_file"])
