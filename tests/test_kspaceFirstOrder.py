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

    @pytest.mark.parametrize("source_kind", ["p0", "p"])
    @pytest.mark.parametrize("save_only", [True, False])
    def test_cpp_backend_rejects_1d(self, source_kind, save_only, tmp_path):
        """The C++ binary is 2D/3D only: 1D must fail early with a clear message, before any file is written."""
        kgrid = kWaveGrid(Vector([64]), Vector([0.1e-3]))
        kgrid.setTime(20, 1e-8)
        source = kSource()
        if source_kind == "p0":
            source.p0 = np.zeros(64)
            source.p0[32] = 1.0
        else:
            source.p_mask = np.zeros(64)
            source.p_mask[32] = 1
            source.p = np.sin(0.3 * np.arange(20))[None, :]
        sensor = kSensor(mask=np.ones(64, dtype=bool))

        with pytest.raises(ValueError, match="backend='cpp' does not support 1D simulations"):
            kspaceFirstOrder(
                kgrid, kWaveMedium(sound_speed=1500), source, sensor, backend="cpp", save_only=save_only, data_path=str(tmp_path)
            )
        assert list(tmp_path.iterdir()) == []

    def test_python_backend_runs(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python")
        assert "p" in result
        assert result["p"].shape == (int(sensor.mask.sum()), int(kgrid.Nt))

    def test_cpp_save_only(self, sim_2d):
        kgrid, medium, source, sensor = sim_2d
        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", save_only=True, data_path=tempfile.mkdtemp())
        assert "input_file" in result and "output_file" in result
        assert os.path.exists(result["input_file"])
