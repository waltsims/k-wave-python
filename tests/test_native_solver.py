"""Integration tests for the native solver path.

Covers physics branches, recording options, and C++ HDF5 serialization.
Validation and compat edge cases are in test_validation.py and test_compat.py.
"""
import os

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder


@pytest.fixture
def grid_2d():
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
    kgrid.makeTime(1500, 0.3)
    kgrid.setTime(10, float(kgrid.dt))
    return kgrid


@pytest.fixture
def grid_1d():
    kgrid = kWaveGrid(Vector([64]), Vector([0.1e-3]))
    kgrid.setTime(20, 1e-8)
    return kgrid


def _p0_source(shape):
    source = kSource()
    source.p0 = np.zeros(shape)
    source.p0[tuple(s // 2 for s in shape)] = 1.0
    return source


class TestNative2D:
    def test_p0_source(self, grid_2d):
        result = kspaceFirstOrder(
            grid_2d, kWaveMedium(sound_speed=1500), _p0_source((64, 64)), kSensor(mask=np.ones((64, 64), dtype=bool)), backend="native"
        )
        assert result["p"].shape == (64 * 64, 10)
        assert np.max(np.abs(result["p"])) > 0

    def test_heterogeneous_medium(self, grid_2d):
        c = 1500 * np.ones((64, 64))
        c[:32, :] = 1600
        result = kspaceFirstOrder(
            grid_2d,
            kWaveMedium(sound_speed=c, density=1000),
            _p0_source((64, 64)),
            kSensor(mask=np.ones((64, 64), dtype=bool)),
            backend="native",
        )
        assert result["p"].shape[0] == 64 * 64

    def test_absorption(self, grid_2d):
        result = kspaceFirstOrder(
            grid_2d,
            kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5),
            _p0_source((64, 64)),
            kSensor(mask=np.ones((64, 64), dtype=bool)),
            backend="native",
        )
        assert result["p"].shape == (64 * 64, 10)

    def test_pml_auto(self):
        kgrid = kWaveGrid(Vector([128, 128]), Vector([0.1e-3, 0.1e-3]))
        kgrid.makeTime(1500, 0.3)
        kgrid.setTime(5, float(kgrid.dt))
        source = kSource()
        source.p0 = np.zeros((128, 128))
        source.p0[64, 64] = 1.0
        result = kspaceFirstOrder(
            kgrid, kWaveMedium(sound_speed=1500), source, kSensor(mask=np.ones((128, 128), dtype=bool)), pml_size="auto", backend="native"
        )
        assert "p" in result

    def test_record_aggregates(self, grid_2d):
        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        sensor.record = ["p", "p_max", "p_rms"]
        result = kspaceFirstOrder(grid_2d, kWaveMedium(sound_speed=1500), _p0_source((64, 64)), sensor, backend="native")
        assert result["p_max"].shape == (64 * 64,)
        assert "p_rms" in result


class TestNative1D:
    def test_tvsp(self, grid_1d):
        source = kSource()
        source.p_mask = np.zeros(64)
        source.p_mask[10] = 1
        source.p = np.sin(2 * np.pi * 1e6 * np.arange(20) * 1e-8).reshape(1, -1)
        sensor = kSensor(mask=np.zeros(64))
        sensor.mask[50] = 1
        result = kspaceFirstOrder(grid_1d, kWaveMedium(sound_speed=1500), source, sensor, backend="native")
        assert result["p"].shape == (1, 20)

    def test_velocity_source(self, grid_1d):
        source = kSource()
        source.u_mask = np.zeros(64)
        source.u_mask[10] = 1
        source.ux = np.sin(2 * np.pi * 1e6 * np.arange(20) * 1e-8).reshape(1, -1)
        sensor = kSensor(mask=np.zeros(64))
        sensor.mask[50] = 1
        result = kspaceFirstOrder(grid_1d, kWaveMedium(sound_speed=1500), source, sensor, backend="native")
        assert "p" in result


class TestNativePhysics:
    def test_nonlinearity_bona(self, grid_2d):
        result = kspaceFirstOrder(
            grid_2d,
            kWaveMedium(sound_speed=1500, density=1000, BonA=6),
            _p0_source((64, 64)),
            kSensor(mask=np.ones((64, 64), dtype=bool)),
            backend="native",
        )
        assert result["p"].shape == (64 * 64, 10)

    def test_stokes_absorption(self, grid_2d):
        result = kspaceFirstOrder(
            grid_2d,
            kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=2.0),
            _p0_source((64, 64)),
            kSensor(mask=np.ones((64, 64), dtype=bool)),
            backend="native",
        )
        assert result["p"].shape == (64 * 64, 10)

    def test_dirichlet_pressure_source(self, grid_1d):
        source = kSource()
        source.p_mask = np.zeros(64)
        source.p_mask[10] = 1
        source.p = np.sin(2 * np.pi * 1e6 * np.arange(20) * 1e-8).reshape(1, -1)
        source.p_mode = "dirichlet"
        sensor = kSensor(mask=np.zeros(64))
        sensor.mask[50] = 1
        result = kspaceFirstOrder(grid_1d, kWaveMedium(sound_speed=1500), source, sensor, backend="native")
        assert result["p"].shape == (1, 20)

    def test_velocity_recording(self, grid_2d):
        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        sensor.record = ["p", "ux", "uy", "ux_max", "uy_rms", "ux_final", "p_final"]
        result = kspaceFirstOrder(grid_2d, kWaveMedium(sound_speed=1500), _p0_source((64, 64)), sensor, backend="native")
        n = 64 * 64
        assert result["ux"].shape == (n, 10)
        assert result["ux_max"].shape == (n,)
        assert "ux_final" in result and "p_final" in result

    def test_intensity_recording(self, grid_2d):
        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        sensor.record = ["p", "ux", "uy", "Ix", "Iy", "Ix_avg", "Iy_avg"]
        result = kspaceFirstOrder(grid_2d, kWaveMedium(sound_speed=1500), _p0_source((64, 64)), sensor, backend="native")
        assert result["Ix"].shape == (64 * 64, 10)
        assert result["Ix_avg"].shape == (64 * 64,)

    def test_record_start_index(self, grid_1d):
        source = kSource()
        source.p0 = np.zeros(64)
        source.p0[32] = 1.0
        sensor = kSensor(mask=np.zeros(64))
        sensor.mask[50] = 1
        sensor.record_start_index = 5  # 1-based; skip first 4 steps → 16 recorded
        result = kspaceFirstOrder(grid_1d, kWaveMedium(sound_speed=1500), source, sensor, backend="native")
        assert result["p"].shape == (1, 16)

    def test_sensor_none_records_everywhere(self, grid_1d):
        source = kSource()
        source.p0 = np.zeros(64)
        source.p0[32] = 1.0
        result = kspaceFirstOrder(grid_1d, kWaveMedium(sound_speed=1500), source, None, backend="native")
        assert result["p"].shape == (64, 20)


class TestCppSaveOnly:
    def _run_save_only(self, grid_2d, medium, source, tmp_path, sensor=None):
        if sensor is None:
            sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        return kspaceFirstOrder(grid_2d, medium, source, sensor, backend="cpp", save_only=True, data_path=str(tmp_path))

    def test_basic(self, grid_2d, tmp_path):
        result = self._run_save_only(grid_2d, kWaveMedium(sound_speed=1500), _p0_source((64, 64)), tmp_path)
        assert os.path.exists(result["input_file"])

    def test_absorption(self, grid_2d, tmp_path):
        result = self._run_save_only(
            grid_2d, kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5), _p0_source((64, 64)), tmp_path
        )
        assert os.path.exists(result["input_file"])

    def test_bona(self, grid_2d, tmp_path):
        result = self._run_save_only(grid_2d, kWaveMedium(sound_speed=1500, density=1000, BonA=6), _p0_source((64, 64)), tmp_path)
        assert os.path.exists(result["input_file"])

    def test_stokes(self, grid_2d, tmp_path):
        result = self._run_save_only(
            grid_2d, kWaveMedium(sound_speed=1500, alpha_coeff=0.5, alpha_power=2.0), _p0_source((64, 64)), tmp_path
        )
        assert os.path.exists(result["input_file"])

    def test_velocity_source(self, grid_2d, tmp_path):
        source = kSource()
        source.u_mask = np.zeros((64, 64))
        source.u_mask[10, 10] = 1
        source.ux = np.sin(2 * np.pi * 1e6 * np.arange(10) * float(grid_2d.dt)).reshape(1, -1)
        result = self._run_save_only(grid_2d, kWaveMedium(sound_speed=1500), source, tmp_path)
        assert os.path.exists(result["input_file"])

    def test_heterogeneous_density(self, grid_2d, tmp_path):
        rho = 1000 * np.ones((64, 64))
        rho[:32, :] = 1200
        result = self._run_save_only(grid_2d, kWaveMedium(sound_speed=1500, density=rho), _p0_source((64, 64)), tmp_path)
        assert os.path.exists(result["input_file"])

    def test_sensor_none(self, grid_2d, tmp_path):
        result = self._run_save_only(grid_2d, kWaveMedium(sound_speed=1500), _p0_source((64, 64)), tmp_path, sensor=None)
        assert os.path.exists(result["input_file"])


class TestCartesianSensor:
    def test_cartesian_2d(self, grid_2d):
        """Cartesian sensor mask with bilinear interpolation."""
        # 3 query points as (ndim, N_pts) array
        cart_mask = np.array([[0.0, 1e-3, -1e-3], [0.0, 0.0, 1e-3]])  # (2, 3)
        sensor = kSensor(mask=cart_mask)
        result = kspaceFirstOrder(grid_2d, kWaveMedium(sound_speed=1500), _p0_source((64, 64)), sensor, backend="native")
        assert result["p"].shape == (3, 10)

    def test_cartesian_1d(self, grid_1d):
        """Cartesian sensor mask with 1D interpolation."""
        cart_mask = np.array([[0.0, 1e-3, -1e-3]])  # (1, 3)
        sensor = kSensor(mask=cart_mask)
        source = kSource()
        source.p0 = np.zeros(64)
        source.p0[32] = 1.0
        result = kspaceFirstOrder(grid_1d, kWaveMedium(sound_speed=1500), source, sensor, backend="native")
        assert result["p"].shape == (3, 20)


class TestMatlabInterop:
    def test_simulate_from_dicts(self):
        from kwave.solvers.kspace_solver import simulate_from_dicts

        kgrid = {"Nx": 64, "dx": 0.1e-3, "Nt": 5, "dt": 1e-8, "pml_size_x": 10, "pml_alpha_x": 2.0}
        medium = {"sound_speed": 1500}
        p0 = np.zeros(64)
        p0[32] = 1.0
        source = {"p0": p0}
        sensor = {"mask": np.ones(64, dtype=bool)}
        result = simulate_from_dicts(kgrid, medium, source, sensor, backend="cpu")
        assert "p" in result
        assert result["p"].shape[0] == 64

    def test_normalize_medium_aliases(self):
        from kwave.solvers.kspace_solver import _normalize_medium

        d = _normalize_medium({"c0": 1500, "rho0": 1000})
        assert "sound_speed" in d and "density" in d
        assert "c0" not in d and "rho0" not in d


class TestSolverFactory:
    def test_get_native_solver(self):
        from kwave.solvers import Backend, get_solver

        s = get_solver("native")
        assert s.backend == Backend.NATIVE

    def test_get_cpp_solver(self):
        from kwave.solvers import Backend, get_solver

        s = get_solver("OMP")
        assert s.backend == Backend.OMP

    def test_get_solver_enum(self):
        from kwave.solvers import Backend, get_solver

        s = get_solver(Backend.NATIVE, use_gpu=True)
        assert s.use_gpu is True

    def test_unknown_backend_raises(self):
        import pytest

        from kwave.solvers import get_solver

        with pytest.raises(ValueError):
            get_solver("bad")
