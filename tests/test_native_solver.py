"""Minimal tests for the native solver path and new unified API.

Covers: kspaceFirstOrder.py, kwave_adapter.py, kspace_solver.py,
        cpp_simulation.py (save_only), validation.py, compat.py
"""
import os
import warnings

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import _normalize_pml, kspaceFirstOrder
from kwave.solvers.validation import validate_cfl, validate_medium, validate_source

# -- Fixtures --


@pytest.fixture
def grid_2d():
    """Small 2D grid for fast tests."""
    kgrid = kWaveGrid(Vector([64, 64]), Vector([0.1e-3, 0.1e-3]))
    kgrid.makeTime(1500, 0.3)
    # Limit to 10 time steps for speed
    kgrid.setTime(10, float(kgrid.dt))
    return kgrid


@pytest.fixture
def grid_1d():
    """Small 1D grid."""
    kgrid = kWaveGrid(Vector([64]), Vector([0.1e-3]))
    kgrid.setTime(20, 1e-8)
    return kgrid


# -- 1. End-to-end native 2D with initial pressure --


class TestNative2D:
    def test_p0_source(self, grid_2d):
        """2D IVP: initial pressure → sensor records pressure."""
        medium = kWaveMedium(sound_speed=1500)
        source = kSource()
        source.p0 = np.zeros((64, 64))
        source.p0[32, 32] = 1.0

        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        sensor.record = ["p", "p_final"]

        result = kspaceFirstOrder(grid_2d, medium, source, sensor, backend="native")

        assert "p" in result
        assert "p_final" in result
        assert result["p"].shape == (64 * 64, 10)
        # p0 energy should propagate — not all zeros after 10 steps
        assert np.max(np.abs(result["p"])) > 0

    def test_heterogeneous_medium(self, grid_2d):
        """2D with spatially varying sound speed and density."""
        c = 1500 * np.ones((64, 64))
        c[:32, :] = 1600
        medium = kWaveMedium(sound_speed=c, density=1000)

        source = kSource()
        source.p0 = np.zeros((64, 64))
        source.p0[32, 32] = 1.0

        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))

        result = kspaceFirstOrder(grid_2d, medium, source, sensor, backend="native")
        assert result["p"].shape[0] == 64 * 64

    def test_absorption(self, grid_2d):
        """2D with power-law absorption."""
        medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)

        source = kSource()
        source.p0 = np.zeros((64, 64))
        source.p0[32, 32] = 1.0

        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))

        result = kspaceFirstOrder(grid_2d, medium, source, sensor, backend="native")
        assert result["p"].shape == (64 * 64, 10)

    def test_pml_auto(self):
        """pml_size='auto' resolves without error."""
        kgrid = kWaveGrid(Vector([128, 128]), Vector([0.1e-3, 0.1e-3]))
        kgrid.makeTime(1500, 0.3)
        kgrid.setTime(5, float(kgrid.dt))

        medium = kWaveMedium(sound_speed=1500)
        source = kSource()
        source.p0 = np.zeros((128, 128))
        source.p0[64, 64] = 1.0
        sensor = kSensor(mask=np.ones((128, 128), dtype=bool))

        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="native", pml_size="auto")
        assert "p" in result

    def test_record_aggregates(self, grid_2d):
        """Recording p_max, p_rms works."""
        medium = kWaveMedium(sound_speed=1500)
        source = kSource()
        source.p0 = np.zeros((64, 64))
        source.p0[32, 32] = 1.0

        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        sensor.record = ["p", "p_max", "p_rms"]

        result = kspaceFirstOrder(grid_2d, medium, source, sensor, backend="native")
        assert "p_max" in result
        assert "p_rms" in result
        assert result["p_max"].shape == (64 * 64,)


# -- 2. Native 1D with time-varying source --


class TestNative1D:
    def test_tvsp(self, grid_1d):
        """1D time-varying pressure source with sparse sensor."""
        medium = kWaveMedium(sound_speed=1500)

        source = kSource()
        source.p_mask = np.zeros(64)
        source.p_mask[10] = 1
        source.p = np.sin(2 * np.pi * 1e6 * np.arange(20) * 1e-8).reshape(1, -1)

        sensor_mask = np.zeros(64)
        sensor_mask[50] = 1
        sensor = kSensor(mask=sensor_mask)

        result = kspaceFirstOrder(grid_1d, medium, source, sensor, backend="native")
        assert "p" in result
        assert result["p"].shape == (1, 20)  # 1 sensor point, 20 time steps

    def test_velocity_source(self, grid_1d):
        """1D velocity source."""
        medium = kWaveMedium(sound_speed=1500)

        source = kSource()
        source.u_mask = np.zeros(64)
        source.u_mask[10] = 1
        source.ux = np.sin(2 * np.pi * 1e6 * np.arange(20) * 1e-8).reshape(1, -1)

        sensor_mask = np.zeros(64)
        sensor_mask[50] = 1
        sensor = kSensor(mask=sensor_mask)

        result = kspaceFirstOrder(grid_1d, medium, source, sensor, backend="native")
        assert "p" in result


# -- 3. C++ save_only (HDF5 serialization) --


class TestCppSaveOnly:
    def test_save_only_creates_hdf5(self, grid_2d, tmp_path):
        """cpp backend with save_only writes HDF5 without running binary."""
        medium = kWaveMedium(sound_speed=1500)
        source = kSource()
        source.p0 = np.zeros((64, 64))
        source.p0[32, 32] = 1.0
        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))

        result = kspaceFirstOrder(
            grid_2d,
            medium,
            source,
            sensor,
            backend="cpp",
            save_only=True,
            data_path=str(tmp_path),
        )

        assert "input_file" in result
        assert os.path.exists(result["input_file"])

    def test_save_only_with_absorption(self, grid_2d, tmp_path):
        """HDF5 serialization with absorbing medium."""
        medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.75, alpha_power=1.5)
        source = kSource()
        source.p0 = np.zeros((64, 64))
        source.p0[32, 32] = 1.0
        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))

        result = kspaceFirstOrder(
            grid_2d,
            medium,
            source,
            sensor,
            backend="cpp",
            save_only=True,
            data_path=str(tmp_path),
        )
        assert os.path.exists(result["input_file"])


# -- 4. Validation and compat edge cases --


class TestValidationEdges:
    def test_negative_density(self):
        kgrid = kWaveGrid(Vector([32, 32]), Vector([0.1e-3, 0.1e-3]))
        kgrid.setTime(10, 1e-8)
        medium = kWaveMedium(sound_speed=1500, density=-1000)
        with pytest.raises(ValueError, match="positive"):
            validate_medium(medium, kgrid)

    def test_negative_alpha_coeff(self):
        kgrid = kWaveGrid(Vector([32, 32]), Vector([0.1e-3, 0.1e-3]))
        kgrid.setTime(10, 1e-8)
        medium = kWaveMedium(sound_speed=1500, alpha_coeff=-0.5, alpha_power=1.5)
        with pytest.raises(ValueError, match="non-negative"):
            validate_medium(medium, kgrid)

    def test_alpha_power_out_of_range_warns(self):
        kgrid = kWaveGrid(Vector([32, 32]), Vector([0.1e-3, 0.1e-3]))
        kgrid.setTime(10, 1e-8)
        medium = kWaveMedium(sound_speed=1500, alpha_coeff=0.5, alpha_power=5.0)
        with pytest.warns(match="alpha_power"):
            validate_medium(medium, kgrid)

    def test_velocity_without_mask(self):
        kgrid = kWaveGrid(Vector([32, 32]), Vector([0.1e-3, 0.1e-3]))
        kgrid.setTime(10, 1e-8)
        source = kSource()
        source.ux = np.ones(10)
        with pytest.raises(ValueError, match="u_mask"):
            validate_source(source, kgrid)

    def test_p_mask_wrong_size(self):
        kgrid = kWaveGrid(Vector([32, 32]), Vector([0.1e-3, 0.1e-3]))
        kgrid.setTime(10, 1e-8)
        source = kSource()
        source.p = np.ones(10)
        source.p_mask = np.ones(100)
        with pytest.raises(ValueError, match="p_mask"):
            validate_source(source, kgrid)

    def test_u_mask_wrong_size(self):
        kgrid = kWaveGrid(Vector([32, 32]), Vector([0.1e-3, 0.1e-3]))
        kgrid.setTime(10, 1e-8)
        source = kSource()
        source.ux = np.ones(10)
        source.u_mask = np.ones(100)
        with pytest.raises(ValueError, match="u_mask"):
            validate_source(source, kgrid)


class TestNormalizePml:
    def test_scalar(self):
        assert _normalize_pml(20, 2) == (20, 20)
        assert _normalize_pml(10, 3) == (10, 10, 10)

    def test_single_element_tuple(self):
        assert _normalize_pml((15,), 3) == (15, 15, 15)

    def test_matching_tuple(self):
        assert _normalize_pml((10, 20), 2) == (10, 20)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="3-element"):
            _normalize_pml((10, 20), 3)

    def test_unknown_backend_raises(self, grid_2d):
        medium = kWaveMedium(sound_speed=1500)
        source = kSource()
        source.p0 = np.zeros((64, 64))
        sensor = kSensor(mask=np.ones((64, 64), dtype=bool))
        with pytest.raises(ValueError, match="Unknown backend"):
            kspaceFirstOrder(grid_2d, medium, source, sensor, backend="bad")


class TestCompatEdges:
    def test_pml_scalar_from_options(self):
        from kwave.compat import options_to_kwargs
        from kwave.options.simulation_options import SimulationOptions

        opts = SimulationOptions()
        opts.pml_x_size = 15
        kwargs = options_to_kwargs(simulation_options=opts)
        assert kwargs["pml_size"] == 15

    def test_pml_alpha_per_axis_collapse(self):
        from kwave.compat import options_to_kwargs
        from kwave.options.simulation_options import SimulationOptions

        opts = SimulationOptions()
        opts.pml_x_alpha = 2.0
        opts.pml_y_alpha = 2.0
        kwargs = options_to_kwargs(simulation_options=opts)
        assert kwargs["pml_alpha"] == 2.0  # collapsed to scalar

    def test_pml_alpha_per_axis_different(self):
        from kwave.compat import options_to_kwargs
        from kwave.options.simulation_options import SimulationOptions

        opts = SimulationOptions()
        opts.pml_x_alpha = 1.0
        opts.pml_y_alpha = 3.0
        opts.pml_z_alpha = 2.0
        kwargs = options_to_kwargs(simulation_options=opts)
        # All 3 axes returned since z defaults to 2.0
        assert kwargs["pml_alpha"] == (1.0, 3.0, 2.0)

    def test_backend_cuda_mapping(self):
        from kwave.compat import options_to_kwargs
        from kwave.options.simulation_execution_options import SimulationExecutionOptions

        opts = SimulationExecutionOptions(backend="CUDA")
        kwargs = options_to_kwargs(execution_options=opts)
        assert kwargs["backend"] == "cpp"
        assert kwargs["use_gpu"] is True

    def test_backend_omp_mapping(self):
        from kwave.compat import options_to_kwargs
        from kwave.options.simulation_execution_options import SimulationExecutionOptions

        opts = SimulationExecutionOptions(backend="OMP")
        kwargs = options_to_kwargs(execution_options=opts)
        assert kwargs["backend"] == "cpp"
        assert kwargs["use_gpu"] is False
