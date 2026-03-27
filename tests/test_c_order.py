"""Tests for C-order migration: _fix_output_order, _reshape_sensor_to_grid, FutureWarnings."""
from types import SimpleNamespace

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import reshape_to_grid

# ---------------------------------------------------------------------------
# _fix_output_order (CppSimulation)
# ---------------------------------------------------------------------------


class TestFixOutputOrder:
    """Unit-test CppSimulation._fix_output_order with synthetic data."""

    def _make_sim(self, grid_shape, sensor_mask=None):
        """Create a minimal CppSimulation without writing HDF5."""
        from kwave.solvers.cpp_simulation import CppSimulation

        kgrid = kWaveGrid(Vector(list(grid_shape)), Vector([1e-4] * len(grid_shape)))
        kgrid.setTime(10, 1e-8)
        medium = kWaveMedium(sound_speed=1500)
        source = kSource()
        sensor = kSensor()
        if sensor_mask is not None:
            sensor.mask = sensor_mask
        else:
            sensor = None
        return CppSimulation(kgrid, medium, source, sensor, pml_size=(5,) * len(grid_shape), pml_alpha=(2.0,) * len(grid_shape))

    def test_transpose_full_grid_2d(self):
        """Full-grid fields are transposed from reversed F-order to C-order."""
        sim = self._make_sim((4, 6))
        # C++ outputs full-grid as (Ny, Nx) = (6, 4) due to F-order HDF5
        result = {"p_final": np.arange(24).reshape(6, 4)}
        fixed = sim._fix_output_order(result)
        assert fixed["p_final"].shape == (4, 6)

    def test_transpose_full_grid_3d(self):
        """Full-grid 3D fields are transposed."""
        sim = self._make_sim((3, 4, 5))
        result = {"p_max": np.arange(60).reshape(5, 4, 3)}
        fixed = sim._fix_output_order(result)
        assert fixed["p_max"].shape == (3, 4, 5)

    def test_sensor_row_reorder_2d(self):
        """Sensor time-series rows are reordered from F-indexed to C-indexed."""
        grid_shape = (3, 4)
        mask = np.zeros(grid_shape, dtype=bool)
        mask[0, 0] = True  # F-index 0, C-index 0
        mask[1, 0] = True  # F-index 1, C-index 4
        mask[0, 1] = True  # F-index 3, C-index 1
        sim = self._make_sim(grid_shape, sensor_mask=mask)

        n_sensor = int(mask.sum())
        Nt = 5
        # F-order sensor data: rows ordered by F-flat index (0, 1, 3)
        f_data = np.array(
            [
                [10, 11, 12, 13, 14],  # F-idx 0 → (0,0)
                [20, 21, 22, 23, 24],  # F-idx 1 → (1,0)
                [30, 31, 32, 33, 34],
            ]
        )  # F-idx 3 → (0,1)
        result = {"p": f_data.copy()}
        fixed = sim._fix_output_order(result)

        # C-order: rows ordered by C-flat index
        # (0,0)=C-idx 0, (0,1)=C-idx 1, (1,0)=C-idx 4
        # So C-order should be: (0,0), (0,1), (1,0)
        assert fixed["p"][0, 0] == 10  # (0,0) stays first
        assert fixed["p"][1, 0] == 30  # (0,1) was F-idx 3, now second
        assert fixed["p"][2, 0] == 20  # (1,0) was F-idx 1, now third

    def test_sensor_none_full_grid(self):
        """When sensor is None, all grid points are sensor points."""
        sim = self._make_sim((2, 3), sensor_mask=None)
        Nt = 4
        result = {"p": np.arange(24).reshape(6, Nt)}
        fixed = sim._fix_output_order(result)
        assert fixed["p"].shape == (6, Nt)

    def test_1d_skips_reorder(self):
        """1D grids skip sensor reordering (F and C order are identical)."""
        sim = self._make_sim((8,), sensor_mask=np.ones(8, dtype=bool))
        result = {"p": np.arange(40).reshape(8, 5)}
        fixed = sim._fix_output_order(result)
        np.testing.assert_array_equal(fixed["p"], np.arange(40).reshape(8, 5))

    def test_non_grid_suffix_skipped(self):
        """Non-grid fields are not transposed."""
        sim = self._make_sim((4, 6))
        result = {"p": np.arange(24).reshape(24, 1), "p_final": np.arange(24).reshape(6, 4)}
        fixed = sim._fix_output_order(result)
        assert fixed["p"].shape == (24, 1)  # unchanged
        assert fixed["p_final"].shape == (4, 6)  # transposed

    def test_1d_aggregate_reorder(self):
        """1D aggregate fields are not reordered."""
        sim = self._make_sim((8,), sensor_mask=np.ones(8, dtype=bool))
        result = {"p_max": np.arange(8)}
        fixed = sim._fix_output_order(result)
        # 1D: ndim=1, so p_max (ndim=1) is treated as full-grid and transposed (no-op for 1D)
        assert fixed["p_max"].shape == (8,)


# ---------------------------------------------------------------------------
# reshape_to_grid helper
# ---------------------------------------------------------------------------


class TestReshapeToGrid:
    def test_time_series_2d(self):
        """(n_sensor, Nt) → (*grid_shape, Nt)."""
        data = np.arange(120).reshape(24, 5)
        out = reshape_to_grid(data, (4, 6))
        assert out.shape == (4, 6, 5)

    def test_aggregate_1d(self):
        """(n_sensor,) → (*grid_shape)."""
        data = np.arange(24)
        out = reshape_to_grid(data, (4, 6))
        assert out.shape == (4, 6)

    def test_3d_grid(self):
        """Works with 3D grids."""
        data = np.arange(60).reshape(60, 1)
        out = reshape_to_grid(data, (3, 4, 5))
        assert out.shape == (3, 4, 5, 1)

    def test_passthrough_higher_dim(self):
        """Higher-dim arrays pass through unchanged."""
        data = np.arange(120).reshape(2, 3, 4, 5)
        out = reshape_to_grid(data, (4, 6))
        assert out.shape == (2, 3, 4, 5)


# ---------------------------------------------------------------------------
# FutureWarning tests
# ---------------------------------------------------------------------------


class TestFutureWarnings:
    def test_cart2grid_warns_on_f_order(self):
        from kwave.utils.conversion import cart2grid

        kgrid = kWaveGrid(Vector([32, 32]), Vector([1e-4, 1e-4]))
        cart = np.array([[0.0], [0.0]])
        with pytest.warns(FutureWarning, match="cart2grid"):
            cart2grid(kgrid, cart)

    def test_cart2grid_no_warn_on_explicit_f(self):
        """Explicit order='F' should NOT warn — only implicit default warns."""
        from kwave.utils.conversion import cart2grid

        kgrid = kWaveGrid(Vector([32, 32]), Vector([1e-4, 1e-4]))
        cart = np.array([[0.0], [0.0]])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            cart2grid(kgrid, cart, order="F")

    def test_cart2grid_no_warn_on_c_order(self):
        from kwave.utils.conversion import cart2grid

        kgrid = kWaveGrid(Vector([32, 32]), Vector([1e-4, 1e-4]))
        cart = np.array([[0.0], [0.0]])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            cart2grid(kgrid, cart, order="C")

    def test_get_distributed_source_signal_warns(self):
        from kwave.utils.kwave_array import kWaveArray

        karray = kWaveArray()
        karray.add_line_element([0, -1e-3], [0, 1e-3])
        kgrid = kWaveGrid(Vector([32, 32]), Vector([1e-4, 1e-4]))
        kgrid.setTime(5, 1e-8)
        signal = np.ones((1, 5))
        with pytest.warns(FutureWarning, match="get_distributed_source_signal"):
            karray.get_distributed_source_signal(kgrid, signal)

    def test_combine_sensor_data_warns(self):
        from kwave.utils.kwave_array import kWaveArray

        karray = kWaveArray()
        karray.add_line_element([0, -1e-3], [0, 1e-3])
        kgrid = kWaveGrid(Vector([32, 32]), Vector([1e-4, 1e-4]))
        kgrid.setTime(5, 1e-8)
        mask = karray.get_array_binary_mask(kgrid)
        n_sensor = int(mask.sum())
        sensor_data = np.ones((n_sensor, 5))
        with pytest.warns(FutureWarning, match="combine_sensor_data"):
            karray.combine_sensor_data(kgrid, sensor_data)
