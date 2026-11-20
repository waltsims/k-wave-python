"""Sensor-data orientation of the C++ backend (issue #760).

``kspaceFirstOrder(..., backend="cpp")`` must return time-series as
``(n_sensor, Nt)`` with sensor points in C-flattened order -- like the Python
backend -- whether ``n_sensor`` is smaller than, equal to, or larger than ``Nt``.

Two layers of tests:

* ``TestReadback*`` replace the binary with a stand-in that writes an output
  file in the layout of the real binary (as seen by h5py: time-series
  ``(1, Nt, n_sensor)``, per-sensor aggregates ``(1, 1, n_sensor)``, full-grid
  fields ``(Nz, Ny, Nx)``).  Always run.
* ``TestCppMatchesPython`` runs the real OMP binary and compares against the
  Python backend.  Skipped when the binary cannot be executed on this machine.
"""

import subprocess

import h5py
import numpy as np
import pytest

import kwave
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.solvers.cpp_simulation import CppSimulation

# (grid_shape, index selecting the sensor points) -> n_sensor = 10, 1-D/2-D/3-D
_SENSORS = {
    1: ((24,), (slice(4, 14),)),
    2: ((12, 8), (slice(3, 5), slice(2, 7))),
    3: ((8, 6, 5), (slice(2, 4), slice(1, 6), 2)),
}
_N_SENSOR = 10
# n_sensor < Nt, n_sensor == Nt, n_sensor > Nt, single time step
_NT_CASES = [25, _N_SENSOR, 4, 1]


def _make_inputs(grid_shape, sensor_index, Nt):
    kgrid = kWaveGrid(Vector(list(grid_shape)), Vector([1e-4] * len(grid_shape)))
    kgrid.setTime(Nt, 1e-8)
    medium = kWaveMedium(sound_speed=1500, density=1000)
    source = kSource()
    # Two unequal off-centre peaks: no symmetry that could hide a wrong sensor order
    p0 = np.zeros(grid_shape)
    p0[tuple(n // 2 for n in grid_shape)] = 1.0
    p0[tuple(n // 3 for n in grid_shape)] = 0.5
    source.p0 = p0
    mask = np.zeros(grid_shape, dtype=bool)
    mask[sensor_index] = True
    return kgrid, medium, source, kSensor(mask=mask)


def _f_index_grid(grid_shape):
    """Array whose value at each grid point is that point's 0-based F-order linear index."""
    return np.arange(int(np.prod(grid_shape))).reshape(grid_shape, order="F")


def _write_fake_output(output_file, Nx, Ny, Nz, Nt, sensor_index):
    """Write an output file in the C++ binary's on-disk layout.

    ``sensor_index`` holds the 0-based F-order linear indices of the sensor points.
    Values encode their position: a time-series sample is
    ``1000 * (F-order linear index of the sensor point) + time step``.
    """
    series = (1000 * sensor_index[None, :] + np.arange(Nt)[:, None]).astype(np.float32)[None]  # (1, Nt, n_sensor)
    per_sensor = sensor_index.astype(np.float32).reshape(1, 1, -1)  # (1, 1, n_sensor)
    full_grid = np.arange(Nx * Ny * Nz, dtype=np.float32).reshape(Nz, Ny, Nx)

    with h5py.File(output_file, "w") as f:
        f["p"] = series
        f["ux"] = -series
        f["uy"] = series + 0.25
        f["uz_non_staggered"] = series + 0.5
        f["p_max"] = per_sensor
        f["p_rms"] = per_sensor + 0.5
        f["ux_min"] = -per_sensor
        f["p_final"] = full_grid
        f["p_max_all"] = full_grid + 0.5
        f["Nt"] = np.array(Nt, dtype=np.uint64).reshape(1, 1, 1)


def _fake_execute(self, input_file, output_file, **kwargs):
    """Stand-in for the C++ binary, driven by the input file the backend wrote."""
    with h5py.File(input_file, "r") as f:
        Nx, Ny, Nz, Nt = (int(f[name][:].ravel()[0]) for name in ("Nx", "Ny", "Nz", "Nt"))
        sensor_index = f["sensor_mask_index"][:].ravel().astype(np.int64) - 1
    _write_fake_output(output_file, Nx, Ny, Nz, Nt, sensor_index)


def _run_fake_cpp(kgrid, medium, source, sensor, tmp_path):
    """Run the cpp backend with the stand-in binary.

    2-D/3-D go through ``kspaceFirstOrder``.  The cpp backend cannot serialize
    1-D inputs yet, so 1-D exercises the readback (parse + reorder) directly.
    """
    if kgrid.dim > 1:
        return kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", pml_inside=True, pml_size=2, quiet=True)
    output_file = tmp_path / "kwave_output.h5"
    _write_fake_output(output_file, int(kgrid.N[0]), 1, 1, int(kgrid.Nt), np.flatnonzero(np.asarray(sensor.mask)))
    sim = CppSimulation(kgrid, medium, source, sensor, pml_size=(2,), pml_alpha=(2.0,))
    return sim._fix_output_order(sim._parse_output(output_file))


@pytest.fixture
def fake_binary(monkeypatch):
    monkeypatch.setattr(CppSimulation, "_execute", _fake_execute)


@pytest.mark.usefixtures("fake_binary")
class TestReadbackOrientation:
    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("Nt", _NT_CASES)
    def test_time_series_are_n_sensor_by_nt(self, ndim, Nt, tmp_path):
        grid_shape, sensor_index = _SENSORS[ndim]
        kgrid, medium, source, sensor = _make_inputs(grid_shape, sensor_index, Nt)

        result = _run_fake_cpp(kgrid, medium, source, sensor, tmp_path)

        # Sensor points in C-flattened order, each labelled by its F-order linear index
        point_ids = _f_index_grid(grid_shape)[np.asarray(sensor.mask, dtype=bool)]
        expected = 1000.0 * point_ids[:, None] + np.arange(Nt)[None, :]
        assert expected.shape == (_N_SENSOR, Nt)

        np.testing.assert_array_equal(result["p"], expected)
        np.testing.assert_array_equal(result["ux"], -expected)
        np.testing.assert_array_equal(result["uy"], expected + 0.25)
        np.testing.assert_array_equal(result["uz_non_staggered"], expected + 0.5)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("Nt", _NT_CASES)
    def test_aggregates_and_full_grid_fields(self, ndim, Nt, tmp_path):
        grid_shape, sensor_index = _SENSORS[ndim]
        kgrid, medium, source, sensor = _make_inputs(grid_shape, sensor_index, Nt)

        result = _run_fake_cpp(kgrid, medium, source, sensor, tmp_path)

        point_ids = _f_index_grid(grid_shape)[np.asarray(sensor.mask, dtype=bool)].astype(float)
        np.testing.assert_array_equal(result["p_max"], point_ids)
        np.testing.assert_array_equal(result["p_rms"], point_ids + 0.5)
        np.testing.assert_array_equal(result["ux_min"], -point_ids)

        np.testing.assert_array_equal(result["p_final"], _f_index_grid(grid_shape))
        np.testing.assert_array_equal(result["p_max_all"], _f_index_grid(grid_shape) + 0.5)

    @pytest.mark.parametrize("Nt", [25, 1])
    def test_single_sensor_point_stays_2d(self, Nt):
        grid_shape = (12, 8)
        kgrid, medium, source, sensor = _make_inputs(grid_shape, (5, 3), Nt)

        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", pml_inside=True, pml_size=2, quiet=True)

        point_id = _f_index_grid(grid_shape)[5, 3]
        np.testing.assert_array_equal(result["p"], 1000.0 * point_id + np.arange(Nt)[None, :])
        assert result["p"].shape == (1, Nt)
        assert result["p_max"].shape == (1,)

    def test_n_sensor_equal_nt_does_not_permute_time_axis(self):
        """With a square (n_sensor == Nt) array every row must still be one sensor's monotonic time trace."""
        grid_shape, sensor_index = _SENSORS[2]
        kgrid, medium, source, sensor = _make_inputs(grid_shape, sensor_index, _N_SENSOR)

        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", pml_inside=True, pml_size=2, quiet=True)

        assert np.all(np.diff(result["p"], axis=1) == 1)

    def test_pml_outside_full_grid_sensor(self):
        """Default pml_inside=False: sensor rows follow the C-order of the user's (unexpanded) grid."""
        grid_shape, Nt = (8, 6), 30
        kgrid, medium, source, _ = _make_inputs(grid_shape, (slice(None),), Nt)
        sensor = kSensor(mask=np.ones(grid_shape, dtype=bool))

        result = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", pml_size=2, quiet=True)

        expanded = (12, 10)
        point_ids = _f_index_grid(expanded)[2:-2, 2:-2].ravel()
        np.testing.assert_array_equal(result["p"], 1000.0 * point_ids[:, None] + np.arange(Nt)[None, :])
        np.testing.assert_array_equal(result["p_final"], _f_index_grid(expanded)[2:-2, 2:-2])


def _omp_binary_runs():
    binary = kwave.BINARY_PATH / "kspaceFirstOrder-OMP"
    if not binary.exists():
        return False
    try:
        return subprocess.run([str(binary), "--version"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _omp_binary_runs(), reason="C++ OMP binary is not runnable on this machine")
class TestCppMatchesPython:
    """End-to-end: the real binary and the Python backend agree in shape and value."""

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("Nt", _NT_CASES)
    def test_sensor_pressure(self, ndim, Nt):
        grid_shape, sensor_index = {2: ((32, 24), (slice(5, 7), slice(3, 8))), 3: ((16, 12, 10), (slice(5, 7), slice(3, 8), 4))}[ndim]
        kgrid, medium, source, sensor = _make_inputs(grid_shape, sensor_index, Nt)
        kwargs = dict(pml_inside=True, pml_size=4, smooth_p0=False, quiet=True)

        res_py = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", **kwargs)
        res_cpp = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", num_threads=1, **kwargs)

        p_py, p_cpp = np.asarray(res_py["p"]), res_cpp["p"]
        assert p_py.shape == (_N_SENSOR, Nt)
        assert p_cpp.shape == (_N_SENSOR, Nt)
        # float32 binary vs float64 Python solver
        np.testing.assert_allclose(p_cpp, p_py, rtol=0, atol=1e-4 * np.abs(p_py).max())
