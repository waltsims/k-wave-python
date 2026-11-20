"""Particle-velocity recording conventions of the Python backend.

Naming differs from MATLAB k-Wave / the C++ binary:

===================  =======================  ==========================
Python backend       MATLAB ``sensor.record``  C++ binary dataset
===================  =======================  ==========================
``ux``               ``u_non_staggered``       ``ux_non_staggered``
``ux_non_staggered`` ``u_non_staggered``       ``ux_non_staggered``
``ux_staggered``     ``u``                     ``ux``
===================  =======================  ==========================
"""
import subprocess
import sys

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
from kwave.utils.mapgen import make_disc

GRID = (64, 48)
SENSOR_POINTS = [(20, 20), (30, 12), (40, 30), (25, 28), (36, 22), (10, 40)]


def _problem(mask=None):
    """Smoothed p0 disc in a homogeneous 2D medium (no source scaling involved)."""
    kgrid = kWaveGrid(Vector(list(GRID)), Vector([1e-4, 1e-4]))
    kgrid.setTime(60, 2e-8)
    medium = kWaveMedium(sound_speed=1500, density=1000)
    source = kSource()
    source.p0 = 5.0 * make_disc(Vector(list(GRID)), Vector([30, 20]), 4).astype(float)
    if mask is None:
        mask = np.zeros(GRID)
        for ij in SENSOR_POINTS:
            mask[ij] = 1
    return kgrid, medium, source, kSensor(mask=mask)


def _run_python(record, mask=None, **kwargs):
    kgrid, medium, source, sensor = _problem(mask)
    sensor.record = record
    return kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", quiet=True, pml_size=10, **kwargs)


class TestNonStaggeredVelocity:
    def test_u_non_staggered_is_returned(self):
        result = _run_python(["u_non_staggered"])
        assert set(result) == {"ux_non_staggered", "uy_non_staggered"}
        assert result["ux_non_staggered"].shape == (len(SENSOR_POINTS), 60)
        assert np.max(np.abs(result["ux_non_staggered"])) > 0

    def test_single_component(self):
        result = _run_python(["p", "uy_non_staggered"])
        assert set(result) == {"p", "uy_non_staggered"}

    def test_u_non_staggered_equals_u(self):
        result = _run_python(["u", "u_non_staggered"])
        for a in "xy":
            np.testing.assert_array_equal(result[f"u{a}_non_staggered"], result[f"u{a}"])

    def test_non_staggered_is_half_cell_shift_of_staggered(self):
        """``u_non_staggered`` is ``u_staggered`` shifted back by half a grid spacing along its own axis."""
        result = _run_python(["u_staggered", "u_non_staggered"], mask=np.ones(GRID), pml_inside=True)
        for axis, a in enumerate("xy"):
            staggered = np.asarray(result[f"u{a}_staggered"]).reshape(*GRID, -1)
            non_staggered = np.asarray(result[f"u{a}_non_staggered"]).reshape(*GRID, -1)
            n = GRID[axis]
            # k * dx / 2 for the FFT wavenumbers of this axis
            half_cell = np.pi * np.fft.fftfreq(n)
            shape = [1, 1, 1]
            shape[axis] = n
            shift = np.exp(-1j * half_cell).reshape(shape)
            expected = np.real(np.fft.ifft(shift * np.fft.fft(staggered, axis=axis), axis=axis))
            scale = np.max(np.abs(staggered))
            np.testing.assert_allclose(non_staggered, expected, rtol=0, atol=1e-12 * scale)
            # the two grids genuinely differ, so mixing them up is detectable
            assert np.max(np.abs(non_staggered - staggered)) > 1e-2 * scale


class TestUnsupportedRecordFields:
    @pytest.mark.parametrize("field", ["u_split_field", "p_max_all", "p_min_all", "u_max_all", "u_min_all", "uz", "uz_final", "pressure"])
    def test_unsupported_field_raises(self, field):
        with pytest.raises(ValueError, match="Unsupported sensor.record"):
            _run_python(["p", field])

    def test_error_names_the_field(self):
        with pytest.raises(ValueError, match="u_split_field"):
            _run_python(["p", "u_split_field"])

    def test_all_supported_fields_are_returned(self):
        record = ["p", "p_max", "p_min", "p_rms", "p_final", "u", "u_staggered", "u_non_staggered", "u_max", "u_min", "u_rms", "u_final"]
        record += ["I", "I_avg"]
        result = _run_python(record)
        expected = {"p", "p_max", "p_min", "p_rms", "p_final"}
        for a in "xy":
            expected |= {f"u{a}{s}" for s in ("", "_staggered", "_non_staggered", "_max", "_min", "_rms", "_final")}
            expected |= {f"I{a}", f"I{a}_avg"}
        assert set(result) == expected


def _omp_binary():
    """Path to the bundled OMP binary, or skip if it cannot be executed here."""
    path = kwave.BINARY_PATH / ("kspaceFirstOrder-OMP.exe" if sys.platform == "win32" else "kspaceFirstOrder-OMP")
    if not path.exists():
        pytest.skip(f"C++ binary not found at {path}")
    try:
        probe = subprocess.run([str(path), "--help"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"C++ binary is not runnable: {exc}")
    if probe.returncode != 0:
        pytest.skip(f"C++ binary is not runnable (exit code {probe.returncode})")
    return path


def test_velocity_matches_cpp_binary(tmp_path):
    """Python velocity time series match the C++ binary sample-by-sample (binary is float32)."""
    binary = _omp_binary()

    py = _run_python(["p", "u", "u_staggered", "u_non_staggered"])

    kgrid, medium, source, sensor = _problem()
    files = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", save_only=True, data_path=str(tmp_path), quiet=True, pml_size=10)
    output_file = str(tmp_path / "velocity_output.h5")
    cmd = [str(binary), "-i", files["input_file"], "-o", output_file, "--p_raw", "--u_raw", "--u_non_staggered_raw"]
    subprocess.run(cmd, check=True, capture_output=True)

    # Binary stores (1, Nt, n_sensor) with sensor points in column-major order; Python uses row-major order.
    mask = np.asarray(sensor.mask).astype(bool)
    f_order_rank = np.cumsum(mask.ravel(order="F")).reshape(GRID, order="F") - 1
    to_c_order = f_order_rank[mask]
    with h5py.File(output_file, "r") as f:
        cpp = {k: np.asarray(f[k])[0].T[to_c_order] for k in ("p", "ux", "uy", "ux_non_staggered", "uy_non_staggered")}

    def max_rel_err(a, b):
        return np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b))

    assert max_rel_err(py["p"], cpp["p"]) < 1e-5
    for a in "xy":
        assert max_rel_err(py[f"u{a}_staggered"], cpp[f"u{a}"]) < 1e-5
        assert max_rel_err(py[f"u{a}_non_staggered"], cpp[f"u{a}_non_staggered"]) < 1e-5
        assert max_rel_err(py[f"u{a}"], cpp[f"u{a}_non_staggered"]) < 1e-5
        # Python "ux" is not the binary's (staggered) "ux": the half-cell offset is far above tolerance
        assert max_rel_err(py[f"u{a}"], cpp[f"u{a}"]) > 1e-2
