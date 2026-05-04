"""Tests for the ``data_cast`` parameter on the modern ``kspaceFirstOrder()`` API.

Issue #695: expose data_cast on the modern API to control compute precision.

The Python backend honors ``data_cast``:
  - ``'off'`` / ``'double'`` -> ``np.float64`` (default)
  - ``'single'``             -> ``np.float32``

The C++ backend uses fixed internal precision regardless; setting ``data_cast``
to anything other than ``'off'`` warns.
"""
from copy import deepcopy

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder


def _build_minimal():
    N = Vector([48, 48])
    dx = Vector([0.1e-3, 0.1e-3])
    kgrid = kWaveGrid(N, dx)
    kgrid.makeTime(1500)
    medium = kWaveMedium(sound_speed=1500 * np.ones(tuple(N)), density=1000 * np.ones(tuple(N)))
    source = kSource()
    source.p_mask = np.zeros(tuple(N), dtype=bool)
    source.p_mask[N.x // 2, N.y // 2] = True
    t = kgrid.t_array.squeeze()
    source.p = (np.sin(2 * np.pi * 1e6 * t) * np.exp(-((t - 5e-7) ** 2) / (2e-7) ** 2))[np.newaxis, :]
    sensor = kSensor(mask=np.ones(tuple(N), dtype=bool))
    return kgrid, medium, source, sensor


@pytest.mark.parametrize("data_cast,expected_dtype", [("off", np.float64), ("double", np.float64), ("single", np.float32)])
def test_python_backend_output_dtype_matches_data_cast(data_cast, expected_dtype):
    """Output ``p`` array dtype must match the precision requested via ``data_cast``."""
    kgrid, medium, source, sensor = _build_minimal()
    result = kspaceFirstOrder(
        kgrid=kgrid,
        medium=medium,
        source=deepcopy(source),
        sensor=sensor,
        backend="python",
        device="cpu",
        data_cast=data_cast,
        pml_inside=False,
        smooth_p0=False,
        quiet=True,
    )
    p = np.asarray(result["p"])
    assert p.dtype == expected_dtype, f"data_cast={data_cast!r} produced {p.dtype}, expected {expected_dtype}"
    assert not np.any(np.isnan(p)), f"data_cast={data_cast!r} produced NaN output"
    assert np.nanmax(np.abs(p)) > 0, f"data_cast={data_cast!r} produced trivially zero output"


def test_default_data_cast_is_off():
    """Calling without data_cast must behave the same as data_cast='off' (float64)."""
    kgrid, medium, source, sensor = _build_minimal()
    result = kspaceFirstOrder(
        kgrid=kgrid,
        medium=medium,
        source=deepcopy(source),
        sensor=sensor,
        backend="python",
        device="cpu",
        pml_inside=False,
        smooth_p0=False,
        quiet=True,
    )
    assert np.asarray(result["p"]).dtype == np.float64


def test_invalid_data_cast_raises():
    """Unknown data_cast values must raise ValueError, not silently fall through."""
    kgrid, medium, source, sensor = _build_minimal()
    with pytest.raises(ValueError, match="data_cast"):
        kspaceFirstOrder(
            kgrid=kgrid,
            medium=medium,
            source=deepcopy(source),
            sensor=sensor,
            backend="python",
            device="cpu",
            data_cast="quad",  # not a valid option
        )


def test_python_single_vs_double_numerical_agreement():
    """Single and double precision runs must agree to within float32 tolerance.

    Sanity check that data_cast='single' isn't producing garbage. Float32 has
    ~7 decimal digits, so a relative tolerance of ~1e-5 is reasonable for a
    short propagation in a uniform medium.
    """
    kgrid, medium, source, sensor = _build_minimal()
    p_double = np.asarray(
        kspaceFirstOrder(
            kgrid=kgrid,
            medium=medium,
            source=deepcopy(source),
            sensor=sensor,
            backend="python",
            device="cpu",
            data_cast="off",
            pml_inside=False,
            smooth_p0=False,
            quiet=True,
        )["p"]
    )
    p_single = np.asarray(
        kspaceFirstOrder(
            kgrid=kgrid,
            medium=medium,
            source=deepcopy(source),
            sensor=sensor,
            backend="python",
            device="cpu",
            data_cast="single",
            pml_inside=False,
            smooth_p0=False,
            quiet=True,
        )["p"]
    )
    # Compare relative error on points with non-trivial signal
    scale = np.max(np.abs(p_double))
    abs_err = np.max(np.abs(p_double.astype(np.float64) - p_single.astype(np.float64)))
    assert abs_err / scale < 1e-4, f"single-vs-double max relative error {abs_err / scale:.2e} > 1e-4"


def test_cpp_backend_warns_on_non_off_data_cast(tmp_path):
    """Setting data_cast='single' with backend='cpp' must warn (binary uses fixed precision)."""
    kgrid, medium, source, sensor = _build_minimal()
    import subprocess

    with pytest.warns(UserWarning, match="data_cast.*has no effect.*cpp"):
        try:
            kspaceFirstOrder(
                kgrid=kgrid,
                medium=medium,
                source=deepcopy(source),
                sensor=sensor,
                backend="cpp",
                device="cpu",
                data_cast="single",
                pml_inside=False,
                smooth_p0=False,
                pml_alpha=10,
                data_path=str(tmp_path),
                quiet=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Binary not available on this box; warning still must have fired
            pass


def test_cpp_backend_silent_on_default_data_cast(tmp_path, recwarn):
    """data_cast='off' (default) must not warn on the C++ backend."""
    kgrid, medium, source, sensor = _build_minimal()
    import subprocess

    try:
        kspaceFirstOrder(
            kgrid=kgrid,
            medium=medium,
            source=deepcopy(source),
            sensor=sensor,
            backend="cpp",
            device="cpu",
            # data_cast left at default 'off'
            pml_inside=False,
            smooth_p0=False,
            pml_alpha=10,
            data_path=str(tmp_path),
            quiet=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    assert not any("data_cast" in str(w.message) for w in recwarn.list)
