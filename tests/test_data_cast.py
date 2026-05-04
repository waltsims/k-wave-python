"""Tests for the ``dtype`` parameter on the modern ``kspaceFirstOrder()`` API.

Issue #695: expose precision control on the modern API.

The Python backend honors ``dtype`` (numpy-style dtype-like input):
  - ``None`` / ``np.float64`` / ``"float64"`` / ``"double"`` / ``float`` /
    ``"off"`` (legacy MATLAB alias) → float64 (default)
  - ``np.float32`` / ``"float32"`` / ``"single"`` → float32

The C++ backend uses fixed internal precision regardless; setting ``dtype``
to anything other than float64 warns.
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


# Cover every input form the resolver should accept.
_FLOAT64_INPUTS = [None, np.float64, "float64", "double", float, "off", np.dtype("f8")]
_FLOAT32_INPUTS = [np.float32, "float32", "single", np.dtype("f4")]


@pytest.mark.parametrize("dtype_arg", _FLOAT64_INPUTS, ids=lambda x: repr(x))
def test_python_backend_float64_inputs(dtype_arg):
    """All float64-equivalent inputs must produce float64 output across every recorded field."""
    kgrid, medium, source, sensor = _build_minimal()
    sensor.record = ("p", "p_final", "p_max", "p_min", "p_rms")
    result = kspaceFirstOrder(
        kgrid=kgrid,
        medium=medium,
        source=deepcopy(source),
        sensor=sensor,
        backend="python",
        device="cpu",
        dtype=dtype_arg,
        pml_inside=False,
        smooth_p0=False,
        quiet=True,
    )
    for field in ("p", "p_final", "p_max", "p_min", "p_rms"):
        arr = np.asarray(result[field])
        assert arr.dtype == np.float64, f"dtype={dtype_arg!r}: {field} produced {arr.dtype}, expected float64"
    assert not np.any(np.isnan(result["p"]))
    assert np.nanmax(np.abs(result["p"])) > 0


@pytest.mark.parametrize("dtype_arg", _FLOAT32_INPUTS, ids=lambda x: repr(x))
def test_python_backend_float32_inputs(dtype_arg):
    """All float32-equivalent inputs must produce float32 output across every recorded field.

    Includes p_final + aggregates (p_max/min/rms) — these caught a real dtype-drift bug
    where k-space operators (kappa, op_grad/div_list) and the PML arrays were float64
    by default and silently upcast self.p back to float64 mid-step().
    """
    kgrid, medium, source, sensor = _build_minimal()
    sensor.record = ("p", "p_final", "p_max", "p_min", "p_rms")
    result = kspaceFirstOrder(
        kgrid=kgrid,
        medium=medium,
        source=deepcopy(source),
        sensor=sensor,
        backend="python",
        device="cpu",
        dtype=dtype_arg,
        pml_inside=False,
        smooth_p0=False,
        quiet=True,
    )
    for field in ("p", "p_final", "p_max", "p_min", "p_rms"):
        arr = np.asarray(result[field])
        assert arr.dtype == np.float32, f"dtype={dtype_arg!r}: {field} produced {arr.dtype}, expected float32"
    assert not np.any(np.isnan(result["p"]))
    assert np.nanmax(np.abs(result["p"])) > 0


def test_default_dtype_is_float64():
    """Calling without dtype must produce float64 (back-compat with pre-#716 behavior)."""
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


@pytest.mark.parametrize("bad", [np.float16, np.complex64, "float16", "complex64", "quad", 42, "garbage"])
def test_invalid_dtype_raises(bad):
    """Non-float64/float32 dtypes (and unparseable values) must raise ValueError."""
    kgrid, medium, source, sensor = _build_minimal()
    with pytest.raises(ValueError, match="dtype"):
        kspaceFirstOrder(
            kgrid=kgrid,
            medium=medium,
            source=deepcopy(source),
            sensor=sensor,
            backend="python",
            device="cpu",
            dtype=bad,
        )


def test_python_backend_dtype_preserved_with_nonlinearity():
    """BonA path must preserve float32 dtype.

    Regression guard for the third dtype-drift bug: ``sum(rho_split)`` in
    ``_nl_factor`` and the equation-of-state line starts with Python ``int 0``,
    which under numpy < 2 (NEP 50) promotes float32 to float64.  Fixed by
    using ``_array_sum`` which starts from the first element.
    """
    kgrid, medium, source, sensor = _build_minimal()
    medium.BonA = 6.0  # water-like nonlinearity
    sensor.record = ("p", "p_final", "p_max")
    result = kspaceFirstOrder(
        kgrid=kgrid,
        medium=medium,
        source=deepcopy(source),
        sensor=sensor,
        backend="python",
        device="cpu",
        dtype=np.float32,
        pml_inside=False,
        smooth_p0=False,
        quiet=True,
    )
    for field in ("p", "p_final", "p_max"):
        arr = np.asarray(result[field])
        assert arr.dtype == np.float32, f"With nonlinearity, {field} produced {arr.dtype}, expected float32"


def test_torch_like_dtype_gets_helpful_error():
    """Non-numpy framework dtypes get a hint pointing to numpy equivalents.

    Uses a fake stand-in (`__module__ = "torch"`) to avoid importing torch.
    """
    kgrid, medium, source, sensor = _build_minimal()

    class FakeTorchDtype:
        pass

    FakeTorchDtype.__module__ = "torch"
    with pytest.raises(ValueError, match=r"torch.*pass the equivalent numpy dtype"):
        kspaceFirstOrder(
            kgrid=kgrid,
            medium=medium,
            source=deepcopy(source),
            sensor=sensor,
            backend="python",
            device="cpu",
            dtype=FakeTorchDtype(),
        )


def test_python_single_vs_double_numerical_agreement():
    """Single and double precision runs must agree to within float32 tolerance.

    Sanity check that float32 runs aren't producing garbage. Float32 has
    ~7 decimal digits, so a relative tolerance of ~1e-4 is reasonable for
    a short propagation in a uniform medium.
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
            dtype=np.float64,
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
            dtype=np.float32,
            pml_inside=False,
            smooth_p0=False,
            quiet=True,
        )["p"]
    )
    scale = np.max(np.abs(p_double))
    abs_err = np.max(np.abs(p_double.astype(np.float64) - p_single.astype(np.float64)))
    assert abs_err / scale < 1e-4, f"single-vs-double max relative error {abs_err / scale:.2e} > 1e-4"


def test_cpp_backend_warns_on_non_float64_dtype(tmp_path):
    """Setting dtype=np.float32 with backend='cpp' must warn (binary uses fixed precision)."""
    kgrid, medium, source, sensor = _build_minimal()
    import subprocess

    with pytest.warns(UserWarning, match="dtype.*has no effect.*cpp"):
        try:
            kspaceFirstOrder(
                kgrid=kgrid,
                medium=medium,
                source=deepcopy(source),
                sensor=sensor,
                backend="cpp",
                device="cpu",
                dtype=np.float32,
                pml_inside=False,
                smooth_p0=False,
                pml_alpha=10,
                data_path=str(tmp_path),
                quiet=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Binary not available on this box; warning still must have fired
            pass


def test_cpp_backend_silent_on_default_dtype(tmp_path, recwarn):
    """Default dtype (None → float64) must not warn on the C++ backend."""
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
            # dtype left at default
            pml_inside=False,
            smooth_p0=False,
            pml_alpha=10,
            data_path=str(tmp_path),
            quiet=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    assert not any("dtype" in str(w.message) and "has no effect" in str(w.message) for w in recwarn.list)
