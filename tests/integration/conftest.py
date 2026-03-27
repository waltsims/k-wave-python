"""Shared fixtures for MATLAB-reference integration tests."""
from pathlib import Path

import numpy as np
import pytest

# Where CI unpacks the MATLAB-generated .mat files
_COLLECTED_VALUES_DIR = Path(__file__).resolve().parents[1] / "matlab_test_data_collectors" / "python_testers" / "collectedValues"


@pytest.fixture(scope="session")
def matlab_refs_dir():
    """Path to collectedValues/. Skips if not present (local dev without MATLAB)."""
    if not _COLLECTED_VALUES_DIR.is_dir():
        pytest.skip("MATLAB reference data not available (run CI or generate locally)")
    return _COLLECTED_VALUES_DIR


@pytest.fixture(scope="session")
def load_matlab_ref(matlab_refs_dir):
    """Return a loader function: load_matlab_ref('example_ivp_2D') -> dict."""
    from scipy.io import loadmat

    def _load(name):
        mat_file = matlab_refs_dir / f"{name}.mat"
        if not mat_file.exists():
            pytest.skip(f"Reference file {mat_file.name} not found")
        raw = loadmat(str(mat_file), simplify_cells=True)
        # TestRecorder prefixes keys with "step_0___" — strip it
        prefix = "step_0___"
        return {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}

    return _load


def _to_matlab_shape(py_val, mat_val):
    """Reshape C-order Python output to match MATLAB F-order reference shape.

    The new API returns full-grid time-series as (Nt, *grid_shape) in C-order.
    MATLAB references store them as (n_sensor, Nt) in F-flat order.
    Similarly, aggregates are (*grid_shape) vs MATLAB (n_sensor,).
    """
    if py_val.shape == mat_val.shape:
        return py_val

    # Time-series: (Nt, *grid_shape) → (n_sensor, Nt) with F-order flatten
    if py_val.ndim >= 3 and mat_val.ndim == 2:
        Nt = py_val.shape[0]
        # Move time axis last, then F-order flatten the grid dims
        return np.moveaxis(py_val, 0, -1).reshape(-1, Nt, order="F")

    # Aggregates: (*grid_shape) → (n_sensor,) with F-order flatten
    if py_val.ndim >= 2 and mat_val.ndim == 1:
        return py_val.ravel(order="F")

    return py_val


def assert_fields_close(result, ref, fields, *, rtol=1e-10, atol=1e-12):
    """Compare Python result dict against MATLAB reference arrays.

    Args:
        result: dict from kspaceFirstOrder()
        ref: dict from scipy.io.loadmat()
        fields: list of (python_key, matlab_key) tuples
        rtol, atol: tolerances passed to np.testing.assert_allclose
    """
    for py_key, mat_key in fields:
        assert py_key in result, f"Python result missing key '{py_key}'"
        assert mat_key in ref, f"MATLAB reference missing key '{mat_key}'"
        py_val = np.atleast_1d(np.squeeze(np.asarray(result[py_key])))
        mat_val = np.atleast_1d(np.squeeze(np.asarray(ref[mat_key])))
        py_val = _to_matlab_shape(py_val, mat_val)
        assert py_val.shape == mat_val.shape, f"Shape mismatch for {py_key}: Python {py_val.shape} vs MATLAB {mat_val.shape}"
        np.testing.assert_allclose(
            py_val, mat_val, rtol=rtol, atol=atol, err_msg=f"Field '{py_key}' differs from MATLAB reference '{mat_key}'"
        )
