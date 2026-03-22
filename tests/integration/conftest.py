"""Shared fixtures for MATLAB-reference integration tests."""
import os
from pathlib import Path

import numpy as np
import pytest

# Where CI unpacks the MATLAB-generated .mat files
_COLLECTED_VALUES_DIR = Path(__file__).resolve().parents[1] / "matlab_test_data_collectors" / "python_testers" / "collectedValues"


@pytest.fixture
def matlab_refs_dir():
    """Path to collectedValues/. Skips if not present (local dev without MATLAB)."""
    if not _COLLECTED_VALUES_DIR.is_dir():
        pytest.skip("MATLAB reference data not available (run CI or generate locally)")
    return _COLLECTED_VALUES_DIR


@pytest.fixture
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
        assert py_val.shape == mat_val.shape, f"Shape mismatch for {py_key}: Python {py_val.shape} vs MATLAB {mat_val.shape}"
        np.testing.assert_allclose(
            py_val, mat_val, rtol=rtol, atol=atol, err_msg=f"Field '{py_key}' differs from MATLAB reference '{mat_key}'"
        )
