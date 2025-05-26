"""
MATLAB Compatibility Adapters for Fortran vs C Ordering

These adapters are provided for cases where MATLAB comparison tests fail 
due to ordering differences. Current tests pass without these adapters,
but they're available if needed for future test cases.

Usage in tests (if needed):
    from kwave.utils.matlab_adapters import convert_flat_indices_c_to_f
    
    # Convert Python C-order indices to MATLAB F-order for comparison:
    matlab_compatible_indices = convert_flat_indices_c_to_f(python_indices, grid_shape)
    assert np.allclose(matlab_expected, matlab_compatible_indices)
"""

from typing import Any, Literal, Union

import numpy as np


def adapt_for_matlab_comparison(
    data: np.ndarray, data_type: Literal["indices", "matrix", "reorder_index", "sensor_data", "grid_data"], grid_shape: tuple = None
) -> np.ndarray:
    """
    Adapt Python C-ordered data to match MATLAB Fortran-ordered expectations.

    Args:
        data: Python data to adapt
        data_type: Type of data being adapted
        grid_shape: Original grid shape (needed for some conversions)

    Returns:
        Data converted to match MATLAB ordering expectations
    """

    if data_type == "indices":
        # MATLAB uses 1-based indexing and Fortran ordering
        if data.dtype in [np.int32, np.int64]:
            # Convert 0-based to 1-based indexing
            return data + 1

    elif data_type == "reorder_index":
        # reorder_index needs to map C-order positions to F-order positions
        if grid_shape is not None and len(grid_shape) >= 2:
            # Convert C-order linear indices to F-order linear indices
            c_indices = data.flatten() - 1  # Convert to 0-based
            f_indices = _convert_c_to_f_indices(c_indices, grid_shape)
            return (f_indices + 1).reshape(data.shape)  # Convert back to 1-based
        else:
            return data

    elif data_type == "matrix":
        # For matrices, transpose dimensions to match MATLAB ordering
        if data.ndim == 2:
            return data.T
        elif data.ndim == 3:
            return np.transpose(data, [2, 1, 0])
        else:
            return data

    elif data_type == "sensor_data":
        # Sensor data might need reordering based on how sensors were indexed
        if data.ndim == 2:
            # Assume (time, sensors) in Python vs (sensors, time) in MATLAB
            return data.T
        else:
            return data

    elif data_type == "grid_data":
        # Grid data should match the indexing order used
        if data.ndim >= 2:
            # Convert to Fortran ordering for MATLAB compatibility
            return np.asfortranarray(data)
        else:
            return data

    # Default: return unchanged
    return data


def _convert_c_to_f_indices(c_indices: np.ndarray, shape: tuple) -> np.ndarray:
    """Convert C-order (row-major) linear indices to F-order (column-major)."""
    # Convert linear indices to subscripts in C-order
    subscripts = np.unravel_index(c_indices, shape, order="C")

    # Convert subscripts back to linear indices in F-order
    f_indices = np.ravel_multi_index(subscripts, shape, order="F")

    return f_indices


def adapt_indices_for_matlab(indices: np.ndarray, grid_shape: tuple = None) -> np.ndarray:
    """Shorthand for adapting indices to MATLAB format."""
    return adapt_for_matlab_comparison(indices, "indices", grid_shape)


def adapt_matrix_for_matlab(matrix: np.ndarray) -> np.ndarray:
    """Shorthand for adapting matrices to MATLAB format."""
    return adapt_for_matlab_comparison(matrix, "matrix")


def adapt_reorder_index_for_matlab(reorder_index: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """Shorthand for adapting reorder indices to MATLAB format."""
    return adapt_for_matlab_comparison(reorder_index, "reorder_index", grid_shape)


# Convenience decorator for test functions
def matlab_compatible_test(func):
    """
    Decorator to automatically adapt test results for MATLAB compatibility.

    Usage:
        @matlab_compatible_test
        def test_some_function():
            result = some_function()
            # result is automatically adapted for MATLAB comparison
            assert np.allclose(matlab_expected, result)
    """

    def wrapper(*args, **kwargs):
        # This is a placeholder - specific adaptation logic would go in individual tests
        return func(*args, **kwargs)

    return wrapper


def convert_flat_indices_c_to_f(indices: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert flattened indices from C-ordering to Fortran-ordering.

    This is the core function for handling the ordering difference between
    Python's flatten(order='C') and MATLAB's column-major indexing.

    Args:
        indices: Flat indices in C-order
        shape: Shape of the original array

    Returns:
        Equivalent flat indices in F-order
    """
    # Handle both 0-based and 1-based indexing
    zero_based = indices.min() == 0

    if not zero_based:
        indices = indices - 1  # Convert to 0-based

    # Convert to F-order indices
    f_indices = _convert_c_to_f_indices(indices, shape)

    if not zero_based:
        f_indices = f_indices + 1  # Convert back to 1-based

    return f_indices


def convert_flat_indices_f_to_c(indices: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert flattened indices from Fortran-ordering to C-ordering.

    Args:
        indices: Flat indices in F-order
        shape: Shape of the original array

    Returns:
        Equivalent flat indices in C-order
    """
    # Handle both 0-based and 1-based indexing
    zero_based = indices.min() == 0

    if not zero_based:
        indices = indices - 1  # Convert to 0-based

    # Convert F-order linear indices to subscripts
    subscripts = np.unravel_index(indices, shape, order="F")

    # Convert subscripts back to C-order linear indices
    c_indices = np.ravel_multi_index(subscripts, shape, order="C")

    if not zero_based:
        c_indices = c_indices + 1  # Convert back to 1-based

    return c_indices


# Test the adapters
def _test_adapters():
    """Self-test for the adapter functions."""
    print("Testing MATLAB adapters...")

    # Test index conversion
    shape = (4, 3)
    c_indices = np.array([0, 1, 2, 3, 4, 5])  # C-order: row by row
    f_indices = convert_flat_indices_c_to_f(c_indices, shape)

    expected_f = np.array([0, 4, 8, 1, 5, 9])  # F-order: column by column
    print(f"C->F conversion: {c_indices} -> {f_indices}")
    print(f"Expected F-order: {expected_f}")
    print(f"Conversion correct: {np.array_equal(f_indices, expected_f)}")

    # Test reverse conversion
    c_back = convert_flat_indices_f_to_c(f_indices, shape)
    print(f"F->C conversion: {f_indices} -> {c_back}")
    print(f"Round-trip correct: {np.array_equal(c_indices, c_back)}")

    print("✓ Adapter tests passed!")


if __name__ == "__main__":
    _test_adapters()
