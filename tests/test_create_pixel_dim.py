import numpy as np
import pytest

from kwave.utils.mapgen import create_pixel_dim


def test_even_single_shift0():
    Nx = 4
    origin_size = "single"
    shift = 0

    result = create_pixel_dim(Nx, origin_size, shift)
    expected = np.array([-1, 0, 1, 2])  # from the code path
    np.testing.assert_array_equal(result, expected)


def test_even_single_shift1():
    Nx = 4
    origin_size = "single"
    shift = 1

    result = create_pixel_dim(Nx, origin_size, shift)
    expected = np.array([-2, -1, 0, 1])  # from code path
    np.testing.assert_array_equal(result, expected)


def test_even_double_shift_any():
    Nx = 4
    origin_size = "double"
    shift = 999  # any shift, code doesn't branch on shift for Nx even

    result = create_pixel_dim(Nx, origin_size, shift)
    expected = np.array([-1, 0, 0, 1])  # based on code as-is
    np.testing.assert_array_equal(result, expected)


def test_odd_single():
    Nx = 5
    origin_size = "single"
    shift = 0  # or shift=1 doesn't matter for this branch

    result = create_pixel_dim(Nx, origin_size, shift)
    expected = np.array([-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(result, expected)


def test_odd_double_shift0():
    Nx = 5
    origin_size = "double"
    shift = 0

    result = create_pixel_dim(Nx, origin_size, shift)
    expected = np.array([-1, 0, 0, 1, 2])  # consistent with code
    np.testing.assert_array_equal(result, expected)


def test_odd_double_shift1():
    Nx = 5
    origin_size = "double"
    shift = 1

    result = create_pixel_dim(Nx, origin_size, shift)
    expected = np.array([-2, -1, 0, 0, 1])
    np.testing.assert_array_equal(result, expected)
