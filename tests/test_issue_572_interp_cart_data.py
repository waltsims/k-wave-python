"""Regression tests for issue #572: interp_cart_data with interp="linear" returned wrong data."""

import numpy as np
import pytest

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.utils.conversion import grid2cart
from kwave.utils.interp import interp_cart_data

TIME_PROFILE = np.array([1.0, -2.0, 0.5, 3.0, 0.0])


def _line_setup():
    """Cartesian points on the line y = 0 carrying data that varies linearly with x."""
    kgrid = kWaveGrid(Vector([8, 8]), Vector([1.0, 1.0]))

    cart_x = np.arange(-3.25, 3.0, 1.0)
    cart_sensor_mask = np.vstack([cart_x, np.zeros_like(cart_x)])
    # reverse the point order so that the data rows are not already sorted by position
    cart_sensor_mask = cart_sensor_mask[:, ::-1]
    cart_sensor_data = np.outer(2.0 * cart_sensor_mask[0] + 1.0, TIME_PROFILE)

    # binary sensor points on the same line (y = 0 is index 4), listed in C order
    binary_sensor_mask = np.zeros((8, 8), dtype=bool)
    binary_sensor_mask[[1, 3, 6], 4] = True
    binary_x = kgrid.x_vec.ravel()[[1, 3, 6]]

    return kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, binary_x


def test_interp_cart_data_linear_weights_two_closest_points_by_distance():
    kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, binary_x = _line_setup()

    result = interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, interp="linear")

    # linear interpolation between the two neighbours reproduces a linear function exactly
    expected = np.outer(2.0 * binary_x + 1.0, TIME_PROFILE)
    assert result.shape == (3, TIME_PROFILE.size)
    assert np.allclose(result, expected)


def test_interp_cart_data_linear_does_not_modify_input():
    kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, _ = _line_setup()
    original = cart_sensor_data.copy()

    interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, interp="linear")

    assert np.array_equal(cart_sensor_data, original)


def test_interp_cart_data_nearest_uses_closest_point():
    kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, binary_x = _line_setup()

    result = interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, interp="nearest")

    # the closest Cartesian point lies 0.25 below each binary point
    expected = np.outer(2.0 * (binary_x - 0.25) + 1.0, TIME_PROFILE)
    assert np.allclose(result, expected)


@pytest.mark.parametrize("interp", ["nearest", "linear"])
def test_interp_cart_data_rows_follow_grid2cart_order(interp):
    """Coincident Cartesian and binary points: each output row is the data of its own grid point."""
    kgrid = kWaveGrid(Vector([6, 5]), Vector([1.0, 1.0]))
    binary_sensor_mask = np.zeros((6, 5), dtype=bool)
    binary_sensor_mask[1, 3] = binary_sensor_mask[2, 0] = binary_sensor_mask[4, 2] = binary_sensor_mask[4, 4] = True

    cart_bsm, _ = grid2cart(kgrid, binary_sensor_mask)
    # Cartesian sensor points at the same positions but in a different order
    permutation = np.array([2, 0, 3, 1])
    cart_sensor_mask = cart_bsm[:, permutation]
    cart_sensor_data = np.outer(10.0 * cart_sensor_mask[0] + cart_sensor_mask[1], TIME_PROFILE)

    result = interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, interp=interp)

    assert np.allclose(result, np.outer(10.0 * cart_bsm[0] + cart_bsm[1], TIME_PROFILE))


def test_interp_cart_data_linear_3d():
    kgrid = kWaveGrid(Vector([6, 6, 6]), Vector([1.0, 1.0, 1.0]))
    cart_z = np.arange(-2.75, 3.0, 1.0)
    cart_sensor_mask = np.vstack([np.zeros_like(cart_z), np.zeros_like(cart_z), cart_z])
    cart_sensor_data = np.outer(3.0 * cart_z - 1.0, TIME_PROFILE)

    binary_sensor_mask = np.zeros((6, 6, 6), dtype=bool)
    binary_sensor_mask[3, 3, [1, 4]] = True
    binary_z = kgrid.z_vec.ravel()[[1, 4]]

    result = interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, interp="linear")

    assert np.allclose(result, np.outer(3.0 * binary_z - 1.0, TIME_PROFILE))
