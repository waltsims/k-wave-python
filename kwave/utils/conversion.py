import math
from typing import Tuple, Union, Any

import numpy as np
from numpy import ndarray

from kwave.kgrid import kWaveGrid
from kwave.utils.matrix import sort_rows


def db2neper(alpha: float, y: int = 1) -> float:
    """
    Convert decibels to nepers.

    Args:
        alpha: Attenuation in dB / (MHz ^ y cm).
        y: Power law exponent (default=1).

    Returns:
        Attenuation in Nepers / ((rad / s) ^ y m).
    """

    # calculate conversion
    alpha = 100 * alpha * (1e-6 / (2 * math.pi)) ** y / (20 * np.log10(np.exp(1)))
    return alpha


def neper2db(alpha: float, y: int = 1) -> float:
    """
    Converts an attenuation coefficient in units of Nepers / ((rad / s) ^ y m) to units of dB / (MHz ^ y cm).

    Args:
        alpha: Attenuation in Nepers / ((rad / s) ^ y m)
        y: Power law exponent (default=1)

    Returns:
        alpha (float): Attenuation in dB / (MHz ^ y cm)
    """

    # calculate conversion
    alpha = 20 * math.log10(math.exp(1)) * alpha * (2 * math.pi * 1e6) ** y / 100
    return alpha


def cast_to_type(data: np.ndarray, matlab_type: str) -> Any:
    """

    Args:
        data:
        matlab_type:

    Returns:

    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    type_map = {
        'single': np.float32,
        'double': np.float64,
        'uint64': np.uint64,
        'uint32': np.uint32,
        'uint16': np.uint16,
    }
    return data.astype(type_map[matlab_type])


def cart2pol(x: float, y: float) -> Tuple[float, float]:
    """
    Convert from cartesian to polar coordinates.

    Args:
    x: The x-coordinate of the point.
    y: The y-coordinate of the point.

    Returns:
    A tuple containing the polar coordinates of the point.
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return phi, rho


def grid2cart(input_kgrid: kWaveGrid, grid_selection: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Returns the Cartesian coordinates of the non-zero points of a binary grid.

    Args:
        input_kgrid: k-Wave grid object returned by kWaveGrid
        grid_selection: binary grid with the same dimensions as the k-Wave grid kgrid

    Returns:
        cart_data: 1 x N, 2 x N, or 3 x N (for 1, 2, and 3 dimensions) array of Cartesian sensor points
        order_index: returns a list of indices of the returned cart_data coordinates.
    Raises:
        ValueError: when input_kgrid.dim is not in [1, 2, 3]
    """
    grid_data = np.array((grid_selection != 0), dtype=bool)
    cart_data = np.zeros((input_kgrid.dim, np.sum(grid_data)))

    if input_kgrid.dim > 0:
        cart_data[0, :] = input_kgrid.x[grid_data]
    if input_kgrid.dim > 1:
        cart_data[1, :] = input_kgrid.y[grid_data]
    if input_kgrid.dim > 2:
        cart_data[2, :] = input_kgrid.z[grid_data]
    if 0 <= input_kgrid.dim > 3:
        raise ValueError("kGrid with unsupported size passed.")

    order_index = np.argwhere(grid_data.squeeze() != 0)
    return cart_data.squeeze(), order_index


def freq2wavenumber(n: int, k_max: float, filter_cutoff: float, c: float, k_dim: Union[int, Tuple[int]]) -> Tuple[
    int, float]:
    """
    Convert the given frequency and maximum wavenumber to a wavenumber cutoff and filter size.

    Args:
        n: The size of the grid.
        k_max: The maximum wavenumber.
        filter_cutoff: The frequency to convert to a wavenumber cutoff.
        c: The speed of sound.
        k_dim: The dimensions of the wavenumber grid.

    Returns:
        A tuple containing the calculated filter size and wavenumber cutoff.
    """
    k_cutoff = 2 * np.pi * filter_cutoff / c

    # set the alpha_filter size
    filter_size = round(n * k_cutoff / k_dim[-1])

    # check the alpha_filter size
    if filter_size > n:
        # set the alpha_filter size to be the same as the grid size
        filter_size = n
        filter_cutoff = k_max * c / (2 * np.pi)
    return filter_size, filter_cutoff


def cart2grid(kgrid: kWaveGrid, cart_data: ndarray, axisymmetric=False):
    """
    Interpolates the set of Cartesian points defined by
    cart_data onto a binary matrix defined by the kWaveGrid object
    kgrid using nearest neighbour interpolation. An error is returned if
    the Cartesian points are outside the computational domain defined by
    kgrid.

    Args:
        kgrid: k:
        cart_data(np.ndarray):
        axisymmetric(bool):

    Returns:
        grid(np.ndarray): binary grid
    """
    # check for axisymmetric input
    if axisymmetric and kgrid.dim != 2:
        raise AssertionError('Axisymmetric flag only supported in 2D.')

    # detect whether the inputs are for one, two, or three dimensions
    if kgrid.dim == 1:
        # one-dimensional
        data_x = cart_data[0, :]

        # scale position values to grid centered pixel coordinates using
        # nearest neighbour interpolation
        data_x = np.round(data_x / kgrid.dx).astype(int)

        # shift pixel coordinates to coincide with matrix indexing
        data_x = data_x + np.floor(kgrid.Nx // 2).astype(int)

        # check if the points all lie within the grid
        if data_x.max() > kgrid.Nx or data_x.min() < 1:
            raise AssertionError('Cartesian points must lie within the grid defined by kgrid.')

        # create empty grid
        grid_data = np.zeros((kgrid.Nx, 1))

        # create index variable
        point_index = np.arange(1, data_x.size + 1)

        # map values
        for data_index in range(data_x.size):
            grid_data[data_x[data_index]] = point_index[data_index]

        # extract reordering index
        reorder_index = np.reshape(grid_data[grid_data != 0], (-1, 1))

    elif kgrid.dim == 2:
        # two-dimensional
        data_x = cart_data[0, :]
        data_y = cart_data[1, :]

        # scale position values to grid centered pixel coordinates using
        # nearest neighbour interpolation
        data_x = np.round(data_x / kgrid.dx).astype(int)
        data_y = np.round(data_y / kgrid.dy).astype(int)

        # shift pixel coordinates to coincide with matrix indexing (leave
        # y-direction = radial-direction if axisymmetric)
        data_x = data_x + np.floor(kgrid.Nx // 2).astype(int)
        if not axisymmetric:
            data_y = data_y + np.floor(kgrid.Ny // 2).astype(int)
        else:
            data_y = data_y

        # check if the points all lie within the grid
        if data_x.max() >= kgrid.Nx or data_y.max() >= kgrid.Ny or data_x.min() < 0 or data_y.min() < 0:
            raise AssertionError('Cartesian points must lie within the grid defined by kgrid.')

        # create empty grid
        grid_data = -1 * np.ones((kgrid.Nx, kgrid.Ny))

        # map values
        for data_index in range(data_x.size):
            grid_data[data_x[data_index], data_y[data_index]] = int(data_index)

        # extract reordering index
        reorder_index = grid_data.flatten(order='F')[
            grid_data.flatten(order='F') != -1
            ]
        reorder_index = reorder_index[:, None] + 1  # [N] => [N, 1]

    elif kgrid.dim == 3:

        # three dimensional
        data_x = cart_data[0, :]
        data_y = cart_data[1, :]
        data_z = cart_data[2, :]

        # scale position values to grid centered pixel coordinates using
        # nearest neighbour interpolation
        data_x = np.round(data_x / kgrid.dx).astype(int)
        data_y = np.round(data_y / kgrid.dy).astype(int)
        data_z = np.round(data_z / kgrid.dz).astype(int)

        # shift pixel coordinates to coincide with matrix indexing
        data_x = data_x + np.floor(kgrid.Nx // 2).astype(int)
        data_y = data_y + np.floor(kgrid.Ny // 2).astype(int)
        data_z = data_z + np.floor(kgrid.Nz // 2).astype(int)

        # check if the points all lie within the grid
        assert 0 <= data_x.min() and 0 <= data_y.min() and 0 <= data_z.min() and \
               data_x.max() < kgrid.Nx and data_y.max() < kgrid.Ny and data_z.max() < kgrid.Nz, \
            "Cartesian points must lie within the grid defined by kgrid."

        # create empty grid
        grid_data = -1 * np.ones((kgrid.Nx, kgrid.Ny, kgrid.Nz), dtype=int)

        # create index variable
        point_index = np.arange(1, data_x.size + 1)

        # map values
        for data_index in range(data_x.size):
            grid_data[data_x[data_index], data_y[data_index], data_z[data_index]] = point_index[data_index]

        # extract reordering index
        reorder_index = grid_data.flatten(order='F')[
            grid_data.flatten(order='F') != -1
            ]
        reorder_index = reorder_index[:, None, None]  # [N] => [N, 1, 1]
    else:
        raise ValueError('Input cart_data must be a 1, 2, or 3 dimensional matrix.')

    # compute the reverse ordering index (i.e., what is the index of each point
    # in the reordering vector)
    order_index = np.ones((reorder_index.size, 2), dtype=int)
    order_index[:, 0] = np.squeeze(reorder_index)
    order_index[:, 1] = np.arange(1, reorder_index.size + 1)
    order_index = sort_rows(order_index, 0)
    order_index = order_index[:, 1]
    order_index = order_index[:, None]  # [N] => [N, 1]

    # reset binary grid values
    if kgrid.dim == 1:
        grid_data[grid_data != 0] = 1
    else:
        grid_data[grid_data != -1] = 1
        grid_data[grid_data == -1] = 0

    # check if any Cartesian points have been mapped to the same grid point,
    # thereby reducing the total number of points
    num_discarded_points = cart_data.shape[1] - np.sum(grid_data)
    if num_discarded_points != 0:
        print(f'  cart2grid: {num_discarded_points} Cartesian points mapped to overlapping grid points')
    return grid_data.astype(np.int), order_index, reorder_index
