from typing import List

import numpy as np
from scipy.interpolate import interpn


def sortrows(arr: np.ndarray, index: int):
    assert arr.ndim == 2, "'sortrows' currently supports only 2-dimensional matrices"
    return arr[arr[:, index].argsort(), ]


def interpolate3D(grid_points: List[np.ndarray], grid_values: np.ndarray, interp_locs: List[np.ndarray]) -> np.ndarray:
    """
        Interpolates input grid values at the given locations
        Added by Farid

        Matlab version of this function assumes unstructured grid. Interpolating such grid in Python using
        SciPy is very expensive. Thankfully, working with structured grid is fine for our purposes.
        We still support 3D arguments for backward compatibility even though they are mapped to 1D grid.
        While mapping we assume that only one axis per 3D grid changes throughout the grid.
    Args:
        grid_points: List of 1D or 3D Numpy arrays
        grid_values: A 3D Numpy array which holds values at grid_points
        interp_locs: List of 1D or 3D Numpy arrays
    Returns:

    """
    assert len(grid_points) == 3, 'interpolate3D supports only 3D interpolation'
    assert len(grid_points) == len(interp_locs)

    def unpack_and_make_1D(pts):
        pts_x, pts_y, pts_z = pts
        if pts_x.ndim == 3:
            pts_x = pts_x[:, 0, 0]
        if pts_y.ndim == 3:
            pts_y = pts_y[0, :, 0]
        if pts_z.ndim == 3:
            pts_z = pts_z[0, 0, :]
        return pts_x, pts_y, pts_z

    g_x, g_y, g_z = unpack_and_make_1D(grid_points)
    q_x, q_y, q_z = unpack_and_make_1D(interp_locs)

    # 'ij' indexing is crucial for Matlab compatibility
    queries = np.array(np.meshgrid(q_x, q_y, q_z, indexing='ij'))
    # Queries are just a list of 3D points
    queries = queries.reshape(3, -1).T

    # Out of bound points will get NaN values
    result = interpn((g_x, g_y, g_z), grid_values, queries, method='linear', bounds_error=False, fill_value=np.nan)
    # Go back from list of interpolated values to 3D volume
    result = result.reshape((g_x.size, g_y.size, g_z.size))
    # set values outside of the interpolation range to original values
    result[np.isnan(result)] = grid_values[np.isnan(result)]
    return result


def interpolate2D(grid_points: List[np.ndarray], grid_values: np.ndarray, interp_locs: List[np.ndarray], copy_nans=True) -> np.ndarray:
    """
        Interpolates input grid values at the given locations
        Added by Farid

        Matlab version of this function assumes unstructured grid. Interpolating such grid in Python using
        SciPy is very expensive. Thankfully, working with structured grid is fine for our purposes.
        We still support 3D arguments for backward compatibility even though they are mapped to 1D grid.
        While mapping we assume that only one axis per 3D grid changes throughout the grid.
    Args:
        grid_points: List of 1D or 3D Numpy arrays
        grid_values: A 3D Numpy array which holds values at grid_points
        interp_locs: List of 1D or 3D Numpy arrays
    Returns:

    """
    assert len(grid_points) == 2, 'interpolate2D supports only 2D interpolation'
    assert len(grid_points) == len(interp_locs)

    def unpack_and_make_1D(pts):
        pts_x, pts_y = pts
        if pts_x.ndim == 2:
            pts_x = pts_x[:, 0]
        if pts_y.ndim == 2:
            pts_y = pts_y[0, :]
        return pts_x, pts_y

    g_x, g_y = unpack_and_make_1D(grid_points)
    q_x, q_y = unpack_and_make_1D(interp_locs)

    # 'ij' indexing is crucial for Matlab compatibility
    queries = np.array(np.meshgrid(q_x, q_y, indexing='ij'))
    # Queries are just a list of 3D points
    queries = queries.reshape(2, -1).T

    # Out of bound points will get NaN values
    result = interpn((g_x, g_y), grid_values, queries, method='linear', bounds_error=False, fill_value=np.nan)
    # Go back from list of interpolated values to 3D volume
    result = result.reshape((q_x.size, q_y.size))
    if copy_nans:
        assert result.shape == grid_values.shape
        # set values outside of the interpolation range to original values
        result[np.isnan(result)] = grid_values[np.isnan(result)]
    return result


def cart2grid(kgrid, cart_data, axisymmetric=False):
    """
        Interpolate a set of Cartesian points onto a binary grid.
    Args:
        kgrid:
        cart_data:
        axisymmetric:

    Returns:
        %     cart2grid interpolates the set of Cartesian points defined by
        %     cart_data onto a binary matrix defined by the kWaveGrid object
        %     kgrid using nearest neighbour interpolation. An error is returned if
        %     the Cartesian points are outside the computational domain defined by
        %     kgrid.
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
        data_x = data_x + np.floor(kgrid.Nx//2).astype(int)

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
        data_x = data_x + np.floor(kgrid.Nx//2).astype(int)
        if not axisymmetric:
            data_y = data_y + np.floor(kgrid.Ny//2).astype(int)
        else:
            data_y = data_y + 1

        # check if the points all lie within the grid
        if data_x.max() > kgrid.Nx or data_y.max() > kgrid.Ny or data_x.min() < 1 or data_y.min() < 1:
            raise AssertionError('Cartesian points must lie within the grid defined by kgrid.')

        # create empty grid
        grid_data = np.zeros((kgrid.Nx, kgrid.Ny))

        # create index variable
        point_index = np.arange(1, data_x.size + 1, dtype=int)

        # map values
        for data_index in range(data_x.size):
            grid_data[data_x[data_index], data_y[data_index]] = point_index[data_index]

        # extract reordering index
        reorder_index = np.reshape(grid_data[grid_data != 0], (-1, 1))

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
        data_x = data_x + np.floor(kgrid.Nx//2).astype(int)
        data_y = data_y + np.floor(kgrid.Ny//2).astype(int)
        data_z = data_z + np.floor(kgrid.Nz//2).astype(int)

        # check if the points all lie within the grid
        assert 1 <= data_x.min() and 1 <= data_y.min() and 1 <= data_z.min() and \
               data_x.max() <= kgrid.Nx and data_y.max() <= kgrid.Ny and data_z.max() <= kgrid.Ny, \
            "Cartesian points must lie within the grid defined by kgrid."

        # create empty grid
        grid_data = np.zeros((kgrid.Nx, kgrid.Ny, kgrid.Nz), dtype=int)

        # create index variable
        point_index = np.arange(1, data_x.size + 1)

        # map values
        for data_index in range(data_x.size):
            grid_data[data_x[data_index], data_y[data_index], data_z[data_index]] = point_index[data_index]

        # extract reordering index
        reorder_index = np.reshape(grid_data[grid_data != 0], (-1, 1, 1))
    else:
        raise ValueError('Input cart_data must be a 1, 2, or 3 dimensional matrix.')

    # compute the reverse ordering index (i.e., what is the index of each point
    # in the reordering vector)
    order_index = np.ones((reorder_index.size, 2))
    order_index[:, 0] = np.squeeze(reorder_index)
    order_index[:, 1] = np.arange(1, reorder_index.size + 1)
    order_index = sortrows(order_index, 1)
    order_index = order_index[:, 1]

    # reset binary grid values
    grid_data[grid_data != 0] = 1

    # check if any Cartesian points have been mapped to the same grid point,
    # thereby reducing the total number of points
    num_discarded_points = cart_data.shape[1] - np.sum(grid_data)
    if num_discarded_points != 0:
        print(f'  cart2grid: {num_discarded_points} Cartesian points mapped to overlapping grid points')
    return grid_data, order_index, reorder_index