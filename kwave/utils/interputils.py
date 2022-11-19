from typing import List

import numpy as np
from numpy.fft import fft, fftshift
from scipy.interpolate import interpn
from scipy.signal import resample
from kwave.utils.tictoc import TicToc
from kwave.utils.checkutils import num_dim
from kwave.utils.conversionutils import scale_time


def sortrows(arr: np.ndarray, index: int):
    assert arr.ndim == 2, "'sortrows' currently supports only 2-dimensional matrices"
    return arr[arr[:, index].argsort(),]


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


def interpolate2D(grid_points: List[np.ndarray], grid_values: np.ndarray, interp_locs: List[np.ndarray],
                  method='linear', copy_nans=True) -> np.ndarray:
    """
        Interpolates input grid values at the given locations
        Added by Farid

        Matlab version of this function assumes unstructured grid. Interpolating such grid in Python using
        SciPy is very expensive. Thankfully, working with structured grid is fine for our purposes.
        We still support 3D arguments for backward compatibility even though they are mapped to 1D grid.
        While mapping we assume that only one axis per 3D grid changes throughout the grid.
    Args:
        copy_nans:
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
    result = interpn((g_x, g_y), grid_values, queries, method=method, bounds_error=False, fill_value=np.nan)
    # Go back from list of interpolated values to 3D volume
    result = result.reshape((q_x.size, q_y.size))
    if copy_nans:
        assert result.shape == grid_values.shape
        # set values outside of the interpolation range to original values
        result[np.isnan(result)] = grid_values[np.isnan(result)]
    return result


def interpolate2D_with_queries(
        grid_points: List[np.ndarray],
        grid_values: np.ndarray,
        queries: np.ndarray,
        method='linear',
        copy_nans=True
) -> np.ndarray:
    """
        Interpolates input grid values at the given locations
        Added by Farid

        Simplified version of `interpolate2D_coords`.
        Expects `interp_locs` to be [N, 2] coordinates of the interpolation locations.
        Does not create meshgrid on the `interp_locs` as `interpolate2D_coords`!
        WARNING: supposed to support only 2D interpolation!
    Args:
        copy_nans:
        grid_points: List of 1D or 3D Numpy arrays
        grid_values: A 3D Numpy array which holds values at grid_points
        queries: Numpy array with shape [N, 2]
    Returns:

    """
    assert len(grid_points) == 2, 'interpolate2D supports only 2D interpolation'

    g_x, g_y = grid_points

    assert g_x.ndim == 1  # is a list
    assert g_y.ndim == 1  # is a list
    assert queries.ndim == 2 and queries.shape[1] == 2

    # Out of bound points will get NaN values
    result = interpn((g_x, g_y), grid_values, queries, method=method, bounds_error=False, fill_value=np.nan)
    if copy_nans:
        assert result.shape == grid_values.shape
        # set values outside the interpolation range to original values
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
        cart2grid interpolates the set of Cartesian points defined by
        cart_data onto a binary matrix defined by the kWaveGrid object
        kgrid using nearest neighbour interpolation. An error is returned if
        the Cartesian points are outside the computational domain defined by
        kgrid.
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
        reorder_index = grid_data.flatten(order='F')[
            grid_data.flatten(order='F') != 0
            ]
        reorder_index = reorder_index[:, None]  # [N] => [N, 1]

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
        assert 1 <= data_x.min() and 1 <= data_y.min() and 1 <= data_z.min() and \
               data_x.max() <= kgrid.Nx and data_y.max() <= kgrid.Ny and data_z.max() <= kgrid.Nz, \
            "Cartesian points must lie within the grid defined by kgrid."

        # create empty grid
        grid_data = np.zeros((kgrid.Nx, kgrid.Ny, kgrid.Nz), dtype=int)

        # create index variable
        point_index = np.arange(1, data_x.size + 1)

        # map values
        for data_index in range(data_x.size):
            grid_data[data_x[data_index], data_y[data_index], data_z[data_index]] = point_index[data_index]

        # extract reordering index
        reorder_index = grid_data.flatten(order='F')[
            grid_data.flatten(order='F') != 0
            ]
        reorder_index = reorder_index[:, None, None]  # [N] => [N, 1, 1]
    else:
        raise ValueError('Input cart_data must be a 1, 2, or 3 dimensional matrix.')

    # compute the reverse ordering index (i.e., what is the index of each point
    # in the reordering vector)
    order_index = np.ones((reorder_index.size, 2), dtype=int)
    order_index[:, 0] = np.squeeze(reorder_index)
    order_index[:, 1] = np.arange(1, reorder_index.size + 1)
    order_index = sortrows(order_index, 0)
    order_index = order_index[:, 1]
    order_index = order_index[:, None]  # [N] => [N, 1]

    # reset binary grid values
    grid_data[grid_data != 0] = 1

    # check if any Cartesian points have been mapped to the same grid point,
    # thereby reducing the total number of points
    num_discarded_points = cart_data.shape[1] - np.sum(grid_data)
    if num_discarded_points != 0:
        print(f'  cart2grid: {num_discarded_points} Cartesian points mapped to overlapping grid points')
    return grid_data, order_index, reorder_index


def get_bli(func, dx=1, up_sampling_factor=20, plot=False):
    """

    Args:
        func: 1d input function
        dx: spatial sampling [m] (default=1)
        up_sampling_factor: up-sampling factor used to sample the underlying BLI (default=20)
        plot:

    Returns:
        bli:    band-limited interpolant
        x_fine: x-grid for BLI
    """

    func = np.squeeze(func)
    assert len(func.shape) == 1, f"func not 1D but rather {len(func.shape)}D"
    nx = len(func)

    dk = 2 * np.pi / (dx * nx)
    if nx % 2:
        # odd
        k_min = -np.pi / dx + dk / 2
        k_max = np.pi / dx - dk / 2
    else:
        # even
        k_min = -np.pi / dx
        k_max = np.pi / dx - dk

    k = np.arange(start=k_min, stop=k_max + dk, step=dk, )
    x_fine = np.arange(start=0, stop=((nx - 1) * dx) + dx / up_sampling_factor, step=dx / up_sampling_factor)

    func_k = fftshift(fft(func)) / nx

    bli = np.real(np.sum(np.matmul(func_k[np.newaxis], np.exp(1j * np.outer(k, x_fine))), axis=0))
    if plot:
        raise NotImplementedError
    return bli, x_fine


def interpCartData(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, interp='nearest'):
    """
     interpCartData takes a matrix of time-series data recorded over a set
     of Cartesian sensor points given by cart_sensor_mask and computes the
     equivalent time-series at each sensor position on the binary sensor
     mask binary_sensor_mask using interpolation. The properties of
     binary_sensor_mask are defined by the k-Wave grid object kgrid.
     Two and three dimensional data are supported.

     Usage:
         binary_sensor_data = interpCartData(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask)
         binary_sensor_data = interpCartData(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, interp)

     Args:
         kgrid:                k-Wave grid object returned by kWaveGrid
         cart_sensor_data:     original sensor data measured over
                              cart_sensor_mask indexed as
                              cart_sensor_data(sensor position, time)
         cart_sensor_mask:     Cartesian sensor mask over which
                              cart_sensor_data is measured
         binary_sensor_mask:   binary sensor mask at which equivalent
                              time-series are computed via interpolation

         interp:               (optional) interpolation mode used to compute the
                              time-series, both 'nearest' and 'linear'
                              (two-point) modes are supported
                              (default = 'nearest')

     returns:
         binary_sensor_data:   array of time-series corresponding to the
                               sensor positions given by binary_sensor_mask
    """

    # make timer
    timer = TicToc()
    # start the clock
    timer.tic()

    # extract the number of data points
    num_cart_data_points, num_time_points = cart_sensor_data.shape
    num_binary_sensor_points = np.sum(binary_sensor_mask.flatten())

    # update command line status
    print('Interpolating Cartesian sensor data...')
    print(f'  interpolation mode: {interp}')
    print(f'  number of Cartesian sensor points:  {num_cart_data_points}')
    print(f'  number of binary sensor points: {num_binary_sensor_points}')

    binary_sensor_data = np.zeros((num_binary_sensor_points, num_time_points))

    # Check dimensionality of data passed
    if kgrid.dim not in [2, 3]:
        raise ValueError('Data must be two- or three-dimensional.')

    from kwave.utils.kutils import grid2cart
    cart_bsm, _ = grid2cart(kgrid, binary_sensor_mask)

    # nearest neighbour interpolation of the data points
    for point_index in range(num_binary_sensor_points):

        # find the measured data point that is closest
        dist = np.linalg.norm(cart_bsm[:, point_index] - cart_sensor_mask.T, ord=2, axis=1)
        if interp == 'nearest':

            dist_min_index = np.argmin(dist)

            # assign value
            binary_sensor_data[point_index, :] = cart_sensor_data[dist_min_index, :]

        elif interp == 'linear':
            # raise NotImplementedError
            # append the distance information onto the data set
            cart_sensor_data_ro = cart_sensor_data
            np.append(cart_sensor_data_ro, dist[:, None], axis=1)
            new_col_pos = -1

            # reorder the data set based on distance information
            cart_sensor_data_ro = sortrows(cart_sensor_data_ro, new_col_pos)

            # linearly interpolate between the two closest points
            perc = cart_sensor_data_ro[2, new_col_pos] / (
                    cart_sensor_data_ro[1, new_col_pos] + cart_sensor_data_ro[2, new_col_pos])
            binary_sensor_data[point_index, :] = perc * cart_sensor_data_ro[1, :] + \
                                                    (1 - perc) * cart_sensor_data_ro[2, :]

        else:
            raise ValueError('Unknown interpolation option.')

        # elif interp == 'linear':
        #
        #         # dist = np.sqrt((cart_bsm[0, point_index] - cart_sensor_mask[0, :])**2 + (cart_bsm[1, point_index] - cart_sensor_mask[1, :])**2)
        #         # dist = np.linalg.norm(cart_bsm[:, point_index] - cart_sensor_mask.T, axis=1)
        #         # append the distance information onto the data set
        #         new_col_pos = len(cart_sensor_data[1, :]) -1
        #         cart_sensor_data_ro = cart_sensor_data
        #         cart_sensor_data_ro[:, new_col_pos] = dist
        #
        #         # reorder the data set based on distance information
        #         cart_sensor_data_ro = sortrows(cart_sensor_data_ro, new_col_pos)
        #
        #         # linearly interpolate between the two closest points
        #         perc = cart_sensor_data_ro[1, new_col_pos] / (cart_sensor_data_ro[0, new_col_pos] + cart_sensor_data_ro[1, new_col_pos] )
        #         binary_sensor_data[point_index, :] = perc * cart_sensor_data_ro[1, :new_col_pos - 1] + (1 - perc) * cart_sensor_data_ro[1, :new_col_pos - 1]
        #
        # else:
        #     raise ValueError('Unknown interpolation option.')

    # update command line status
    print(f'  computation completed in {scale_time(timer.toc())}')
    return binary_sensor_data


def interpftn(x, sz: tuple, win=None):
    """
     Resamples an N-D matrix to the size given in sz using Fourier interpolation.

     USAGE:
         y = interpftn(x, sz)
         y = interpftn(x, sz, win)

     Args:
         x:           matrix to interpolate
         sz:          list or tupple of new size
         win:         (optional) name of windowing function to use

     Returns:
         y:           resampled matrix
    """

    # extract the size of the input matrix
    x_sz = x.shape

    # check enough coefficients have been given
    if sum([x != 1 for x in x_sz]) != len(sz):
        raise ValueError('The number of scaling coefficients must equal the number of dimensions in x.')

    # interpolate for each matrix dimension (dimensions with no interpolation required are skipped)
    y = x
    for p_idx, p in enumerate(sz):
        if p != x_sz[p_idx]:
            y = resample(y, p, axis=p_idx, window=win)

    return y
