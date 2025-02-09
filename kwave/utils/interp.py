import logging

import numpy as np
from numpy.fft import fft, fftshift
from scipy.interpolate import interpn
from scipy.signal import resample
from beartype import beartype as typechecker
from beartype.typing import Union, List, Tuple, Optional

from .conversion import grid2cart
from .data import scale_time
from .matrix import sort_rows
from .tictoc import TicToc


def interpolate3d(grid_points: List[np.ndarray], grid_values: np.ndarray, interp_locs: List[np.ndarray]) -> np.ndarray:
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

    """

    assert len(grid_points) == 3, "interpolate3D supports only 3D interpolation"
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
    queries = np.array(np.meshgrid(q_x, q_y, q_z, indexing="ij"))
    # Queries are just a list of 3D points
    queries = queries.reshape(3, -1).T

    # Out of bound points will get NaN values
    result = interpn((g_x, g_y, g_z), grid_values, queries, method="linear", bounds_error=False, fill_value=np.nan)
    # Go back from list of interpolated values to 3D volume
    result = result.reshape((q_x.size, q_y.size, q_z.size))
    # set values outside of the interpolation range to original values
    result[np.isnan(result)] = grid_values[np.isnan(result)]
    return result


def interpolate2d(
    grid_points: List[np.ndarray], grid_values: np.ndarray, interp_locs: List[np.ndarray], method="linear", copy_nans=True
) -> np.ndarray:
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

    """

    assert len(grid_points) == 2, "interpolate2D supports only 2D interpolation"
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
    queries = np.array(np.meshgrid(q_x, q_y, indexing="ij"))
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


def interpolate2d_with_queries(
    grid_points: List[np.ndarray], grid_values: np.ndarray, queries: np.ndarray, method="linear", copy_nans=True
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

    """
    assert len(grid_points) == 2, "interpolate2D supports only 2D interpolation"

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


def get_bli(
    func: np.ndarray,
    dx: Optional[float] = 1,
    up_sampling_factor: Optional[int] = 20,
    plot: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the band-limited interpolant of a 1D input function.

    Args:
        func: The 1D input function.
        dx: Spatial sampling in meters. Defaults to 1.
        up_sampling_factor: Up-sampling factor used to sample the underlying BLI. Defaults to 20.
        plot: Whether to plot the BLI. Defaults to False.

    Returns:
        A tuple containing the BLI and the x-grid for the BLI.

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

    k = np.arange(
        start=k_min,
        stop=k_max + dk,
        step=dk,
    )
    x_fine = np.arange(start=0, stop=((nx - 1) * dx) + dx / up_sampling_factor, step=dx / up_sampling_factor)

    func_k = fftshift(fft(func)) / nx

    bli = np.real(np.sum(np.matmul(func_k[np.newaxis], np.exp(1j * np.outer(k, x_fine))), axis=0))
    if plot:
        raise NotImplementedError
    return bli, x_fine


def interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, interp="nearest"):
    """
    Takes a matrix of time-series data recorded over a set
    of Cartesian sensor points given by cart_sensor_mask and computes the
    equivalent time-series at each sensor position on the binary sensor
    mask binary_sensor_mask using interpolation. The properties of
    binary_sensor_mask are defined by the k-Wave grid object kgrid.
    Two and three-dimensional data are supported.

    Usage:
        binary_sensor_data = interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask)
        binary_sensor_data = interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, interp)

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

    Returns:
        array of time-series corresponding to the sensor positions given by binary_sensor_mask

    """

    # make timer
    timer = TicToc()
    # start the clock
    timer.tic()

    # extract the number of data points
    num_cart_data_points, num_time_points = cart_sensor_data.shape
    num_binary_sensor_points = np.sum(binary_sensor_mask.flatten())

    # update command line status
    logging.log(logging.INFO, "Interpolating Cartesian sensor data...")
    logging.log(logging.INFO, f"  interpolation mode: {interp}")
    logging.log(logging.INFO, f"  number of Cartesian sensor points:  {num_cart_data_points}")
    logging.log(logging.INFO, f"  number of binary sensor points: {num_binary_sensor_points}")

    binary_sensor_data = np.zeros((num_binary_sensor_points, num_time_points))

    # Check dimensionality of data passed
    if kgrid.dim not in [2, 3]:
        raise ValueError("Data must be two- or three-dimensional.")

    cart_bsm, _ = grid2cart(kgrid, binary_sensor_mask)

    # nearest neighbour interpolation of the data points
    for point_index in range(num_binary_sensor_points):
        # find the measured data point that is closest
        dist = np.linalg.norm(cart_bsm[:, point_index] - cart_sensor_mask[: kgrid.dim, :].T, ord=2, axis=1)
        if interp == "nearest":
            dist_min_index = np.argmin(dist)

            # assign value
            binary_sensor_data[point_index, :] = cart_sensor_data[dist_min_index, :]

        elif interp == "linear":
            # raise NotImplementedError
            # append the distance information onto the data set
            cart_sensor_data_ro = cart_sensor_data
            np.append(cart_sensor_data_ro, dist[:, None], axis=1)
            new_col_pos = -1

            # reorder the data set based on distance information
            cart_sensor_data_ro = sort_rows(cart_sensor_data_ro, new_col_pos)

            # linearly interpolate between the two closest points
            perc = cart_sensor_data_ro[2, new_col_pos] / (cart_sensor_data_ro[1, new_col_pos] + cart_sensor_data_ro[2, new_col_pos])
            binary_sensor_data[point_index, :] = perc * cart_sensor_data_ro[1, :] + (1 - perc) * cart_sensor_data_ro[2, :]

        else:
            raise ValueError("Unknown interpolation option.")

        # elif interp == 'linear':
        #
        #         # dist = np.sqrt((cart_bsm[0, point_index] - cart_sensor_mask[0, :])**2 +
        #                                                   (cart_bsm[1, point_index] - cart_sensor_mask[1, :])**2)
        #         # dist = np.linalg.norm(cart_bsm[:, point_index] - cart_sensor_mask.T, axis=1)
        #         # append the distance information onto the data set
        #         new_col_pos = len(cart_sensor_data[1, :]) -1
        #         cart_sensor_data_ro = cart_sensor_data
        #         cart_sensor_data_ro[:, new_col_pos] = dist
        #
        #         # reorder the data set based on distance information
        #         cart_sensor_data_ro = sort_rows(cart_sensor_data_ro, new_col_pos)
        #
        #         # linearly interpolate between the two closest points
        #         perc = cart_sensor_data_ro[1, new_col_pos] /
        #                           (cart_sensor_data_ro[0, new_col_pos] + cart_sensor_data_ro[1, new_col_pos] )
        #         binary_sensor_data[point_index, :] = perc * cart_sensor_data_ro[1, :new_col_pos - 1] +
        #                                                       (1 - perc) * cart_sensor_data_ro[1, :new_col_pos - 1]
        #
        # else:
        #     raise ValueError('Unknown interpolation option.')

    # update command line status
    logging.log(logging.INFO, f"  computation completed in {scale_time(timer.toc())}")
    return binary_sensor_data


def interpftn(x, sz: tuple, win=None):
    """
    Resamples an N-D matrix to the size given in sz using Fourier interpolation.


    Args:
        x:           matrix to interpolate
        sz:          list or tupple of new size
        win:         (optional) name of windowing function to use

    Returns:
        Resampled matrix

    Examples:
        >>> y = interpftn(x, sz)
        >>> y = interpftn(x, sz, win)

    """

    # extract the size of the input matrix
    x_sz = x.shape

    # check enough coefficients have been given
    if sum([x != 1 for x in x_sz]) != len(sz):
        raise ValueError("The number of scaling coefficients must equal the number of dimensions in x.")

    # interpolate for each matrix dimension (dimensions with no interpolation required are skipped)
    y = x
    for p_idx, p in enumerate(sz):
        if p != x_sz[p_idx]:
            y = resample(y, p, axis=p_idx, window=win)

    return y


@typechecker
def get_delta_bli(Nx: int, dx: float, x: np.ndarray, x0: Union[int, float], include_imag: bool = False) -> np.ndarray:
    """
    Exact BLI of an arbitrarily positioned delta function.

    Calculates the exact Band-Limited Interpolation (BLI) of an arbitrarily positioned delta function.
    For grid dimensions with an evenly-sampled periodicity, a small Nyquist frequency sinusoid is added.
    This sinusoid is invisible on grid samples and has zero amplitude when the delta function lies on a grid node.
    It is important when the evaluation points aren't grid nodes, and when the delta function is off-grid.
    It serves to ensure conjugate symmetry in the BLI's Fourier transform.

    Args:
        Nx: Number of grid points in the relevant Cartesian direction.
        dx: Grid point spacing [m].
        x: Coordinates at which the BLI is evaluated [m].
        x0: Coordinate at which the BLI is centered [m].
        include_imag: Whether to include the imaginary component of the off-grid delta function.
                      Defaults to False.

    Returns:
        f: Value of the BLI at the specified coordinates.

    """

    # ignore imaginary component of even function by default
    if include_imag is None:
        include_imag = False

    # check whether the grid has even or odd samples per period
    is_even = Nx % 2 == 0

    # compute BLI
    if is_even:
        # compute periodic sinc function
        f = np.sin(np.pi * (x - x0) / dx) / (Nx * np.tan(np.pi * (x - x0) / (Nx * dx)))

        # correct indeterminate points
        f[(x - x0) == 0] = 1

        # add Nyquist sinusoid to ensure conjugate symmetry
        f = f - np.sin(np.pi * x0 / dx) / Nx * np.sin(np.pi * x / dx)
        if include_imag:
            f = f + 1j * np.cos(np.pi * x0 / dx) / Nx * np.sin(np.pi * x / dx)
    else:
        # compute periodic sinc function
        f = np.sin(np.pi * (x - x0) / dx) / (Nx * np.sin(np.pi * (x - x0) / (Nx * dx)))

        # correct indeterminate points
        f[(x - x0) == 0] = 1

    return f
