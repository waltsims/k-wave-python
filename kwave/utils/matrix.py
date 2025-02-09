import logging

import numpy as np
from scipy.interpolate import interpn, interp1d
from beartype import beartype as typechecker
from beartype.typing import Union, List, Tuple, Optional
from jaxtyping import Int, Num, Shaped, Real, Bool
import kwave.utils.typing as kt

from .data import scale_time
from .tictoc import TicToc


@typechecker
def trim_zeros(data: Num[np.ndarray, "..."]) -> Tuple[Num[np.ndarray, "..."], List[Tuple[Int[kt.ScalarLike, ""], Int[kt.ScalarLike, ""]]]]:
    """
    Create a tight bounding box by removing zeros.

    Args:
        data: Matrix to trim.

    Returns:
        Tuple containing the trimmed matrix and indices used to trim the matrix.

    Raises:
        ValueError: If the input data is not 1D, 2D, or 3D.

    Example:
        data = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 3, 0, 0],
                         [0, 0, 1, 3, 4, 0],
                         [0, 0, 1, 3, 4, 0],
                         [0, 0, 1, 3, 0, 0],
                         [0, 0, 0, 0, 0, 0]])

        trimmed_data, indices = trim_zeros(data)

        # Output:
        # trimmed_data:
        # [[0 3 0]
        #  [1 3 4]
        #  [1 3 4]
        #  [1 3 0]]
        #
        # indices:
        # [(1, 4), (2, 5), (3, 5)]

    """
    data = np.squeeze(data)

    # only allow 1D, 2D, and 3D
    if data.ndim > 3:
        raise ValueError("Input data must be 1D, 2D, or 3D.")

    # set collapse directions for each dimension
    collapse = {2: [1, 0], 3: [(1, 2), (0, 2), (0, 1)]}

    # preallocate output to store indices
    ind = []

    # loop through dimensions
    for dim_index in range(data.ndim):
        # collapse to 1D vector
        if data.ndim == 1:
            summed_values = data
        else:
            summed_values = np.sum(np.abs(data), axis=collapse[data.ndim][dim_index])

        # find the first and last non-empty values
        non_zeros = np.where(summed_values > 0)[0]
        ind_first = non_zeros[0]
        ind_last = non_zeros[-1] + 1

        # trim data
        if data.ndim == 1:
            data = data[ind_first:ind_last]
            ind.append((ind_first, ind_last))
        else:
            if dim_index == 0:
                data = data[ind_first:ind_last, ...]
                ind.append((ind_first, ind_last))
            elif dim_index == 1:
                data = data[:, ind_first:ind_last, ...]
                ind.append((ind_first, ind_last))
            elif dim_index == 2:
                data = data[..., ind_first:ind_last]
                ind.append((ind_first, ind_last))

    return data, ind


@typechecker
def expand_matrix(
    matrix: Union[Num[np.ndarray, "..."], Bool[np.ndarray, "..."]],
    exp_coeff: Union[Shaped[kt.ArrayLike, "dim"], List],
    edge_val: Optional[Real[kt.ScalarLike, ""]] = None,
):
    """
    Enlarge a matrix by extending the edge values.

    expandMatrix enlarges an input matrix by extension of the values at
    the outer faces of the matrix (endpoints in 1D, outer edges in 2D,
    outer surfaces in 3D). Alternatively, if an input for edge_val is
    given, all expanded matrix elements will have this value. The values
    for exp_coeff are forced to be real positive integers (or zero).

    Note, indexing is done inline with other k-Wave functions using
    mat(x) in 1D, mat(x, y) in 2D, and mat(x, y, z) in 3D.

    Args:
        matrix: the matrix to enlarge
        exp_coeff: the number of elements to add in each dimension
                    in 1D: [a] or [x_start, x_end]
                    in 2D: [a] or [x, y] or
                           [x_start, x_end, y_start, y_end]
                    in 3D: [a] or [x, y, z] or
                           [x_start, x_end, y_start, y_end, z_start, z_end]
                           (here 'a' is applied to all dimensions)
        edge_val: value to use in the matrix expansion

    Returns:
        expanded matrix

    """

    opts = {}
    matrix = np.squeeze(matrix)

    if edge_val is None:
        opts["mode"] = "edge"
    else:
        opts["mode"] = "constant"
        opts["constant_values"] = edge_val

    exp_coeff = np.array(exp_coeff).astype(int).squeeze()
    n_coeff = exp_coeff.size
    assert n_coeff > 0

    if n_coeff == 1:
        opts["pad_width"] = exp_coeff
    elif len(matrix.shape) == 1:
        assert n_coeff <= 2
        opts["pad_width"] = exp_coeff
    elif len(matrix.shape) == 2:
        if n_coeff == 2:
            opts["pad_width"] = [(exp_coeff[0],), (exp_coeff[1],)]
        if n_coeff == 4:
            opts["pad_width"] = [(exp_coeff[0], exp_coeff[1]), (exp_coeff[2], exp_coeff[3])]
    elif len(matrix.shape) == 3:
        if n_coeff == 3:
            opts["pad_width"] = np.tile(np.expand_dims(exp_coeff, axis=-1), [1, 2])
        if n_coeff == 6:
            opts["pad_width"] = [(exp_coeff[0], exp_coeff[1]), (exp_coeff[2], exp_coeff[3]), (exp_coeff[4], exp_coeff[5])]

    return np.pad(matrix, **opts)


def resize(mat: np.ndarray, new_size: Union[int, List[int]], interp_mode: str = "linear") -> np.ndarray:
    """
    Resizes a matrix of spatial samples to a desired resolution or spatial sampling frequency
    via interpolation.

    Parameters:
        mat: Matrix to be resized (i.e., resampled).
        new_size: Desired output resolution.
        interp_mode: Interpolation method.

    Returns:
        Resized matrix.
    """
    # start the timer
    TicToc.tic()

    # update command line status
    logging.log(logging.INFO, "Resizing matrix...")
    # check inputs
    assert num_dim2(mat) == len(new_size), "Resolution input must have the same number of elements as data dimensions."

    mat = mat.squeeze()

    axis = []
    for dim in range(len(mat.shape)):
        dim_size = mat.shape[dim]
        axis.append(np.linspace(0, 1, dim_size))

    new_axis = []
    for dim in range(len(new_size)):
        dim_size = new_size[dim]
        new_axis.append(np.linspace(0, 1, dim_size))

    points = tuple(p for p in axis)
    xi = np.meshgrid(*new_axis)
    xi = np.array([x.flatten() for x in xi]).T
    new_points = xi
    mat_rs = np.squeeze(interpn(points, mat, new_points, method=interp_mode))
    # TODO: fix this hack.
    if dim + 1 == 3:
        mat_rs = mat_rs.reshape([new_size[1], new_size[0], new_size[2]])
        mat_rs = np.transpose(mat_rs, (1, 0, 2))
    else:
        mat_rs = mat_rs.reshape(new_size, order="F")
    # update command line status
    logging.log(logging.INFO, f"  completed in {scale_time(TicToc.toc())}")
    assert mat_rs.shape == tuple(new_size), "Resized matrix does not match requested size."
    return mat_rs


def gradient_fd(f, dx=None, dim=None, deriv_order=None, accuracy_order=None) -> List[np.ndarray]:
    """
    Calculate the gradient of an n-dimensional input matrix using the finite-difference method.

    This function is a wrapper of the numpy gradient method for use in the k-wave library.
    For one-dimensional inputs, the gradient is always computed along the non-singleton dimension.
    For higher dimensional inputs, the gradient for singleton dimensions is returned as 0.
    For elements in the center of the grid, the gradient is computed using centered finite-differences.
    For elements on the edge of the grid, the gradient is computed using forward or backward finite-differences.
    The order of accuracy of the finite-difference approximation is controlled by `accuracy_order` (default = 2).
    The calculations are done using sparse multiplication, so the input matrix is always cast to double precision.

    Args:
        f: Input matrix.
        dx: Array of values for the grid point spacing in each dimension.
                If a value for `dim` is given, `dn` is the spacing in dimension `dim`.
        dim: Optional input to specify a single dimension over which to compute the gradient for
        deriv_order: Order of the derivative to compute,
                        e.g., use 1 to compute df/dx, 2 to compute df^2/dx^2, etc. (default = 1).
        accuracy_order: Order of accuracy for the finite difference coefficients.
                            Because centered differences are used, this must be set to an integer
                            multiple of 2 (default = 2).

    Returns:
        A list of ndarrays (or a single ndarray if there is only one dimension)
        corresponding to the derivatives of f with respect to each dimension.
        Each derivative has the same shape as f.

    """

    if deriv_order:
        logging.log(logging.WARN, f"{DeprecationWarning.__name__}: deriv_order is no longer a supported argument.")
    if accuracy_order:
        logging.log(logging.WARN, f"{DeprecationWarning.__name__}: accuracy_order is no longer a supported argument.")

    if dim is not None and dx is not None:
        return np.gradient(f, dx, axis=dim)
    elif dim is not None:
        return np.gradient(f, axis=dim)
    elif dx is not None:
        return np.gradient(f, dx)
    else:
        return np.gradient(f)


def min_nd(matrix: np.ndarray) -> Tuple[float, Tuple]:
    """
    Find the minimum value and its indices in a numpy array.

    Args:
        matrix: A numpy array of any value type.

    Returns:
        A tuple containing the minimum value and a tuple of indices in the form (row, column, ...).
            Indices are 1-based, following the convention used in MATLAB.

    Examples:
        >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> min_nd(matrix)
        (1, (1, 1))

    """

    min_val, linear_index = np.min(matrix), matrix.argmin()
    numpy_index = np.unravel_index(linear_index, matrix.shape)
    matlab_index = tuple(idx + 1 for idx in numpy_index)
    return min_val, matlab_index


def max_nd(matrix: np.ndarray) -> Tuple[float, Tuple]:
    """
    Returns the maximum value in a n-dimensional array and its index.

    Args:
        matrix: n-dimensional array of values.

    Returns:
        A tuple containing the maximum value in the array, and a tuple containing the index of the
        maximum value. The index is given in the MATLAB convention, where indexing starts at 1.

    """

    # Get the maximum value and its linear index
    max_val, linear_index = np.max(matrix), matrix.argmax()

    # Convert the linear index to a tuple of indices in the original matrix
    numpy_index = np.unravel_index(linear_index, matrix.shape)

    # Convert the tuple of indices to 1-based indices (as used in Matlab)
    matlab_index = tuple(idx + 1 for idx in numpy_index)

    # Return the maximum value and the 1-based index
    return max_val, matlab_index


def broadcast_axis(data: np.ndarray, ndims: int, axis: int) -> np.ndarray:
    """
    Broadcast the given axis of the data to the specified number of dimensions.

    Args:
        data: The data to broadcast.
        ndims: The number of dimensions to broadcast the axis to.
        axis: The axis to broadcast.

    Returns:
        The broadcasted data.

    """

    newshape = [1] * ndims
    newshape[axis] = -1
    return data.reshape(*newshape)


def revolve2d(mat2d: np.ndarray) -> np.ndarray:
    """
    Revolve a 2D numpy array in a clockwise direction to form a 3D numpy array.

    Args:
        mat2d: A 2D numpy array of any value type.

    Returns:
        A 3D numpy array formed by revolving the input array in a clockwise direction.

    Examples:
        >>> mat2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> revolve2d(mat2d)
        array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               [[7, 4, 1],
                [8, 5, 2],
                [9, 6, 3]],
               [[9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]]])

    """

    # Start timer
    TicToc.tic()

    # Update command line status
    logging.log(logging.INFO, "Revolving 2D matrix to form a 3D matrix...")

    # Get size of matrix
    m, n = mat2d.shape

    # Create the reference axis for the 2D image
    r_axis_one_sided = np.arange(0, n)
    r_axis_two_sided = np.arange(-(n - 1), n)

    # Compute the distance from every pixel in the z-y cross-section of the 3D
    # matrix to the rotation axis
    z, y = np.meshgrid(r_axis_two_sided, r_axis_two_sided)
    r = np.sqrt(y**2 + z**2)

    # Create empty image matrix
    mat3D = np.zeros((m, 2 * n - 1, 2 * n - 1))

    # Loop through each cross-section and create 3D matrix
    for x_index in range(m):
        interp = interp1d(x=r_axis_one_sided, y=mat2d[x_index, :], kind="linear", bounds_error=False, fill_value=0)
        mat3D[x_index, :, :] = interp(r)

    # Update command line status
    logging.log(logging.INFO, f"  completed in {scale_time(TicToc.toc())}s")
    return mat3D


def sort_rows(arr: np.ndarray, index: int) -> np.ndarray:
    """
    Sort the rows of a 2D numpy array by the values in a specific column.

    Args:
        arr: A 2D numpy array.
        index: The index of the column to sort by.

    Returns:
        A copy of the input array with the rows sorted by the values in the specified column.

    Raises:
        AssertionError: If `arr` is not a 2D numpy array.

    Examples:
        >>> arr = np.array([[3, 2, 1], [1, 3, 2], [2, 1, 3]])
        >>> sort_rows(arr, 0)
        array([[1, 3, 2],
               [2, 1, 3],
               [3, 2, 1]])

    """

    assert arr.ndim == 2, "'sort_rows' currently supports only 2-dimensional matrices"
    return arr[arr[:, index].argsort()]


def num_dim(x: np.ndarray) -> int:
    """
    Returns the number of dimensions in x, after collapsing any singleton dimensions.

    Args:
        x: The input array.

    Returns:
        The number of dimensions in x.

    """

    return len(x.squeeze().shape)


def num_dim2(x: np.ndarray) -> int:
    """
    Get the number of dimensions of an array after collapsing singleton dimensions.

    Args:
        x: The input array.

    Returns:
        The number of dimensions of the array after collapsing singleton dimensions.

    """

    sz = np.squeeze(x).shape

    if len(sz) > 2:
        return len(sz)
    else:
        return np.sum(np.array(sz) > 1)
