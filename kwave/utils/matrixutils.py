from typing import Tuple
from skimage.transform import resize as si_resize
from scipy.interpolate import interpn
import numpy as np
import warnings

from .tictoc import TicToc
from .conversionutils import scale_time
from .checkutils import num_dim2, is_number
from .interputils import interpolate2d


def expand_matrix(matrix, exp_coeff, edge_val=None):
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
        opts['mode'] = 'edge'
    else:
        opts['mode'] = 'constant'
        opts['constant_values'] = edge_val

    exp_coeff = np.array(exp_coeff).astype(int).squeeze()
    n_coeff = exp_coeff.size
    assert n_coeff > 0

    if n_coeff == 1:
        opts['pad_width'] = exp_coeff
    elif len(matrix.shape) == 1:
        assert n_coeff <= 2
        opts['pad_width'] = exp_coeff
    elif len(matrix.shape) == 2:
        if n_coeff == 2:
            opts['pad_width'] = exp_coeff
        if n_coeff == 4:
            opts['pad_width'] = [(exp_coeff[0], exp_coeff[1]), (exp_coeff[2], exp_coeff[3])]
    elif len(matrix.shape) == 3:
        if n_coeff == 3:
            opts['pad_width'] = np.tile(np.expand_dims(exp_coeff, axis=-1), [1, 2])
        if n_coeff == 6:
            opts['pad_width'] = [(exp_coeff[0], exp_coeff[1]), (exp_coeff[2], exp_coeff[3]), (exp_coeff[4], exp_coeff[5])]

    return np.pad(matrix, **opts)


def matlab_find(arr, val=0, mode='neq'):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if mode == 'neq':
        arr = np.where(arr.flatten(order='F') != val)[0] + 1  # +1 due to matlab indexing
    else:  # 'eq'
        arr = np.where(arr.flatten(order='F') == val)[0] + 1  # +1 due to matlab indexing
    return np.expand_dims(arr, -1)  # compatibility, n => [n, 1]


def matlab_mask(arr, mask, diff=None):
    if diff is None:
        return np.expand_dims(arr.ravel(order='F')[mask.ravel(order='F')], axis=-1)  # compatibility, n => [n, 1]
    else:
        return np.expand_dims(arr.ravel(order='F')[mask.ravel(order='F') + diff], axis=-1)  # compatibility, n => [n, 1]


def unflatten_matlab_mask(arr, mask, diff=None):
    if diff is None:
        return np.unravel_index(mask.ravel(order='F'), arr.shape, order='F')
    else:
        return np.unravel_index(mask.ravel(order='F') + diff, arr.shape, order='F')

def matlab_assign(matrix: np.ndarray, indices, values):
    original_shape = matrix.shape
    matrix = matrix.flatten(order='F')
    matrix[indices] = values
    return matrix.reshape(original_shape, order='F')

def resize(mat, new_size, interp_mode='linear'):
    """
    resize: resamples a "matrix" of spatial samples to a desired "resolution" or spatial sampling frequency via interpolation

    Args:
        mat:                matrix to be "resized" i.e. resampled
        new_size:         desired output resolution
        interp_mode:        interpolation method

    Returns:
        res_mat:            "resized" matrix

    """
    # start the timer
    TicToc.tic()

    # update command line status
    print('Resizing matrix...')

    # check inputs
    assert num_dim2(mat) == len(new_size), \
        'Resolution input must have the same number of elements as data dimensions.'

    mat = mat.squeeze()
    mat_shape = mat.shape

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
        mat_rs = mat_rs.reshape(new_size, order='F')
    # update command line status
    print(f'  completed in {scale_time(TicToc.toc())}')
    assert mat_rs.shape == tuple(new_size), "Resized matrix does not match requested size."
    return mat_rs


def smooth(mat, restore_max=False, window_type='Blackman'):
    """
        Smooth a matrix
    Returns:

    """
    DEF_USE_ROTATION = True

    assert is_number(mat) and np.all(~np.isinf(mat))
    assert isinstance(restore_max, bool)
    assert isinstance(window_type, str)

    # get the grid size
    grid_size = mat.shape

    # remove singleton dimensions
    if num_dim2(mat) != len(grid_size):
        grid_size = np.squeeze(grid_size)

    # use a symmetric filter for odd grid sizes, and a non-symmetric filter for
    # even grid sizes to ensure the DC component of the window has a value of
    # unity
    window_symmetry = (np.array(grid_size) % 2).astype(bool)

    # get the window, taking the absolute value to discard machine precision
    # negative values
    from .kutils import get_win
    win, _ = get_win(grid_size, type_=window_type,
                     rotation=DEF_USE_ROTATION, symmetric=window_symmetry)
    win = np.abs(win)

    # rotate window if input mat is (1, N)
    if mat.shape[0] == 1:  # is row?
        win = win.T

    # apply the filter
    mat_sm = np.real(np.fft.ifftn(np.fft.fftn(mat) * np.fft.ifftshift(win)))

    # restore magnitude if required
    if restore_max:
        mat_sm = (np.abs(mat).max() / np.abs(mat_sm).max()) * mat_sm
    return mat_sm


def gradient_fd(f, dx=None, dim=None, deriv_order=None, accuracy_order=None):
    """
    A wrapper of the numpy gradient method for use in the k-wave library.

    gradient_fd calculates the gradient of an n-dimensional input matrix
    using the finite-difference method. For one-dimensional inputs, the
    gradient is always computed along the non-singleton dimension. For
    higher dimensional inputs, the gradient for singleton dimensions is
    returned as 0. For elements in the centre of the grid, the gradient
    is computed using centered finite-differences. For elements on the
    edge of the grid, the gradient is computed using forward or backward
    finite-differences. The order of accuracy of the finite-difference
    approximation is controlled by accuracy_order (default = 2). The
    calculations are done using sparse multiplication, so the input
    matrix is always cast to double precision.

    Args:
        f:
        dx:                 array of values for the grid point spacing in each
                            dimension. If a value for dim is given, dn is the
                            spacing in dimension dim.
        dim:                optional input to specify a single dimension over which to compute the gradient for
                            n-dimension input functions
        deriv_order:        order of the derivative to compute, e.g., use 1 to
                            compute df/dx, 2 to compute df^2/dx^2, etc.
                            (default = 1)
        accuracy_order:     order of accuracy for the finite difference
                            coefficients. Because centered differences are
                            used, this must be set to an integer multiple of
                            2 (default = 2)

    Returns:
        fx, fy, ...         gradient

    """
    if deriv_order:
        warnings.warn("deriv_order is no longer a supported argument.", DeprecationWarning)
    if accuracy_order:
        warnings.warn("accuracy_order is no longer a supported argument.", DeprecationWarning)

    if dim is not None and dx is not None:
        return np.gradient(f, dx, axis=dim)
    elif dim is not None:
        return np.gradient(f, axis=dim)
    elif dx is not None:
        return np.gradient(f, dx)
    else:
        return np.gradient(f)


def min_nd(matrix: np.ndarray) -> Tuple[float, Tuple]:
    min_val, linear_index = np.min(matrix), matrix.argmin()
    numpy_index = np.unravel_index(linear_index, matrix.shape)
    matlab_index = tuple(idx + 1 for idx in numpy_index)
    return min_val, matlab_index


def max_nd(matrix: np.ndarray) -> Tuple[float, Tuple]:
    max_val, linear_index = np.max(matrix), matrix.argmax()
    numpy_index = np.unravel_index(linear_index, matrix.shape)
    matlab_index = tuple(idx + 1 for idx in numpy_index)
    return max_val, matlab_index
