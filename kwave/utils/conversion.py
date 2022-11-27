import math
from math import floor
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from kwave.utils.tictoc import TicToc


def scale_time(seconds):
    # switch to calculating years, weeks, and days if larger than 100 hours
    if seconds > (60 * 60 * 100):
        years = floor(seconds / (60 * 60 * 24 * 365))
        seconds = seconds - years * 60 * 60 * 24 * 365
        weeks = floor(seconds / (60 * 60 * 24 * 7))
        seconds = seconds - weeks * 60 * 60 * 24 * 7
        days = floor(seconds / (60 * 60 * 24))
        seconds = seconds - days * 60 * 60 * 24
    else:
        years = 0
        weeks = 0
        days = 0

    # calculate hours, minutes, and seconds
    hours = floor(seconds / (60 * 60))
    seconds = seconds - hours * 60 * 60
    minutes = floor(seconds / 60)
    seconds = seconds - minutes * 60

    # write out as a string, to keep the output manageable, only the largest
    # three units are written
    if years > 0:
        time = f'{years} years, {weeks} weeks, and {days} days'
    elif weeks > 0:
        time = f'{weeks} weeks, {days} days, and {hours} hours'
    elif days > 0:
        time = f'{days} days, {hours} hours, and {minutes} min'
    elif hours > 0:
        seconds = np.round(seconds, 4)
        if np.abs(seconds - int(seconds)) < 1e-4:
            seconds = int(seconds)
        time = f'{hours}hours {minutes}min {seconds}s'
    elif minutes > 0:
        seconds = np.round(seconds, 4)
        if np.abs(seconds - int(seconds)) < 1e-4:
            seconds = int(seconds)
        time = f'{minutes}min {seconds}s'
    else:
        precision = 10  # manually tuned number
        seconds = round(seconds, precision)
        time = f'{seconds}s'
    return time


def scale_SI(x):
    # force the input to be a scalar
    x = np.max(x)

    # check for a negative input
    if x < 0:
        x = -x
        negative = True
    else:
        negative = False

    if x == 0:

        # if x is zero, don't scale
        x_sc = x
        prefix = ''
        prefix_fullname = ''
        scale = 1

    elif x < 1:

        # update index and input
        x_sc = x * 1e3
        sym_index = 1

        # find scaling parameter
        while x_sc < 1 and sym_index < 8:
            x_sc = x_sc * 1e3
            sym_index = sym_index + 1

        # define SI unit scalings
        units = {
            1: ('m', 'milli', 1e3),
            2: ('u', 'micro', 1e6),
            3: ('n', 'nano', 1e9),
            4: ('p', 'pico', 1e12),
            5: ('f', 'femto', 1e15),
            6: ('a', 'atto', 1e18),
            7: ('z', 'zepto', 1e21),
            8: ('y', 'yocto', 1e24),
        }
        prefix, prefix_fullname, scale = units[sym_index]

    elif x >= 1000:

        # update index and input
        x_sc = x * 1e-3
        sym_index = 1

        # find scaling parameter
        while x_sc >= 1000 and sym_index < 8:
            x_sc = x_sc * 1e-3
            sym_index = sym_index + 1

        # define SI unit scalings
        units = {
            1: ('k', 'kilo', 1e-3),
            2: ('M', 'mega', 1e-6),
            3: ('G', 'giga', 1e-9),
            4: ('T', 'tera', 1e-12),
            5: ('P', 'peta', 1e-15),
            6: ('E', 'exa', 1e-18),
            7: ('Z', 'zetta', 1e-21),
            8: ('Y', 'yotta', 1e-24)
        }
        prefix, prefix_fullname, scale = units[sym_index]

    else:
        # if x is between 1 and 1000, don't scale
        x_sc = x
        prefix = ''
        prefix_fullname = ''
        scale = 1

    # form scaling into a string
    round_decimals = 6  # TODO this needs to be tuned
    x_sc = x_sc.round(round_decimals)
    if (x_sc - int(x_sc)) < (0.1 ** round_decimals):
        # avoid values like X.0, instead have only X
        x_sc = int(x_sc)
    x_sc = f'-{x_sc}{prefix}' if negative else f'{x_sc}{prefix}'
    return x_sc, scale, prefix, prefix_fullname


def db2neper(alpha, y=1):
    """
    DB2NEPER Convert decibels to nepers.

    DESCRIPTION:
    db2neper converts an attenuation coefficient in units of
    dB / (MHz ^ y cm) to units of Nepers / ((rad / s) ^ y m).

    USAGE:
    alpha = db2neper(alpha)
    alpha = db2neper(alpha, y)

    INPUTS:
    alpha - attenuation in dB / (MHz ^ y cm)

    OPTIONAL INPUTS:
    y - power law exponent(default=1)

    OUTPUTS:
    alpha - attenuation in Nepers / ((rad / s) ^ y m)

    set default y value if not given by user
    """

    # calculate conversion
    alpha = 100 * alpha * (1e-6 / (2 * math.pi)) ** y / (20 * np.log10(np.exp(1)))
    return alpha


def neper2db(alpha, y=1):
    """
    NEPER2DB Convert nepers to decibels.

    DESCRIPTION:
     neper2db converts an attenuation coefficient in units of
     Nepers / ((rad / s) ^ y m) to units of dB / (MHz ^ y cm).

     USAGE:
     alpha = neper2db(alpha)
     alpha = neper2db(alpha, y)

     INPUTS:
     alpha - attenuation in Nepers / ((rad / s) ^ y m)

     OPTIONAL INPUTS:
     y - power law exponent(default=1)

     OUTPUTS:
     alpha - attenuation in dB / (MHz ^ y cm)

    """

    # calculate conversion
    alpha = 20 * math.log10(math.exp(1)) * alpha * (2 * math.pi * 1e6) ** y / 100
    return alpha


def cast_to_type(data, matlab_type: str):
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


def scan_conversion(
    scan_lines: np.ndarray,
    steering_angles,
    image_size: Tuple[float, float],
    c0,
    dt,
    resolution: Optional[Tuple[int, int]]
) -> np.ndarray:

    if resolution is None:
        resolution = (256, 256)  # in pixels

    x_resolution, y_resolution = resolution

    # assign the inputs
    x, y = image_size

    # start the timer
    TicToc.tic()

    # update command line status
    print('Computing ultrasound scan conversion...')

    # extract a_line parameters
    Nt = scan_lines.shape[1]

    # calculate radius variable based on the sound speed in the medium and the
    # round trip distance
    r = c0 * np.arange(1, Nt + 1) * dt / 2     # [m]

    # create regular Cartesian grid to remap to
    pos_vec_y_new = np.linspace(0, 1, y_resolution) * y - y / 2
    pos_vec_x_new = np.linspace(0, 1, x_resolution) * x
    [pos_mat_x_new, pos_mat_y_new] = np.array(np.meshgrid(pos_vec_x_new, pos_vec_y_new, indexing='ij'))

    # convert new points to polar coordinates
    [th_cart, r_cart] = cart2pol(pos_mat_x_new, pos_mat_y_new)

    # TODO: move this import statement at the top of the file
    # Not possible now due to cyclic dependencies
    from kwave.utils.interp import interpolate2d_with_queries

    # below part has some modifications
    # we flatten the _cart matrices and build queries
    # then we get values at the query locations
    # and reshape the values to the desired size
    # These three steps can be accomplished in one step in Matlab
    # However, we don't want to add custom logic to the `interpolate2D_with_queries` method.

    # Modifications -start
    queries = np.array([r_cart.flatten(), th_cart.flatten()]).T

    b_mode = interpolate2d_with_queries(
        [r, 2 * np.pi * steering_angles / 360],
        scan_lines.T,
        queries,
        method='linear',
        copy_nans=False
    )
    image_size_points = (len(pos_vec_x_new), len(pos_vec_y_new))
    b_mode = b_mode.reshape(image_size_points)
    # Modifications -end

    b_mode[np.isnan(b_mode)] = 0

    # update command line status
    print(f'  completed in {scale_time(TicToc.toc())}')

    return b_mode


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return phi, rho


def revolve2D(mat2D):
    # start timer
    TicToc.tic()

    # update command line status
    print('Revolving 2D matrix to form a 3D matrix...')

    # get size of matrix
    m, n = mat2D.shape

    # create the reference axis for the 2D image
    r_axis_one_sided = np.arange(0, n)
    r_axis_two_sided = np.arange(-(n-1), n)

    # compute the distance from every pixel in the z-y cross-section of the 3D
    # matrix to the rotation axis
    z, y = np.meshgrid(r_axis_two_sided, r_axis_two_sided)
    r = np.sqrt(y**2 + z**2)

    # create empty image matrix
    mat3D = np.zeros((m, 2 * n - 1, 2 * n - 1))

    # loop through each cross section and create 3D matrix
    for x_index in range(m):
        interp = interp1d(x=r_axis_one_sided, y=mat2D[x_index, :], kind='linear', bounds_error=False, fill_value=0)
        mat3D[x_index, :, :] = interp(r)

    # update command line status
    print(f'  completed in {scale_time(TicToc.toc())}s')
    return mat3D
