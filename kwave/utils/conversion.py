import math
from math import floor
from typing import Tuple

import numpy as np
from numpy import ndarray

from kwave import kWaveGrid


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
        Attenuation in dB / (MHz ^ y cm)
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


def cart2pol(x, y):
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
