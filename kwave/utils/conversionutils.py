import math
from math import floor

import numpy as np


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
        time = f'{hours} hours {minutes} min seconds s'
    elif minutes > 0:
        time = f'{minutes} min {seconds} s'
    else:
        time = f'{seconds} s'
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
            7: ('a', 'zepto', 1e21),
            8: ('a', 'yocto', 1e24),
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
    x_sc = x_sc.round(4)
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

    ABOUT:
    author - Bradley Treeby
    date - 27 th March 2009
    last update - 4 th June 2017

    This function is part of the k - Wave Toolbox(http: // www.k - wave.org)
    Copyright(C) 2009 - 2017 Bradley Treeby

    See also neper2db

    This file is part of k - Wave.k - Wave is free software: you can
    redistribute it and / or modify it under the terms of the GNU Lesser
    General Public License as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.

    k - Wave is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.See the GNU Lesser General Public License for
    more details.

    You should have received a copy of the GNU Lesser General Public License
    along with k - Wave.If not, see < http:// www.gnu.org / licenses / >.

    set default y value if not given by user
    """

    # calculate conversion
    alpha = 100 * alpha * (1e-6 / (2 * math.pi)) ** y / (20 * np.log10(np.exp(1)))
    return alpha


def neper2db(alpha, y=1):
    """
    NEPER2DB Convert nepers to decibels.

    DESCRIPTION:
    % neper2db converts an attenuation coefficient in units of
    % Nepers / ((rad / s) ^ y m) to units of dB / (MHz ^ y cm).
    %
    % USAGE:
    % alpha = neper2db(alpha)
    % alpha = neper2db(alpha, y)
    %
    % INPUTS:
    % alpha - attenuation in Nepers / ((rad / s) ^ y m)
    %
    % OPTIONAL INPUTS:
    % y - power law exponent(default=1)
    %
    % OUTPUTS:
    % alpha - attenuation in dB / (MHz ^ y cm)
    %
    % ABOUT:
    % author - Bradley Treeby
    % date - 3 rd December 2009
    % last update - 7 th June 2017
    %
    % This function is part of the k - Wave Toolbox(http: // www.k - wave.org)
    % Copyright(C) 2009 - 2017 Bradley Treeby
    %
    % See also db2neper

    % This file is part of k - Wave.k - Wave is free software: you can
    % redistribute it and / or modify it under the terms of the GNU Lesser
    % General Public License as published by the Free Software Foundation,
    % either version 3 of the License, or (at your option) any later version.
    %
    % k - Wave is distributed in the hope that it will be useful, but WITHOUT ANY
    % WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    % FOR A PARTICULAR PURPOSE.See the GNU Lesser General Public License for
        % more details.
    %
    % You should have received a copy of the GNU Lesser General Public License
    % along with k - Wave.If not, see < http:// www.gnu.org / licenses / >.
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
