from datetime import datetime
from math import floor
from beartype.typing import Tuple, Union, Optional
from beartype import beartype as typechecker

import numpy as np
from kwave.data import Vector

from kwave.utils.typing import NUMERIC_WITH_COMPLEX


@typechecker
def get_smallest_possible_type(
    max_array_val: Union[NUMERIC_WITH_COMPLEX, Vector], target_type_group: str, default: Optional[str] = None
) -> Union[str, None]:
    """
    Returns the smallest possible type for the given array.
    Args:
        max_array_val: The maximum value in the array.
        target_type_group: The type group to search for the smallest possible type.
        default: The default type to return if no type is found.

    Returns:
        The smallest possible type for the given array.

    """

    types = {"uint", "int"}
    assert target_type_group in types

    for bit_count in [8, 16, 32]:
        type_ = f"{target_type_group}{bit_count}"
        if max_array_val < intmax(type_):
            return type_

    type_ = default
    return type_


@typechecker
def intmax(dtype: str) -> int:
    """
    Returns the maximum value for the given integer type.

    Args:
        dtype: The integer type.

    Returns
        The maximum value for the given integer type.

    """

    return np.iinfo(getattr(np, dtype)).max


@typechecker
def scale_time(seconds: Union[int, float]) -> str:
    """
    Converts an integer number of seconds into hours, minutes,
    and seconds, and returns a string with this information.

    Args:
        seconds: number of seconds

    Returns:
        String of scaled time.

    """

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
        time = f"{years} years, {weeks} weeks, and {days} days"
    elif weeks > 0:
        time = f"{weeks} weeks, {days} days, and {hours} hours"
    elif days > 0:
        time = f"{days} days, {hours} hours, and {minutes} min"
    elif hours > 0:
        seconds = np.round(seconds, 4)
        if np.abs(seconds - int(seconds)) < 1e-4:
            seconds = int(seconds)
        time = f"{hours}hours {minutes}min {seconds}s"
    elif minutes > 0:
        seconds = np.round(seconds, 4)
        if np.abs(seconds - int(seconds)) < 1e-4:
            seconds = int(seconds)
        time = f"{minutes}min {seconds}s"
    else:
        precision = 10  # manually tuned number
        seconds = round(seconds, precision)
        time = f"{seconds}s"
    return time


@typechecker
def scale_SI(x: Union[float, np.ndarray]) -> Tuple[str, Union[int, float], str, str]:
    """
    Scale a number to the nearest SI unit prefix.

    Args:
        x: The number to scale.

    Returns:
        A tuple containing a string of the scaled number, a numeric scaling factor, the prefix, and the unit.

    """

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
        prefix = ""
        prefix_fullname = ""
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
            1: ("m", "milli", 1e3),
            2: ("u", "micro", 1e6),
            3: ("n", "nano", 1e9),
            4: ("p", "pico", 1e12),
            5: ("f", "femto", 1e15),
            6: ("a", "atto", 1e18),
            7: ("z", "zepto", 1e21),
            8: ("y", "yocto", 1e24),
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
            1: ("k", "kilo", 1e-3),
            2: ("M", "mega", 1e-6),
            3: ("G", "giga", 1e-9),
            4: ("T", "tera", 1e-12),
            5: ("P", "peta", 1e-15),
            6: ("E", "exa", 1e-18),
            7: ("Z", "zetta", 1e-21),
            8: ("Y", "yotta", 1e-24),
        }
        prefix, prefix_fullname, scale = units[sym_index]

    else:
        # if x is between 1 and 1000, don't scale
        x_sc = x
        prefix = ""
        prefix_fullname = ""
        scale = 1

    # form scaling into a string
    round_decimals = 6  # TODO this needs to be tuned
    x_sc = x_sc.round(round_decimals)
    if (x_sc - int(x_sc)) < (0.1**round_decimals):
        # avoid values like X.0, instead have only X
        x_sc = int(x_sc)
    x_sc = f"-{x_sc}{prefix}" if negative else f"{x_sc}{prefix}"
    return x_sc, scale, prefix, prefix_fullname


def get_date_string() -> str:
    return datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
