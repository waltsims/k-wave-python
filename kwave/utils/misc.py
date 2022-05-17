from datetime import datetime

import numpy as np


def get_date_string():
    return datetime.now().strftime("%d-%b-%Y-%H-%M-%S")


def gaussian(x, magnitude=None, mean=0, variance=1):
    if magnitude is None:
        magnitude = np.sqrt(2 * np.pi * variance)
    return magnitude * np.exp(-(x - mean) ** 2 / (2 * variance))


def ndgrid(*args):
    return np.array(np.meshgrid(*args, indexing='ij'))


def sinc(x):
    return np.sin(x) / x


def round_even(x):
    """
    Rounds to the nearest even integer.

    Args:
        x (float): inpput value

    Returns:
        (int): nearest odd integer.
    """
    return 2 * round(x / 2)


def round_odd(x):
    """
    Rounds to the nearest odd integer.

    Args:
        x (float): input value

    Returns:
        (int): nearest odd integer.

    """
    return 2 * round((x + 1) / 2) - 1
