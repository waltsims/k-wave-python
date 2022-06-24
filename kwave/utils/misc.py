from datetime import datetime
import numpy as np

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
    return np.sinc(x / np.pi)


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


def find_closest(A, a):
    """
    find_closest returns the value and index of the item in A that is
    closest to the value a. For vectors, value and index correspond to
    the closest element in A. For matrices, value and index are row
    vectors corresponding to the closest element from each column. For
    N-D arrays, the function finds the closest value along the first
    matrix dimension (singleton dimensions are removed before the
    search). If there is more than one element with the closest value,
    the index of the first one is returned.

    Args:
        A: matrix to search
        a: value to find

    Returns:
        val
        idx
    """

    assert isinstance(A, np.ndarray), "A must be an np.array"

    idx = np.unravel_index(np.argmin(abs(A - a)), A.shape)
    return A[idx], idx


