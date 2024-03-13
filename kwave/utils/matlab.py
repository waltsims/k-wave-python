from typing import Tuple, Union, Optional, List

import numpy as np


def rem(x, y, rtol=1e-05, atol=1e-08):
    """
    Returns the remainder after division of x by y, taking into account the floating point precision.
    x and y must be real and have compatible sizes.
    This function should be equivalent to the MATLAB rem function.

    Args:
        x (float, list, or ndarray): The dividend(s).
        y (float, list, or ndarray): The divisor(s).
        rtol (float): The relative tolerance parameter (see numpy.isclose).
        atol (float): The absolute tolerance parameter (see numpy.isclose).

    Returns:
        float or ndarray: The remainder after division.
    """
    if np.any(y == 0):
        return np.nan

    quotient = x / y
    closest_int = np.round(quotient)

    # check if quotient is close to an integer value
    if np.isclose(quotient, closest_int, rtol=rtol, atol=atol).all():
        return np.zeros_like(x)

    remainder = x - np.fix(quotient) * y

    return remainder


def matlab_assign(matrix: np.ndarray, indices: Union[int, np.ndarray], values: Union[int, float, np.ndarray]) -> np.ndarray:
    """
    Assigns values to elements of a matrix using subscript indices.

    Args:
        matrix: The matrix to which values will be assigned.
        indices: The subscript indices of the elements to be assigned. Can be a single integer or a NumPy array.
        values: The values to be assigned. Can be a single integer, float, or a NumPy array.

    Returns:
        The modified matrix.

    """
    original_shape = matrix.shape
    matrix = np.ravel(matrix, order="F")
    matrix[indices] = values
    return matrix.reshape(original_shape, order="F")


def matlab_find(arr: Union[List[int], np.ndarray], val: int = 0, mode: str = "neq") -> np.ndarray:
    """
    Finds the indices of elements in an array that satisfy a given condition.

    Args:
        arr: The array to search. Can be a list or a NumPy array.
        val: The value to compare against. Default is 0.
        mode: The comparison mode. Can be either 'neq' (not equal) or 'eq' (equal). Default is 'neq'.

    Returns:
        A NumPy array of indices.

    """

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if mode == "neq":
        arr = np.where(arr.flatten(order="F") != val)[0] + 1  # +1 due to matlab indexing
    else:  # 'eq'
        arr = np.where(arr.flatten(order="F") == val)[0] + 1  # +1 due to matlab indexing
    return np.expand_dims(arr, -1)  # compatibility, n => [n, 1]


def matlab_mask(arr: np.ndarray, mask: np.ndarray, diff: Optional[int] = None) -> np.ndarray:
    """
    Applies a mask to an array and returns the masked elements.

    Args:
        arr: The array to be masked.
        mask: The mask array, which must be of the same shape as arr.
        diff: An optional integer to add to the mask indices before applying the mask.

    Returns:
        A NumPy array containing the masked elements.

    """

    if diff is None:
        return np.expand_dims(arr.ravel(order="F")[mask.ravel(order="F")], axis=-1)  # compatibility, n => [n, 1]
    else:
        return np.expand_dims(arr.ravel(order="F")[mask.ravel(order="F") + diff], axis=-1)  # compatibility, n => [n, 1]


def unflatten_matlab_mask(arr: np.ndarray, mask: np.ndarray, diff: Optional[int] = None) -> Tuple[Union[int, np.ndarray], ...]:
    """
    Converts a mask array to a tuple of subscript indices for an n-dimensional array.

    Args:
        arr: The n-dimensional array for which the mask was created.
        mask: The mask array, which can be of any dimensions.
        diff: An optional integer to add to the mask indices before converting them to subscript indices.

    Returns:
        A tuple of integers or NumPy arrays representing the corresponding subscript indices.

    """

    if diff is None:
        return np.unravel_index(mask.ravel(order="F"), arr.shape, order="F")
    else:
        return np.unravel_index(mask.ravel(order="F") + diff, arr.shape, order="F")


def ind2sub(array_shape: Tuple[int, ...], ind: int) -> Tuple[int, ...]:
    """
    Converts a linear index to a tuple of subscript indices for an n-dimensional array.

    Args:
        array_shape: A tuple of integers representing the shape of the array.
        ind: The linear index to be converted.

    Returns:
        A tuple of integers representing the corresponding subscript indices.

    """

    indices = np.unravel_index(ind - 1, array_shape, order="F")
    indices = (np.squeeze(index) + 1 for index in indices)
    return indices


def sub2ind(array_shape: Tuple[int, int, int], x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Convert 3D subscript indices to a linear index.

    This function converts 3D subscript indices to a linear index in a way that is consistent with the way
    that MATLAB handles indexing. The output is a 1D numpy array containing the linear indices.

    Args:
        array_shape: A tuple containing the shape of the array.
        x: A 1D numpy array of subscript indices for the x-dimension.
        y: A 1D numpy array of subscript indices for the y-dimension.
        z: A 1D numpy array of subscript indices for the z-dimension.

    Returns:
        A 1D numpy array containing the linear indices.

    """

    results = []
    x, y, z = np.squeeze(x), np.squeeze(y), np.squeeze(z)
    for x_i, y_i, z_i in zip(x, y, z):
        index = np.ravel_multi_index((x_i, y_i, z_i), dims=array_shape, order="F")
        results.append(index)
    return np.array(results)
