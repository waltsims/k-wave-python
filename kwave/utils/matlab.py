import numpy as np


def matlab_assign(matrix: np.ndarray, indices, values):
    original_shape = matrix.shape
    matrix = matrix.flatten(order='F')
    matrix[indices] = values
    return matrix.reshape(original_shape, order='F')


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
 