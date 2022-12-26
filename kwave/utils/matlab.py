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


def ind2sub(array_shape, ind):
    # Matlab style ind2sub
    # row, col = np.unravel_index(ind - 1, array_shape, order='F')
    # return np.squeeze(row) + 1, np.squeeze(col) + 1

    indices = np.unravel_index(ind - 1, array_shape, order='F')
    indices = (np.squeeze(index) + 1 for index in indices)
    return indices


def sub2ind(array_shape, x, y, z) -> np.ndarray:
    results = []
    x, y, z = np.squeeze(x), np.squeeze(y), np.squeeze(z)
    for x_i, y_i, z_i in zip(x, y, z):
        index = np.ravel_multi_index((x_i, y_i, z_i), dims=array_shape, order='F')
        results.append(index)
    return np.array(results)
