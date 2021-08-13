from typing import List

import numpy as np


def enforce_fields(dictionary, *fields):
    # from kwave
    for f in fields:
        assert f in dictionary.keys(), [f'The field {f} must be defined in the given dictionary']


def enforce_fields_obj(obj, *fields):
    # from kwave
    for f in fields:
        assert getattr(obj, f) is not None, f'The field {f} must be not None in the given object'


def check_field_names(dictionary, *fields):
    # from kwave
    for k in dictionary.keys():
        assert k in fields, f'The field {k} is not a valid field for the given dictionary'


def num_dim(x):
    # get the size collapsing any singleton dimensions
    return len(np.squeeze(x).shape)

    # check for 1D vectors
    # if len(sz) > 2:
    #     dim = len(sz)
    # elif len(sz) == 0:
    #     dim = 1
    # elif sz[0] == 1 or sz[1] == 1:
    #     dim = 1
    # else:
    #     dim = 2
    # return dim


def num_dim2(x: np.ndarray):
    # get the size collapsing any singleton dimensions
    sz = np.squeeze(x).shape

    if len(sz) > 2:
        return len(sz)
    else:
        return np.sum(np.array(sz) > 1)


def check_str_eq(value, target: str):
    """
        String equality check only if the value is string. Helps to avoid FutureWarnings when value is not a string.
        Added by @Farid
    Args:
        value:
        target:

    Returns:

    """
    return isinstance(value, str) and value == target


def check_str_in(value, target: List[str]):
    """
        Check if value is in the given list only if the value is string.
        Helps to avoid FutureWarnings when value is not a string.
        Added by @Farid
    Args:
        value:
        target:

    Returns:

    """
    # added by Farid
    return isinstance(value, str) and value in target


def is_number(value):
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return False
    if value.dtype in [np.float32, np.float64]:
        return True
    return np.issubdtype(np.array(value), np.number)