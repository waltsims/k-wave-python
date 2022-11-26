import numpy as np


def get_smallest_possible_type(max_array_val, target_type_group, default=None):
    types = {'uint', 'int'}
    assert target_type_group in types

    for bit_count in [8, 16, 32]:
        type_ = f'{target_type_group}{bit_count}'
        if max_array_val < intmax(type_):
            return type_

    return default


def intmax(dtype: str):
    return np.iinfo(getattr(np, dtype)).max