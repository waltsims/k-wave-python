from dataclasses import dataclass
from typing import Any

import numpy as np


class Vector(np.ndarray):
    def __new__(cls, elements: list):
        assert 1 <= len(elements) <= 3
        elements = list(elements)
        obj = np.array(elements).view(cls)
        return obj

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        if self.shape[0] > 1:
            return self[1]
        return np.nan

    @y.setter
    def y(self, value):
        if self.shape[0] > 1:
            self[1] = value

    @property
    def z(self):
        if self.shape[0] > 2:
            return self[2]
        return np.nan

    @z.setter
    def z(self, value):
        if self.shape[0] > 2:
            self[2] = value

    def assign_dim(self, dim: int, val: Any):
        self[dim - 1] = val

    def append(self, value):
        new_coordinates = list(self) + [value]
        return Vector(new_coordinates)


@dataclass
class FlexibleVector(object):
    """
    This class is very similar to Numpy.ndarray but there are differences:
        - It can have 3 elements at max
        - Its elements can be anything
        - The elements do not have to be same type,
                e.g. this is valid: Array([<scalar>, <List>, <Tuple of Tuples>])

    WARNING: The class will be deprecated once we refactor the kWaveGrid class to use the Vector class instead!
    """

    data: list

    def __post_init__(self):
        assert 1 <= len(self) <= 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, val):
        self[0] = val

    @property
    def y(self):
        return self[1] if len(self) >= 2 else np.nan

    @y.setter
    def y(self, val):
        assert len(self) >= 2
        self[1] = val

    @property
    def z(self):
        """
        :return: 3rd dimension element. 0 if not defined
        """
        return self[2] if len(self) == 3 else np.nan

    @z.setter
    def z(self, val):
        assert len(self) == 3
        self[2] = val

    def numpy(self):
        return np.asarray(self.data)

    def assign_dim(self, dim, val):
        if dim == 1:
            self.x = val
        if dim == 2:
            self.y = val
        if dim == 3:
            self.z = val

    def append(self, val):
        assert len(self.data) <= 2
        self.data.append(val)
        return self
