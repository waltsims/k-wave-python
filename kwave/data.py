from dataclasses import dataclass
from typing import Any, Optional

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


@dataclass
class SimulationResult:
    """
    Structured return type for kWave simulation results.

    Contains all possible fields that can be returned by the kWave C++ binaries.
    Fields are populated based on the sensor.record configuration.
    """

    # Grid information (always present)
    Nx: int
    Ny: int
    Nz: int
    Nt: int
    pml_x_size: int
    pml_y_size: int
    pml_z_size: int
    axisymmetric_flag: bool

    # Pressure fields (optional - based on sensor.record)
    p_raw: Optional[np.ndarray] = None
    p_max: Optional[np.ndarray] = None
    p_min: Optional[np.ndarray] = None
    p_rms: Optional[np.ndarray] = None
    p_max_all: Optional[np.ndarray] = None
    p_min_all: Optional[np.ndarray] = None
    p_final: Optional[np.ndarray] = None

    # Velocity fields (optional - based on sensor.record)
    u_raw: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    u_min: Optional[np.ndarray] = None
    u_rms: Optional[np.ndarray] = None
    u_max_all: Optional[np.ndarray] = None
    u_min_all: Optional[np.ndarray] = None
    u_final: Optional[np.ndarray] = None
    u_non_staggered_raw: Optional[np.ndarray] = None

    # Intensity fields (optional - based on sensor.record)
    I_avg: Optional[np.ndarray] = None
    I: Optional[np.ndarray] = None

    def __getitem__(self, key: str):
        """
        Enable dictionary-style access for backward compatibility.

        Args:
            key: Field name to access

        Returns:
            Value of the field

        Raises:
            KeyError: If the field does not exist
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' field not found in SimulationResult")

    def __contains__(self, key: str) -> bool:
        """
        Enable dictionary-style membership testing for backward compatibility.

        Args:
            key: Field name to check

        Returns:
            True if the field exists, False otherwise
        """
        return hasattr(self, key)

    @classmethod
    def from_dotdict(cls, data: dict) -> "SimulationResult":
        """
        Create SimulationResult from dotdict returned by parse_executable_output.

        Args:
            data: Dictionary containing simulation results from HDF5 file

        Returns:
            SimulationResult instance with all available fields populated
        """
        return cls(
            # Grid information
            Nx=int(data.get("Nx", 0)),
            Ny=int(data.get("Ny", 0)),
            Nz=int(data.get("Nz", 0)),
            Nt=int(data.get("Nt", 0)),
            pml_x_size=int(data.get("pml_x_size", 0)),
            pml_y_size=int(data.get("pml_y_size", 0)),
            pml_z_size=int(data.get("pml_z_size", 0)),
            axisymmetric_flag=bool(data.get("axisymmetric_flag", False)),
            # Pressure fields
            p_raw=data.get("p_raw"),
            p_max=data.get("p_max"),
            p_min=data.get("p_min"),
            p_rms=data.get("p_rms"),
            p_max_all=data.get("p_max_all"),
            p_min_all=data.get("p_min_all"),
            p_final=data.get("p_final"),
            # Velocity fields
            u_raw=data.get("u_raw"),
            u_max=data.get("u_max"),
            u_min=data.get("u_min"),
            u_rms=data.get("u_rms"),
            u_max_all=data.get("u_max_all"),
            u_min_all=data.get("u_min_all"),
            u_final=data.get("u_final"),
            u_non_staggered_raw=data.get("u_non_staggered_raw"),
            # Intensity fields
            I_avg=data.get("I_avg"),
            I=data.get("I"),
        )
