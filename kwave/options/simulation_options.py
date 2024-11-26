from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from tempfile import gettempdir
from typing import Literal, Optional, Union
from beartype.typing import List, Tuple

from beartype import beartype
import numpy as np

from kwave.utils.data import get_date_string
from kwave.utils.io import get_h5_literals
from kwave.utils.pml import get_optimal_pml_size
from kwave.utils.typing import ArrayLike
from kwave.data import Vector


class SimulationType(str, Enum):
    """Simulation type with clearer string representations"""

    FLUID = "fluid"
    AXISYMMETRIC = "axisymmetric"
    ELASTIC = "elastic"
    ELASTIC_WITH_KSPACE = "elastic_with_kspace"

    @property
    def is_elastic(self) -> bool:
        return self in [self.ELASTIC, self.ELASTIC_WITH_KSPACE]

    @property
    def is_axisymmetric(self) -> bool:
        return self == self.AXISYMMETRIC


InterpolationMode = Literal["linear", "nearest"]
RadialSymmetry = Literal["WSWA", "WSWS", "WSWA-FFT", "WSWS-FFT"]
DataCastMode = Literal["off", "single", "double"]


@beartype
@dataclass
class SimulationOptions:
    """Configuration options for k-Wave simulations."""

    # Core simulation settings
    simulation_type: SimulationType = SimulationType.FLUID
    dimension: Optional[int] = None

    # Interpolation settings
    cart_interp: InterpolationMode = "linear"
    radial_symmetry: RadialSymmetry = "WSWA-FFT"

    # Computation flags
    use_kspace: bool = True
    use_sg: bool = True  # staggered grid
    use_fd: Optional[int] = None
    scale_source_terms: bool = True
    data_cast: DataCastMode = "off"
    data_recast: bool = False

    # Smoothing flags
    smooth_p0: bool = True
    smooth_c0: bool = False
    smooth_rho0: bool = False

    # PML Configuration
    pml_inside: bool = True
    pml_auto: bool = False
    pml_alpha: Union[ArrayLike, Vector] = field(default_factory=lambda: np.array([2.0]))
    pml_search_range: Union[ArrayLike, Vector] = field(default_factory=lambda: [10, 40])
    pml_size: Union[ArrayLike, Vector] = field(default_factory=lambda: [20])
    pml_x_size: int = 20
    pml_y_size: int = 20
    pml_z_size: int = 10
    pml_x_alpha: float = field(init=False)
    pml_y_alpha: float = field(init=False)
    pml_z_alpha: float = field(init=False)
    pml_multi_axial_ratio: float = 0.1

    # File Configuration
    data_path: str = field(default_factory=gettempdir)
    input_filename: str = field(default_factory=lambda: f"{get_date_string()}_kwave_input.h5")
    output_filename: str = field(default_factory=lambda: f"{get_date_string()}_kwave_output.h5")
    save_to_disk: bool = False
    save_to_disk_exit: bool = False
    stream_to_disk: bool = False
    stream_to_disk_steps: int = 200
    compression_level: int = field(default_factory=lambda: get_h5_literals().HDF_COMPRESSION_LEVEL)

    def __post_init__(self) -> None:
        """Validate the simulation options"""
        self._validate_pml()
        self._validate_files()
        self._validate_dimension()
        self._validate_fd()

    def _validate_pml(self) -> None:
        """Validate PML configuration"""
        # Set PML alphas
        self.pml_x_alpha = self.pml_alpha
        self.pml_y_alpha = self.pml_alpha
        self.pml_z_alpha = self.pml_alpha

        if isinstance(self.pml_size, int):
            if self.pml_size < 0:
                raise ValueError("PML size value must be positive")
        else:
            if not all(size > 0 for size in self.pml_size):
                raise ValueError("All PML size values must be positive")

        # Validate other parameters
        if self.pml_auto and self.pml_inside:
            raise ValueError("'pml_size=auto' requires 'pml_inside=False'")

        if not (isinstance(self.pml_multi_axial_ratio, (int, float)) and self.pml_multi_axial_ratio >= 0):
            raise ValueError("pml_multi_axial_ratio must be a non-negative number")

    def _validate_files(self) -> None:
        """Validate file configuration"""
        if not (0 <= self.compression_level <= 9):
            raise ValueError("Compression level must be between 0 and 9")

    def _validate_dimension(self) -> None:
        """Validate grid dimension"""
        if self.dimension is not None and not (1 <= self.dimension <= 3):
            raise ValueError("Grid dimension must be 1, 2, or 3")

    def _validate_fd(self) -> None:
        """Validate finite difference settings"""
        if self.use_fd is not None:
            if self.use_fd not in (2, 4):
                raise ValueError("Finite difference order must be 2 or 4")
            if self.dimension != 1:
                raise ValueError("Finite difference is only supported in 1D")
            if self.simulation_type.is_elastic:
                raise ValueError("Finite difference is not supported for elastic simulations")

    def get_file_paths(self) -> Tuple[str, str]:
        """Get full paths for input and output files"""
        return (os.path.join(self.data_path, self.input_filename), os.path.join(self.data_path, self.output_filename))

    def configure_dimensions(self, kgrid: "kWaveGrid") -> None:
        """Configure dimension-dependent settings"""
        self.dimension = kgrid.dim

        # Configure PML sizes based on dimension
        self._configure_pml_size(kgrid)

        # Configure automatic PML if needed
        if self.simulation_type.is_axisymmetric or self.pml_auto:
            self._configure_automatic_pml(kgrid)

        # Validate dimension-specific constraints
        self._validate_dimension_constraints()

    def _configure_pml_size(self, kgrid: "kWaveGrid") -> None:
        """Configure PML sizes based on grid dimensions"""
        # Convert single int to list if needed
        if isinstance(self.pml_size, int):
            self.pml_size = [self.pml_size]

        if len(self.pml_size) > kgrid.dim:
            raise ValueError(f"PML size must be 1 or {kgrid.dim} elements")

        # Set dimension-specific PML sizes
        if kgrid.dim == 1:
            self.pml_x_size = self.pml_size[0]
        elif kgrid.dim == 2:
            if len(self.pml_size) == 2:
                self.pml_x_size, self.pml_y_size = self.pml_size
            else:
                self.pml_x_size = self.pml_y_size = self.pml_size[0]
        else:  # 3D
            if len(self.pml_size) == 3:
                self.pml_x_size, self.pml_y_size, self.pml_z_size = self.pml_size
            else:
                self.pml_x_size = self.pml_y_size = self.pml_z_size = self.pml_size[0]

    def _configure_automatic_pml(self, kgrid: "kWaveGrid") -> None:
        """Configure automatic PML sizes"""
        pml_size = get_optimal_pml_size(
            kgrid, self.pml_search_range, self.radial_symmetry[:4] if self.simulation_type.is_axisymmetric else None
        )

        if self.dimension == 1:
            self.pml_x_size = int(pml_size[0])
        elif self.dimension == 2:
            self.pml_x_size, self.pml_y_size = map(int, pml_size)
        else:  # 3D
            self.pml_x_size, self.pml_y_size, self.pml_z_size = map(int, pml_size)

    def _validate_dimension_constraints(self) -> None:
        """Validate dimension-specific constraints"""
        if self.dimension == 1:
            if self.save_to_disk or self.save_to_disk_exit:
                raise ValueError("save_to_disk is not compatible with 1D simulations")

        if self.stream_to_disk:
            if self.simulation_type.is_elastic or self.dimension != 3:
                raise ValueError("stream_to_disk is only compatible with 3D fluid simulations")
