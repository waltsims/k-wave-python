from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from tempfile import gettempdir
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Found here: https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
    from kwave.kgrid import kWaveGrid
from kwave.utils.data import get_date_string
from kwave.utils.io import get_h5_literals
from kwave.utils.pml import get_optimal_pml_size


class SimulationType(Enum):
    """
    Enum for the simulation type

    In the original matlab code, simulation type was determined
        by looking at the calling function name and the user args.

    Rules from the original matlab code:
        AXISYMMETRIC => if the calling function name started with 'kspaceFirstOrderAS'
                            or if the userarg_axisymmetric is set to true
        ELASTIC => if the calling function name started with 'pstdElastic' or 'kspaceElastic'
        ELASTIC_WITH_KSPACE_CORRECTION => if the calling function name started with 'kspaceElastic'
    """

    FLUID = 1
    AXISYMMETRIC = 2
    ELASTIC = 3
    ELASTIC_WITH_KSPACE_CORRECTION = 4

    def is_elastic_simulation(self):
        return self in [SimulationType.ELASTIC, SimulationType.ELASTIC_WITH_KSPACE_CORRECTION]

    def is_axisymmetric(self):
        return self == SimulationType.AXISYMMETRIC


@dataclass
class SimulationOptions(object):
    """
    Args:
        axisymmetric: Flag that indicates whether axisymmetric simulation is used
        cart_interp: Interpolation mode used to extract the pressure when a Cartesian sensor mask is given.
                     If set to 'nearest' and more than one Cartesian point maps to the same grid point,
                     duplicated data points are discarded and sensor_data will be returned
                     with less points than that specified by sensor.mask (default = 'linear').
        pml_inside: put the PML inside the grid defined by the user
        pml_alpha: Absorption within the perfectly matched layer in Nepers per grid point (default = 2).
        save_to_disk: save the input data to a HDF5 file
        save_to_disk_exit: exit the simulation after saving the HDF5 file
        scale_source_terms: apply the source scaling term to time varying sources
        smooth_c0: smooth the sound speed distribution
        smooth_rho0: smooth the density distribution
        smooth_p0: smooth the initial pressure distribution
        use_kspace: use the k-space correction
        use_sg: use a staggered grid
        use_fd: Use finite difference gradients instead of spectral (in 1D)
        pml_auto: automatically choose the PML size to give small prime factors
        create_log: create a log using diary
        use_finite_difference: use finite difference gradients instead of spectral (in 1D)
        stream_to_disk:  String containing a filename (including pathname if required).
                         If set, after the precomputation phase, the input variables used in the time loop are saved
                         the specified location in HDF5 format. The simulation then exits.
                         The saved variables can be used to run simulations using the C++ code.
        data_recast: recast the sensor data back to double precision
        cartesian_interp: interpolation mode for Cartesian sensor mask
        hdf_compression_level: zip compression level for HDF5 input files
        data_cast: data cast
        pml_search_range: search range used when automatically determining PML size
        radial_symmetry: radial symmetry used in axisymmetric code
        multi_axial_PML_ratio: MPML settings
        pml_x_alpha: PML Alpha for x-axis
        pml_y_alpha: PML Alpha for y-axis
        pml_z_alpha: PML Alpha for z-axis
        pml_x_size: PML Size for x-axis
        pml_y_size: PML Size for y-axis
        pml_z_size: PML Size for z-axis
    """

    simulation_type: SimulationType = SimulationType.FLUID
    cart_interp: str = "linear"
    pml_inside: bool = True
    pml_alpha: float = 2.0
    save_to_disk: bool = False
    save_to_disk_exit: bool = False
    scale_source_terms: bool = True
    smooth_c0: bool = False
    smooth_rho0: bool = False
    smooth_p0: bool = True
    use_kspace: bool = True
    use_sg: bool = True
    use_fd: Optional[int] = None
    pml_auto: bool = False
    create_log: bool = False
    use_finite_difference: bool = False
    stream_to_disk: bool = False
    data_recast: Optional[bool] = False
    cartesian_interp: str = "linear"
    hdf_compression_level: Optional[int] = None
    data_cast: str = "off"
    pml_search_range: List[int] = field(default_factory=lambda: [10, 40])
    radial_symmetry: str = "WSWA-FFT"
    multi_axial_PML_ratio: float = 0.1
    data_path: Optional[str] = field(default_factory=lambda: gettempdir())
    input_filename: Optional[str] = field(default_factory=lambda: f"{get_date_string()}_kwave_input.h5")
    output_filename: Optional[str] = field(default_factory=lambda: f"{get_date_string()}_kwave_output.h5")
    pml_x_alpha: Optional[float] = None
    pml_y_alpha: Optional[float] = None
    pml_z_alpha: Optional[float] = None
    pml_size: Optional[List[int]] = None
    pml_x_size: Optional[int] = None
    pml_y_size: Optional[int] = None
    pml_z_size: Optional[int] = None

    def __post_init__(self):
        assert self.cartesian_interp in [
            "linear",
            "nearest",
        ], "Optional input ''cartesian_interp'' must be set to ''linear'' or ''nearest''."

        assert isinstance(self.data_cast, str), "Optional input ''data_cast'' must be a string."

        assert self.data_cast in ["off", "double", "single"], "Invalid input for ''data_cast''."

        if self.data_cast == "double":
            self.data_cast = "off"

        # load the HDF5 literals (for the default compression level)
        h5_literals = get_h5_literals()
        self.hdf_compression_level = h5_literals.HDF_COMPRESSION_LEVEL
        # check value is an integer between 0 and 9
        assert (
            isinstance(self.hdf_compression_level, int) and 0 <= self.hdf_compression_level <= 9
        ), "Optional input ''hdf_compression_level'' must be an integer between 0 and 9."

        assert (
            np.isscalar(self.multi_axial_PML_ratio) and self.multi_axial_PML_ratio >= 0
        ), "Optional input ''multi_axial_PML_ratio'' must be a single positive value."

        assert np.isscalar(self.stream_to_disk) or isinstance(
            self.stream_to_disk, bool
        ), "Optional input ''stream_to_disk'' must be a single scalar or Boolean value."

        boolean_inputs = {
            "use_sg": self.use_sg,
            "data_recast": self.data_recast,
            "save_to_disk_exit": self.save_to_disk_exit,
            "use_kspace": self.use_kspace,
            "save_to_disk": self.save_to_disk,
            "pml_inside": self.pml_inside,
            "create_log": self.create_log,
            "scale_source_terms": self.scale_source_terms,
        }

        for key, val in boolean_inputs.items():
            assert isinstance(val, bool), f"Optional input ''{key}'' must be Boolean."

        assert self.radial_symmetry in [
            "WSWA",
            "WSWS",
            "WSWA-FFT",
            "WSWS-FFT",
        ], "Optional input ''RadialSymmetry'' must be set to ''WSWA'', ''WSWS'', ''WSWA-FFT'', ''WSWS-FFT''."

        # automatically assign the PML size to give small prime factors
        if self.pml_auto and self.pml_inside:
            raise NotImplementedError("''pml_size'' set to ''auto'' is only supported with ''pml_inside'' set to false.")

        if self.pml_size is not None:
            # TODO(walter): remove auto option in exchange for pml_auto=True
            if isinstance(self.pml_size, int):
                self.pml_size = np.array([self.pml_size])
            if not isinstance(self.pml_size, (list, np.ndarray)):
                raise ValueError("Optional input ''PMLSize'' must be a integer array of 1, 2 or 3 dimensions.")

        # Check if each member variable is None, and set it to self.pml_alpha if it is
        self.pml_x_alpha = self.pml_alpha if self.pml_x_alpha is None else self.pml_x_alpha
        self.pml_y_alpha = self.pml_alpha if self.pml_y_alpha is None else self.pml_y_alpha
        self.pml_z_alpha = self.pml_alpha if self.pml_z_alpha is None else self.pml_z_alpha

        # add pathname to input and output filenames
        self.input_filename = os.path.join(self.data_path, self.input_filename)
        self.output_filename = os.path.join(self.data_path, self.output_filename)

        assert self.use_fd is None or (
            np.issubdtype(self.use_fd, np.number) and self.use_fd in [2, 4]
        ), "Optional input ''UseFD'' can only be set to 2, 4."

    @staticmethod
    def option_factory(kgrid: "kWaveGrid", options: SimulationOptions):
        """
        Initialize the Simulation Options

        Args:
            kgrid: kWaveGrid instance
            elastic_code: Flag that indicates whether elastic simulation is used
            **kwargs: Dictionary that holds following optional simulation properties:

                * cart_interp: Interpolation mode used to extract the pressure when a Cartesian sensor mask is given.
                               If set to 'nearest', duplicated data points are discarded and sensor_data
                               will be returned with fewer points than specified by sensor.mask (default = 'linear').
                * create_log: Boolean controlling whether the command line output is saved using the diary function
                              with a date and time stamped filename (default = false).
                * data_cast: String input of the data type that variables are cast to before computation.
                             For example, setting to 'single' will speed up the computation time
                             (due to the improved efficiency of fftn and ifftn for this data type) at the expense
                             of a loss in precision. This variable is also useful for utilising GPU parallelisation
                             through libraries such as the Parallel Computing Toolbox
                             by setting 'data_cast' to 'gpuArray-single' (default = 'off').
                * data_recast: Boolean controlling whether the output data is cast back to double precision.
                               If set to false, sensor_data will be returned in
                               the data format set using the 'data_cast' option.
                * hdf_compression_level: Compression level used for writing the input HDF5 file when using
                                         'save_to_disk' or kspaceFirstOrder3DC. Can be set to an integer
                                         between 0 (no compression, the default) and 9 (maximum compression).
                                         The compression is lossless. Increasing the compression level will reduce
                                         the file size if there are portions of the medium that are homogeneous,
                                         but will also increase the time to create the HDF5 file.
                * multi_axial_pml_ratio: MPML settings
                * pml_alpha: Absorption within the perfectly matched layer in Nepers per grid point (default = 2).
                * pml_inside: Boolean controlling whether the perfectly matched layer is inside or outside the grid.
                              If set to false, the input grids are enlarged by pml_size
                              before running the simulation (default = true).
                * pml_range: Search range used when automatically determining PML size. Tuple of two elements
                * pml_size: Size of the perfectly matched layer in grid points. By default, the PML is added evenly to
                            all sides of the grid, however, both pml_size and pml_alpha can be given as three element
                            arrays to specify the x, y, and z properties, respectively.
                            To remove the PML, set the appropriate pml_alpha to zero rather than forcing
                            the PML to be of zero size (default = 10).
                * radial_symmetry: Radial symmetry used in axisymmetric code
                * stream_to_disk: Boolean controlling whether sensor_data is periodically saved to disk to avoid storing
                                  the complete matrix in memory. StreamToDisk may also be given as an integer which
                                  specifies the number of time steps that are taken before the data
                                  is saved to disk (default = 200).
                * save_to_disk: String containing a filename (including pathname if required).
                                If set, after the precomputation phase, the input variables used in the time loop are
                                saved the specified location in HDF5 format. The simulation then exits.
                                The saved variables can be used to run simulations using the C++ code.
                * save_to_disk_exit: Exit the simulation after saving the HDF5 file
                * scale_source_terms: Apply the source scaling term to time varying sources
                * use_fd: Use finite difference gradients instead of spectral (in 1D)
                * use_k_space: use the k-space correction
                * use_sg: Use a staggered grid


        Returns:
            SimulationOptions instance
        """

        STREAM_TO_DISK_STEPS_DEF = 200  # number of steps before streaming to disk

        if options.pml_size is not None and not isinstance(options.pml_size, bool):
            if len(options.pml_size) > kgrid.dim:
                if kgrid.dim > 1:
                    raise ValueError(f"Optional input ''pml_size'' must be a 1 or {kgrid.dim} element numerical array.")
                else:
                    raise ValueError("Optional input ''pml_size'' must be a single numerical value.")

        if kgrid.dim == 1:
            options.pml_x_size = options.pml_size if options.pml_size else 20
            options.plot_scale = [-1.1, 1.1]
        elif kgrid.dim == 2:
            if options.pml_size is not None:
                if len(options.pml_size) == kgrid.dim:
                    options.pml_x_size, options.pml_y_size = np.asarray(options.pml_size, dtype=int).ravel()
                else:
                    options.pml_x_size, options.pml_y_size = (options.pml_size[0], options.pml_size[0])
            else:
                options.pml_x_size, options.pml_y_size = (20, 20)
            options.plot_scale = [-1, 1]
        elif kgrid.dim == 3:
            if (options.pml_size is not None) and (len(options.pml_size) == kgrid.dim):
                options.pml_x_size, options.pml_y_size, options.pml_z_size = np.asarray(options.pml_size).ravel()
            else:
                if options.pml_size is None:
                    options.pml_x_size = 10
                    options.pml_y_size = 10
                    options.pml_z_size = 10
                else:
                    options.pml_x_size = options.pml_size[0]
                    options.pml_y_size = options.pml_x_size
                    options.pml_z_size = options.pml_x_size
                options.plot_scale = [-1, 1]

        # replace defaults with user defined values if provided and check inputs
        if (val := options.pml_alpha) is not None and not isinstance(options.pml_alpha, str):
            # check input is correct size
            val = np.atleast_1d(val)
            if val.size > kgrid.dim:
                if kgrid.dim > 1:
                    raise ValueError(f"Optional input ''pml_alpha'' must be a 1 or {kgrid.dim} element numerical array.")
                else:
                    raise ValueError("Optional input ''pml_alpha'' must be a single numerical value.")

            # assign input based on number of dimensions
            if kgrid.dim == 1:
                options.pml_x_alpha = val
            elif kgrid.dim == 2:
                options.pml_x_alpha = val[0]
                options.pml_y_alpha = val[-1]
            elif kgrid.dim == 2:
                options.pml_x_alpha = val[0]
                options.pml_y_alpha = val[len(val) // 2]
                options.pml_z_alpha = val[-1]

        if options.save_to_disk_exit:
            assert kgrid.dim != 1, "Optional input ''save_to_disk'' is not compatible with 1D simulations."

        if options.stream_to_disk:
            assert (
                not options.simulation_type.is_elastic_simulation() and kgrid.dim == 3
            ), "Optional input ''stream_to_disk'' is currently only compatible with 3D fluid simulations."
            # if given as a Boolean, replace with the default number of time steps
            if isinstance(options.stream_to_disk, bool) and options.stream_to_disk:
                options.stream_to_disk = STREAM_TO_DISK_STEPS_DEF

        if options.save_to_disk or options.save_to_disk_exit:
            assert kgrid.dim != 1, "Optional input ''save_to_disk'' is not compatible with 1D simulations."

        if options.use_fd:
            # input only supported in 1D fluid code
            assert kgrid.dim == 1 and not options.simulation_type.is_elastic_simulation(), "Optional input ''use_fd'' only supported in 1D."
        # get optimal pml size
        if options.simulation_type.is_axisymmetric() or options.pml_auto:
            if options.simulation_type.is_axisymmetric():
                pml_size_temp = get_optimal_pml_size(kgrid, options.pml_search_range, options.radial_symmetry[:4])
            else:
                pml_size_temp = get_optimal_pml_size(kgrid, options.pml_search_range)

            # assign to individual variables
            if kgrid.dim == 1:
                options.pml_x_size = int(pml_size_temp[0])
            elif kgrid.dim == 2:
                options.pml_x_size = int(pml_size_temp[0])
                options.pml_y_size = int(pml_size_temp[1])
            elif kgrid.dim == 3:
                options.pml_x_size = int(pml_size_temp[0])
                options.pml_y_size = int(pml_size_temp[1])
                options.pml_z_size = int(pml_size_temp[2])

            # cleanup unused variables
            del pml_size_temp
        return options
