import numbers
import os
from dataclasses import dataclass, field
from tempfile import gettempdir
from typing import List, Optional

import numpy as np

from kwave.utils import get_h5_literals, get_date_string
from kwave.utils import get_optimal_pml_size


@dataclass
class SimulationOptions(object):
    """
    Args:
        axisymmetric: Flag that indicates whether axisymmetric simulation is used
        cart_interp: Interpolation mode used to extract the pressure when a Cartesian sensor mask is given. If set to 'nearest' and more than one Cartesian point maps to the same grid point, duplicated data points are discarded and sensor_data will be returned with less points than that specified by sensor.mask (default = 'linear').
        pml_inside: put the PML inside the grid defined by the user
        pml_alpha: Absorption within the perfectly matched layer in Nepers per grid point (default = 2).
        save_to_disk: save the input data to a HDF5 file
        save_to_disk_exit: exit the simulation after saving the HDF5 file
        scale_source_terms: apply the source scaling term to time varying sources
        # TODO(Walter): work out directional smoothing logic
        smooth: Boolean controlling whether source.p0, medium.sound_speed, and medium.density are smoothed using smooth before computation. 'Smooth' can either be given as a single Boolean value or as a 3 element array to control the smoothing of source.p0, medium.sound_speed, and medium.density, independently (default = [true, false, false]).
        smooth_c0: smooth the sound speed distribution
        smooth_rho0: smooth the density distribution
        smooth_p0: smooth the initial pressure distribution
        use_kspace: use the k-space correction
        use_sg: use a staggered grid
        use_fd: Use finite difference gradients instead of spectral (in 1D)
        pml_auto: automatically choose the PML size to give small prime factors
        create_log: create a log using diary
        use_finite_difference: use finite difference gradients instead of spectral (in 1D)
        stream_to_disk: buffer the sensor data to disk (in 3D)
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

    # FLAGS WHICH CAN BE CONTROLLED WITH OPTIONAL INPUTS (THESE CAN BE MODIFIED)
    # flags which control the behaviour of the simulations
    axisymmetric: bool = False
    cart_interp: str = 'linear'
    pml_inside: Optional[bool] = None
    pml_alpha: float = 2.0
    save_to_disk: bool = False
    save_to_disk_exit: bool = True
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

    # VARIABLES THAT CAN BE CHANGED USING OPTIONAL INPUTS (THESE CAN BE MODIFIED)
    # general settings
    cartesian_interp: str = 'linear'
    hdf_compression_level: Optional[int] = None
    data_cast: str = 'off'
    pml_search_range: List[int] = field(default_factory=lambda: [10, 40])
    radial_symmetry: str = 'WSWA-FFT'
    multi_axial_PML_ratio: float = 0.1
    data_path: str = gettempdir()
    data_name: Optional[str] = None
    input_filename: Optional[str] = None
    output_filename: Optional[str] = None

    pml_x_alpha: Optional[float] = None
    pml_y_alpha: Optional[float] = None
    pml_z_alpha: Optional[float] = None
    pml_size: Optional[int] = None
    pml_x_size: Optional[int] = None
    pml_y_size: Optional[int] = None
    pml_z_size: Optional[int] = None

    def __post_init__(self):
        assert self.cartesian_interp in ['linear', 'nearest'], \
            "Optional input ''CartInterp'' must be set to ''linear'' or ''nearest''."

        assert isinstance(self.create_log, bool), "Optional input ''CreateLog'' must be Boolean."

        assert isinstance(self.data_cast, str), "Optional input ''DataCast'' must be a string."

        assert self.data_cast in ['off', 'double', 'single', 'gpuArray-single', 'gpuArray-double'], \
            "Invalid input for ''DataCast''."
        # replace double with off
        if self.data_cast == 'double':
            self.data_cast = 'off'

        # replace PCT options with gpuArray
        if self.data_cast == 'gpuArray-single':
            self.data_cast = 'gpuArray'
            self.data_cast_prepend = 'single'
        elif self.data_cast == 'gpuArray-double':
            self.data_cast = 'gpuArray'
        if self.data_cast == 'gpuArray':
            raise NotImplementedError("gpuArray is not supported in Python-version")

        assert isinstance(self.data_recast, bool), "Optional input ''DataRecast'' must be Boolean."

        assert np.isscalar(self.stream_to_disk) or isinstance(self.stream_to_disk, bool), \
            "Optional input ''StreamToDisk'' must be a single scalar or Boolean value."


        # TODO: figure this logic out
        # assert isinstance(self.pml_inside, bool), "Optional input ''PMLInside'' must be Boolean."

        # load the HDF5 literals (for the default compression level)
        h5_literals = get_h5_literals()
        self.hdf_compression_level = h5_literals.HDF_COMPRESSION_LEVEL
        # check value is an integer between 0 and 9
        assert isinstance(self.hdf_compression_level, int) and 0 <= self.hdf_compression_level <= 9, \
            "Optional input ''HDFCompressionLevel'' must be an integer between 0 and 9."

        assert np.isscalar(self.multi_axial_PML_ratio) and self.multi_axial_PML_ratio >= 0, \
            "Optional input ''MultiAxialPMLRatio'' must be a single positive value."
        assert isinstance(self.use_sg, bool), "Optional input ''use_sg'' must be Boolean."

        assert isinstance(self.save_to_disk_exit, bool), "Optional input ''save_to_disk_exit'' must be Boolean."

        assert isinstance(self.use_kspace, bool), "Optional input ''UsekSpace'' must be Boolean."

        assert isinstance(self.scale_source_terms, bool), "Optional input ''ScaleSourceTerms'' must be Boolean."

        assert self.radial_symmetry in ['WSWA', 'WSWS', 'WSWA-FFT', 'WSWS-FFT'], \
            "Optional input ''RadialSymmetry'' must be set to ''WSWA'', ''WSWS'', ''WSWA-FFT'', ''WSWS-FFT''."

        assert isinstance(self.save_to_disk_exit,
                          (bool, str)), "Optional input ''save_to_disk'' must be Boolean or a String."
        # automatically assign the PML size to give small prime factors
        if self.pml_auto:
            if self.pml_inside:
                raise NotImplementedError(
                    "''PMLSize'' set to ''auto'' is only supported with ''PMLInside'' set to false.")

        if self.pml_size is not None:
            # TODO(walter): remove auto option in exchange for pml_auto=True
            if isinstance(self.pml_size, str):
                raise ValueError(f"Optional input ''PMLSize'' must be a integer array of 1, 2 or 3 dimensions.")
            if isinstance(self.pml_size, float):
                raise ValueError(f"Optional input ''PMLSize'' must be a integer array of 1, 2 or 3 dimensions.")
            if isinstance(self.pml_size, int):
                self.pml_size = np.array([self.pml_size])

        # Check if each member variable is None, and set it to self.pml_alpha if it is
        self.pml_x_alpha = self.pml_alpha if self.pml_x_alpha is None else self.pml_x_alpha
        self.pml_y_alpha = self.pml_alpha if self.pml_y_alpha is None else self.pml_y_alpha
        self.pml_z_alpha = self.pml_alpha if self.pml_z_alpha is None else self.pml_z_alpha

        # unimplimented_params = ['display_mask', 'log_scale', 'mesh_plot', 'plot_freq',
        #              'plot_layout', 'plot_scale', 'plot_sim', 'plot_pml']
        # for param in unimplimented_params:
        # raise NotImplementedError(f'Plotting is not supported! Parameter {key} is related to plotting.')
        # check for a user defined location for the input and output files

        # check for a user defined name for the input and output files
        if self.data_name:
            name_prefix = self.data_name
            input_filename = f'{name_prefix}_input.h5'
            output_filename = f'{name_prefix}_output.h5'
        else:
            # set the filename inputs to store data in the default temp directory
            date_string = get_date_string()
            input_filename = 'kwave_input_data' + date_string + '.h5'
            output_filename = 'kwave_output_data' + date_string + '.h5'

        # add pathname to input and output filenames
        self.input_filename = os.path.join(self.data_path, input_filename)
        self.output_filename = os.path.join(self.data_path, output_filename)

    @staticmethod
    def option_factory(kgrid, elastic_code: bool, axisymmetric: bool, **kwargs):
        """
            Initialize the Simulation Options

        Args:
            kgrid: kWaveGrid instance
            elastic_code: Flag that indicates whether elastic simulation is used
            **kwargs: Dictionary that holds following optional simulation properties:

                * CartInterp: Interpolation mode used to extract the pressure when a Cartesian sensor mask is given. If set to 'nearest' and more than one Cartesian point maps to the same grid point, duplicated data points are discarded and sensor_data will be returned with less points than that specified by sensor.mask (default = 'linear').
                * CreateLog: Boolean controlling whether the command line output is saved using the diary function with a date and time stamped filename (default = false).
                * DataCast: String input of the data type that variables are cast to before computation. For example, setting to 'single' will speed up the computation time (due to the improved efficiency of fftn and ifftn for this data type) at the expense of a loss in precision.  This variable is also useful for utilising GPU parallelisation through libraries such as the Parallel Computing Toolbox by setting 'DataCast' to 'gpuArray-single' (default = 'off').
                * DataRecast: Boolean controlling whether the output data is cast back to double precision. If set to false, sensor_data will be returned in the data format set using the 'DataCast' option.
                * HDFCompressionLevel: Compression level used for writing the input HDF5 file when using 'save_to_dis
' or kspaceFirstOrder3DC. Can be set to an integer between 0 (no compression, the default) and 9 (maximum compression). The compression is lossless. Increasing the compression level will reduce the file size if there are portions of the medium that are homogeneous, but will also increase the time to create the HDF5 file.
                * MultiAxialPMLRatio: MPML settings
                * PMLAlpha: Absorption within the perfectly matched layer in Nepers per grid point (default = 2).
                * PMLInside: Boolean controlling whether the perfectly matched layer is inside or outside the grid. If set to false, the input grids are enlarged by PMLSize before running the simulation (default = true).
                * PMLRange: Search range used when automatically determining PML size. Tuple of two elements
                * PMLSize: Size of the perfectly matched layer in grid  points. By default, the PML is added evenly to all sides of the grid, however, both PMLSize and PMLAlpha can be given as three element arrays to specify the x, y, and z properties, respectively. To remove the PML, set the appropriate PMLAlpha to zero rather than forcing the PML to be of zero size (default = 10).
                * RadialSymmetry: Radial symmetry used in axisymmetric code
                * StreamToDisk: Boolean controlling whether sensor_data is periodically saved to disk to avoid storing the complete matrix in memory. StreamToDisk may also be given as an integer which specifies the number of times steps that are taken before the data is saved to disk (default = 200).
                * save_to_dis
: String containing a filename (including  pathname if required). If set, after the precomputation phase, the input variables used in the time loop are saved the specified location in HDF5 format. The simulation then exits. The saved variables can be used to run simulations using the C++ code.
                * save_to_dis
Exit: Exit the simulation after saving the HDF5 file
                * ScaleSourceTerms: Apply the source scaling term to time varying sources
                * UseFD:
                * UsekSpace: use the k-space correction
                * UseSG: Use a staggered grid

        Returns:
            SimulationOptions instance
        """
        # =========================================================================
        # FIXED LITERALS USED IN THE CODE (THESE CAN BE MODIFIED)
        # =========================================================================

        # Literals used to set default parameters end with _DEF. These are cleared
        # at the end of kspaceFirstOrder_inputChecking. Literals used at other
        # places in the code are not cleared.

        # general
        STREAM_TO_DISK_STEPS_DEF = 200  # number of steps before streaming to disk

        options = SimulationOptions(**kwargs)

        if options.pml_size is not None or not isinstance(options.pml_size, bool):
            options.pml_size = np.atleast_1d(options.pml_size)
            # TODO: pml_size option must always be an np.array OR ensure
            if len(options.pml_size) > kgrid.dim:
                if kgrid.dim > 1:
                    raise ValueError(
                        f"Optional input ''PMLSize'' must be a 1 or {kgrid.dim} element numerical array.")
                else:
                    raise ValueError(f"Optional input ''PMLSize'' must be a single numerical value.")

        if kgrid.dim == 1:
            options.pml_x_alpha = 2
            options.pml_x_size = options.pml_size if options.pml_size else 20
            options.plot_scale = [-1.1, 1.1]
        elif kgrid.dim == 2:
            options.pml_x_alpha = 2
            options.pml_y_alpha = options.pml_x_alpha
            options.pml_x_size = options.pml_size if options.pml_size else 20
            options.pml_y_size = options.pml_x_size
            options.plot_scale = [-1, 1]
        elif kgrid.dim == 3:
            # TODO: take into acount multi dimensional pml size case
            if len(options.pml_size) == kgrid.dim:
                options.pml_x_size, options.pml_y_size, options.pml_z_size = options.pml_size.ravel()
            else:
                options.pml_x_alpha = 2
                options.pml_y_alpha = options.pml_x_alpha
                options.pml_z_alpha = options.pml_x_alpha
                options.pml_x_size = options.pml_size if isinstance(options.pml_size, numbers.Number) else 10
                options.pml_y_size = options.pml_x_size
                options.pml_z_size = options.pml_x_size
                options.plot_scale = [-1, 1]

        # replace defaults with user defined values if provided and check inputs
        for key, val in kwargs.items():
            if key == 'pml_alpha':
                # check input is correct size
                val = np.atleast_1d(val)
                if val.size > kgrid.dim:
                    if kgrid.dim > 1:
                        raise ValueError(
                            f"Optional input ''PMLAlpha'' must be a 1 or {kgrid.dim} element numerical array.")
                    else:
                        raise ValueError(f"Optional input ''PMLAlpha'' must be a single numerical value.")

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

            elif options.save_to_disk_exit:
                assert kgrid.dim != 1, "Optional input ''save_to_disk'' is not compatible with 1D simulations."

            elif options.stream_to_disk:
                assert not options.elastic_code and kgrid.dim == 3, \
                    "Optional input ''stream_to_disk'' is currently only compatible with 3D fluid simulations."
                # if given as a Boolean, replace with the default number of time steps
                if isinstance(options.stream_to_disk, bool) and options.stream_to_disk:
                    options.stream_to_disk = STREAM_TO_DISK_STEPS_DEF

            elif options.save_to_disk or options.save_to_disk_exit:
                assert kgrid.dim != 1, "Optional input ''save_to_disk'' is not compatible with 1D simulations."

            else:
                # raise NotImplementedError(f"Unknown optional input: {key}.")
                pass

            assert options.use_fd is None or (np.issubdtype(options.use_fd, np.number) and options.use_fd in [2, 4]), \
                "Optional input ''UseFD'' can only be set to 2, 4."

            if options.use_fd:
                # input only supported in 1D fluid code
                assert kgrid.dim == 1 and not options.elastic_code, "Optional input ''UseFD'' only supported in 1D."
            # get optimal pml size
            if options.axisymmetric or options.pml_auto:
                pml_size_temp = get_optimal_pml_size(kgrid, options.pml_search_range, options.radial_symmetry[:4])

                # assign to individual variables
                if kgrid.dim == 1:
                    options.pml_x_size = float(pml_size_temp[0])
                elif kgrid.dim == 2:
                    options.pml_x_size = float(pml_size_temp[1])
                    options.pml_y_size = float(pml_size_temp[2])
                elif kgrid.dim == 3:
                    options.pml_x_size = float(pml_size_temp[0])
                    options.pml_y_size = float(pml_size_temp[1])
                    options.pml_z_size = float(pml_size_temp[2])

                # cleanup unused variables
                del pml_size_temp
            return options
