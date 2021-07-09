from dataclasses import dataclass
import numpy as np
from kwave_py.utils import get_h5_literals


@dataclass
class SimulationOptions(object):
    # =========================================================================
    # FLAGS WHICH CAN BE CONTROLLED WITH OPTIONAL INPUTS (THESE CAN BE MODIFIED)
    # =========================================================================

    # flags which control the behaviour of the simulations
    pml_inside                = True                 # put the PML inside the grid defined by the user
    save_to_disk              = False                # save the input data to a HDF5 file
    save_to_disk_exit         = True                 # exit the simulation after saving the HDF5 file
    scale_source_terms        = True                 # apply the source scaling term to time varying sources
    smooth_c0                 = False                # smooth the sound speed distribution
    smooth_rho0               = False                # smooth the density distribution
    smooth_p0                 = True                 # smooth the initial pressure distribution
    use_kspace                = True                 # use the k-space correction
    use_sg                    = True                 # use a staggered grid
    pml_auto                  = False                # automatically choose the PML size to give small prime factors
    create_log                = False                # create a log using diary
    use_finite_difference     = False                # use finite difference gradients instead of spectral (in 1D)
    stream_to_disk            = False                # buffer the sensor data to disk (in 3D)
    data_recast               = False                # recast the sensor data back to double precision

    # =========================================================================
    # VARIABLES THAT CAN BE CHANGED USING OPTIONAL INPUTS (THESE CAN BE MODIFIED)
    # =========================================================================

    # general settings
    cartesian_interp                = 'linear'                          # interpolation mode for Cartesian sensor mask
    hdf_compression_level           = None                              # zip compression level for HDF5 input files
    data_cast                       = 'off'                             # data cast
    pml_search_range                = [10, 40]                          # search range used when automatically determining PML size
    radial_symmetry                 = 'WSWA-FFT'                        # radial symmetry used in axisymmetric code
    multi_axial_PML_ratio           = 0.1                               # MPML settings

    # default PML properties and plot scale
    pml_x_alpha, pml_y_alpha, pml_z_alpha = None, None, None  # farid | for now
    pml_x_size, pml_y_size, pml_z_size = None, None, None  # farid | for now

    def __post_init__(self):
        # load the HDF5 literals (for the default compression level)
        h5_literals = get_h5_literals()
        self.hdf_compression_level = h5_literals.HDF_COMPRESSION_LEVEL

    @staticmethod
    def init(kgrid, elastic_code: bool, axisymmetric: bool, **kwargs):
        # =========================================================================
        # FIXED LITERALS USED IN THE CODE (THESE CAN BE MODIFIED)
        # =========================================================================

        # Literals used to set default parameters end with _DEF. These are cleared
        # at the end of kspaceFirstOrder_inputChecking. Literals used at other
        # places in the code are not cleared.

        # general
        STREAM_TO_DISK_STEPS_DEF        = 200                  # number of steps before streaming to disk

        # filenames
        SAVE_TO_DISK_FILENAME_DEF       = 'kwave_input_data.h5'

        options = SimulationOptions()

        if kgrid.dim  == 1:
            options.pml_x_alpha = 2
            options.pml_x_size = 20
            options.plot_scale = [-1.1, 1.1]
        elif kgrid.dim == 2:
            options.pml_x_alpha = 2
            options.pml_y_alpha = options.pml_x_alpha
            options.pml_x_size = 20
            options.pml_y_size = options.pml_x_size
            options.plot_scale = [-1, 1]
        elif kgrid.dim == 3:
            options.pml_x_alpha = 2
            options.pml_y_alpha = options.pml_x_alpha
            options.pml_z_alpha = options.pml_x_alpha
            options.pml_x_size = 10
            options.pml_y_size = options.pml_x_size
            options.pml_z_size = options.pml_x_size
            options.plot_scale = [-1, 1]

        # replace defaults with user defined values if provided and check inputs
        for key, val in kwargs.items():
            if key == 'CartInterp':
                assert val in ['linear', 'nearest'], \
                    "Optional input ''CartInterp'' must be set to ''linear'' or ''nearest''."
            elif key == 'CreateLog':
                # assign input
                options.create_log = val
                assert isinstance(val, bool), "Optional input ''CreateLog'' must be Boolean."
            elif key == 'DataCast':
                data_cast = val
                assert isinstance(data_cast, str), "Optional input ''DataCast'' must be a string."
                assert data_cast in ['off', 'double', 'single', 'gpuArray-single', 'gpuArray-double'], \
                    "Invalid input for ''DataCast''."

                # replace double with off
                if data_cast == 'double':
                    data_cast = 'off'

                # create empty string to hold extra cast variable for use with the parallel computing toolbox
                data_cast_prepend = ''

                # replace PCT options with gpuArray
                if data_cast == 'gpuArray-single':
                    data_cast = 'gpuArray'
                    data_cast_prepend = 'single'
                elif data_cast == 'gpuArray-double':
                    data_cast = 'gpuArray'

                if data_cast == 'gpuArray':
                    raise NotImplementedError("gpuArray is not supported in Python-version")
                options.data_cast = data_cast
                options.data_cast_prepend = data_cast_prepend

            elif key == 'DataRecast':
                options.data_recast = val
                assert isinstance(val, bool), "Optional input ''DataRecast'' must be Boolean."

            elif key == 'HDFCompressionLevel':
                # assign input
                hdf_compression_level = val

                # check value is an integer between 0 and 9
                assert isinstance(hdf_compression_level, int) and 0 <= hdf_compression_level <= 9, \
                    "Optional input ''HDFCompressionLevel'' must be an integer between 0 and 9."
                options.hdf_compression_level = hdf_compression_level

            elif key == 'MultiAxialPMLRatio':
                assert np.isscalar(val) and val >= 0, \
                    "Optional input ''MultiAxialPMLRatio'' must be a single positive value."
                # assign input
                options.multi_axial_PML_ratio = val

            elif key == 'PMLAlpha':
                # check input is correct size
                val = np.atleast_1d(val)
                if val.size > kgrid.dim:
                    if kgrid.dim > 1:
                        raise ValueError(f"Optional input ''PMLAlpha'' must be a 1 or {kgrid.dim} element numerical array.")
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

            elif key == 'PMLInside':
                assert isinstance(val, bool), "Optional input ''PMLInside'' must be Boolean."
                options.pml_inside = val

            elif key == 'PMLRange':
                options.pml_search_range = val

            elif key == 'PMLSize':
                if isinstance(val, str):
                    # check for 'auto'
                    if val == 'auto':
                        options.pml_auto = True
                    else:
                        raise ValueError(f"Optional input ''PMLSize'' must be a 1 or {kgrid.dim} element "
                                         f"numerical array, or set to ''auto''.")
                else:
                    val = np.atleast_1d(val)
                    if len(val) > kgrid.dim:
                        if kgrid.dim > 1:
                            raise ValueError(f"Optional input ''PMLSize'' must be a 1 or {kgrid.dim} element numerical array.")
                        else:
                            raise ValueError(f"Optional input ''PMLSize'' must be a single numerical value.")

                # assign input based on number of dimensions, rounding to
                # the nearest integer
                val = np.atleast_1d(np.squeeze(val))
                if kgrid.dim == 1:
                    options.pml_x_size = float(np.round(val))
                elif kgrid.dim == 2:
                    options.pml_x_size = float(np.round(val[0]))
                    options.pml_y_size = float(np.round(val[-1]))
                elif kgrid.dim == 3:
                    options.pml_x_size = float(np.round(val[0]))
                    options.pml_y_size = float(np.round(val[len(val) // 2]))
                    options.pml_z_size = float(np.round(val[-1]))

            elif key == 'RadialSymmetry':
                assert val in ['WSWA', 'WSWS', 'WSWA-FFT', 'WSWS-FFT'], \
                    "Optional input ''RadialSymmetry'' must be set to ''WSWA'', ''WSWS'', ''WSWA-FFT'', ''WSWS-FFT''."
                options.radial_symmetry = val

            elif key == 'StreamToDisk':
                assert not elastic_code and kgrid.dim == 3, \
                    "Optional input ''StreamToDisk'' is currently only compatible with 3D fluid simulations."

                assert np.isscalar(val) or isinstance(val, bool), \
                    "Optional input ''StreamToDisk'' must be a single scalar or Boolean value."

                options.stream_to_disk = val

                # if given as a Boolean, replace with the default number of time steps
                if isinstance(options.stream_to_disk, bool) and options.stream_to_disk:
                    options.stream_to_disk = STREAM_TO_DISK_STEPS_DEF

            elif key == 'SaveToDisk':
                assert kgrid.dim != 1, "Optional input ''SaveToDisk'' is not compatible with 1D simulations."

                assert isinstance(val, (bool, str)), "Optional input ''SaveToDisk'' must be Boolean or a String."
                options.save_to_disk = val

                if isinstance(options.save_to_disk, bool) and options.save_to_disk:
                    options.save_to_disk = SAVE_TO_DISK_FILENAME_DEF

            elif key == 'SaveToDiskExit':
                assert kgrid.dim != 1, "Optional input ''SaveToDisk'' is not compatible with 1D simulations."

                assert isinstance(val, bool), "Optional input ''SaveToDiskExit'' must be Boolean."
                options.save_to_disk_exit = val

            elif key == 'ScaleSourceTerms':
                assert isinstance(val, bool), "Optional input ''ScaleSourceTerms'' must be Boolean."
                options.scale_source_terms = val

            elif key == 'Smooth':
                val = np.atleast_1d(val)
                assert len(val) <= 3 and np.array(val).dtype == bool, "Optional input ''Smooth'' must be a 1, 2 or 3 element Boolean array."
                options.smooth_p0 = val[0]
                options.smooth_c0 = val[len(val) // 2]
                options.smooth_rho0 = val[-1]

            elif key == 'UseFD':
                # input only supported in 1D fluid code
                assert kgrid.dim == 1 and not elastic_code, "Optional input ''UseFD'' only supported in 1D."

                assert (isinstance(val, bool) and not val) or (np.issubdtype(val, np.number) and val in [2, 4]), \
                    "Optional input ''UseFD'' must be set to 2, 4, or false."
                options.use_finite_difference = val

            elif key == 'UsekSpace':
                assert isinstance(val, bool), "Optional input ''UsekSpace'' must be Boolean."
                options.use_kspace = val

            elif key == 'UseSG':
                assert isinstance(val, bool), "Optional input ''UseSG'' must be Boolean."
                options.use_sg = val

            elif key in ['DisplayMask', 'LogScale', 'MeshPlot', 'PlotFreq',
                         'PlotLayout', 'PlotScale', 'PlotSim', 'PlotPML']:
                raise NotImplementedError(f'Plotting is not supported! Parameter {key} is related to plotting.')

            else:
                raise NotImplementedError(f"Unknown optional input: {key}.")

        # automatically assign the PML size to give small prime factors
        if options.pml_auto:
            if options.pml_inside:
                raise NotImplementedError("''PMLSize'' set to ''auto'' is only supported with ''PMLInside'' set to false.")
            else:
                # get optimal pml size
                if axisymmetric:
                    pml_size_temp = getOptimalPMLSize(kgrid, options.pml_search_range, options.radial_symmetry[:4])
                else:
                    pml_size_temp = getOptimalPMLSize(kgrid, options.pml_search_range)

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
